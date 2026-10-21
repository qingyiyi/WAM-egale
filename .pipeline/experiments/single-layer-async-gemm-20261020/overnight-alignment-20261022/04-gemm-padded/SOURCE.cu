// 单层 Transformer：一次 cooperative launch，CTA 常驻并领取各阶段的 tile。
// 固定实验输入为 BF16 hidden/context，D=3072、24 heads、head_dim=128。
// 输入校验与结果核验在外部实验脚本；这里直接呈现计算和必要同步。
#include "norm.cuh" // pinned Torch Welford arithmetic, read-only helper
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cutlass/gemm/device/gemm.h>
#include "cutlass_gemm_mainloop.cuh"
#include <map>
#include <mma.h>
#include <optional>
#include <torch/extension.h>
#define FLASH_NAMESPACE wam_whole_fa2
#include "fa2_device_generated.cuh"

namespace wam_whole_layer_v1 {
namespace cg = cooperative_groups;
namespace wm = nvcuda::wmma;
using B = __nv_bfloat16;
using at::Tensor;
constexpr int D = 3072, H = 24, HD = 128, F = 14336, THREADS = 256;
constexpr int MAX_STAGES = 32; // 29个顺序阶段，30/31用于attention split合并。
using FaStandard = Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using FaCandidate = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
using FaSplit = Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t>;
constexpr int FA_SMEM =
    FaStandard::kSmemSize > FaSplit::kSmemSize ? FaStandard::kSmemSize : FaSplit::kSmemSize;
enum Buf {
    MOD,
    N1,
    M1,
    SQ0,
    SK0,
    SV,
    SQN,
    SKN,
    SQR,
    SKR,
    SA,
    SO,
    R1,
    N2,
    CQ0,
    CK0,
    CV,
    CQ,
    CK,
    CA,
    CO,
    R2,
    N3,
    M3,
    F1,
    ACT,
    F2,
    OUT,
    NB
};
enum Point { TABLE, MODULATE, RESIDUAL, ROPE, GELU };
// CTA内8个warp的固定分工：0领取tile；3–6计算；Norm/小M由7写回。
// 小M的1/2分别加载A/W；大M的CUTLASS由计算warp自行加载，epilogue用全部8warp。
// a[]是中间buffer，w[]对应whole_layer.py绑定的原始参数，均不跨候选混用。
struct Params {
    const void *w[27];
    void *a[NB];
    const B *x, *context;
    const void *temb;
    const c10::complex<float> *freq;
    B *pk, *pv;
    int64_t *ids, *rows, *slots, *free, *info;
    bool *mask, *pred;
    int *queues;
    int b, q, c, l, fb, pool, update;
    bool temb_float, cache, candidate, affine;
    float eps[7];
    float *fa_lse, *fa_lse_partial, *fa_out_partial;
    int64_t *fa_meta;
    int sms;
    float attn_scale, attn_scale_log2;
};
template <int TN, int TK> struct GemmStorage {
    B x[2][64 * (TK + 8)], w[2][TN * (TK + 8)];
    float y[64 * TN], partial_y[64 * TN];
};
using GemmShared = GemmStorage<64, 32>;
#include "cutlass_gemm.cuh"
struct NormShared {
    alignas(16) B input[D];
    float x[D], y[D], reduce[6];
};
union alignas(32) TileShared {
    GemmShared gemm;
    CutlassGemmStorage cutlass_gemm;
    NormShared norm;
    NormShared norm_pair[2]; // 两行各占4个warp；仍小于FA2已有的共享内存上限。
    char fa2_workspace[FA_SMEM];
};
struct Shared {
    TileShared tile;
    int task;
    wam_whole_fa2::WholeParams fa2;
};
__device__ inline int warp() {
    return threadIdx.x / 32;
}
__device__ inline int lane() {
    return threadIdx.x % 32;
}
__device__ inline bool compute() {
    return warp() >= 3 && warp() < 7;
}
__device__ inline int ct() {
    return int(threadIdx.x) - 96;
}
__device__ inline float value(B x) {
    return __bfloat162float(x);
}
__device__ inline B bf(float x) {
    return __float2bfloat16_rn(x);
}
__device__ inline float *fp(const Params &p, int i) {
    return (float *)p.a[i];
}
__device__ inline B *bp(const Params &p, int i) {
    return (B *)p.a[i];
}
__device__ inline void group_sync() {
    asm volatile("bar.sync 1, 128;" ::: "memory");
}

// 每个阶段有一个全局计数器；一个CTA一次领取一个tile，做完再领。
// 阶段开始/结束的grid.sync保证所有前置输出完成；目前还不是跨阶段ready队列。
__device__ void begin_stage(const Params &p, cg::grid_group &grid, int stage) {
    if (grid.thread_rank() == 0)
        p.queues[stage] = 0;
    grid.sync();
}
__device__ int claim_tile(const Params &p, Shared &s, int stage) {
    if (threadIdx.x == 0)
        s.task = atomicAdd(p.queues + stage, 1);
    __syncthreads(); // 将warp0领到的编号交给整个CTA。
    return s.task;
}
__device__ void finish_tile() {
    __syncthreads(); // 共享workspace不可在上一tile的读写结束前被复用。
}

// Small-M: 64x64 output / K32 with separate loader warps. Action M>=128:
// 64x128 / K64. Context M>=1024: compile-selected 64/128 x128 / K64.
// Every path retains increasing-K and the recorded BF16/bias boundaries.
#include "gemm_pipeline.cuh"
// B14的六个小投影：224行hidden，3072→3072。整个阶段只调用一次
// 固定形状设备函数，让编译器直接生成M64×N128的CUTLASS路径。
// 仍是96个输出tile、原两个K1536分段及BF16提交；领取和同步均不改。
__device__ __noinline__ void action_projection(const Params &p, Shared &s, cg::grid_group &g,
                                              int st, const B *a, int wi, B *out) {
    constexpr int M = 224, tiles = ((M + 63) / 64) * (D / 128);
    begin_stage(p, g, st);
    for (int t; (t = claim_tile(p, s, st)) < tiles;) {
        cutlass_gemm_tile<64>(s.tile.cutlass_gemm.action, a, (const B *)p.w[wi],
                             (const B *)p.w[wi + 1], out, M, D, D, t, true);
        finish_tile();
    }
    g.sync();
}

__device__ void linear(const Params &p, Shared &s, cg::grid_group &g, int st, const B *a, int wi,
                       B *out, int m, int n, int k) {
    if (p.candidate && m == 224 && n == D && k == D) {
        action_projection(p, s, g, st, a, wi, out);
        return;
    }
    const int tn = m >= 128 ? 128 : 64;
    const int tm = gemm_tile_rows(m, n);
    int ntiles = ((m + tm - 1) / tm) * ((n + tn - 1) / tn);
    begin_stage(p, g, st);
    for (int t; (t = claim_tile(p, s, st)) < ntiles;) {
        gemm_tile(s, a, (const B *)p.w[wi], (const B *)p.w[wi + 1], out, m, n, k, t, true);
        finish_tile();
    }
    g.sync();
}

// Context的N/K均固定3072，实际M为1024或7168，无分段BF16提交。
// 独立设备函数只移除通用GEMM的形状分派；tile、领取顺序和同步不变。
__device__ __noinline__ void context_linear(const Params &p, Shared &s, cg::grid_group &g,
                                           int st, int wi, B *out, int m) {
    const int ntiles = ((m + WAM_CONTEXT_TILE_M - 1) / WAM_CONTEXT_TILE_M) * (D / 128);
    begin_stage(p, g, st);
    for (int t; (t = claim_tile(p, s, st)) < ntiles;) {
        cutlass_gemm_tile<WAM_CONTEXT_TILE_M, 136>(s.tile.cutlass_gemm.context, p.context,
            (const B *)p.w[wi], (const B *)p.w[wi + 1], out, m, D, D, t, false);
        finish_tile();
    }
    g.sync();
}

// B14 FFN2：M128输出tile共48个，每个K分两段，共96个CTA任务。
// 各CTA独立算一段，FP32片段暂存在自己的shared；阶段代码负责有序提交。
__device__ __forceinline__ void ffn2_partition(CutlassTile<128>::Storage &s, const __nv_bfloat16 *a,
                                const __nv_bfloat16 *w, int tile, int partition) {
    using T = CutlassTile<128>; using Mma = T::Mma;
    const int physical_warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (physical_warp >= 3 && physical_warp < 7) {
        const int thread = int(threadIdx.x) - 96, warp = physical_warp - 3;
        typename Mma::IteratorA::Params ap{CutlassRow(14336)};
        typename Mma::IteratorB::Params bp{CutlassCol(14336)};
        typename Mma::IteratorA ai(ap, reinterpret_cast<CutlassB *>(const_cast<__nv_bfloat16 *>(a)) + partition * 7168,
                                  {224, 7168}, thread, {(tile / 24) * 128, 0});
        typename Mma::IteratorB bi(bp, reinterpret_cast<CutlassB *>(const_cast<__nv_bfloat16 *>(w)) + partition * 7168,
                                  {7168, 3072}, thread, {0, (tile % 24) * 128});
        Mma mma(s.mainloop, thread, warp, lane);
        typename Mma::FragmentC acc; acc.clear();
        mma(7168 / 64, acc, ai, bi, acc);
        asm volatile("bar.sync 2, 128;" ::: "memory");
        typename Mma::Operator::IteratorC store({s.result, CutlassRow(136)}, lane);
        store.add_tile_offset({warp % 2, warp / 2}); store.store(acc);
    }
    __syncthreads();
}

__device__ __forceinline__ void ffn2_commit(CutlassTile<128>::Storage &s, const __nv_bfloat16 *bias,
                             __nv_bfloat16 *out, int tile, bool second) {
    for (int i = 2 * int(threadIdx.x); i < 128 * 128; i += 2 * int(blockDim.x)) {
        const int row = (tile / 24) * 128 + i / 128, col = (tile % 24) * 128 + i % 128;
        if (row < 224) {
            float2 z = *reinterpret_cast<const float2 *>(s.result + (i / 128) * 136 + i % 128);
            if (second) {
                float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(bias + col));
                float2 prev = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(out + row * 3072 + col));
                z.x = __fadd_rn(__fadd_rn(z.x, b.x), prev.x);
                z.y = __fadd_rn(__fadd_rn(z.y, b.y), prev.y);
            }
            *reinterpret_cast<__nv_bfloat162 *>(out + row * 3072 + col) = __floats2bfloat162_rn(z.x, z.y);
        }
    }
    __syncthreads();
}

// 固定B14 FFN2的48个输出tile×两K分段。专用阶段函数包住整个过程，
// 两个数学体在函数内展开，避免再次从通用linear逐任务进入大设备函数。
// 只并行同一FFN2的两个原K分段；不是跨Transformer阶段ready调度。
__device__ __noinline__ void action_ffn2_parallel(const Params &p, Shared &s,
                                                cg::grid_group &g, int st) {
    begin_stage(p, g, st);
    const int task = int(blockIdx.x), tile = task % 48;
    auto &workspace = s.tile.cutlass_gemm.ffn1;
    if (task < 96)
        ffn2_partition(workspace, bp(p, ACT), (const B *)p.w[25], tile, task / 48);
    if (task < 48)
        ffn2_commit(workspace, (const B *)p.w[26], bp(p, F2), tile, false);
    g.sync(); // 第一段BF16可见；第二段FP32仍保存在各自CTA的shared。
    if (task >= 48 && task < 96)
        ffn2_commit(workspace, (const B *)p.w[26], bp(p, F2), tile, true);
    g.sync();
}

// FFN1只有224个M128×N128输出tile。216个保留大tile，最后8个
// 输出tile各拆成两个M64任务，缩短只有少量CTA工作的最后一波。
// 这是FFN1内部划分，不提前运行任何依赖它的Transformer阶段。
__device__ __noinline__ void action_ffn1_tail(const Params &p, Shared &s, cg::grid_group &g, int st) {
    begin_stage(p, g, st);
    for (int tile = int(blockIdx.x); tile < 216; tile += int(gridDim.x))
        cutlass_gemm_tile<128>(s.tile.cutlass_gemm.ffn1, bp(p, M3),
            (const B *)p.w[23], (const B *)p.w[24], bp(p, F1), 224, F, D, tile, true);
    g.sync();
    for (int tail = int(blockIdx.x); tail < 16; tail += int(gridDim.x)) {
        const int tile64 = (2 + tail / 8) * 112 + 104 + tail % 8;
        cutlass_gemm_tile<64>(s.tile.cutlass_gemm.action, bp(p, M3),
            (const B *)p.w[23], (const B *)p.w[24], bp(p, F1), 224, F, D, tile64, true);
    }
    g.sync();
}

#include "norm_device_generated.cuh"

// Norm tile为完整3072维行。8个warp搬入/写回，warp3–6仍按原Torch顺序归约。
// 仅扩大搬运参与度；FP32转换、Welford树与BF16舍入顺序保持原样。
__device__ void norm_tile(Shared &sh, const B *x, void *y, const void *gamma, const float *beta,
                          float eps, bool rms, bool output_float, int row) {
    auto &s = sh.tile.norm;
    {
        for (int i = threadIdx.x; i < D / 8; i += THREADS)
            copy16_async(s.input + i * 8, x + row * D + i * 8, true);
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    if (compute())
        for (int v = ct(); v < D / 4; v += 128) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                s.x[v * 4 + j] = value(s.input[v * 4 + j]);
        }
    __syncthreads();
    if (compute() && !rms) {
        whole_norm_forward<float, float, false>(D, eps, s.x, static_cast<const float *>(gamma),
                                                beta, nullptr, nullptr, s.y, s.reduce);
    } else if (compute()) {
        using namespace wam_optimized_norm;
        int tid = ct(), cw = tid / 32, cl = tid % 32;
        WelfordDataLN wd;
        for (int v = tid; v < D / 4; v += 128)
            for (int j = 0; j < 4; ++j)
                wd = rms ? cuWelfordOnlineSum<float, true>(s.x[v * 4 + j], wd)
                         : cuWelfordOnlineSum<float, false>(s.x[v * 4 + j], wd);
        for (int offset = 16; offset; offset >>= 1) {
            WelfordDataLN b{__shfl_down_sync(0xffffffff, wd.mean, offset),
                            __shfl_down_sync(0xffffffff, wd.sigma2, offset),
                            __shfl_down_sync(0xffffffff, wd.count, offset)};
            wd = rms ? cuWelfordCombine<true>(wd, b) : cuWelfordCombine<false>(wd, b);
        }
        for (int offset = 2; offset; offset >>= 1) {
            if (cl == 0 && cw >= offset && cw < 2 * offset) {
                int j = cw - offset;
                s.reduce[2 * j] = wd.mean;
                s.reduce[2 * j + 1] = wd.sigma2;
                s.reduce[4 + j] = wd.count;
            }
            group_sync();
            if (cl == 0 && cw < offset) {
                WelfordDataLN b{s.reduce[2 * cw], s.reduce[2 * cw + 1], s.reduce[4 + cw]};
                wd = rms ? cuWelfordCombine<true>(wd, b) : cuWelfordCombine<false>(wd, b);
            }
            group_sync();
        }
        if (tid == 0) {
            s.reduce[0] = wd.mean;
            s.reduce[1] = wd.sigma2 / float(D);
        }
        group_sync();
        float rstd = c10::cuda::compat::rsqrt(s.reduce[1] + eps), mean = s.reduce[0];
        for (int v = tid; v < D / 4; v += 128)
            for (int j = 0; j < 4; ++j) {
                int i = v * 4 + j;
                float z = rms ? rstd * s.x[i] : rstd * (s.x[i] - mean);
                if (gamma)
                    z = (rms ? value(((const B *)gamma)[i]) : ((const float *)gamma)[i]) * z;
                if (beta)
                    z = z + beta[i];
                s.y[i] = z;
            }
    }
    __syncthreads();
    {
        if (output_float) {
            for (int i = threadIdx.x; i < D / 4; i += THREADS)
                reinterpret_cast<float4 *>((float *)y + row * D)[i] =
                    reinterpret_cast<const float4 *>(s.y)[i];
        } else {
            for (int i = threadIdx.x; i < D / 8; i += THREADS) {
                uint32_t packed[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned lo = __bfloat16_as_ushort(bf(s.y[i * 8 + j * 2]));
                    const unsigned hi = __bfloat16_as_ushort(bf(s.y[i * 8 + j * 2 + 1]));
                    packed[j] = lo | (hi << 16);
                }
                reinterpret_cast<uint4 *>((B *)y + row * D)[i] =
                    make_uint4(packed[0], packed[1], packed[2], packed[3]);
            }
        }
    }
    __syncthreads();
}
// B14 RMS每个128线程组归约一行；256线程CTA的两个组分别处理两行。
// 行宽3072、vec4、行内归约树均与原版一致；逻辑任务数为ceil(rows/2)。
// 这里8个warp
// 分成两个独立的128线程组，各算一行。没有跨阶段调度，也不改变行内归约树。
__device__ inline void rms_pair_sync(int group) {
    if (group == 0)
        asm volatile("bar.sync 1, 128;" ::: "memory");
    else
        asm volatile("bar.sync 2, 128;" ::: "memory");
}
__device__ void rms_pair_tile(Shared &sh, const B *x, B *y, const B *gamma,
                              float eps, int first_row, int rows) {
    using namespace wam_optimized_norm;
    const int group = threadIdx.x / 128, tid = threadIdx.x % 128;
    const int row = first_row + group, cw = tid / 32, cl = tid % 32;
    auto &s = sh.tile.norm_pair[group];
    if (row < rows) {
        // 与原版一样按每线程4元素向量读取；BF16转FP32后按原顺序归约。
        // 不再先搬到shared，再展开成另一份FP32 shared tensor。
        using Vec = aligned_vector<B, 4>;
        const auto *input = reinterpret_cast<const Vec *>(x + row * D);
        WelfordDataLN wd;
        for (int v = tid; v < D / 4; v += 128) {
            Vec values = input[v];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                wd = cuWelfordOnlineSum<float, true>(value(values.val[j]), wd);
        }
        for (int offset = 16; offset; offset >>= 1) {
            WelfordDataLN b{__shfl_down_sync(0xffffffff, wd.mean, offset),
                            __shfl_down_sync(0xffffffff, wd.sigma2, offset),
                            __shfl_down_sync(0xffffffff, wd.count, offset)};
            wd = cuWelfordCombine<true>(wd, b);
        }
        for (int offset = 2; offset; offset >>= 1) {
            if (cl == 0 && cw >= offset && cw < 2 * offset) {
                int j = cw - offset;
                s.reduce[2 * j] = wd.mean;
                s.reduce[2 * j + 1] = wd.sigma2;
                s.reduce[4 + j] = wd.count;
            }
            rms_pair_sync(group);
            if (cl == 0 && cw < offset) {
                WelfordDataLN b{s.reduce[2 * cw], s.reduce[2 * cw + 1], s.reduce[4 + cw]};
                wd = cuWelfordCombine<true>(wd, b);
            }
            rms_pair_sync(group);
        }
        if (tid == 0)
            s.reduce[1] = wd.sigma2 / float(D);
        rms_pair_sync(group);
        float rstd = c10::cuda::compat::rsqrt(s.reduce[1] + eps);
        for (int v = tid; v < D / 4; v += 128) {
            Vec values = input[v], output;
            Vec weights;
            if (gamma) weights = reinterpret_cast<const Vec *>(gamma)[v];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float z = rstd * value(values.val[j]);
                if (gamma) z = value(weights.val[j]) * z;
                output.val[j] = bf(z);
            }
            reinterpret_cast<Vec *>(y + row * D)[v] = output;
        }
    }
    __syncthreads(); // 两行都写完后，整个CTA才复用共享空间。
}
__device__ void norm(const Params &p, Shared &s, cg::grid_group &g, int st, const B *x, void *y,
                     int rows, int gi, int bi, float eps, bool rms, bool out_float) {
    begin_stage(p, g, st);
    if (p.candidate && rms && !out_float && bi < 0) {
        for (int t = int(blockIdx.x); t < (rows + 1) / 2; t += int(gridDim.x)) {
            rms_pair_tile(s, x, (B *)y, gi >= 0 ? (const B *)p.w[gi] : nullptr,
                          eps, 2 * t, rows);
            // rms_pair_tile内部末尾已经同步整个CTA。
        }
        g.sync();
        return;
    }
    for (int t = int(blockIdx.x); t < rows; t += int(gridDim.x)) {
        norm_tile(s, x, y, gi >= 0 ? p.w[gi] : nullptr, bi >= 0 ? (const float *)p.w[bi] : nullptr,
                  eps, rms, out_float, t);
        // norm_tile内部末尾已经同步整个CTA。
    }
    g.sync();
}

__device__ void layout(const Params &p, Shared &s) {
    if (blockIdx.x != 0)
        return;
    if (threadIdx.x == 0) {
        p.info[0] = 0;
        p.info[1] = 0;
    }
    if (!p.cache)
        return;
    if (p.candidate) {
        // B14只读取公共历史。按物理slot顺序压紧mask，不排序、不淘汰、
        // 不把候选private suffix写入历史。8个warp各处理32个相邻slot。
        // workspace复用现有union；没有增加persistent CTA的shared占用。
        int *counts = reinterpret_cast<int *>(s.tile.norm.input);
        if (threadIdx.x == 0)
            s.task = 0;
        __syncthreads();
        for (int base = 0; base < p.pool; base += THREADS) {
            int i = base + threadIdx.x;
            bool valid = i < p.pool && p.mask[i];
            unsigned bits = __ballot_sync(0xffffffff, valid);
            if (lane() == 0)
                counts[warp()] = __popc(bits);
            __syncthreads();
            int before = 0;
            for (int w = 0; w < warp(); ++w)
                before += counts[w];
            if (valid)
                p.slots[s.task + before + __popc(bits & ((1u << lane()) - 1))] = i;
            __syncthreads();
            if (threadIdx.x == 0)
                for (int w = 0; w < 8; ++w)
                    s.task += counts[w];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            p.info[0] = s.task;
        return;
    }
    // B2原始路径有真实cache写入和稳定淘汰，暂时保留已对齐的实现。
    if (threadIdx.x != 0)
        return;
    if (!p.candidate) {
        int free = 0;
        for (int i = 0; i < p.pool; ++i)
            if (!p.mask[i])
                ++free;
        int need = max(0, p.q - free);
        // Stable sort equivalent for the needed smallest IDs; ties use physical
        // slot order. It is intentionally a slow first correctness path.
        for (int e = 0; e < need; ++e) {
            int chosen = -1;
            int64_t low = INT64_MAX;
            for (int i = 0; i < p.pool; ++i)
                if (p.mask[i] && (chosen < 0 || p.ids[i] < low)) {
                    chosen = i;
                    low = p.ids[i];
                }
            p.mask[chosen] = false;
            p.ids[chosen] = -1;
        }
        int n = 0;
        int64_t high = -1;
        for (int i = 0; i < p.pool; ++i) {
            if (p.mask[i])
                high = max(high, p.ids[i]);
            else if (n < p.q)
                p.free[n++] = i;
        }
        for (int j = 0; j < p.q; ++j) {
            int i = p.free[j];
            p.mask[i] = true;
            p.ids[i] = high + 1;
            p.pred[i] = (p.update == 1);
        }
    }
    int n = 0;
    for (int i = 0; i < p.pool; ++i)
        if (p.mask[i])
            p.slots[n++] = i;
    p.info[0] = n;
}

struct alignas(8) Point4 { B value[4]; };

// 固定D3072的pointwise：按类型向量搬运；每元素公式、精度和任务队列不变。
__device__ void point(const Params &p, Shared &s, cg::grid_group &g, int st, Point op, int dst,
                      int a, int b, int param, int64_t count) {
    constexpr int tile_elements = 2048;
    const int tiles = (count + tile_elements - 1) / tile_elements;
    begin_stage(p, g, st);
    // 每个tile独立；按CTA编号固定分配，省去每tile原子领取/广播。
    for (int task = int(blockIdx.x); task < tiles; task += int(gridDim.x)) {
        if (op == TABLE && (p.temb_float || p.l != p.q)) {
            // 非逐token/BF16输入的兼容分支，保持原广播寻址与FP32相加。
            for (int j = threadIdx.x; j < tile_elements; j += THREADS) {
                int64_t i = int64_t(task) * tile_elements + j;
                if (i < count) {
                    int cell = i % (6 * D), row = i / (6 * D), batch = row / p.q, token = row % p.q;
                    int64_t src = (int64_t(batch) * p.l + (p.l == 1 ? 0 : token)) * 6 * D + cell;
                    float v = p.temb_float ? ((const float *)p.temb)[src] : value(((const B *)p.temb)[src]);
                    fp(p, dst)[i] = __fadd_rn(((const float *)p.w[0])[cell], v);
                }
            }
        } else if (op == ROPE) {
            // count以复数对计数；一次搬两个复数对。原模型的FP64乘加保持不变。
            for (int j = 2 * int(threadIdx.x); j < tile_elements; j += 2 * THREADS) {
                int64_t i = int64_t(task) * tile_elements + j;
                if (i < count) {
                    Point4 v = *reinterpret_cast<const Point4 *>(bp(p, a) + 2 * i), out;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        int64_t pair = i + c, r = pair / 64, batch = r / (p.q * H), token = (r / H) % p.q, dim = pair % 64;
                        auto f = p.freq[((p.fb == 1 ? 0 : batch) * p.q + token) * 64 + dim];
                        double x = double(value(v.value[2 * c])), y = double(value(v.value[2 * c + 1]));
                        out.value[2 * c] = bf(float(x * double(f.real()) - y * double(f.imag())));
                        out.value[2 * c + 1] = bf(float(x * double(f.imag()) + y * double(f.real())));
                    }
                    *reinterpret_cast<Point4 *>(bp(p, dst) + 2 * i) = out;
                }
            }
        } else {
            for (int j = 4 * int(threadIdx.x); j < tile_elements; j += 4 * THREADS) {
                int64_t i = int64_t(task) * tile_elements + j;
                if (i < count) {
                    Point4 x, y, out; float4 xf, shift, scale;
                    if (op == MODULATE) {
                        xf = *reinterpret_cast<const float4 *>(fp(p, a) + i);
                        int64_t mod = (i / D * 6 + param) * D + i % D;
                        shift = *reinterpret_cast<const float4 *>(fp(p, MOD) + mod);
                        scale = *reinterpret_cast<const float4 *>(fp(p, MOD) + mod + D);
                    } else {
                        const B *input = op == TABLE ? (const B *)p.temb : a < 0 ? p.x : bp(p, a);
                        x = *reinterpret_cast<const Point4 *>(input + i);
                    }
                    if (op == TABLE) shift = *reinterpret_cast<const float4 *>((const float *)p.w[0] + i % (6 * D));
                    if (op == RESIDUAL) {
                        y = *reinterpret_cast<const Point4 *>(bp(p, b) + i);
                        if (param >= 0) scale = *reinterpret_cast<const float4 *>(fp(p, MOD) + (i / D * 6 + param) * D + i % D);
                    }
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        if (op == TABLE) ((float *)&shift)[c] = __fadd_rn(((float *)&shift)[c], value(x.value[c]));
                        else if (op == MODULATE)
                            out.value[c] = bf(__fadd_rn(__fmul_rn(((float *)&xf)[c], __fadd_rn(1.f, ((float *)&scale)[c])), ((float *)&shift)[c]));
                        else if (op == RESIDUAL) {
                            float v = value(y.value[c]);
                            if (param >= 0) v = __fmul_rn(v, ((float *)&scale)[c]);
                            out.value[c] = bf(__fadd_rn(value(x.value[c]), v));
                        } else {
                            float v = value(x.value[c]);
                            constexpr float beta = M_SQRT2 * M_2_SQRTPI * float(.5), kappa = .044715;
                            float cube = v * v * v, inner = beta * (v + kappa * cube);
                            out.value[c] = bf(.5f * v * (1.f + c10::cuda::compat::tanh(inner)));
                        }
                    }
                    if (op == TABLE) *reinterpret_cast<float4 *>(fp(p, dst) + i) = shift;
                    else *reinterpret_cast<Point4 *>(bp(p, dst) + i) = out;
                }
            }
        }
        // 此阶段只使用线程私有寄存器和互不重叠的输出，无shared复用屏障。
    }
    g.sync();
}
__device__ void insert_cache(const Params &p, Shared &s, cg::grid_group &g, int st) {
    int count = (p.cache && !p.candidate) ? p.b * p.q * D : 0;
    constexpr int tile_elements = 2048;
    int tiles = (count + tile_elements - 1) / tile_elements;
    begin_stage(p, g, st);
    for (int t; (t = claim_tile(p, s, st)) < tiles;) {
        if (warp() == 1 || warp() == 2)
            for (int i = t * tile_elements + lane(); i < min(count, (t + 1) * tile_elements);
                 i += 32) {
                int64_t dest =
                    ((int64_t(i / (p.q * D)) * p.pool + p.free[(i / D) % p.q]) * D + i % D);
                if (warp() == 1)
                    p.pk[dest] = bp(p, SKR)[i];
                else
                    p.pv[dest] = bp(p, SV)[i];
            }
        finish_tile();
    }
    g.sync();
}
// Query tiles always stay within one (candidate, head) instance. B14 is NOT a
// Q112 sequence. Candidate private suffix visibility is preserved explicitly.
#include "fa2_stage.cuh"

// All threads read the same immutable launch descriptors. Avoid an implicit
// per-thread local copy when a device helper takes Params by reference.
__global__ void whole_layer(const __grid_constant__ Params p) {
    extern __shared__ __align__(32) unsigned char memory[];
    Shared &s = *reinterpret_cast<Shared *>(memory);
    auto g = cg::this_grid();
    // 缓存只整理本次可见slot；B14不会把候选的private KV写入公共历史。
    layout(p, s);
    g.sync();
    int st = 0, m = p.b * p.q, mc = p.b * p.c;
    // 1. 按token生成6组调制量，得到self-attention的输入。
    point(p, s, g, st++, TABLE, MOD, 0, 0, 0, int64_t(m) * 6 * D);
    norm(p, s, g, st++, p.x, p.a[N1], m, -1, -1, p.eps[0], false, true);
    point(p, s, g, st++, MODULATE, M1, N1, 0, 0, int64_t(m) * D);
    // 2. Self attention：Q/K/V投影→Q/K的RMSNorm与RoPE→FA2→输出投影。
    linear(p, s, g, st++, bp(p, M1), 3, bp(p, SQ0), m, D, D);
    linear(p, s, g, st++, bp(p, M1), 5, bp(p, SK0), m, D, D);
    linear(p, s, g, st++, bp(p, M1), 7, bp(p, SV), m, D, D);
    norm(p, s, g, st++, bp(p, SQ0), p.a[SQN], m, 11, -1, p.eps[3], true, false);
    norm(p, s, g, st++, bp(p, SK0), p.a[SKN], m, 12, -1, p.eps[4], true, false);
    if (p.freq) {
        point(p, s, g, st++, ROPE, SQR, SQN, 0, 0, int64_t(m) * D / 2);
        point(p, s, g, st++, ROPE, SKR, SKN, 0, 0, int64_t(m) * D / 2);
    }
    insert_cache(p, s, g, st++);
    attention(p, s, g, st++, false);
    linear(p, s, g, st++, bp(p, SA), 9, bp(p, SO), m, D, D);
    // The external hidden input has an explicit operand, not a copied Params.
    point(p, s, g, st++, RESIDUAL, R1, -1, SO, 2, int64_t(m) * D);
    // 3. Cross attention：Q来自当前hidden；K/V只来自context。
    // 当前依次执行；K/V尚未提前作为ready任务填入其他阶段的尾波。
    if (p.affine)
        norm(p, s, g, st++, bp(p, R1), p.a[N2], m, 1, 2, p.eps[1], false, false);
    linear(p, s, g, st++, bp(p, N2), 13, bp(p, CQ0), m, D, D);
    context_linear(p, s, g, st++, 15, bp(p, CK0), mc);
    context_linear(p, s, g, st++, 17, bp(p, CV), mc);
    norm(p, s, g, st++, bp(p, CQ0), p.a[CQ], m, 21, -1, p.eps[5], true, false);
    norm(p, s, g, st++, bp(p, CK0), p.a[CK], mc, 22, -1, p.eps[6], true, false);
    attention(p, s, g, st++, true);
    linear(p, s, g, st++, bp(p, CA), 19, bp(p, CO), m, D, D);
    point(p, s, g, st++, RESIDUAL, R2, R1, CO, -1, int64_t(m) * D);
    // 4. FFN：3072→14336→3072；保留GEMM和GELU之间原有BF16舍入。
    norm(p, s, g, st++, bp(p, R2), p.a[N3], m, -1, -1, p.eps[2], false, true);
    point(p, s, g, st++, MODULATE, M3, N3, 0, 3, int64_t(m) * D);
    if (p.candidate && m == 224)
        action_ffn1_tail(p, s, g, st++);
    else
        linear(p, s, g, st++, bp(p, M3), 23, bp(p, F1), m, F, D);
    point(p, s, g, st++, GELU, ACT, F1, 0, 0, int64_t(m) * F);
    if (p.candidate && m == 224)
        action_ffn2_parallel(p, s, g, st++);
    else
        linear(p, s, g, st++, bp(p, ACT), 25, bp(p, F2), m, D, F);
    // 5. 门控残差与临时KV状态恢复，输出仍为[B,Q,3072]。
    point(p, s, g, st++, RESIDUAL, OUT, R2, F2, 5, int64_t(m) * D);
    if (p.cache && !p.candidate && p.update == 0 && blockIdx.x == 0 && warp() == 0)
        for (int j = lane(); j < p.q; j += 32)
            p.mask[p.free[j]] = false;
    g.sync();
}

// Host只绑定指针、分配workspace并发射一次kernel，不执行GEMM/attention。
// 调用方提供固定实验中的合法输入；shape/dtype/RNG/KV检查放在外部脚本。
std::map<std::string, Tensor>
forward(const std::vector<Tensor> &w, const Tensor &x, const Tensor &context, const Tensor &temb,
        const std::optional<Tensor> &freq, const std::vector<Tensor> &cache,
        const std::optional<Tensor> &rows, int update, const std::vector<double> &eps,
        bool debug) {
    c10::cuda::CUDAGuard guard(x.device());
    Params p{};
    p.b = x.size(0);
    p.q = x.size(1);
    p.c = context.size(1);
    p.l = temb.size(1);
    p.update = update;
    p.x = (const B *)x.data_ptr();
    p.context = (const B *)context.data_ptr();
    p.temb = temb.data_ptr();
    p.temb_float = temb.scalar_type() == at::kFloat;
    for (int i = 0; i < 27; ++i)
        p.w[i] = w[i].data_ptr();
    for (int i = 0; i < 7; ++i)
        p.eps[i] = eps[i];
    p.affine = w[1].numel() != 0;
    if (freq) {
        p.freq = freq->data_ptr<c10::complex<float>>();
        p.fb = freq->size(0);
    }
    if (!cache.empty()) {
        p.cache = true;
        p.pk = (B *)cache[0].data_ptr();
        p.pv = (B *)cache[1].data_ptr();
        p.ids = cache[2].data_ptr<int64_t>();
        p.mask = cache[3].data_ptr<bool>();
        p.pred = cache[4].data_ptr<bool>();
        p.pool = cache[0].size(1);
        p.candidate = rows.has_value();
        if (rows)
            p.rows = rows->data_ptr<int64_t>();
    }

    // 所有中间量按原BF16/FP32边界落地；当前版本没有把整层中间量全部留在shared。
    const auto bo = x.options(), fo = bo.dtype(at::kFloat), lo = bo.dtype(at::kLong),
               io = bo.dtype(at::kInt);
    std::vector<Tensor> b(NB);
    for (int i = 0; i < NB; ++i) {
        const int rows = (i == CK0 || i == CK || i == CV) ? p.b * p.c : p.b * p.q;
        if (i == MOD)
            b[i] = at::empty({rows, 6, D}, fo);
        else
            b[i] = at::empty({rows, (i == F1 || i == ACT) ? F : D}, (i == N1 || i == N3) ? fo : bo);
        p.a[i] = b[i].data_ptr();
    }
    if (!p.freq) {
        b[SQR] = b[SQN];
        b[SKR] = b[SKN];
        p.a[SQR] = p.a[SQN];
        p.a[SKR] = p.a[SKN];
    }
    if (!p.affine) {
        b[N2] = b[R1];
        p.a[N2] = p.a[R1];
    }

    // slot表和任务计数属于执行所需状态；逐tile visits/stats审计已移出核心。
    auto slots = at::empty({std::max(p.pool, 1)}, lo), free = at::empty({p.q}, lo),
         info = at::empty({2}, lo);
    auto queues = at::empty({MAX_STAGES}, io);
    p.slots = slots.data_ptr<int64_t>();
    p.free = free.data_ptr<int64_t>();
    p.info = info.data_ptr<int64_t>();
    p.queues = queues.data_ptr<int>();
    p.sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    p.attn_scale = float(1.0 / std::sqrt(128.0));
    p.attn_scale_log2 = float(double(p.attn_scale) * M_LOG2E);

    // FA2仅在小batch时需要split-KV；B14正常路径num_splits=1。
    const int max_splits = (p.b * H * ((p.q + 63) / 64) >= 0.8f * p.sms * 2)
                               ? 1
                               : std::min(128, (std::max(p.pool + p.q, p.c) + 127) / 128);
    auto lse = at::empty({p.b, H, p.q}, fo), partial_lse = at::empty({max_splits, p.b, H, p.q}, fo);
    auto partial_out = at::empty({max_splits, p.b, H, p.q, HD}, fo),
         fa_meta = at::empty({2, 4}, lo);
    p.fa_lse = lse.data_ptr<float>();
    p.fa_lse_partial = partial_lse.data_ptr<float>();
    p.fa_out_partial = partial_out.data_ptr<float>();
    p.fa_meta = fa_meta.data_ptr<int64_t>();

    // Cooperative grid必须全部驻留，否则grid.sync无法完成；按实际资源计算CTA数量。
    int active = 0;
    C10_CUDA_CHECK(cudaFuncSetAttribute(whole_layer, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        sizeof(Shared)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, whole_layer, THREADS,
                                                                 sizeof(Shared)));
    const int blocks = p.sms * active;
    void *launch_args[] = {&p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)whole_layer, dim3(blocks), dim3(THREADS),
                                               launch_args, sizeof(Shared),
                                               at::cuda::getCurrentCUDAStream()));

    // 普通路径到此结束；中间量仍是计算workspace，但不再构造审计字典/资源统计。
    std::map<std::string, Tensor> result{{"output", b[OUT].view(x.sizes())}};
    if (!debug)
        return result;

    // debug只暴露已经存在的中间量，外部脚本据此核验，不在kernel里重复检查。
    const char *names[NB] = {"modulation",    "norm1",       "modulated1",     "self_q_linear",
                             "self_k_linear", "self_v",      "self_q_norm",    "self_k_norm",
                             "self_q_rope",   "self_k_rope", "self_attention", "self_out",
                             "residual1",     "norm2",       "cross_q_linear", "cross_k_linear",
                             "cross_v",       "cross_q",     "cross_k",        "cross_attention",
                             "cross_out",     "residual2",   "norm3",          "modulated3",
                             "ffn1",          "gelu",        "ffn2",           "output"};
    for (int i = 0; i < NB; ++i)
        result[names[i]] = b[i];
    result["output"] = b[OUT].view(x.sizes());
    result["cache_info"] = info;
    result["logical_slots"] = slots;
    result["attention_meta"] = fa_meta;
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, whole_layer));
    result["resources"] =
        at::tensor({int64_t(blocks), int64_t(active), int64_t(sizeof(Shared)),
                    int64_t(attr.numRegs), int64_t(attr.localSizeBytes), int64_t(THREADS)},
                   at::TensorOptions().dtype(at::kLong));
    return result;
}
} // namespace wam_whole_layer_v1
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 原始扩展调用默认保留debug字典兼容旧重放脚本；Python普通入口显式传false。
    m.def("forward", &wam_whole_layer_v1::forward,
          py::arg("weights"), py::arg("hidden"), py::arg("context"), py::arg("temb"),
          py::arg("rotary"), py::arg("cache"), py::arg("rows"), py::arg("update"),
          py::arg("eps"), py::arg("debug") = true);
    m.attr("production_admitted") = false;
    m.attr("computation_launches") = 1;
    m.attr("context_tile_rows") = WAM_CONTEXT_TILE_M;
    m.attr("point_tile_elements") = 2048;
    m.attr("cooperative_epilogue") = true;
}

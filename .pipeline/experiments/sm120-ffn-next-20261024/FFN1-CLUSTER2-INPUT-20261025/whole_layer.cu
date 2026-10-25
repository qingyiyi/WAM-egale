#ifndef FFN1_SHARE_A
#define FFN1_SHARE_A 1
#endif
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
#include <cuda.h>
#include <cuda/atomic>
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
// CTA共8个warp：SM120 action投影W0做TMA、W1–3做epilogue、W4–7做MMA；
// FFN1采用相同1T/4M/3E、完整K后做GELU；B14 attention分两个4warp组，RMS分两组各算一行。
// 动态队列由warp0领取；静态point/norm/FFN/双组attention不经过队列。
// a[]是中间buffer，w[]对应whole_layer.py绑定的原始参数，均不跨候选混用。
struct Params {
    alignas(64) CUtensorMap projection_maps[7][2];
    int64_t *ffn1_role_trace;
    int64_t *ffn1_transfer_trace;
    int64_t *projection_trace;
    int64_t *projection_role_trace;
    const void *w[27];
    void *a[NB];
    const B *x, *context;
    const void *temb;
    const c10::complex<float> *freq;
    B *pk, *pv;
    int64_t *ids, *rows, *slots, *free, *info;
    const int64_t *context_rows, *context_roots; // 条件分组独立于self历史映射。
    bool *mask, *pred;
    int *queues, *ffn_state;
    int64_t *ffn_trace; // 仅debug记录四组就绪/开始/完成时间，不参与计时路径。
    int b, q, c, cb, l, fb, pool, update;
    bool temb_float, cache, candidate, affine, debug_modulation, context_cached;
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
// 两个K64输入槽与分段结果同时活跃；不与输出union别名。
struct alignas(1024) ProjectionTmaStorage {
    CutlassB a[2][64 * 64], b[2][128 * 64];
    float result[64 * 136];
    alignas(8) uint64_t ready[2], empty[2], result_full, result_empty, peer_ready[2];
};

struct NormShared {
    alignas(16) B input[D];
    float x[D], y[D], reduce[6];
};
union alignas(1024) TileShared {
    GemmShared gemm;
    CutlassGemmStorage cutlass_gemm;
    ProjectionTmaStorage projection_tma;
    NormShared norm;
    NormShared norm_pair[2]; // 两行各占4个warp；仍小于FA2已有的共享内存上限。
    char fa2_workspace[FA_SMEM];
    char fa2_dual[2][FaCandidate::kSmemSize]; // 两个4-warp组各有独立FA2 workspace。
};
struct Shared {
    TileShared tile;
    int task;
    wam_whole_fa2::WholeParams fa2;
    wam_whole_fa2::GroupParams group[2];
};
__device__ inline int warp() { return threadIdx.x / 32; }
__device__ inline int lane() { return threadIdx.x % 32; }
__device__ inline bool compute() { return warp() >= 3 && warp() < 7; }
__device__ inline int ct() { return int(threadIdx.x) - 96; }
__device__ inline float value(B x) { return __bfloat162float(x); }
__device__ inline B bf(float x) { return __float2bfloat16_rn(x); }
// 输入已按原FFN1输出舍入为BF16；融合只改变数据传递，不跨越舍入边界。
__device__ __forceinline__ B gelu_value(B x) {
    float v = value(x);
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * float(.5), kappa = .044715;
    float cube = v * v * v, inner = beta * (v + kappa * cube);
    return bf(.5f * v * (1.f + c10::cuda::compat::tanh(inner)));
}
__device__ inline float *fp(const Params &p, int i) { return (float *)p.a[i]; }
__device__ inline B *bp(const Params &p, int i) { return (B *)p.a[i]; }
__device__ inline void group_sync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }

// 仅动态分派阶段使用全局计数器；每CTA一次领取一个tile。
// 静态阶段复用上一阶段的出口依赖同步，仍不是跨阶段ready队列。
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
// Q/K/V只依赖同一份M1，合并独立tile队列；三份权重和输出仍各自独立。
// 仅B14 M224使用：3×96任务、原64×128 tile和两段BF16边界。

__device__ __forceinline__ unsigned projection_smem(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void projection_wait(uint64_t *barrier, unsigned phase) {
    unsigned done;
    do {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
                     "selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(projection_smem(barrier)), "r"(phase) : "memory");
    } while (!done);
}
// 一个线程发起A/B两次TMA；24KiB合计完成后ready才翻转phase。
__device__ __forceinline__ void projection_load(const Params &p, ProjectionTmaStorage &s,
                                               int projection, int step, int row, int col) {
    const int slot = step & 1;
    const unsigned barrier = projection_smem(&s.ready[slot]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier), "r"(24576) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(projection_smem(s.a[slot])), "l"(&p.projection_maps[projection][0]),
                    "r"(step * 64), "r"(row), "r"(barrier) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(projection_smem(s.b[slot])), "l"(&p.projection_maps[projection][1]),
                    "r"(step * 64), "r"(col), "r"(barrier) : "memory");
}

// 每个缓冲分别交接：TMA完成 -> ready；全部MMA warp读完 -> empty。
__device__ __forceinline__ void projection_arrive(uint64_t *barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(projection_smem(barrier)) : "memory");
}
__device__ __forceinline__ void projection_epilogue(ProjectionTmaStorage &s,
        const B *bias, B *out, int rm, int cn, int partition, int thread) {
    constexpr int m = 224, n = 3072, splits = 10;
            // 四元素向量仅减少搬运指令；FP32加法及BF16舍入仍逐元素保持。
            struct alignas(8) B4 {
                __nv_bfloat162 lo, hi;
            };
            for (int i = 4 * thread; i < 64 * 128; i += 4 * 96) {
                const int r = rm + i / 128, c = cn + i % 128;
                if (r < m && c < n) {
                    float4 z =
                        *reinterpret_cast<const float4 *>(s.result + (i / 128) * 136 + i % 128);
#if __CUDA_ARCH__ < 1200
                    if (partition == splits - 1) {
                        B4 b = *reinterpret_cast<const B4 *>(bias + c);
                        float2 lo = __bfloat1622float2(b.lo), hi = __bfloat1622float2(b.hi);
                        z.x = __fadd_rn(z.x, lo.x);
                        z.y = __fadd_rn(z.y, lo.y);
                        z.z = __fadd_rn(z.z, hi.x);
                        z.w = __fadd_rn(z.w, hi.y);
                    }
#endif
                    if (partition > 0) {
                        B4 p = *reinterpret_cast<const B4 *>(out + r * n + c);
                        float2 lo = __bfloat1622float2(p.lo), hi = __bfloat1622float2(p.hi);
                        z.x = __fadd_rn(z.x, lo.x);
                        z.y = __fadd_rn(z.y, lo.y);
                        z.z = __fadd_rn(z.z, hi.x);
                        z.w = __fadd_rn(z.w, hi.y);
                    }
#if __CUDA_ARCH__ >= 1200
                    if (partition == splits - 1) {
                        B4 b = *reinterpret_cast<const B4 *>(bias + c);
                        float2 lo = __bfloat1622float2(b.lo), hi = __bfloat1622float2(b.hi);
                        z.x = __fadd_rn(z.x, lo.x);
                        z.y = __fadd_rn(z.y, lo.y);
                        z.z = __fadd_rn(z.z, hi.x);
                        z.w = __fadd_rn(z.w, hi.y);
                    }
#endif
                    *reinterpret_cast<B4 *>(out + r * n + c) = {__floats2bfloat162_rn(z.x, z.y),
                                                                __floats2bfloat162_rn(z.z, z.w)};
                }
            }

}

// W0: TMA；W1–3: BF16 epilogue；W4–7: 四warp MMA。
// 单个shared结果槽：MMA可以计算下一段，但覆盖结果前必须等epilogue释放。
__device__ __forceinline__ void projection_tma_tile(const Params &p, ProjectionTmaStorage &s,
        int projection, const B *bias, B *out, int tile) {
    using Op = typename CutlassTile<64>::Mma::Operator;
    const int rm = (tile / 24) * 64, cn = (tile % 24) * 128;
    const int w = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(projection_smem(&s.ready[i])) : "memory");
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;"
                         :: "r"(projection_smem(&s.empty[i])) : "memory");
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;"
                     :: "r"(projection_smem(&s.result_full)) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 3;"
                     :: "r"(projection_smem(&s.result_empty)) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads(); // 只在角色进入流水前共同初始化。
    int waits = 0, releases = 0, finished = 0;
    if (w == 0) {
        for (int step = 0; step < 48; ++step) {
            if (step >= 2) {
                projection_wait(&s.empty[step & 1], ((step / 2) - 1) & 1);
                ++releases;
            }
            if (lane == 0) projection_load(p, s, projection, step, rm, cn);
            ++waits;
            __syncwarp();
        }
        // 退出前确认最后两块都被消费者读完，不提前失效仍活跃的barrier。
        projection_wait(&s.empty[0], 1); projection_wait(&s.empty[1], 1);
        releases += 2;
        finished = 96; // 每个K64各一次A和B。
        if (lane == 0 && p.projection_trace) {
            auto dst = p.projection_trace + (projection * 96 + tile) * 4;
            dst[0] = waits; dst[1] = waits;
        }
    } else if (w >= 4) {
        const int warp = w - 4;
        int step = 0;
        for (int partition = 0; partition < 10; ++partition) {
            typename Op::FragmentC accum;
            accum.clear();
            const int iterations = partition == 9 ? 3 : 5;
            for (int k = 0; k < iterations; ++k, ++step) {
                const int slot = step & 1;
                // 所有MMA线程获得acquire；无整个CTA集合。
                projection_wait(&s.ready[slot], (step / 2) & 1);
                ++waits;
                typename Op::IteratorA ai({s.a[slot], typename Op::LayoutA(64)}, lane);
                typename Op::IteratorB bi({s.b[slot], typename Op::LayoutB(64)}, lane);
                ai.add_tile_offset({warp % 2, 0});
                bi.add_tile_offset({0, warp / 2});
                typename Op::FragmentA af[2];
                typename Op::FragmentB bf[2];
                ai.set_kgroup_index(0); bi.set_kgroup_index(0);
                ai.load(af[0]); bi.load(bf[0]); ++ai; ++bi;
                Op mma;
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    if (group < 3) {
                        ai.set_kgroup_index(group + 1); bi.set_kgroup_index(group + 1);
                        ai.load(af[(group + 1) & 1]); bi.load(bf[(group + 1) & 1]); ++ai; ++bi;
                    }
                    if (group == 2) {
                        // 最后shared读取已完成，剩余group2/3只读寄存器。
                        __syncwarp();
                        if (lane == 0) projection_arrive(&s.empty[slot]);
                        ++releases;
                    }
                    typename Op::TransformedFragmentA at;
                    typename Op::TransformedFragmentB bt;
                    mma.transform(at, bt, af[group & 1], bf[group & 1]);
                    mma(accum, at, bt, accum);
                }
            }
            if (partition > 0) projection_wait(&s.result_empty, (partition - 1) & 1);
            typename Op::IteratorC store({s.result, CutlassRow(136)}, lane);
            store.add_tile_offset({warp % 2, warp / 2});
            store.store(accum);
            __syncwarp();
            if (lane == 0) projection_arrive(&s.result_full);
            ++finished;
        }
        if (w == 4 && lane == 0 && p.projection_trace)
            p.projection_trace[(projection * 96 + tile) * 4 + 3] = waits;
    } else {
        for (int partition = 0; partition < 10; ++partition) {
            projection_wait(&s.result_full, partition & 1);
            ++waits;
            projection_epilogue(s, bias, out, rm, cn, partition, int(threadIdx.x) - 32);
            __syncwarp();
            if (lane == 0) projection_arrive(&s.result_empty);
            ++releases; ++finished;
        }
        if (w == 1 && lane == 0 && p.projection_trace)
            p.projection_trace[(projection * 96 + tile) * 4 + 2] = finished;
    }
    if (lane == 0 && p.projection_role_trace) {
        auto dst = p.projection_role_trace + ((projection * 96 + tile) * 8 + w) * 3;
        dst[0] = waits; dst[1] = releases; dst[2] = finished;
    }
    __syncthreads(); // 全部角色结束，才允许整层workspace复用。
    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; ++i) {
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&s.ready[i])) : "memory");
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&s.empty[i])) : "memory");
        }
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&s.result_full)) : "memory");
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&s.result_empty)) : "memory");
    }
    __syncthreads();
}

__device__ __noinline__ void self_qkv_projection_group(const Params &p, Shared &s,
                                                     cg::grid_group &g, int st, const B *a) {
    constexpr int per_projection = 96;
#if __CUDA_ARCH__ >= 1200
    // 发布前一阶段M1写入，后续TMA async proxy读取同一global输入。
    asm volatile("fence.proxy.async.global;" ::: "memory");
#endif
    begin_stage(p, g, st);
    for (int task; (task = claim_tile(p, s, st)) < 3 * per_projection;) {
        const int projection = task / per_projection;
        const int wi = 3 + 2 * projection;
        B *out = bp(p, projection == 0 ? SQ0 : projection == 1 ? SK0 : SV);
#if __CUDA_ARCH__ >= 1200
        // Q/K/V沿用288项队列，各自权重、输出和数值分段保持独立。
        projection_tma_tile(p, s.tile.projection_tma, 3 + projection,
                            (const B *)p.w[wi + 1], out, task % per_projection);
#else
        self_qkv8_tile(s.tile.cutlass_gemm.self_qkv8, a, (const B *)p.w[wi],
                      (const B *)p.w[wi + 1], out, task % per_projection);
#endif
        finish_tile();
    }
    g.sync();
}

__device__ __noinline__ void self_qkv_projection8(const Params &p, Shared &s, cg::grid_group &g,
                                               int st, const B *a, int wi, B *out) {
    constexpr int M = 224, tiles = ((M + 63) / 64) * (D / 128);
#if __CUDA_ARCH__ >= 1200
    if (wi == 9 || wi == 13 || wi == 19) {
        // 前一阶段写出的global输入将被TMA async proxy读取；每个写线程发布。
        asm volatile("fence.proxy.async.global;" ::: "memory");
        const int projection = wi == 9 ? 0 : (wi == 13 ? 1 : 2);
        begin_stage(p, g, st);
        for (int tile; (tile = claim_tile(p, s, st)) < tiles;) {
            projection_tma_tile(p, s.tile.projection_tma, projection,
                                (const B *)p.w[wi + 1], out, tile);
            finish_tile();
        }
        g.sync();
        return;
    }
#endif
    begin_stage(p, g, st);
    for (int t; (t = claim_tile(p, s, st)) < tiles;) {
        self_qkv8_tile(s.tile.cutlass_gemm.self_qkv8, a, (const B *)p.w[wi],
                      (const B *)p.w[wi + 1], out, t);
        finish_tile();
    }
    g.sync();
}

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
        self_qkv_projection8(p, s, g, st, a, wi, out);
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
__device__ __noinline__ void context_linear(const Params &p, Shared &s, cg::grid_group &g, int st,
                                            int wi, B *out, int m) {
    const int group_tiles = ((p.c + WAM_CONTEXT_TILE_M - 1) / WAM_CONTEXT_TILE_M) * (D / 128);
    const int ntiles = p.context_roots ? p.cb * group_tiles
                                     : ((m + WAM_CONTEXT_TILE_M - 1) / WAM_CONTEXT_TILE_M) * (D / 128);
    begin_stage(p, g, st);
    for (int t; (t = claim_tile(p, s, st)) < ntiles;) {
        const int group = p.context_roots ? t / group_tiles : 0;
        const B *input = p.context_roots ? p.context + p.context_roots[group] * p.c * D : p.context;
        B *output = p.context_roots ? out + group * p.c * D : out;
        cutlass_gemm_tile<WAM_CONTEXT_TILE_M, 136>(s.tile.cutlass_gemm.context, input,
                                                   (const B *)p.w[wi], (const B *)p.w[wi + 1], output,
                                                   p.context_roots ? p.c : m, D, D,
                                                   p.context_roots ? t % group_tiles : t, false);
        finish_tile();
    }
    g.sync();
}

// B14 FFN2：M128输出tile共48个，每个K分两段，共96个CTA任务。
// 各CTA独立算一段，FP32片段暂存在自己的shared；阶段代码负责有序提交。
__device__ __forceinline__ void ffn2_partition(CutlassTile<128>::Storage &s, const __nv_bfloat16 *a,
                                               const __nv_bfloat16 *w, int tile, int partition) {
    using T = CutlassTile<128>;
    using Mma = T::Mma;
#if __CUDA_ARCH__ >= 1200
    constexpr int kspan = 2048; // Current Torch SM120: seven BF16 partial sums.
#else
    constexpr int kspan = 7168;
#endif
    const int physical_warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (physical_warp >= 3 && physical_warp < 7) {
        const int thread = int(threadIdx.x) - 96, warp = physical_warp - 3;
        typename Mma::IteratorA::Params ap{CutlassRow(14336)};
        typename Mma::IteratorB::Params bp{CutlassCol(14336)};
        typename Mma::IteratorA ai(
            ap, reinterpret_cast<CutlassB *>(const_cast<__nv_bfloat16 *>(a)) + partition * kspan,
            {224, kspan}, thread, {(tile / 24) * 128, 0});
        typename Mma::IteratorB bi(
            bp, reinterpret_cast<CutlassB *>(const_cast<__nv_bfloat16 *>(w)) + partition * kspan,
            {kspan, 3072}, thread, {0, (tile % 24) * 128});
        Mma mma(s.mainloop, thread, warp, lane);
        typename Mma::FragmentC acc;
        acc.clear();
        mma(kspan / 64, acc, ai, bi, acc);
        asm volatile("bar.sync 2, 128;" ::: "memory");
        typename Mma::Operator::IteratorC store({s.result, CutlassRow(136)}, lane);
        store.add_tile_offset({warp % 2, warp / 2});
        store.store(acc);
    }
    __syncthreads();
}

// 每线程四个相邻元素；同一线程的列不随循环变化，bias/固定调制只读一次。
// SM80保留原两段顺序；SM120先合并已舍入前段，再在最后段加入bias。
// 每段BF16舍入，最后再做FP32门控残差；不跨越原Torch的舍入边界。
__device__ __forceinline__ void ffn2_commit(const Params &p, CutlassTile<128>::Storage &s,
                                            int tile, int partition, int partitions = 2) {
    const bool second = partition == partitions - 1;
    struct alignas(8) B4 { __nv_bfloat162 lo, hi; };
    B *out = bp(p, F2);
    const int col_in_tile = (int(threadIdx.x) % 32) * 4;
    const int col = (tile % 24) * 128 + col_in_tile;
    const int cell = 5 * D + col;
    float4 bias{}, table{};
    if (second) {
        const B4 b = *reinterpret_cast<const B4 *>((const B *)p.w[26] + col);
        const float2 lo = __bfloat1622float2(b.lo), hi = __bfloat1622float2(b.hi);
        bias = {lo.x, lo.y, hi.x, hi.y};
        table = *reinterpret_cast<const float4 *>((const float *)p.w[0] + cell);
    }
    for (int i = 4 * int(threadIdx.x); i < 128 * 128; i += 4 * THREADS) {
        const int row = (tile / 24) * 128 + i / 128;
        if (row < 224) {
            float4 z = *reinterpret_cast<const float4 *>(s.result + (i / 128) * 136 + col_in_tile);
#if __CUDA_ARCH__ < 1200
            if (second) {
                z.x = __fadd_rn(z.x, bias.x);
                z.y = __fadd_rn(z.y, bias.y);
                z.z = __fadd_rn(z.z, bias.z);
                z.w = __fadd_rn(z.w, bias.w);
            }
#endif
            if (partition > 0) {
                const B4 prev = *reinterpret_cast<const B4 *>(out + row * D + col);
                const float2 lo = __bfloat1622float2(prev.lo), hi = __bfloat1622float2(prev.hi);
                z.x = __fadd_rn(z.x, lo.x);
                z.y = __fadd_rn(z.y, lo.y);
                z.z = __fadd_rn(z.z, hi.x);
                z.w = __fadd_rn(z.w, hi.y);
            }
#if __CUDA_ARCH__ >= 1200
            if (second) {
                z.x = __fadd_rn(z.x, bias.x);
                z.y = __fadd_rn(z.y, bias.y);
                z.z = __fadd_rn(z.z, bias.z);
                z.w = __fadd_rn(z.w, bias.w);
            }
#endif
            const B4 rounded = {__floats2bfloat162_rn(z.x, z.y), __floats2bfloat162_rn(z.z, z.w)};
            if (!second || p.debug_modulation)
                *reinterpret_cast<B4 *>(out + row * D + col) = rounded;
            if (second) {
                const int batch = row / p.q, token = row % p.q;
                const int64_t src = (int64_t(batch) * p.l + (p.l == 1 ? 0 : token)) * 6 * D + cell;
                float4 t;
                if (p.temb_float) {
                    t = *reinterpret_cast<const float4 *>((const float *)p.temb + src);
                } else {
                    const B4 v = *reinterpret_cast<const B4 *>((const B *)p.temb + src);
                    const float2 lo = __bfloat1622float2(v.lo), hi = __bfloat1622float2(v.hi);
                    t = {lo.x, lo.y, hi.x, hi.y};
                }
                const float4 gate = {__fadd_rn(table.x, t.x), __fadd_rn(table.y, t.y),
                                     __fadd_rn(table.z, t.z), __fadd_rn(table.w, t.w)};
                const float2 lo = __bfloat1622float2(rounded.lo), hi = __bfloat1622float2(rounded.hi);
                const B4 residual = *reinterpret_cast<const B4 *>(bp(p, R2) + row * D + col);
                const float2 rlo = __bfloat1622float2(residual.lo), rhi = __bfloat1622float2(residual.hi);
                *reinterpret_cast<B4 *>(bp(p, OUT) + row * D + col) = {
                    __floats2bfloat162_rn(__fadd_rn(rlo.x, __fmul_rn(lo.x, gate.x)),
                                          __fadd_rn(rlo.y, __fmul_rn(lo.y, gate.y))),
                    __floats2bfloat162_rn(__fadd_rn(rhi.x, __fmul_rn(hi.x, gate.z)),
                                          __fadd_rn(rhi.y, __fmul_rn(hi.y, gate.w)))};
            }
        }
    }
    __syncthreads();
}

// SM120 M64粒度候选：输出tile从48变96；七段归约和提交算术不改。
__device__ __forceinline__ void ffn2_partition64(CutlassTile<64>::Storage &s, const __nv_bfloat16 *a,
                                               const __nv_bfloat16 *w, int tile, int partition) {
    using T = CutlassTile<64>;
    using Mma = T::Mma;
#if __CUDA_ARCH__ >= 1200
    constexpr int kspan = 2048; // Current Torch SM120: seven BF16 partial sums.
#else
    constexpr int kspan = 7168;
#endif
    const int physical_warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (physical_warp >= 3 && physical_warp < 7) {
        const int thread = int(threadIdx.x) - 96, warp = physical_warp - 3;
        typename Mma::IteratorA::Params ap{CutlassRow(14336)};
        typename Mma::IteratorB::Params bp{CutlassCol(14336)};
        typename Mma::IteratorA ai(
            ap, reinterpret_cast<CutlassB *>(const_cast<__nv_bfloat16 *>(a)) + partition * kspan,
            {224, kspan}, thread, {(tile / 24) * 64, 0});
        typename Mma::IteratorB bi(
            bp, reinterpret_cast<CutlassB *>(const_cast<__nv_bfloat16 *>(w)) + partition * kspan,
            {kspan, 3072}, thread, {0, (tile % 24) * 128});
        Mma mma(s.mainloop, thread, warp, lane);
        typename Mma::FragmentC acc;
        acc.clear();
        mma(kspan / 64, acc, ai, bi, acc);
        asm volatile("bar.sync 2, 128;" ::: "memory");
        typename Mma::Operator::IteratorC store({s.result, CutlassRow(136)}, lane);
        store.add_tile_offset({warp % 2, warp / 2});
        store.store(acc);
    }
    __syncthreads();
}

__device__ __forceinline__ void ffn2_commit64(const Params &p, CutlassTile<64>::Storage &s,
                                            int tile, int partition, int partitions = 2) {
    const bool second = partition == partitions - 1;
    struct alignas(8) B4 { __nv_bfloat162 lo, hi; };
    B *out = bp(p, F2);
    const int col_in_tile = (int(threadIdx.x) % 32) * 4;
    const int col = (tile % 24) * 128 + col_in_tile;
    const int cell = 5 * D + col;
    float4 bias{}, table{};
    if (second) {
        const B4 b = *reinterpret_cast<const B4 *>((const B *)p.w[26] + col);
        const float2 lo = __bfloat1622float2(b.lo), hi = __bfloat1622float2(b.hi);
        bias = {lo.x, lo.y, hi.x, hi.y};
        table = *reinterpret_cast<const float4 *>((const float *)p.w[0] + cell);
    }
    for (int i = 4 * int(threadIdx.x); i < 64 * 128; i += 4 * THREADS) {
        const int row = (tile / 24) * 64 + i / 128;
        if (row < 224) {
            float4 z = *reinterpret_cast<const float4 *>(s.result + (i / 128) * 136 + col_in_tile);
#if __CUDA_ARCH__ < 1200
            if (second) {
                z.x = __fadd_rn(z.x, bias.x);
                z.y = __fadd_rn(z.y, bias.y);
                z.z = __fadd_rn(z.z, bias.z);
                z.w = __fadd_rn(z.w, bias.w);
            }
#endif
            if (partition > 0) {
                const B4 prev = *reinterpret_cast<const B4 *>(out + row * D + col);
                const float2 lo = __bfloat1622float2(prev.lo), hi = __bfloat1622float2(prev.hi);
                z.x = __fadd_rn(z.x, lo.x);
                z.y = __fadd_rn(z.y, lo.y);
                z.z = __fadd_rn(z.z, hi.x);
                z.w = __fadd_rn(z.w, hi.y);
            }
#if __CUDA_ARCH__ >= 1200
            if (second) {
                z.x = __fadd_rn(z.x, bias.x);
                z.y = __fadd_rn(z.y, bias.y);
                z.z = __fadd_rn(z.z, bias.z);
                z.w = __fadd_rn(z.w, bias.w);
            }
#endif
            const B4 rounded = {__floats2bfloat162_rn(z.x, z.y), __floats2bfloat162_rn(z.z, z.w)};
            if (!second || p.debug_modulation)
                *reinterpret_cast<B4 *>(out + row * D + col) = rounded;
            if (second) {
                const int batch = row / p.q, token = row % p.q;
                const int64_t src = (int64_t(batch) * p.l + (p.l == 1 ? 0 : token)) * 6 * D + cell;
                float4 t;
                if (p.temb_float) {
                    t = *reinterpret_cast<const float4 *>((const float *)p.temb + src);
                } else {
                    const B4 v = *reinterpret_cast<const B4 *>((const B *)p.temb + src);
                    const float2 lo = __bfloat1622float2(v.lo), hi = __bfloat1622float2(v.hi);
                    t = {lo.x, lo.y, hi.x, hi.y};
                }
                const float4 gate = {__fadd_rn(table.x, t.x), __fadd_rn(table.y, t.y),
                                     __fadd_rn(table.z, t.z), __fadd_rn(table.w, t.w)};
                const float2 lo = __bfloat1622float2(rounded.lo), hi = __bfloat1622float2(rounded.hi);
                const B4 residual = *reinterpret_cast<const B4 *>(bp(p, R2) + row * D + col);
                const float2 rlo = __bfloat1622float2(residual.lo), rhi = __bfloat1622float2(residual.hi);
                *reinterpret_cast<B4 *>(bp(p, OUT) + row * D + col) = {
                    __floats2bfloat162_rn(__fadd_rn(rlo.x, __fmul_rn(lo.x, gate.x)),
                                          __fadd_rn(rlo.y, __fmul_rn(lo.y, gate.y))),
                    __floats2bfloat162_rn(__fadd_rn(rhi.x, __fmul_rn(hi.x, gate.z)),
                                          __fadd_rn(rhi.y, __fmul_rn(hi.y, gate.w)))};
            }
        }
    }
    __syncthreads();
}

using DeviceCounter = cuda::atomic_ref<int, cuda::thread_scope_device>;
__device__ __forceinline__ int64_t ffn_timestamp() {
    int64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// 每个写线程先发布ACT/F2/OUT，再由CTA leader递增完成数。
// acq_rel RMW串起同组生产者；消费者acquire后才读取对应全局buffer。
__device__ void ffn_publish(const Params &p, int offset, int group, int expected) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int64_t at = p.debug_modulation ? ffn_timestamp() : 0;
        const int done = DeviceCounter(p.ffn_state[offset + group])
                             .fetch_add(1, cuda::memory_order_acq_rel) + 1;
        if (p.debug_modulation && done == expected)
            p.ffn_trace[(offset == 0 ? 0 : 8) + group] = at;
    }
    __syncthreads();
}

// 尾部4块各有96有效行，拆成12个32行任务；所有任务仍完成整个K3072。
// 原28个组3生产者变为24+12=36个，其他组与FFN2接口保持不变。
__device__ __noinline__ void action_ffn1_tail(const Params &p, Shared &s, int tail) {
    using Mma = CutlassTailFFN::Mma;
    auto &workspace = s.tile.cutlass_gemm.tail_ffn;
    const int rm = 128 + (tail % 3) * 32, cn = (52 + tail / 3) * 256;
    const int thread = int(threadIdx.x), warp = thread / 32, lane = thread % 32;
    if (warp < 4) {
        typename Mma::IteratorA::Params ap{CutlassRow(D)};
        typename Mma::IteratorB::Params bp{CutlassCol(D)};
        typename Mma::IteratorA ai(ap, reinterpret_cast<CutlassB *>(p.a[M3]),
                                  {224, D}, thread, {rm, 0});
        typename Mma::IteratorB bi(bp, reinterpret_cast<CutlassB *>(const_cast<void *>(p.w[23])),
                                  {D, F}, thread, {0, cn});
        Mma mma(workspace.mainloop, thread, warp, lane);
        typename Mma::FragmentC acc;
        acc.clear();
        mma(D / 64, acc, ai, bi, acc);
        asm volatile("bar.sync 2, 128;" ::: "memory");
        typename Mma::Operator::IteratorC store({workspace.result, CutlassRow(264)}, lane);
        store.add_tile_offset({0, warp});
        store.store(acc);
    }
    __syncthreads();
    struct alignas(8) B4 { __nv_bfloat162 lo, hi; };
    for (int i = 4 * thread; i < 32 * 256; i += 4 * THREADS) {
        const int row = rm + i / 256, col = cn + i % 256;
        const float4 v = *reinterpret_cast<const float4 *>(workspace.result +
                                                          (i / 256) * 264 + i % 256);
        const B4 b = *reinterpret_cast<const B4 *>(reinterpret_cast<const B *>(p.w[24]) + col);
        const float2 lo = __bfloat1622float2(b.lo), hi = __bfloat1622float2(b.hi);
        const B4 rounded = {
            __floats2bfloat162_rn(__fadd_rn(v.x, lo.x), __fadd_rn(v.y, lo.y)),
            __floats2bfloat162_rn(__fadd_rn(v.z, hi.x), __fadd_rn(v.w, hi.y))};
        if (p.debug_modulation)
            *reinterpret_cast<B4 *>(reinterpret_cast<B *>(p.a[F1]) + row * F + col) = rounded;
        *reinterpret_cast<B4 *>(reinterpret_cast<B *>(p.a[ACT]) + row * F + col) = {
            __halves2bfloat162(gelu_value(__low2bfloat16(rounded.lo)),
                               gelu_value(__high2bfloat16(rounded.lo))),
            __halves2bfloat162(gelu_value(__low2bfloat16(rounded.hi)),
                               gelu_value(__high2bfloat16(rounded.hi)))};
    }
    ffn_publish(p, 0, 3, 36);
}

// FFN1前108块M128×N256、8个MMA warp，尾部12块M32×N256。
// 8warp的FP32 accumulator分两半经同一128×136 shared区写回，避免
// 为整层抬高shared预留；完整K3072、bias和BF16舍入边界不变。
__device__ __forceinline__ void action_ffn1_legacy(const Params &p, Shared &s) {
    using Mma = CutlassWideFFN::Mma;
    // 静态任务无需入口全局屏障；保留出口及分段依赖。
    auto &workspace = s.tile.cutlass_gemm.wide_ffn;
    for (int tile = int(blockIdx.x); tile < 120; tile += int(gridDim.x)) {
        if (tile >= 108) {
            action_ffn1_tail(p, s, tile - 108);
            continue;
        }
        const int rm = (tile / 56) * 128, cn = (tile % 56) * 256;
        const int thread = int(threadIdx.x), warp = thread / 32, lane = thread % 32;
        typename Mma::IteratorA::Params ap{CutlassRow(D)};
        typename Mma::IteratorB::Params bp{CutlassCol(D)};
        typename Mma::IteratorA ai(ap, reinterpret_cast<CutlassB *>(p.a[M3]), {224, D}, thread,
                                   {rm, 0});
        typename Mma::IteratorB bi(bp, reinterpret_cast<CutlassB *>(const_cast<void *>(p.w[23])),
                                   {D, F}, thread, {0, cn});
        Mma mma(workspace.mainloop, thread, warp, lane);
        typename Mma::FragmentC acc;
        acc.clear();
        mma(D / 32, acc, ai, bi, acc);
        __syncthreads();
        for (int half = 0; half < 2; ++half) {
            if (warp / 4 == half) {
                typename Mma::Operator::IteratorC store({workspace.result, CutlassRow(136)}, lane);
                store.add_tile_offset({warp % 2, (warp % 4) / 2});
                store.store(acc);
            }
            __syncthreads();
            struct alignas(8) B4 {
                __nv_bfloat162 lo, hi;
            };
            for (int i = 4 * thread; i < 128 * 128; i += 4 * THREADS) {
                const int row = rm + i / 128, col = cn + half * 128 + i % 128;
                if (row < 224) {
                    float4 v = *reinterpret_cast<const float4 *>(workspace.result +
                                                                 (i / 128) * 136 + i % 128);
                    B4 b =
                        *reinterpret_cast<const B4 *>(reinterpret_cast<const B *>(p.w[24]) + col);
                    float2 lo = __bfloat1622float2(b.lo), hi = __bfloat1622float2(b.hi);
                    B4 rounded = {
                        __floats2bfloat162_rn(__fadd_rn(v.x, lo.x), __fadd_rn(v.y, lo.y)),
                        __floats2bfloat162_rn(__fadd_rn(v.z, hi.x), __fadd_rn(v.w, hi.y))};
                    if (p.debug_modulation)
                        *reinterpret_cast<B4 *>(reinterpret_cast<B *>(p.a[F1]) + row * F + col) = rounded;
                    // 已完成此输出tile的完整K归约，直接生成GELU结果。
                    B4 activated = {
                        __halves2bfloat162(gelu_value(__low2bfloat16(rounded.lo)),
                                           gelu_value(__high2bfloat16(rounded.lo))),
                        __halves2bfloat162(gelu_value(__low2bfloat16(rounded.hi)),
                                           gelu_value(__high2bfloat16(rounded.hi)))};
                    *reinterpret_cast<B4 *>(reinterpret_cast<B *>(p.a[ACT]) + row * F + col) = activated;
                }
            }
            __syncthreads();
        }
        ffn_publish(p, 0, tile / 28, tile / 28 == 3 ? 36 : 28);
    }
}


// FFN1独立三角色流水：448个M64/N128输出tile，48个K64一次完整累加。
// 与projection共用82KiB双输入/单结果布局，但不使用projection的10段舍入。
// 一CTA静态承接2或3个输出tile；GELU写global，阶段出口仍全局同步。
__device__ __forceinline__ void ffn1_epilogue(const Params &p, ProjectionTmaStorage &s,
                                             int tile, int thread) {
    const int rm = (tile / 112) * 64, cn = (tile % 112) * 128;
    struct alignas(8) B4 { __nv_bfloat162 lo, hi; };
    for (int i = 4 * thread; i < 64 * 128; i += 4 * 96) {
        const int row = rm + i / 128, col = cn + i % 128;
        if (row < 224) {
            const float4 v = *reinterpret_cast<const float4 *>(s.result + (i / 128) * 136 + i % 128);
            const B4 bias = *reinterpret_cast<const B4 *>(static_cast<const B *>(p.w[24]) + col);
            const float2 lo = __bfloat1622float2(bias.lo), hi = __bfloat1622float2(bias.hi);
            const B4 rounded = {
                __floats2bfloat162_rn(__fadd_rn(v.x, lo.x), __fadd_rn(v.y, lo.y)),
                __floats2bfloat162_rn(__fadd_rn(v.z, hi.x), __fadd_rn(v.w, hi.y))};
            if (p.debug_modulation)
                *reinterpret_cast<B4 *>(bp(p, F1) + row * F + col) = rounded;
            *reinterpret_cast<B4 *>(bp(p, ACT) + row * F + col) = {
                __halves2bfloat162(gelu_value(__low2bfloat16(rounded.lo)), gelu_value(__high2bfloat16(rounded.lo))),
                __halves2bfloat162(gelu_value(__low2bfloat16(rounded.hi)), gelu_value(__high2bfloat16(rounded.hi)))};
        }
    }
}

// 双CTA共享输入A：相同M64行、不同N128列；各自完整累加器，不跨CTA归约输出。
__device__ __forceinline__ int ffn1_cluster_tile(int cid,int rank,int clusters,int it) {
    const int job=cid+it*clusters;
    return (job/56)*112+2*(job%56)+rank;
}
__device__ __forceinline__ void ffn1_remote_arrive(uint64_t *barrier) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                 :: "r"(projection_smem(barrier)) : "memory");
}
__device__ __forceinline__ void ffn1_cluster_load(const Params &p,ProjectionTmaStorage &s,
                                                int step,int tile,int rank) {
    const int slot=step&1;
    if (rank == 0) {
        projection_load(p,s,6,step%48,(tile/112)*64,(tile%112)*128);
    } else {
        auto ready=projection_smem(&s.ready[slot]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 24576;" :: "r"(ready) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            :: "r"(projection_smem(s.b[slot])), "l"(&p.projection_maps[6][1]),
               "r"((step%48)*64), "r"((tile%112)*128), "r"(ready) : "memory");
        // 发布目的A槽可写+事务计数已arm，生产者随后才发送A。
        ffn1_remote_arrive(cg::this_cluster().map_shared_rank(&s.peer_ready[slot],0));
    }
}

__device__ __forceinline__ void action_ffn1_produce(const Params &p, Shared &s) {
#if __CUDA_ARCH__ >= 1200
    using Op = typename CutlassTile<64>::Mma::Operator;
    auto &pipe = s.tile.projection_tma;
    const int w = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    const int rank = int(blockIdx.x) % 2, cid = int(blockIdx.x) / 2;
    const int clusters = int(gridDim.x) / 2;
    const int tiles = (224 - cid + clusters - 1) / clusters;
    auto cluster = cg::this_cluster();
    const int steps = tiles * 48;
    // M3是前一阶段普通store的结果，先完成generic->async proxy发布。
    asm volatile("fence.proxy.async.global;" ::: "memory");
    cg::this_grid().sync();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(projection_smem(&pipe.ready[i])) : "memory");
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(projection_smem(&pipe.empty[i])), "r"(FFN1_SHARE_A && rank == 0 ? 5 : 4) : "memory");
        }
        for (int i = 0; i < 2; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(projection_smem(&pipe.peer_ready[i])) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" :: "r"(projection_smem(&pipe.result_full)) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 3;" :: "r"(projection_smem(&pipe.result_empty)) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads(); // 初始化角色共享的mbarrier；热循环不做全CTA同步。
    cluster.sync(); // 仅初始化：所有peer mbarrier存在，才允许远端到达。
    int waits = 0, releases = 0, finished = 0;
    int a_loads=0,b_loads=0,peer_copies=0,peer_acks=0,peer_credits=0;
    if (w == 0) {
#if FFN1_SHARE_A
        // 先装满两个槽；每处理一个槽才回填step+2，保留双缓冲前瞻。
        for (int issue = 0; issue < 2; ++issue) {
            const int tile = ffn1_cluster_tile(cid,rank,clusters,issue/48);
            if (lane == 0) ffn1_cluster_load(p,pipe,issue,tile,rank);
            ++waits; a_loads += rank == 0; ++b_loads;
            finished += rank == 0 ? 2 : 1;
            peer_credits += rank == 1;
            __syncwarp();
        }
        for (int step = 0; step < steps; ++step) {
            const int slot=step&1;
            projection_wait(&pipe.ready[slot], (step/2)&1);
            if (rank == 0) {
                projection_wait(&pipe.peer_ready[slot], (step/2)&1);
                if (lane == 0) {
                    auto remote_a=cluster.map_shared_rank(pipe.a[slot],1);
                    auto remote_ready=cluster.map_shared_rank(&pipe.ready[slot],1);
                    // 源已由TMA填好，目标已arm expect_tx；8KiB原样复制A的swizzle布局。
                    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], 8192, [%2];"
                        :: "r"(projection_smem(remote_a)), "r"(projection_smem(pipe.a[slot])),
                           "r"(projection_smem(remote_ready)) : "memory");
                }
                ++peer_copies;
            } else {
                // ready表示B TMA+A DSM都已到达，源A才允许回收；不必等peer MMA。
                if (lane == 0) ffn1_remote_arrive(cluster.map_shared_rank(&pipe.empty[slot],0));
                ++peer_acks;
            }
            __syncwarp();
            if (step+2 < steps) {
                projection_wait(&pipe.empty[slot], (step/2)&1);
                ++releases;
                const int next=step+2;
                const int tile=ffn1_cluster_tile(cid,rank,clusters,next/48);
                if (lane == 0) ffn1_cluster_load(p,pipe,next,tile,rank);
                ++waits; a_loads += rank == 0; ++b_loads;
                finished += rank == 0 ? 2 : 1;
                peer_credits += rank == 1;
                __syncwarp();
            }
        }
#else
        for (int step = 0; step < steps; ++step) {
            if (step >= 2) {
                projection_wait(&pipe.empty[step & 1], ((step / 2) - 1) & 1);
                ++releases;
            }
            const int tile=ffn1_cluster_tile(cid,rank,clusters,step/48);
            if (lane == 0) projection_load(p,pipe,6,step%48,(tile/112)*64,(tile%112)*128);
            ++waits; finished+=2; ++a_loads; ++b_loads;
            __syncwarp();
        }
#endif
        projection_wait(&pipe.empty[0], ((steps / 2) - 1) & 1);
        projection_wait(&pipe.empty[1], ((steps / 2) - 1) & 1);
        releases += 2;
    } else if (w >= 4) {
        const int warp = w - 4;
        for (int it = 0; it < tiles; ++it) {
            typename Op::FragmentC accum;
            accum.clear(); // FFN1只在输出tile起点清零，不在K64之间舍入。
            for (int k = 0; k < 48; ++k) {
                const int step = it * 48 + k, slot = step & 1;
                projection_wait(&pipe.ready[slot], (step / 2) & 1);
                ++waits;
                typename Op::IteratorA ai({pipe.a[slot], typename Op::LayoutA(64)}, lane);
                typename Op::IteratorB bi({pipe.b[slot], typename Op::LayoutB(64)}, lane);
                ai.add_tile_offset({warp % 2, 0}); bi.add_tile_offset({0, warp / 2});
                typename Op::FragmentA af[2];
                typename Op::FragmentB bf[2];
                ai.set_kgroup_index(0); bi.set_kgroup_index(0);
                ai.load(af[0]); bi.load(bf[0]); ++ai; ++bi;
                Op mma;
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    if (group < 3) {
                        ai.set_kgroup_index(group + 1); bi.set_kgroup_index(group + 1);
                        ai.load(af[(group + 1) & 1]); bi.load(bf[(group + 1) & 1]); ++ai; ++bi;
                    }
                    if (group == 2) {
                        __syncwarp();
                        if (lane == 0) projection_arrive(&pipe.empty[slot]);
                        ++releases; // 所有shared输入已进入寄存器，可以覆盖输入槽。
                    }
                    typename Op::TransformedFragmentA at;
                    typename Op::TransformedFragmentB bt;
                    mma.transform(at, bt, af[group & 1], bf[group & 1]);
                    mma(accum, at, bt, accum);
                }
            }
            // 下一tile的MMA可与上一tile epilogue重叠，只有覆盖结果时等待EMPTY。
            if (it > 0) projection_wait(&pipe.result_empty, (it - 1) & 1);
            typename Op::IteratorC store({pipe.result, CutlassRow(136)}, lane);
            store.add_tile_offset({warp % 2, warp / 2});
            store.store(accum);
            __syncwarp();
            if (lane == 0) projection_arrive(&pipe.result_full);
            ++finished;
        }
    } else {
        for (int it = 0; it < tiles; ++it) {
            projection_wait(&pipe.result_full, it & 1);
            ++waits;
            ffn1_epilogue(p, pipe, ffn1_cluster_tile(cid,rank,clusters,it), int(threadIdx.x) - 32);
            __syncwarp();
            if (lane == 0) projection_arrive(&pipe.result_empty);
            ++releases; ++finished;
        }
    }
    if (lane == 0 && p.ffn1_role_trace) {
        auto dst = p.ffn1_role_trace + (int(blockIdx.x) * 8 + w) * 3;
        dst[0] = waits; dst[1] = releases; dst[2] = finished;
    }
    if (w == 0 && lane == 0 && p.ffn1_transfer_trace) {
        auto out=p.ffn1_transfer_trace+int(blockIdx.x)*5;
        out[0]=a_loads; out[1]=b_loads; out[2]=peer_copies;
        out[3]=peer_acks; out[4]=peer_credits;
    }
    cluster.sync(); // 远端所有ACK/DSM使用已排空，之后才销毁barrier。
    __syncthreads(); // 排空全部角色，结果和输入槽才能交还下一阶段。
    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; ++i) {
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&pipe.ready[i])) : "memory");
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&pipe.empty[i])) : "memory");
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&pipe.peer_ready[i])) : "memory");
        }
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&pipe.result_full)) : "memory");
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(projection_smem(&pipe.result_empty)) : "memory");
    }
    __syncthreads();
    // 真实ACT覆盖计数：四个128行×7168列区域各112个新tile；不沿用旧120任务审计。
    for (int it=0; it<tiles; ++it) {
        const int tile=ffn1_cluster_tile(cid,rank,clusters,it);
        ffn_publish(p,0,2*((tile/112)/2)+(tile%112)/56,112);
    }
#else
    action_ffn1_legacy(p, s);
#endif
}

// SM120 FFN2：96个输出tile × 7个K2048片段，仍使用原M64/N128/K64主循环。
// 每个行组按partition-major领取：前一片段一定先取得owner。所有cooperative
// CTA均驻留，且每CTA先完成自己的FFN1任务；提交等待只指向更小partition，
// 因而不会出现等待尚未领取任务的环。FP32部分和在本CTA shared中等待提交。
// helper不含grid.sync，独立FFN2测量可复用；调用者负责初始化和最终汇合。
__device__ __forceinline__ void action_ffn2_splitk(const Params &p, Shared &s) {
    constexpr int columns = 24, partitions = 7, tasks_per_row = columns * partitions;
    while (true) {
        if (threadIdx.x == 0) {
            s.task = -1;
            for (int scan = 0; scan < 4; ++scan) {
                const int row = (int(blockIdx.x) + scan) % 4;
                const int first = 2 * (row / 2);
                if (DeviceCounter(p.ffn_state[first]).load(cuda::memory_order_acquire) != 112 ||
                    DeviceCounter(p.ffn_state[first + 1]).load(cuda::memory_order_acquire) !=
                        112)
                    continue;
                auto next = DeviceCounter(p.ffn_state[4 + row]);
                int job = next.load(cuda::memory_order_relaxed);
                while (job < tasks_per_row && !next.compare_exchange_weak(
                           job, job + 1, cuda::memory_order_relaxed)) {}
                if (job < tasks_per_row) {
                    s.task = row * tasks_per_row + job;
                    break;
                }
            }
            // 所有任务已有owner才退出；暂时没有ready行时继续等待生产者。
            if (s.task == -1 &&
                DeviceCounter(p.ffn_state[4]).load(cuda::memory_order_relaxed) == tasks_per_row &&
                DeviceCounter(p.ffn_state[5]).load(cuda::memory_order_relaxed) == tasks_per_row &&
                DeviceCounter(p.ffn_state[6]).load(cuda::memory_order_relaxed) == tasks_per_row &&
                DeviceCounter(p.ffn_state[7]).load(cuda::memory_order_relaxed) == tasks_per_row)
                s.task = -2;
        }
        __syncthreads();
        const int task = s.task;
        if (task == -2) break;
        if (task < 0) { __nanosleep(128); __syncthreads(); continue; }
        const int row = task / tasks_per_row;
        const int job = task % tasks_per_row;
        const int partition = job / columns, tile = row * columns + job % columns;
        auto &workspace = s.tile.cutlass_gemm.action;
        ffn2_partition64(workspace, bp(p, ACT), (const B *)p.w[25], tile, partition);

        // 各K片段可并行做MMA；只有BF16部分和、末段bias和残差提交必须有序。
        // acquire后经CTA同步将前一片段的全tile写入对所有写回线程可见。
        if (threadIdx.x == 0)
            while (DeviceCounter(p.ffn_state[16 + tile]).load(cuda::memory_order_acquire) != partition)
                __nanosleep(128);
        __syncthreads();
        ffn2_commit64(p, workspace, tile, partition, partitions);
        // 每个写线程先发布自己的F2/OUT，再由leader释放后继片段。
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
            DeviceCounter(p.ffn_state[16 + tile]).store(partition + 1, cuda::memory_order_release);
        __syncthreads();
        if (partition == partitions - 1)
            ffn_publish(p, 8, row, columns); // 仍只统计24个最终完成输出tile。
    }
}

// FFN两阶段：先生成全部GELU结果到global ACT，再开始FFN2。
// SM120仅增加阶段边界；七个K2048任务的BF16递推、tile和warp分工不变。
__device__ __forceinline__ void action_ffn_stages(const Params &p, Shared &s, cg::grid_group &g) {
    action_ffn1_produce(p, s);
#if __CUDA_ARCH__ >= 1200
    g.sync(); // 所有ACT写入完成且对整个cooperative grid可见。
    action_ffn2_splitk(p, s);
    g.sync();
#else
    while (true) {
        if (threadIdx.x == 0) {
            s.task = -1;
            for (int scan = 0; scan < 4; ++scan) {
                const int group = (int(blockIdx.x) + scan) % 4;
                if (DeviceCounter(p.ffn_state[group]).load(cuda::memory_order_acquire) !=
                    (group == 3 ? 36 : 28))
                    continue;
                auto next = DeviceCounter(p.ffn_state[4 + group]);
                int col = next.load(cuda::memory_order_relaxed);
                while (col < 24 && !next.compare_exchange_weak(
                                       col, col + 1, cuda::memory_order_relaxed)) {}
                if (col < 24) {
                    s.task = group * 24 + col;
                    if (p.debug_modulation && col == 0) {
                        p.ffn_trace[4 + group] = ffn_timestamp();
                        int produced = 0;
                        for (int j = 0; j < 4; ++j)
                            produced += DeviceCounter(p.ffn_state[j]).load(cuda::memory_order_relaxed);
                        p.ffn_state[12 + group] = produced;
                    }
                    break;
                }
            }
            // 全部96任务已有owner后，空闲CTA直接进入出口屏障，不反复轮询。
            if (s.task == -1) {
                bool all_claimed = true;
                for (int group = 0; group < 4; ++group)
                    all_claimed &= DeviceCounter(p.ffn_state[4 + group])
                                       .load(cuda::memory_order_relaxed) == 24;
                if (all_claimed)
                    s.task = -2;
            }
        }
        __syncthreads();
        const int task = s.task;
        if (task == -2)
            break;
        if (task < 0) {
            __nanosleep(128);
            __syncthreads(); // 所有warp已读完s.task，leader才可写下一轮编号。
            continue;
        }
        const int group = task / 24, tile = (group / 2) * 24 + task % 24;
        auto &workspace = s.tile.cutlass_gemm.ffn1;
        ffn2_partition(workspace, bp(p, ACT), (const B *)p.w[25], tile, group & 1);
        if (group & 1) {
            if (threadIdx.x == 0)
                while (DeviceCounter(p.ffn_state[8 + group - 1])
                           .load(cuda::memory_order_acquire) != 24)
                    __nanosleep(128);
            __syncthreads(); // FP32第二段仍留在本CTA shared，第一段BF16现已可见。
        }
        ffn2_commit(p, workspace, tile, group & 1);
        ffn_publish(p, 8, group, 24);
    }
    g.sync(); // FFN所有输出已完成；取代FFN1出口和FFN2两处grid同步。
#endif
}

#include "norm_device_generated.cuh"

__device__ __forceinline__ float4 modulation4(const Params &p, int64_t element, int param);

// Norm tile为完整3072维行。8个warp搬入/写回，warp3–6仍按原Torch顺序归约。
// 仅扩大搬运参与度；FP32转换、Welford树与BF16舍入顺序保持原样。
__device__ void norm_tile(Shared &sh, const B *x, void *y, const void *gamma, const float *beta,
                          float eps, bool rms, bool output_float, int row,
                          const Params *mod = nullptr, int mod_param = 0, B *mod_out = nullptr) {
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
        if (mod) {
            // Row's FP32 Welford output is ready in shared. Keep the exact
            // FP32 add/multiply order and final BF16 conversion of MODULATE.
            for (int v = int(threadIdx.x); v < D / 4; v += THREADS) {
                const int64_t element = int64_t(row) * D + v * 4;
                const float4 z = reinterpret_cast<const float4 *>(s.y)[v];
                if (mod->debug_modulation)
                    reinterpret_cast<float4 *>((float *)y + row * D)[v] = z;
                const float4 shift = modulation4(*mod, element, mod_param);
                const float4 scale = modulation4(*mod, element, mod_param + 1);
                float out[4];
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    out[c] = __fadd_rn(__fmul_rn(((const float *)&z)[c],
                               __fadd_rn(1.f, ((const float *)&scale)[c])),
                               ((const float *)&shift)[c]);
                auto *dest = reinterpret_cast<__nv_bfloat162 *>(mod_out + element);
                dest[0] = __floats2bfloat162_rn(out[0], out[1]);
                dest[1] = __floats2bfloat162_rn(out[2], out[3]);
            }
        } else if (output_float) {
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
__device__ void rms_pair_tile(const Params &p, Shared &sh, const B *x, B *y, const B *gamma,
                              float eps, int first_row, int rows, B *rotated = nullptr) {
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
            if (gamma)
                weights = reinterpret_cast<const Vec *>(gamma)[v];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float z = rstd * value(values.val[j]);
                if (gamma)
                    z = value(weights.val[j]) * z;
                output.val[j] = bf(z);
            }
            // RMS先按原边界舍入BF16。debug保留此中间值；普通融合路径不落地。
            if (y)
                reinterpret_cast<Vec *>(y + row * D)[v] = output;
            if (rotated) {
                Vec result;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int batch = row / p.q, token = row % p.q;
                    const int dim = (2 * v + c) % 64;
                    auto f = p.freq[((p.fb == 1 ? 0 : batch) * p.q + token) * 64 + dim];
                    double real = double(value(output.val[2 * c]));
                    double imag = double(value(output.val[2 * c + 1]));
                    // 保留原模型FP64旋转及最终转换，不将RMS与旋转跨舍入合并。
                    result.val[2 * c] = bf(float(real * double(f.real()) - imag * double(f.imag())));
                    result.val[2 * c + 1] = bf(float(real * double(f.imag()) + imag * double(f.real())));
                }
                reinterpret_cast<Vec *>(rotated + row * D)[v] = result;
            }
        }
    }
    __syncthreads(); // 两行都写完后，整个CTA才复用共享空间。
}
__device__ void norm(const Params &p, Shared &s, cg::grid_group &g, int st, const B *x, void *y,
                     int rows, int gi, int bi, float eps, bool rms, bool out_float) {
    // 静态任务无需重置队列；前一阶段出口已保证输入就绪。
    if (p.candidate && rms && !out_float && bi < 0) {
        for (int t = int(blockIdx.x); t < (rows + 1) / 2; t += int(gridDim.x)) {
            rms_pair_tile(p, s, x, (B *)y, gi >= 0 ? (const B *)p.w[gi] : nullptr, eps, 2 * t, rows);
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

// B14 self Q/K：每组4warp完成一行RMS，随后在寄存器旋转相邻元素。
// 只在完整行归约后做RoPE；两阶段合为一个出口，GEMM与归约树不变。
__device__ void norm_rope(const Params &p, Shared &s, cg::grid_group &g,
                          const B *x, B *normalized, B *rotated, int gi, float eps) {
    const int rows = p.b * p.q;
    for (int t = int(blockIdx.x); t < (rows + 1) / 2; t += int(gridDim.x))
        rms_pair_tile(p, s, x, p.debug_modulation ? normalized : nullptr,
                      (const B *)p.w[gi], eps, 2 * t, rows, rotated);
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

struct alignas(8) Point4 {
    B value[4];
};

// 固定D3072的pointwise：向量搬运和静态tile分配；每元素公式与精度不变。
// 在消费位置计算4个调制参数，保持原TABLE的FP32相加与广播寻址。
// 普通路径不再写/读完整[m,6,D]中间表；这里只返回线程私有寄存器值。
__device__ __forceinline__ float4 modulation4(const Params &p, int64_t element, int param) {
    const int row = element / D, col = element % D;
    const int batch = row / p.q, token = row % p.q;
    const int cell = param * D + col;
    const int64_t src = (int64_t(batch) * p.l + (p.l == 1 ? 0 : token)) * 6 * D + cell;
    float4 result = *reinterpret_cast<const float4 *>((const float *)p.w[0] + cell);
    if (p.temb_float) {
        const float4 t = *reinterpret_cast<const float4 *>((const float *)p.temb + src);
#pragma unroll
        for (int c = 0; c < 4; ++c)
            ((float *)&result)[c] = __fadd_rn(((float *)&result)[c], ((const float *)&t)[c]);
    } else {
        const Point4 t = *reinterpret_cast<const Point4 *>((const B *)p.temb + src);
#pragma unroll
        for (int c = 0; c < 4; ++c)
            ((float *)&result)[c] = __fadd_rn(((float *)&result)[c], value(t.value[c]));
    }
    return result;
}

__device__ void point(const Params &p, Shared &s, cg::grid_group &g, int st, Point op, int dst,
                      int a, int b, int param, int64_t count) {
    constexpr int tile_elements = 2048;
    const int tiles = (count + tile_elements - 1) / tile_elements;
    // 静态任务无需重置队列；前一阶段出口已保证输入就绪。
    // 每个tile独立；按CTA编号固定分配，省去每tile原子领取/广播。
    for (int task = int(blockIdx.x); task < tiles; task += int(gridDim.x)) {
        if (op == TABLE && (p.temb_float || p.l != p.q)) {
            // 非逐token/BF16输入的兼容分支，保持原广播寻址与FP32相加。
            for (int j = threadIdx.x; j < tile_elements; j += THREADS) {
                int64_t i = int64_t(task) * tile_elements + j;
                if (i < count) {
                    int cell = i % (6 * D), row = i / (6 * D), batch = row / p.q, token = row % p.q;
                    int64_t src = (int64_t(batch) * p.l + (p.l == 1 ? 0 : token)) * 6 * D + cell;
                    float v = p.temb_float ? ((const float *)p.temb)[src]
                                           : value(((const B *)p.temb)[src]);
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
                        int64_t pair = i + c, r = pair / 64, batch = r / (p.q * H),
                                token = (r / H) % p.q, dim = pair % 64;
                        auto f = p.freq[((p.fb == 1 ? 0 : batch) * p.q + token) * 64 + dim];
                        double x = double(value(v.value[2 * c])),
                               y = double(value(v.value[2 * c + 1]));
                        out.value[2 * c] = bf(float(x * double(f.real()) - y * double(f.imag())));
                        out.value[2 * c + 1] =
                            bf(float(x * double(f.imag()) + y * double(f.real())));
                    }
                    *reinterpret_cast<Point4 *>(bp(p, dst) + 2 * i) = out;
                }
            }
        } else {
            for (int j = 4 * int(threadIdx.x); j < tile_elements; j += 4 * THREADS) {
                int64_t i = int64_t(task) * tile_elements + j;
                if (i < count) {
                    Point4 x, y, out;
                    float4 xf, shift, scale;
                    if (op == MODULATE) {
                        xf = *reinterpret_cast<const float4 *>(fp(p, a) + i);
                        shift = modulation4(p, i, param);
                        scale = modulation4(p, i, param + 1);
                    } else {
                        const B *input = op == TABLE ? (const B *)p.temb : a < 0 ? p.x : bp(p, a);
                        x = *reinterpret_cast<const Point4 *>(input + i);
                    }
                    if (op == TABLE)
                        shift =
                            *reinterpret_cast<const float4 *>((const float *)p.w[0] + i % (6 * D));
                    if (op == RESIDUAL) {
                        y = *reinterpret_cast<const Point4 *>(bp(p, b) + i);
                        if (param >= 0)
                            scale = modulation4(p, i, param);
                    }
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        if (op == TABLE)
                            ((float *)&shift)[c] =
                                __fadd_rn(((float *)&shift)[c], value(x.value[c]));
                        else if (op == MODULATE)
                            out.value[c] = bf(__fadd_rn(
                                __fmul_rn(((float *)&xf)[c], __fadd_rn(1.f, ((float *)&scale)[c])),
                                ((float *)&shift)[c]));
                        else if (op == RESIDUAL) {
                            float v = value(y.value[c]);
                            if (param >= 0)
                                v = __fmul_rn(v, ((float *)&scale)[c]);
                            out.value[c] = bf(__fadd_rn(value(x.value[c]), v));
                        } else {
                            out.value[c] = gelu_value(x.value[c]);
                        }
                    }
                    if (op == TABLE)
                        *reinterpret_cast<float4 *>(fp(p, dst) + i) = shift;
                    else
                        *reinterpret_cast<Point4 *>(bp(p, dst) + i) = out;
                }
            }
        }
        // 此阶段只使用线程私有寄存器和互不重叠的输出，无shared复用屏障。
    }
    g.sync();
}
__device__ void insert_cache(const Params &p, Shared &s, cg::grid_group &g, int st) {
    if (p.candidate)
        return; // B14的private KV从不写入公共history；无写入就无队列任务。
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

// One owner computes the complete LN row then its modulation. Only the
// FP32 global intermediate and the intervening grid barrier disappear.
__device__ void norm_modulate(const Params &p, Shared &s, cg::grid_group &g,
                               const B *x, void *audit, B *out, float eps, int param) {
    for (int row = int(blockIdx.x); row < p.b * p.q; row += int(gridDim.x))
        norm_tile(s, x, audit, nullptr, nullptr, eps, false, true, row, &p, param, out);
    g.sync();
}

// All threads read the same immutable launch descriptors. Avoid an implicit
// per-thread local copy when a device helper takes Params by reference.
__global__ void whole_layer(const __grid_constant__ Params p) {
    extern __shared__ __align__(1024) unsigned char memory[];
    Shared &s = *reinterpret_cast<Shared *>(memory);
    auto g = cg::this_grid();
    if (blockIdx.x == 0 && threadIdx.x < 16)
        p.ffn_state[threadIdx.x] = threadIdx.x < 12 ? 0 : -1;
#if __CUDA_ARCH__ >= 1200
    if (blockIdx.x == 0 && threadIdx.x < 96)
        p.ffn_state[16 + threadIdx.x] = 0;
#endif
    if (p.debug_modulation && blockIdx.x == 0 && threadIdx.x < 12)
        p.ffn_trace[threadIdx.x] = 0;
    // 缓存只整理本次可见slot；B14不会把候选的private KV写入公共历史。
    layout(p, s);
    g.sync();
    int st = 0, m = p.b * p.q, mc = p.cb * p.c;
    // 1. 调制量在M1/M3与门控残差处按需计算。debug才物化表供外部核验，
    // 后续计算不依赖该表；普通路径少一个全局阶段屏障。
    if (p.debug_modulation)
        point(p, s, g, st, TABLE, MOD, 0, 0, 0, int64_t(m) * 6 * D);
    ++st;
    if (p.candidate && m == 224) {
        norm_modulate(p, s, g, p.x, p.a[N1], bp(p, M1), p.eps[0], 0);
        st += 2; // Preserve all later queue IDs.
    } else {
    norm(p, s, g, st++, p.x, p.a[N1], m, -1, -1, p.eps[0], false, true);
    point(p, s, g, st++, MODULATE, M1, N1, 0, 0, int64_t(m) * D);
    }
    // 2. Self attention：Q/K/V投影→Q/K的RMSNorm与RoPE→FA2→输出投影。
    if (p.candidate && m == 224) {
        self_qkv_projection_group(p, s, g, st, bp(p, M1));
        st += 3; // 后续队列编号不变，独立Q/K/V共享一个入口和出口。
    } else {
        linear(p, s, g, st++, bp(p, M1), 3, bp(p, SQ0), m, D, D);
        linear(p, s, g, st++, bp(p, M1), 5, bp(p, SK0), m, D, D);
        linear(p, s, g, st++, bp(p, M1), 7, bp(p, SV), m, D, D);
    }
    if (p.candidate && p.freq) {
        norm_rope(p, s, g, bp(p, SQ0), bp(p, SQN), bp(p, SQR), 11, p.eps[3]);
        norm_rope(p, s, g, bp(p, SK0), bp(p, SKN), bp(p, SKR), 12, p.eps[4]);
        st += 4; // 保留后续动态投影队列的阶段编号。
    } else {
        norm(p, s, g, st++, bp(p, SQ0), p.a[SQN], m, 11, -1, p.eps[3], true, false);
        norm(p, s, g, st++, bp(p, SK0), p.a[SKN], m, 12, -1, p.eps[4], true, false);
        if (p.freq) {
            point(p, s, g, st++, ROPE, SQR, SQN, 0, 0, int64_t(m) * D / 2);
            point(p, s, g, st++, ROPE, SKR, SKN, 0, 0, int64_t(m) * D / 2);
        }
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
    if (!p.context_cached) {
        context_linear(p, s, g, st, 15, bp(p, CK0), mc);
        context_linear(p, s, g, st + 1, 17, bp(p, CV), mc);
    }
    st += 2;
    norm(p, s, g, st++, bp(p, CQ0), p.a[CQ], m, 21, -1, p.eps[5], true, false);
    if (!p.context_cached)
        norm(p, s, g, st, bp(p, CK0), p.a[CK], mc, 22, -1, p.eps[6], true, false);
    ++st;
    attention(p, s, g, st++, true);
    linear(p, s, g, st++, bp(p, CA), 19, bp(p, CO), m, D, D);
    point(p, s, g, st++, RESIDUAL, R2, R1, CO, -1, int64_t(m) * D);
    // 4. FFN：3072→14336→3072；保留GEMM和GELU之间原有BF16舍入。
    if (p.candidate && m == 224) {
        norm_modulate(p, s, g, bp(p, R2), p.a[N3], bp(p, M3), p.eps[2], 3);
        st += 2; // Preserve all later queue IDs.
    } else {
    norm(p, s, g, st++, bp(p, R2), p.a[N3], m, -1, -1, p.eps[2], false, true);
    point(p, s, g, st++, MODULATE, M3, N3, 0, 3, int64_t(m) * D);
    }
    const bool fused_ffn = p.candidate && m == 224;
    if (fused_ffn) {
        action_ffn_stages(p, s, g);
        st += 3;
    } else {
        linear(p, s, g, st++, bp(p, M3), 23, bp(p, F1), m, F, D);
        point(p, s, g, st++, GELU, ACT, F1, 0, 0, int64_t(m) * F);
        linear(p, s, g, st++, bp(p, ACT), 25, bp(p, F2), m, D, F);
    }
    // 5. 门控残差与临时KV状态恢复，输出仍为[B,Q,3072]。
    if (!fused_ffn)
        point(p, s, g, st, RESIDUAL, OUT, R2, F2, 5, int64_t(m) * D);
    ++st;
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
        const std::optional<Tensor> &rows, int update, const std::vector<double> &eps, bool debug,
        const std::optional<Tensor> &context_rows, const std::optional<Tensor> &context_roots,
        const std::vector<Tensor> &context_kv) {
    c10::cuda::CUDAGuard guard(x.device());
    Params p{};
    p.b = x.size(0);
    p.q = x.size(1);
    p.c = context.size(1);
    p.cb = context_roots ? context_roots->numel() : p.b;
    if (context_roots) {
        p.context_roots = context_roots->data_ptr<int64_t>();
        p.context_rows = context_rows->data_ptr<int64_t>();
    }
    p.l = temb.size(1);
    p.update = update;
    p.x = (const B *)x.data_ptr();
    p.context = (const B *)context.data_ptr();
    p.temb = temb.data_ptr();
    p.temb_float = temb.scalar_type() == at::kFloat;
    p.debug_modulation = debug;
    p.context_cached = !context_kv.empty();
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
        const int rows = (i == CK0 || i == CK || i == CV) ? p.cb * p.c : p.b * p.q;
        // 预计算缓存按K0/K/V排列；K0仅保留供debug审计，warm kernel读取K/V。
        if (p.context_cached && (i == CK0 || i == CK || i == CV))
            b[i] = context_kv[i == CK0 ? 0 : (i == CK ? 1 : 2)].view({rows, D});
        else if (i == MOD)
            b[i] = debug ? at::empty({rows, 6, D}, fo) : at::empty({0}, fo);
        else if ((i == SQN || i == SKN) && p.candidate && p.freq && !debug)
            b[i] = at::empty({0}, bo);
        else if ((i == N1 || i == N3) && p.candidate && p.b * p.q == 224 && !debug)
            b[i] = at::empty({0}, fo);
        else if (i == F1 && p.candidate && p.b * p.q == 224 && !debug)
            b[i] = at::empty({0}, bo);
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
    auto queues = at::empty({MAX_STAGES + 16 + 96}, io); // 末96项为SM120逐输出tile提交计数。
    auto ffn1_transfer_trace = at::empty({debug ? at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 5 : 0}, lo);
    p.ffn1_transfer_trace = debug ? ffn1_transfer_trace.data_ptr<int64_t>() : nullptr;
    auto ffn1_role_trace = at::empty({debug ? at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8 * 3 : 0}, lo);
    p.ffn1_role_trace = debug ? ffn1_role_trace.data_ptr<int64_t>() : nullptr;
    auto ffn_trace = at::empty({debug ? 12 : 0}, lo);
    auto projection_trace = at::empty({debug ? 6 * 96 * 4 : 0}, lo);
    p.projection_trace = debug ? projection_trace.data_ptr<int64_t>() : nullptr;
    auto projection_role_trace = at::empty({debug ? 6 * 96 * 8 * 3 : 0}, lo);
    p.projection_role_trace = debug ? projection_role_trace.data_ptr<int64_t>() : nullptr;
    p.slots = slots.data_ptr<int64_t>();
    p.free = free.data_ptr<int64_t>();
    p.info = info.data_ptr<int64_t>();
    p.queues = queues.data_ptr<int>();
    p.ffn_state = p.queues + MAX_STAGES;
    p.ffn_trace = debug ? ffn_trace.data_ptr<int64_t>() : nullptr;
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

    if (p.candidate && p.b * p.q == 224) {
        const cuuint64_t dim_a[2] = {3072, 224}, dim_b[2] = {3072, 3072};
        const cuuint64_t stride[1] = {3072 * 2};
        const cuuint32_t box_a[2] = {64, 64}, box_b[2] = {64, 128}, elem[2] = {1, 1};
        auto encode = [&](CUtensorMap *map, void *ptr, const cuuint64_t *dim, const cuuint32_t *box) {
            CUresult status = cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                2, ptr, dim, stride, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            TORCH_CHECK(status == CUDA_SUCCESS, "Projection TMA tensor map status ", int(status));
        };
        const int input[6] = {SA, N2, CA, M1, M1, M1}, weight[6] = {9, 13, 19, 3, 5, 7};
        for (int projection = 0; projection < 6; ++projection) {
            encode(&p.projection_maps[projection][0], p.a[input[projection]], dim_a, box_a);
            encode(&p.projection_maps[projection][1], const_cast<void *>(p.w[weight[projection]]), dim_b, box_b);
        }
    }

        // FFN1 M3与W1仍是row-major(K-contiguous)，第四个row tile的OOB由TMA补零。
    if (p.candidate && p.b * p.q == 224) {
        const cuuint64_t dim_a[2] = {3072,224}, dim_b[2] = {3072,14336}, stride[1] = {3072*2};
        const cuuint32_t box_a[2] = {64,64}, box_b[2] = {64,128}, elem[2] = {1,1};
        for (int i=0; i<2; ++i) {
            CUresult status=cuTensorMapEncodeTiled(&p.projection_maps[6][i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                2, i == 0 ? p.a[M3] : const_cast<void *>(p.w[23]), i == 0 ? dim_a : dim_b,
                stride, i == 0 ? box_a : box_b, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            TORCH_CHECK(status == CUDA_SUCCESS, "FFN1 TMA map status ", int(status));
        }
    }

    // Cooperative grid必须全部驻留，否则grid.sync无法完成；按实际资源计算CTA数量。
    int active = 0;
    C10_CUDA_CHECK(cudaFuncSetAttribute(whole_layer, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        sizeof(Shared)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, whole_layer, THREADS,
                                                                 sizeof(Shared)));
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id=cudaLaunchAttributeCooperative; attrs[0].val.cooperative=1;
    attrs[1].id=cudaLaunchAttributeClusterDimension; attrs[1].val.clusterDim={2,1,1};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim=dim3((p.sms*active/2)*2); cfg.blockDim=dim3(THREADS);
    cfg.dynamicSmemBytes=sizeof(Shared); cfg.stream=at::cuda::getCurrentCUDAStream();
    cfg.attrs=attrs; cfg.numAttrs=2;
    int clusters=0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters,whole_layer,&cfg));
    TORCH_CHECK(clusters>0,"cluster occupancy unavailable");
    const int blocks=clusters*2; cfg.gridDim=dim3(blocks);
    void *launch_args[] = {&p};
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,whole_layer,p));

    // 普通路径到此结束；中间量仍是计算workspace，但不再构造审计字典/资源统计。
    std::map<std::string, Tensor> result{{"output", b[OUT].view(x.sizes())}};
    if (!debug)
        return result;

    // debug额外物化调制表以保留审计字段；普通计时路径不生成它。
    const char *names[NB] = {"modulation",    "norm1",       "modulated1",     "self_q_linear",
                             "self_k_linear", "self_v",      "self_q_norm",    "self_k_norm",
                             "self_q_rope",   "self_k_rope", "self_attention", "self_out",
                             "residual1",     "norm2",       "cross_q_linear", "cross_k_linear",
                             "cross_v",       "cross_q",     "cross_k",        "cross_attention",
                             "cross_out",     "residual2",   "norm3",          "modulated3",
                             "ffn1",          "gelu",        "ffn2",           "output"};
    for (int i = 0; i < NB; ++i) {
        // debug才展开条件结果，用于与原B14中间值比较；普通路径不复制14份。
        result[names[i]] = context_roots && (i == CK0 || i == CK || i == CV)
                              ? b[i].view({p.cb, p.c, D}).index_select(0, *context_rows).reshape({p.b * p.c, D})
                              : b[i];
    }
    result["output"] = b[OUT].view(x.sizes());
    result["cache_info"] = info;
    result["logical_slots"] = slots;
    result["attention_meta"] = fa_meta;
    result["ffn_pipeline_state"] = queues.narrow(0, MAX_STAGES, 16);
    if (p.candidate && p.b * p.q == 224 && at::cuda::getCurrentDeviceProperties()->major >= 12)
        result["ffn_splitk_commit"] = queues.narrow(0, MAX_STAGES + 16, 96);
    result["ffn1_transfer_counts"] = ffn1_transfer_trace.view({p.sms,5});
    result["ffn1_role_counts"] = ffn1_role_trace.view({p.sms,8,3});
    result["ffn_pipeline_timestamps"] = ffn_trace;
    if (p.candidate && p.b * p.q == 224 && at::cuda::getCurrentDeviceProperties()->major >= 12)
        result["projection_tma_counts"] = projection_trace.view({6, 96, 4});
    if (debug && p.candidate && p.b * p.q == 224)
        result["projection_role_counts"] = projection_role_trace.view({6,96,8,3});
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
    m.def("forward", &wam_whole_layer_v1::forward, py::arg("weights"), py::arg("hidden"),
          py::arg("context"), py::arg("temb"), py::arg("rotary"), py::arg("cache"), py::arg("rows"),
          py::arg("update"), py::arg("eps"), py::arg("debug") = true,
          py::arg("context_rows") = std::nullopt, py::arg("context_roots") = std::nullopt,
          py::arg("context_kv") = std::vector<at::Tensor>{});
    m.attr("production_admitted") = false;
    m.attr("computation_launches") = 1;
    m.attr("context_tile_rows") = WAM_CONTEXT_TILE_M;
    m.attr("point_tile_elements") = 2048;
    m.attr("cooperative_epilogue") = true;
    m.attr("ffn_epilogue_fused") = true;
    m.attr("ffn_four_group_pipeline") = true;
    m.attr("ffn_tail_row32") = false;
    m.attr("ffn1_warp_pipeline") = true;
    m.attr("ffn1_output_tiles") = 448;
    m.attr("ffn1_cluster2_input_share") = bool(FFN1_SHARE_A);
}

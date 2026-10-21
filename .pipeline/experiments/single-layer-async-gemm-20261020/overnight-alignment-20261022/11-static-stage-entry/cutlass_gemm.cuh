// 一个任务计算C中的TM×128输出tile，沿K每次推进64，CUTLASS内部三级流水。
// warp3–6执行cp.async/MMA；全部8warp按四元素向量写回。
// action用TM64，context用TM128；保留原cuBLASLt的分段BF16与bias加法顺序。
#pragma once

#ifndef WAM_CONTEXT_TILE_M
#define WAM_CONTEXT_TILE_M 64
#endif
static_assert(WAM_CONTEXT_TILE_M == 64 || WAM_CONTEXT_TILE_M == 128);

using CutlassB = cutlass::bfloat16_t;
using CutlassRow = cutlass::layout::RowMajor;
using CutlassCol = cutlass::layout::ColumnMajor;
template <int TM, int Stride = 136> struct CutlassTile {
    using Config = cutlass::gemm::device::Gemm<
        CutlassB, CutlassRow, CutlassB, CutlassCol, CutlassB, CutlassRow, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, 128, 64>,
        cutlass::gemm::GemmShape<TM / 2, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<CutlassB, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
    using Base = typename Config::GemmKernel::Mma;
    using Mma = cutlass::gemm::threadblock::WamMmaMultistage<
        typename Base::Shape, typename Base::IteratorA, typename Base::SmemIteratorA,
        Base::kCacheOpA, typename Base::IteratorB, typename Base::SmemIteratorB, Base::kCacheOpB,
        float, CutlassRow, typename Base::Policy, 3>;
    union Storage {
        typename Mma::SharedStorage mainloop;
        float result[TM * Stride];
    };
};
// FFN1真实M224/N14336：M128×N256输出tile，8个warp均执行MMA。
// K32三级pipeline与分半epilogue将临时shared保持在现有整层配额内。
struct CutlassWideFFN {
    using Config = cutlass::gemm::device::Gemm<
        CutlassB, CutlassRow, CutlassB, CutlassCol, CutlassB, CutlassRow, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,256,32>,cutlass::gemm::GemmShape<64,64,32>,
        cutlass::gemm::GemmShape<16,8,16>,
        cutlass::epilogue::thread::LinearCombination<CutlassB,8,float,float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3>;
    using Mma = typename Config::GemmKernel::Mma;
    union Storage { typename Mma::SharedStorage mainloop; float result[128*136]; };
};
union CutlassGemmStorage {
    CutlassWideFFN::Storage wide_ffn;
    typename CutlassTile<64>::Storage action;
    typename CutlassTile<128>::Storage ffn1;
    // 128列结果按136列stride落shared，减少MMA accumulator转存的bank冲突。
    typename CutlassTile<WAM_CONTEXT_TILE_M, 136>::Storage context;
    float result[WAM_CONTEXT_TILE_M * 128];
};

__device__ __forceinline__ int gemm_tile_rows(int m, int n = 0) {
    // B14 FFN1的M224可用两个128行tile，减少重复读权重；K归约次序不变。
    if (m == 224 && n == 14336) return 128;
    return m >= 1024 ? WAM_CONTEXT_TILE_M : 64;
}

template <int TM, int Stride = 136>
__device__ __forceinline__ void cutlass_gemm_tile(typename CutlassTile<TM, Stride>::Storage &s,
                                                  const __nv_bfloat16 *a, const __nv_bfloat16 *w,
                                                  const __nv_bfloat16 *bias, __nv_bfloat16 *out,
                                                  int m, int n, int k, int task, bool bf16_splits) {
    using Mma = typename CutlassTile<TM, Stride>::Mma;
    const int rm = (task / ((n + 127) / 128)) * TM, cn = (task % ((n + 127) / 128)) * 128;
    int splits = 1;
    if (bf16_splits && n == 3072) {
        if (k == 14336)
            splits = m == 224 ? 2 : m == 256 ? 4 : 1;
        else if (k == 3072 && m == 224)
            splits = 2;
    }
    const int span = k / splits, physical_warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int partition = 0; partition < splits; ++partition) {
        if (physical_warp >= 3 && physical_warp < 7) {
            const int thread = int(threadIdx.x) - 96, warp = physical_warp - 3;
            typename Mma::IteratorA::Params ap{CutlassRow(k)};
            typename Mma::IteratorB::Params bp{CutlassCol(k)};
            typename Mma::IteratorA ai(
                ap, reinterpret_cast<CutlassB *>(const_cast<__nv_bfloat16 *>(a)) + partition * span,
                {m, span}, thread, {rm, 0});
            typename Mma::IteratorB bi(
                bp, reinterpret_cast<CutlassB *>(const_cast<__nv_bfloat16 *>(w)) + partition * span,
                {span, n}, thread, {0, cn});
            Mma mma(s.mainloop, thread, warp, lane);
            typename Mma::FragmentC acc;
            acc.clear();
            mma(span / 64, acc, ai, bi, acc);
            // All compute warps finish reading the input ring before aliasing it.
            asm volatile("bar.sync 2, 128;" ::: "memory");
            typename Mma::Operator::IteratorC store({s.result, CutlassRow(Stride)}, lane);
            store.add_tile_offset({warp % 2, warp / 2});
            store.store(acc);
        }
        __syncthreads();
        if constexpr (Stride == 136) {
            // 四元素向量仅减少搬运指令；FP32加法及BF16舍入仍逐元素保持。
            struct alignas(8) B4 { __nv_bfloat162 lo, hi; };
            for (int i = 4 * int(threadIdx.x); i < TM * 128; i += 4 * int(blockDim.x)) {
                const int r = rm + i / 128, c = cn + i % 128;
                if (r < m && c < n) {
                    float4 z = *reinterpret_cast<const float4 *>(s.result + (i / 128) * Stride + i % 128);
                    if (partition == splits - 1) {
                        B4 b = *reinterpret_cast<const B4 *>(bias + c);
                        float2 lo = __bfloat1622float2(b.lo), hi = __bfloat1622float2(b.hi);
                        z.x = __fadd_rn(z.x, lo.x); z.y = __fadd_rn(z.y, lo.y);
                        z.z = __fadd_rn(z.z, hi.x); z.w = __fadd_rn(z.w, hi.y);
                    }
                    if (partition > 0) {
                        B4 p = *reinterpret_cast<const B4 *>(out + r * n + c);
                        float2 lo = __bfloat1622float2(p.lo), hi = __bfloat1622float2(p.hi);
                        z.x = __fadd_rn(z.x, lo.x); z.y = __fadd_rn(z.y, lo.y);
                        z.z = __fadd_rn(z.z, hi.x); z.w = __fadd_rn(z.w, hi.y);
                    }
                    *reinterpret_cast<B4 *>(out + r * n + c) = {
                        __floats2bfloat162_rn(z.x, z.y), __floats2bfloat162_rn(z.z, z.w)};
                }
            }
        } else {
            // All warps cooperate after MMA completes. Pair stores preserve each
            // element's original FP32 add order and final BF16 rounding.
            for (int i = 2 * int(threadIdx.x); i < TM * 128; i += 2 * int(blockDim.x)) {
                const int r = rm + i / 128, c = cn + i % 128;
                if (r < m && c < n) {
                    float2 z = *reinterpret_cast<const float2 *>(s.result + (i / 128) * Stride + i % 128);
                    if (partition == splits - 1) {
                        const float2 b =
                            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(bias + c));
                        z.x = __fadd_rn(z.x, b.x);
                        z.y = __fadd_rn(z.y, b.y);
                    }
                    if (partition > 0) {
                        const float2 prev = __bfloat1622float2(
                            *reinterpret_cast<const __nv_bfloat162 *>(out + r * n + c));
                        z.x = __fadd_rn(z.x, prev.x);
                        z.y = __fadd_rn(z.y, prev.y);
                    }
                    *reinterpret_cast<__nv_bfloat162 *>(out + r * n + c) =
                        __floats2bfloat162_rn(z.x, z.y);
                }
            }
        }
        __syncthreads();
    }
}

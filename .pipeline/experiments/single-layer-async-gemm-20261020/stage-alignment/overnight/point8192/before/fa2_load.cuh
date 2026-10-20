// Address adapter only. Original FA2 helpers and licenses remain frozen.
#pragma once
#include "vendor/flash_attn/flash.h"
#include "vendor/flash_attn/flash_fwd_kernel.h"

namespace FLASH_NAMESPACE {
struct WholeParams : Flash_fwd_params {
    const void *prefix_k, *prefix_v, *current_k, *current_v;
    const int64_t *valid_slots, *cfg_rows;
    int prefix_length;
    int64_t prefix_batch, current_batch;
};

__device__ __forceinline__ void whole_group_sync() {
    // Only physical warps 3..6 call the adapted FA2 functions.
    asm volatile("bar.sync 2, 128;" ::: "memory");
}

template<bool Value, bool ClearTail, int BlockN, typename Copy,
         typename Engine, typename Layout, typename CE, typename CL>
__device__ __forceinline__ void copy_whole_kv(
    const WholeParams& p, int batch, int head, int block, Copy copy_atom,
    Tensor<Engine, Layout>& dst, const Tensor<CE, CL>& coords) {
    using E = cutlass::bfloat16_t;
    CUTE_STATIC_ASSERT_V(rank(dst) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(dst) == Int<8>{});
    #pragma unroll
    for (int m = 0; m < size<1>(dst); ++m) {
        const int row = block * BlockN + get<0>(coords(0,m,0));
        if (row < p.seqlen_k) {
            const bool prefix = row < p.prefix_length;
            const int owner = prefix && p.cfg_rows ? int(p.cfg_rows[batch]) : batch;
            const int source_row = prefix ? int(p.valid_slots[row]) : row - p.prefix_length;
            const auto* base = reinterpret_cast<const E*>(
                prefix ? (Value ? p.prefix_v : p.prefix_k)
                       : (Value ? p.current_v : p.current_k));
            const int64_t bs = prefix ? p.prefix_batch : p.current_batch;
            #pragma unroll
            for (int k = 0; k < size<2>(dst); ++k) {
                const int col = get<1>(coords(0,m,k));
                auto src = make_tensor(make_gmem_ptr(base + owner*bs +
                    source_row*p.k_row_stride + head*p.k_head_stride + col),
                    cute::Layout<Shape<_8>,Stride<_1>>{});
                cute::copy(copy_atom, src, dst(_,m,k));
            }
        } else if (ClearTail) {
            cute::clear(dst(_,m,_));
        }
    }
}
} // namespace FLASH_NAMESPACE

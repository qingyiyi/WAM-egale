#pragma once
#include "vendor/flash_attn/flash_fwd_kernel.h"
namespace FLASH_NAMESPACE {
struct GroupParams : Flash_fwd_params {
    const void *prefix_k, *prefix_v, *current_k, *current_v;
    const int64_t *valid_slots;
    int prefix_length, candidates, group_base, group_cfg;
    int64_t pk_batch, pk_row, pk_head, pv_batch, pv_row, pv_head;
    int64_t ck_batch, ck_row, ck_head, cv_batch, cv_row, cv_head;
};

template<bool Value, bool ClearTail, typename TiledCopy, typename Engine, typename Layout,
         typename CoordEngine, typename CoordLayout>
__forceinline__ __device__ void copy_group_kv(
        const GroupParams &p, int bidb, int bidh, int n_block, TiledCopy tiled_copy,
        Tensor<Engine, Layout> &dest, const Tensor<CoordEngine, CoordLayout> &coord) {
    using Element = cutlass::bfloat16_t;
    CUTE_STATIC_ASSERT_V(rank(dest) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(dest) == Int<8>{});
    #pragma unroll
    for (int m=0; m<size<1>(dest); ++m) {
        const int row=n_block*64+get<0>(coord(0,m,0));
        if (row<p.seqlen_k) {
            const bool prefix=row<p.prefix_length;
            const int private_row=row-p.prefix_length;
            const int64_t owner=prefix?p.group_cfg:p.group_base+private_row/16;
            const int64_t source_row=prefix?p.valid_slots[row]:private_row%16;
            const Element *base=reinterpret_cast<const Element *>(
                prefix?(Value?p.prefix_v:p.prefix_k):(Value?p.current_v:p.current_k));
            const int64_t bs=prefix?(Value?p.pv_batch:p.pk_batch):(Value?p.cv_batch:p.ck_batch);
            const int64_t rs=prefix?(Value?p.pv_row:p.pk_row):(Value?p.cv_row:p.ck_row);
            const int64_t hs=prefix?(Value?p.pv_head:p.pk_head):(Value?p.cv_head:p.ck_head);
            #pragma unroll
            for (int k=0;k<size<2>(dest);++k) {
                const int col=get<1>(coord(0,m,k));
                auto source=make_tensor(make_gmem_ptr(base+owner*bs+source_row*rs+bidh*hs+col),
                    cute::Layout<Shape<_8>,Stride<_1>>{});
                cute::copy(tiled_copy,source,dest(_,m,k));
            }
        } else if (ClearTail) {cute::clear(dest(_,m,_));}
    }
}

template<typename Acc,typename Coord>
__forceinline__ __device__ void apply_group_visibility(const GroupParams &p,Acc &acc,const Coord &coord,int nb) {
    #pragma unroll
    for (int i=0;i<size(acc);++i) {
        const int qr=get<0>(coord(i));
        const int kr=nb*64+get<1>(coord(i));
        if (kr>=p.prefix_length && (kr-p.prefix_length)/16 != qr/16) acc(i)=-INFINITY;
    }
}
}

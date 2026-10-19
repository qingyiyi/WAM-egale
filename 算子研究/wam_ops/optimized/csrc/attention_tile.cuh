// Device-callable output-query tiles over the pinned FlashAttention mainloop.
// The vendor math, traversal and split/combine order are unchanged.
#pragma once

namespace wam_tiles {
struct AttentionTile {
    int batch, head, query_block, kv_split, kv_splits;
};

__host__ __device__ inline AttentionTile standard_task(int id, int batches, int heads,
                                                       int query_blocks) {
    return {(id / query_blocks) % batches,
            id / (query_blocks * batches), id % query_blocks, 0, 1};
}

__host__ __device__ inline AttentionTile split_task(int id, int heads,
                                                    int query_blocks, int splits) {
    const int bh = id / (query_blocks * splits);
    return {bh / heads, bh % heads, id % query_blocks,
            (id / query_blocks) % splits, splits};
}

template<class Traits, bool EvenMN, class Params>
__device__ __forceinline__ void attention_tile(const Params& p, AttentionTile task) {
    FLASH_NAMESPACE::compute_attn_1rowblock<Traits,
        false, false, false, false, EvenMN, true, false, false>(
            p, task.batch, task.head, task.query_block);
}

template<class Traits, bool EvenMN, class Params>
__device__ __forceinline__ void attention_split_tile(const Params& p, AttentionTile task) {
    FLASH_NAMESPACE::compute_attn_1rowblock_splitkv<Traits,
        false, false, false, EvenMN, true, false, true, false>(
            p, task.batch, task.head, task.query_block, task.kv_split, task.kv_splits);
}
} // namespace wam_tiles

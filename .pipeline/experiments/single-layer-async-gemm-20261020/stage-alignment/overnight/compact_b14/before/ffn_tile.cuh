// Adapted from NVIDIA CUTLASS v3.5.1 gemm/kernel/gemm.h (BSD-3-Clause).
// Original license: .pipeline/cache/cutlass-v3.5.1/LICENSE.txt.
// Only non-split-K execution; explicit coordinates replace implicit blockIdx.
#pragma once

namespace wam_tiles {
struct GemmTile { int row, column; };

template<class Kernel>
__device__ __forceinline__ void ffn_tile(
    const typename Kernel::Params& p, typename Kernel::SharedStorage& storage,
    GemmTile tile) {
    static_assert(!Kernel::kSplitKSerial, "tile ABI does not implement serial split-K");
    using Mma = typename Kernel::Mma;
    using Epilogue = typename Kernel::Epilogue;
    using OutputOp = typename Kernel::OutputOp;
    if (tile.row >= p.grid_tiled_shape.m() || tile.column >= p.grid_tiled_shape.n()) return;
    const int thread = threadIdx.x;
    const int warp = cutlass::canonical_warp_idx_sync();
    const int lane = thread % 32;
    const int k_iterations = (p.problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
    typename Mma::IteratorA a(p.params_A, p.ref_A.data(),
        {p.problem_size.m(), p.problem_size.k()}, thread,
        {tile.row * Mma::Shape::kM, 0}, p.gather_A_indices);
    typename Mma::IteratorB b(p.params_B, p.ref_B.data(),
        {p.problem_size.k(), p.problem_size.n()}, thread,
        {0, tile.column * Mma::Shape::kN}, p.gather_B_indices);
    Mma mma(storage.main_loop, thread, warp, lane);
    typename Mma::FragmentC accumulator;
    accumulator.clear();
    mma(k_iterations, accumulator, a, b, accumulator);
    const cutlass::MatrixCoord offset(tile.row * Mma::Shape::kM, tile.column * Mma::Shape::kN);
    typename Epilogue::OutputTileIterator c(p.params_C, p.ref_C.data(),
        p.problem_size.mn(), thread, offset, p.scatter_D_indices);
    typename Epilogue::OutputTileIterator d(p.params_D, p.ref_D.data(),
        p.problem_size.mn(), thread, offset, p.scatter_D_indices);
    OutputOp output_op(p.output_op);
    Epilogue epilogue(storage.epilogue, thread, warp, lane);
    epilogue(output_op, d, accumulator, c);
}
} // namespace wam_tiles

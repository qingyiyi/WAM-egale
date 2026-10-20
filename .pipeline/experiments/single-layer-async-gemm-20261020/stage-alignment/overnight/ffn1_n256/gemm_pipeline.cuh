// GEMM设备入口：大M交给CUTLASS；B2 action的小M32保留原来的64×64/K32路径。
// 小M路径的warp1/2加载A/W，warp3–6计算，warp7写回；双buffer交替复用。
// split归约对应已观测的原库算术顺序，不可作为一般的调参开关删除。
#pragma once

__device__ __forceinline__ void copy16_async(B *dst, const B *src, bool valid) {
    const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem), "l"(src), "r"(bytes)
                 : "memory");
}

template <int TN, int TK>
__device__ __forceinline__ void stage_gemm_copy(GemmStorage<TN, TK> &s, const B *a, const B *w,
                                                int rm, int cn, int m, int n, int k, int z,
                                                int buffer) {
    // Every loader lane owns eight non-overlapping 16-byte destinations.
    if (warp() == 1 || warp() == 2) {
        const int rows = warp() == 1 ? 64 : TN;
        for (int i = lane(); i < rows * TK / 8; i += 32) {
            const int row = i / (TK / 8), col = (i % (TK / 8)) * 8;
            const bool valid = (warp() == 1 ? rm + row < m : cn + row < n);
            const B *src = (warp() == 1 ? a : w);
            // Invalid padded rows use an in-bounds pointer and zero source size.
            if (valid)
                src += (warp() == 1 ? rm + row : cn + row) * k + z + col;
            B *dst = (warp() == 1 ? s.x[buffer] : s.w[buffer]) + row * (TK + 8) + col;
            copy16_async(dst, src, valid);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
}

__device__ __forceinline__ void wait_gemm_copy() {
    if (warp() == 1 || warp() == 2)
        asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template <int TN, int TK>
__device__ void gemm_tile_impl(GemmStorage<TN, TK> &s, const B *a, const B *w, const B *bias,
                               B *out, int m, int n, int k, int task, bool bf16_splits = false) {
    constexpr int LD = TK + 8, WN = TN / 2, FRAG = 2 * (WN / 16);
    const int cols = (n + TN - 1) / TN, rm = (task / cols) * 64, cn = (task % cols) * TN;
    // Fixed signatures observed in the actual cuBLASLt call log; no search.
    int splits = 1;
    if (bf16_splits && n == 3072) {
        if (k == 14336)
            splits = (m == 32 ? 7 : m == 224 ? 2 : m == 256 ? 4 : 1);
        else if (k == 3072 && m == 224)
            splits = 2;
    }
    // Observed sliced1x2 kernel: two interleaved FP32 K slices, bias before their sum.
    const bool internal_split = bf16_splits && m == 32 && n == 3072 && k == 3072;
    const int span = k / splits;
    for (int partition = 0; partition < splits; ++partition) {
        const int first = partition * span, last = first + span;
        wm::fragment<wm::accumulator, 16, 16, 16, float> acc[FRAG], second[FRAG];
        if (compute())
            for (int j = 0; j < FRAG; ++j) {
                wm::fill_fragment(acc[j], 0.f);
                if (internal_split)
                    wm::fill_fragment(second[j], 0.f);
            }

        stage_gemm_copy(s, a, w, rm, cn, m, n, k, first, 0);
        wait_gemm_copy();
        __syncthreads(); // First buffer ready for all consumer warps.
        for (int z = first; z < last; z += TK) {
            const int current = ((z - first) / TK) & 1, next = current ^ 1;
            // The other buffer is free: its consumers finished at the prior barrier.
            if (z + TK < last)
                stage_gemm_copy(s, a, w, rm, cn, m, n, k, z + TK, next);
            if (compute()) {
                int cw = warp() - 3, m0 = (cw / 2) * 32, n0 = (cw % 2) * WN;
#pragma unroll
                for (int t = 0; t < TK / 16; ++t) {
                    wm::fragment<wm::matrix_a, 16, 16, 16, B, wm::row_major> af[2];
                    wm::fragment<wm::matrix_b, 16, 16, 16, B, wm::col_major> wf[WN / 16];
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wm::load_matrix_sync(af[j], s.x[current] + (m0 + j * 16) * LD + t * 16, LD);
                    }
#pragma unroll
                    for (int j = 0; j < WN / 16; ++j)
                        wm::load_matrix_sync(wf[j], s.w[current] + (n0 + j * 16) * LD + t * 16, LD);
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < WN / 16; ++j)
                            if (internal_split && ((z / 32) & 1))
                                wm::mma_sync(second[i * (WN / 16) + j], af[i], wf[j],
                                             second[i * (WN / 16) + j]);
                            else
                                wm::mma_sync(acc[i * (WN / 16) + j], af[i], wf[j],
                                             acc[i * (WN / 16) + j]);
                }
            }
            wait_gemm_copy();
            // Both obligations: next input ready AND current buffer no longer read.
            __syncthreads();
        }
        if (compute()) {
            int cw = warp() - 3, m0 = (cw / 2) * 32, n0 = (cw % 2) * WN;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < WN / 16; ++j) {
                    wm::store_matrix_sync(s.y + (m0 + i * 16) * TN + n0 + j * 16,
                                          acc[i * (WN / 16) + j], TN, wm::mem_row_major);
                    if (internal_split)
                        wm::store_matrix_sync(s.partial_y + (m0 + i * 16) * TN + n0 + j * 16,
                                              second[i * (WN / 16) + j], TN, wm::mem_row_major);
                }
        }
        __syncthreads();
        if (warp() == 7)
            for (int i = lane(); i < 64 * TN; i += 32) {
                int r = rm + i / TN, c = cn + i % TN;
                if (r < m && c < n) {
                    // Bias is a final epilogue; intermediate partials remain unbiased.
                    float z = s.y[i];
                    if (internal_split)
                        z = __fadd_rn(__fadd_rn(z, value(bias[c])), s.partial_y[i]);
                    if (partition == splits - 1 && !internal_split)
                        z = __fadd_rn(z, value(bias[c]));
                    if (partition > 0)
                        z = __fadd_rn(z, value(out[r * n + c]));
                    out[r * n + c] = bf(z);
                }
            }
        __syncthreads();
    } // serial partitions, same output tile owner
}

// Large-M profiles are dominated by projection waves. Reuse one 64/128-row A
// tile across twice as many columns, with fewer K staging barriers. Tiny M32
// retains its original tile and observed sliced reduction.
__device__ void gemm_tile(Shared &shared, const B *a, const B *w, const B *bias, B *out, int m,
                          int n, int k, int task, bool bf16_splits = false) {
    if (m == 224 && n == 14336)
        cutlass_gemm_tile<128, 256>(shared.tile.cutlass_gemm.ffn1, a, w, bias, out, m,
                               n, k, task, bf16_splits);
    else if (m >= 1024)
        cutlass_gemm_tile<WAM_CONTEXT_TILE_M>(shared.tile.cutlass_gemm.context, a, w, bias, out, m,
                                              n, k, task, bf16_splits);
    else if (m >= 128)
        cutlass_gemm_tile<64>(shared.tile.cutlass_gemm.action, a, w, bias, out, m, n, k, task,
                              bf16_splits);
    else
        gemm_tile_impl<64, 32>(shared.tile.gemm, a, w, bias, out, m, n, k, task, bf16_splits);
}

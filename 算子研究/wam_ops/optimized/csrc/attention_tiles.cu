// Independent SM80/BF16/D128 SDPA forward extension.
// Baseline: installed PyTorch 2.9.0+cu126, commit
// 0fabc3ba44823f257e70ce397d989c8de5e362c1. Its pinned FlashAttention forward
// and CUTLASS headers are copied byte-for-byte into csrc/vendor; provenance,
// exact commits and licenses are in vendor/flash_attn/UPSTREAM.json.
// Only this translation unit is compiled. No ATen SDPA, pytorch_flash entry
// point, or installed FlashAttention extension is invoked. The wrapper is
// specialized to the real C0 inference contract, not the full PyTorch API.
#define FLASH_NAMESPACE wam_tile_flash
#define UNFUSE_FMA
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_UNEVEN_K
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "vendor/flash_attn/flash_fwd_launch_template.h"

#include "attention_tile.cuh"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>

namespace wam_optimized_sdpa {
using Params = wam_tile_flash::Flash_fwd_params;
using Standard = Flash_fwd_kernel_traits<
    128, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using SplitKV = Flash_fwd_kernel_traits<
    128, 64, 128, 4, false, false, cutlass::bfloat16_t>;

// Static launch adapter. A task is one query-output tile (or one legal KV split).
// These device calls accept explicit coordinates and are reusable by a future
// persistent worker; this adapter itself launches one CTA per task.
template<bool EvenMN>
__global__ void standard_tiles(KERNEL_PARAM_MODIFIER const Params params) {
    const int qblocks=(params.seqlen_q+127)/128;
    auto task=wam_tiles::standard_task(blockIdx.x,params.b,params.h,qblocks);
    wam_tiles::attention_tile<Standard,EvenMN>(params,task);
}
template<bool EvenMN>
__global__ void split_tiles(KERNEL_PARAM_MODIFIER const Params params) {
    const int qblocks=(params.seqlen_q+63)/64;
    auto task=wam_tiles::split_task(blockIdx.x,params.h,qblocks,params.num_splits);
    wam_tiles::attention_split_tile<SplitKV,EvenMN>(params,task);
}

// PyTorch 2.9.0 flash_api.cpp's occupancy-based split heuristic, specialized
// to D128 and no dropout. Choosing splits is NOT a different attention method.
int choose_splits(int batch, int heads, int sq, int sk, int sms) {
    const int mblocks = (sq + 63) / 64;
    const int nblocks = (sk + 127) / 128;
    const int work = batch * heads * mblocks;
    const int capacity = sms * 2;
    if (work >= 0.8f * capacity) return 1;
    const int limit = std::min({128, capacity, nblocks});
    std::array<float, 128> efficiency{};
    float best = 0;
    for (int n = 1; n <= limit; ++n) {
        if (n > 1 && (nblocks + n - 1) / n == (nblocks + n - 2) / (n - 1))
            continue;
        const float waves = float(work * n) / capacity;
        efficiency[n - 1] = waves / std::ceil(waves);
        best = std::max(best, efficiency[n - 1]);
    }
    for (int n = 1; n <= limit; ++n)
        if (efficiency[n - 1] >= 0.85 * best) return n;
    return 1;
}

template <auto Kernel, size_t Smem>
void configure_smem(int device) {
    // cudaFuncSetAttribute applies per device. Only the first call on each
    // device needs it, and warmup keeps it out of steady-state measurements.
    static std::array<std::atomic<bool>, 64> ready{};
    static std::mutex mutex;
    TORCH_CHECK(device >= 0 && device < 64, "unsupported CUDA device index");
    if (!ready[device].load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ready[device].load(std::memory_order_relaxed)) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem));
            ready[device].store(true, std::memory_order_release);
        }
    }
}

template <bool EvenMN>
void launch_standard(const Params& params, cudaStream_t stream, int device, bool tiled) {
    if(tiled) {
        constexpr auto kernel=&standard_tiles<EvenMN>;
        configure_smem<kernel,Standard::kSmemSize>(device);
        int count=((params.seqlen_q+127)/128)*params.b*params.h;
        kernel<<<count,Standard::kNThreads,Standard::kSmemSize,stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
    }
    constexpr auto kernel = &wam_tile_flash::flash_fwd_kernel<
        Standard, false, false, false, false, EvenMN, true, false, false>;
    configure_smem<kernel, Standard::kSmemSize>(device);
    const dim3 grid((params.seqlen_q + 127) / 128, params.b, params.h);
    kernel<<<grid, Standard::kNThreads, Standard::kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool EvenMN>
void launch_split(const Params& params, cudaStream_t stream, int device, bool tiled) {
    if(tiled) {
        constexpr auto kernel=&split_tiles<EvenMN>;
        configure_smem<kernel,SplitKV::kSmemSize>(device);
        int count=((params.seqlen_q+63)/64)*params.num_splits*params.b*params.h;
        kernel<<<count,SplitKV::kNThreads,SplitKV::kSmemSize,stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
    }
    constexpr auto kernel = &wam_tile_flash::flash_fwd_splitkv_kernel<
        SplitKV, false, false, false, EvenMN, true, false, true, false>;
    configure_smem<kernel, SplitKV::kSmemSize>(device);
    const dim3 grid((params.seqlen_q + 63) / 64,
                    params.num_splits, params.b * params.h);
    kernel<<<grid, SplitKV::kNThreads, SplitKV::kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int LogSplits>
void launch_combine(const Params& params, cudaStream_t stream) {
    const dim3 grid((params.b * params.h * params.seqlen_q + 3) / 4);
    wam_tile_flash::flash_fwd_splitkv_combine_kernel<SplitKV, 4, LogSplits, true>
        <<<grid, SplitKV::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_inputs(const at::Tensor& q, const at::Tensor& k,
                  const at::Tensor& v, double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q/K/V devices differ");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16
                && v.scalar_type() == at::kBFloat16, "only BF16 is supported");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B,S,H,128]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "Q/K/V must be contiguous; no hidden timed copies");
    TORCH_CHECK(k.sizes() == v.sizes(), "K/V shapes differ");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2)
                && q.size(3) == 128 && k.size(3) == 128, "Q/K/V dimensions differ or D != 128");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && k.size(1) > 0,
                "empty attention is outside the C0 contract");
    TORCH_CHECK(q.numel() <= std::numeric_limits<int32_t>::max()
                && k.numel() <= std::numeric_limits<int32_t>::max(), "tensor exceeds int32 kernel indexing");
    TORCH_CHECK(std::isfinite(scale) && scale > 0, "scale must be finite and positive");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
                "this SDPA extension is inference-only");
}

at::Tensor sdpa(const at::Tensor& q, const at::Tensor& k,
                const at::Tensor& v, std::optional<double> scale_arg, bool tiled) {
    // The unchanged input contract below requires D128. Resolve the optional
    // default in the native entry, as PyTorch does, without Python shape/pow.
    const double scale = scale_arg.value_or(1.0 / std::sqrt(128.0));
    check_inputs(q, k, v, scale);
    c10::cuda::CUDAGuard guard(q.device());
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 8 && props->minor == 0, "this build supports SM80 (A100) only");
    auto out = at::empty_like(q);
    const int b = q.size(0), sq = q.size(1), sk = k.size(1), h = q.size(2);
    auto lse = at::empty({b, h, sq}, q.options().dtype(at::kFloat));
    Params p{};
    p.q_ptr = q.data_ptr(); p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr();
    p.o_ptr = out.data_ptr(); p.softmax_lse_ptr = lse.data_ptr();
    p.q_batch_stride = q.stride(0); p.q_row_stride = q.stride(1); p.q_head_stride = q.stride(2);
    p.k_batch_stride = k.stride(0); p.k_row_stride = k.stride(1); p.k_head_stride = k.stride(2);
    p.v_batch_stride = v.stride(0); p.v_row_stride = v.stride(1); p.v_head_stride = v.stride(2);
    p.o_batch_stride = out.stride(0); p.o_row_stride = out.stride(1); p.o_head_stride = out.stride(2);
    p.b = b; p.h = h; p.h_k = h; p.h_h_k_ratio = 1;
    p.seqlen_q = sq; p.seqlen_k = sk;
    p.seqlen_q_rounded = (sq + 127) / 128 * 128;
    p.seqlen_k_rounded = (sk + 127) / 128 * 128;
    p.d = 128; p.d_rounded = 128;
    p.scale_softmax = float(scale);
    p.scale_softmax_log2 = float(scale) * M_LOG2E;
    p.p_dropout = 1.f; p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.f; p.scale_softmax_rp_dropout = float(scale);
    p.window_size_left = -1; p.window_size_right = -1;
    p.softcap = 0.f;
    p.unpadded_lse = false; p.seqlenq_ngroups_swapped = false;
    p.is_bf16 = true; p.is_causal = false; p.is_seqlens_k_cumulative = true;
    p.num_splits = choose_splits(b, h, sq, sk, props->multiProcessorCount);
    at::Tensor partial_lse, partial_out;
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (p.num_splits > 1) {
        partial_lse = at::empty({p.num_splits, b, h, sq}, q.options().dtype(at::kFloat));
        partial_out = at::empty({p.num_splits, b, h, sq, 128}, q.options().dtype(at::kFloat));
        p.softmax_lseaccum_ptr = partial_lse.data_ptr();
        p.oaccum_ptr = partial_out.data_ptr();
        if (sq % 64 == 0 && sk % 128 == 0) launch_split<true>(p, stream, q.get_device(), tiled);
        else launch_split<false>(p, stream, q.get_device(), tiled);
        if (p.num_splits <= 2) launch_combine<1>(p, stream);
        else if (p.num_splits <= 4) launch_combine<2>(p, stream);
        else if (p.num_splits <= 8) launch_combine<3>(p, stream);
        else if (p.num_splits <= 16) launch_combine<4>(p, stream);
        else if (p.num_splits <= 32) launch_combine<5>(p, stream);
        else if (p.num_splits <= 64) launch_combine<6>(p, stream);
        else launch_combine<7>(p, stream);
    } else {
        if (sq % 128 == 0 && sk % 64 == 0) launch_standard<true>(p, stream, q.get_device(), tiled);
        else launch_standard<false>(p, stream, q.get_device(), tiled);
    }
    return out;
}
} // namespace wam_optimized_sdpa

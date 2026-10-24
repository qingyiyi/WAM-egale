// Direct original-torch comparison and a private shared-address specialization.
// The original FlashAttention/CUTLASS commits and licenses live in csrc/vendor.
#define FLASH_NAMESPACE speculative_flash
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "vendor/flash_attn/flash_fwd_launch_template.h"
#include "shared_flash_fwd_kernel.h"
#include <cmath>

using Params = speculative_flash::SharedParams;
using Traits = Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>;

__global__ void shared_kernel(KERNEL_PARAM_MODIFIER const Params p) {
#if defined(ARCH_SUPPORTS_FLASH)
    speculative_flash::compute_shared_attn_1rowblock<
        Traits, false, false, false, false, false, true, false, false>(
            p, blockIdx.y, blockIdx.z, blockIdx.x);
#endif
}

void check(const at::Tensor& q) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.is_contiguous(),
                "Q must be contiguous CUDA BF16");
    TORCH_CHECK(q.sizes() == at::IntArrayRef({14, 16, 24, 128}), "fixed Q contract [14,16,24,128]");
    TORCH_CHECK(!q.requires_grad(), "inference only");
}

void check_kv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int batch, int length) {
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device()
                && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "K/V device or dtype mismatch");
    TORCH_CHECK(k.sizes() == at::IntArrayRef({batch, length, 24, 128}) && v.sizes() == k.sizes(),
                "K/V shape mismatch");
    for (const auto& x : {k, v}) {
        TORCH_CHECK(x.stride(3) == 1 && x.stride(2) == 128 && x.stride(1) % 8 == 0
                    && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                    "K/V must preserve aligned BF16 vectors");
        TORCH_CHECK(!x.requires_grad(), "inference only");
    }
}

Params base_params(const at::Tensor& q, const at::Tensor& out, const at::Tensor& lse, int sk) {
    Params p{};
    const float scale = float(1.0 / std::sqrt(128.0));
    p.q_ptr = q.data_ptr(); p.o_ptr = out.data_ptr(); p.softmax_lse_ptr = lse.data_ptr();
    p.q_batch_stride = q.stride(0); p.q_row_stride = q.stride(1); p.q_head_stride = q.stride(2);
    p.o_batch_stride = out.stride(0); p.o_row_stride = out.stride(1); p.o_head_stride = out.stride(2);
    p.b = 14; p.h = 24; p.h_k = 24; p.h_h_k_ratio = 1;
    p.seqlen_q = 16; p.seqlen_k = sk;
    p.seqlen_q_rounded = 128; p.seqlen_k_rounded = (sk + 127) / 128 * 128;
    p.d = p.d_rounded = 128;
    p.scale_softmax = scale; p.scale_softmax_log2 = scale * M_LOG2E;
    p.p_dropout = 1.f; p.p_dropout_in_uint8_t = 255; p.rp_dropout = 1.f;
    p.scale_softmax_rp_dropout = scale;
    p.window_size_left = -1; p.window_size_right = -1;
    p.is_bf16 = true; p.is_causal = false; p.is_seqlens_k_cumulative = true;
    p.num_splits = 1; // 14*24 = 336 CTAs: original torch SM80 heuristic selects one split.
    return p;
}

void device_check() {
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 12 && props->minor == 0,
                "independent RTX5090 SM120 migration contract required");
}

at::Tensor dense(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    check(q);
    TORCH_CHECK(k.dim() == 4 && k.size(1) > 0, "nonempty K required");
    check_kv(q, k, v, 14, k.size(1));
    c10::cuda::CUDAGuard guard(q.device()); device_check();
    auto out = at::empty_like(q), lse = at::empty({14,24,16}, q.options().dtype(at::kFloat));
    Params p = base_params(q, out, lse, k.size(1));
    p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr();
    p.k_batch_stride = k.stride(0); p.k_row_stride = k.stride(1); p.k_head_stride = k.stride(2);
    p.v_batch_stride = v.stride(0); p.v_row_stride = v.stride(1); p.v_head_stride = v.stride(2);
    constexpr auto kernel = &speculative_flash::flash_fwd_kernel<
        Traits, false, false, false, false, false, true, false, false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    kernel<<<dim3(1,14,24),128,Traits::kSmemSize,at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor shared(const at::Tensor& q, const at::Tensor& pk, const at::Tensor& pv,
                  const at::Tensor& ck, const at::Tensor& cv,
                  const at::Tensor& slots, const at::Tensor& rows) {
    check(q);
    TORCH_CHECK(pk.dim() == 4 && pk.size(1) > 0, "nonempty prefix pool required");
    check_kv(q, pk, pv, 2, pk.size(1)); check_kv(q, ck, cv, 14, 16);
    TORCH_CHECK(slots.device() == q.device() && rows.device() == q.device()
                && slots.scalar_type() == at::kLong && rows.scalar_type() == at::kLong
                && slots.dim() == 1 && rows.sizes() == at::IntArrayRef({14})
                && slots.is_contiguous() && rows.is_contiguous()
                && slots.numel() > 0 && slots.numel() <= pk.size(1), "slot/row metadata contract");
    // Trusted private adapter checks slot bounds, order, CFG rows and tensor versions.
    c10::cuda::CUDAGuard guard(q.device()); device_check();
    auto out = at::empty_like(q), lse = at::empty({14,24,16}, q.options().dtype(at::kFloat));
    Params p = base_params(q, out, lse, slots.numel() + 16);
    p.k_ptr = pk.data_ptr(); p.v_ptr = pv.data_ptr();
    p.prefix_k = pk.data_ptr(); p.prefix_v = pv.data_ptr();
    p.current_k = ck.data_ptr(); p.current_v = cv.data_ptr();
    p.valid_slots = slots.data_ptr<int64_t>(); p.cfg_rows = rows.data_ptr<int64_t>();
    p.prefix_length = slots.numel();
    p.pk_batch=pk.stride(0); p.pk_row=pk.stride(1); p.pk_head=pk.stride(2);
    p.pv_batch=pv.stride(0); p.pv_row=pv.stride(1); p.pv_head=pv.stride(2);
    p.ck_batch=ck.stride(0); p.ck_row=ck.stride(1); p.ck_head=ck.stride(2);
    p.cv_batch=cv.stride(0); p.cv_row=cv.stride(1); p.cv_head=cv.stride(2);
    C10_CUDA_CHECK(cudaFuncSetAttribute(shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    shared_kernel<<<dim3(1,14,24),128,Traits::kSmemSize,at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense", &dense, "Pinned original torch Flash math (dense comparison only)");
    m.def("shared", &shared, "Same Flash math, CFG2 prefix addresses plus private current KV");
    m.attr("pytorch_commit") = "0fabc3ba44823f257e70ce397d989c8de5e362c1";
    m.attr("flash_commit") = "979702c87a8713a8e0a5e9fee122b90d2ef13be5";
    m.attr("compiled_cuda_major") = __CUDACC_VER_MAJOR__;
    m.attr("compiled_cuda_minor") = __CUDACC_VER_MINOR__;
}

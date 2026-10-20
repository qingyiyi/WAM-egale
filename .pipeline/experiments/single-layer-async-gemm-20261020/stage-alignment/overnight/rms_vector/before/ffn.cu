// SM80 FFN device primitives. Research only; NOT production admitted.
// CUTLASS: NVIDIA BSD-3-Clause, pinned by tools/fetch_cutlass.py.
// GELU expression follows PyTorch 0fabc3b ActivationGeluKernel.cu (BSD-3-Clause).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cmath>

using BF16 = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Shape = cutlass::gemm::GemmShape<64, 128, 64>;
using Warp = cutlass::gemm::GemmShape<32, 64, 64>;
using Instruction = cutlass::gemm::GemmShape<16, 8, 16>;
using Epilogue = cutlass::epilogue::thread::LinearCombination<BF16, 8, float, float>;
__device__ __forceinline__ c10::BFloat16 gelu_value(c10::BFloat16 value);

struct RoundedGelu : Epilogue {
    using Epilogue::Epilogue;
    CUTLASS_DEVICE FragmentOutput activate(FragmentOutput values) const {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Epilogue::kCount; ++i)
            values[i] = BF16(float(gelu_value(c10::BFloat16(float(values[i])))));
        return values;
    }
    CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& a, FragmentSource const& c) const {
        return activate(Epilogue::operator()(a, c)); // BF16 rounding happens BEFORE GELU.
    }
    CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& a) const {
        return activate(Epilogue::operator()(a));
    }
};

template<class OutputOp>
using FFNGemm = cutlass::gemm::device::Gemm<BF16, Row, BF16, Col, BF16, Row, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Shape, Warp, Instruction,
    OutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
    8, 8, false>;
using Gemm = FFNGemm<Epilogue>;
using FusedGemm = FFNGemm<RoundedGelu>;
using Kernel = Gemm::GemmKernel;

// The actual MMA/mainloop/epilogue execute inside this CTA, not via a host BLAS call.
template<class K>
__global__ void linear_tile(typename K::Params params) {
    extern __shared__ __align__(16) unsigned char storage[];
    K operation;
    operation(params, *reinterpret_cast<typename K::SharedStorage*>(storage));
}

void tensor_contract(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
        "requires contiguous CUDA BF16 tensors");
}

template<class G>
void launch_linear(const torch::Tensor& x, const torch::Tensor& weight,
                const torch::Tensor& bias, torch::Tensor out) {
    using K0 = typename G::GemmKernel;
    for (const auto& t : {x, weight, bias, out}) tensor_contract(t);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && bias.dim() == 1 && out.dim() == 2,
        "linear matrix ranks");
    const int M = x.size(0), K = x.size(1), N = weight.size(0);
    TORCH_CHECK((M == 32 || M == 224) && weight.size(1) == K && bias.numel() == N &&
        out.size(0) == M && out.size(1) == N, "registered FFN shapes only");
    TORCH_CHECK((K == 3072 && N == 14336) || (K == 14336 && N == 3072),
        "registered FFN widths only");
    TORCH_CHECK(x.device() == weight.device() && x.device() == bias.device() && x.device() == out.device(),
        "all matrices must use the same device");
    c10::cuda::CUDAGuard guard(x.device());
    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 8 && props->minor == 0, "SM80 only");
    auto A = reinterpret_cast<const BF16*>(x.data_ptr());
    auto B = reinterpret_cast<const BF16*>(weight.data_ptr());
    auto C = reinterpret_cast<const BF16*>(bias.data_ptr());
    auto D = reinterpret_cast<BF16*>(out.data_ptr());
    // weight[N,K] is B[K,N] column-major. C stride0 broadcasts the original bias.
    typename G::Arguments args({M, N, K}, {A, K}, {B, K}, {C, 0}, {D, N}, {1.f, 1.f});
    TORCH_CHECK(G::can_implement(args) == cutlass::Status::kSuccess, "CUTLASS shape/alignment gate");
    typename G::ThreadblockSwizzle swizzle;
    auto tiled = swizzle.get_tiled_shape(args.problem_size, {Shape::kM, Shape::kN, Shape::kK}, 1);
    typename K0::Params params(args.problem_size, tiled, args.ref_A.non_const_ref(),
        args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D, args.epilogue);
    const int smem = sizeof(typename K0::SharedStorage);
    C10_CUDA_CHECK(cudaFuncSetAttribute(linear_tile<K0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    linear_tile<K0><<<swizzle.get_grid_shape(tiled), K0::kThreadCount, smem,
        at::cuda::getCurrentCUDAStream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void device_linear_out(const torch::Tensor& x, const torch::Tensor& weight,
                       const torch::Tensor& bias, torch::Tensor out) {
    launch_linear<Gemm>(x, weight, bias, out);
}

void ffn1_gelu_out(const torch::Tensor& x, const torch::Tensor& weight,
                   const torch::Tensor& bias, torch::Tensor out) {
    TORCH_CHECK(x.size(1) == 3072 && weight.size(0) == 14336, "FFN1+GELU only");
    launch_linear<FusedGemm>(x, weight, bias, out);
}

__device__ __forceinline__ c10::BFloat16 gelu_value(c10::BFloat16 value) {
    // Keep input BF16 -> FP32 expression -> BF16 output, including original tanh.
    const float x = static_cast<float>(value);
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float kappa = 0.044715;
    const auto cube = x * x * x;
    const auto inner = beta * (x + kappa * cube);
    return 0.5f * x * (1.f + c10::cuda::compat::tanh(inner));
}

__global__ void gelu_tile(const c10::BFloat16* input, c10::BFloat16* output, int64_t size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        output[i] = gelu_value(input[i]);
}

void device_gelu_out(const torch::Tensor& x, torch::Tensor output) {
    tensor_contract(x);
    tensor_contract(output);
    TORCH_CHECK(x.sizes() == output.sizes() && x.device() == output.device(), "GELU output contract");
    c10::cuda::CUDAGuard guard(x.device());
    gelu_tile<<<(x.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<c10::BFloat16>(), output.data_ptr<c10::BFloat16>(), x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ffn_out(const torch::Tensor& x, const torch::Tensor& w1, const torch::Tensor& b1,
             const torch::Tensor& w2, const torch::Tensor& b2,
             torch::Tensor projected, torch::Tensor activated, torch::Tensor output) {
    // Three original rounding boundaries; this stage deliberately does not fuse them.
    device_linear_out(x, w1, b1, projected);
    device_gelu_out(projected, activated);
    device_linear_out(activated, w2, b2, output);
}

pybind11::dict metadata() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, linear_tile<Kernel>));
    int active;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, linear_tile<Kernel>,
        Kernel::kThreadCount, sizeof(Kernel::SharedStorage)));
    pybind11::dict result;
    result["CTA_MNK"] = pybind11::make_tuple(64, 128, 64);
    result["warp_MNK"] = pybind11::make_tuple(32, 64, 64);
    result["threads"] = Kernel::kThreadCount;
    result["registers_per_thread"] = attributes.numRegs;
    result["shared_memory_bytes"] = sizeof(Kernel::SharedStorage);
    result["theoretical_active_CTAs_per_SM"] = active;
    result["achieved_occupancy"] = "UNKNOWN_NOT_NCU_MEASURED";
    result["production_admitted"] = false;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, linear_tile<FusedGemm::GemmKernel>));
    result["fused_registers_per_thread"] = attributes.numRegs;
    result["fused_shared_memory_bytes"] = sizeof(FusedGemm::GemmKernel::SharedStorage);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("linear_out", &device_linear_out);
    module.def("gelu_out", &device_gelu_out);
    module.def("ffn1_gelu_out", &ffn1_gelu_out);
    module.def("ffn_out", &ffn_out);
    module.def("metadata", &metadata);
    module.attr("production_admitted") = false;
}

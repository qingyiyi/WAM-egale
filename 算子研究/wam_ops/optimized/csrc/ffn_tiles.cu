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
#include "ffn_tile.cuh"
#include "persistent.cuh"

namespace wam_optimized_ffn {

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
    const auto tile=wam_tiles::GemmTile{int(blockIdx.x),int(blockIdx.y)};
    wam_tiles::ffn_tile<K>(params, *reinterpret_cast<typename K::SharedStorage*>(storage),tile);
}

template<class K>
__global__ void persistent_linear(typename K::Params params,int count,int* audit) {
    extern __shared__ __align__(16) unsigned char storage[];
    wam_tiles::PersistentTasks tasks{count};
    int done=0;
    for(int id=tasks.first(); id<count; id=tasks.next(id)) {
        // Match the original x-major tile order. GEMM reduction order is fixed.
        const int rows=params.grid_tiled_shape.m();
        wam_tiles::ffn_tile<K>(params,*reinterpret_cast<typename K::SharedStorage*>(storage),
                             {id%rows,id/rows});
        wam_tiles::finish_tile();
        wam_tiles::record_tile(audit,id);
        ++done;
    }
    wam_tiles::record_worker(audit,count,done);
}

void tensor_contract(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
        "requires contiguous CUDA BF16 tensors");
}

template<class G>
void launch_linear(const torch::Tensor& x, const torch::Tensor& weight,
                const torch::Tensor& bias, torch::Tensor out, bool persistent=false,
                at::Tensor* audit_out=nullptr, int worker_limit=0, at::Tensor* meta_out=nullptr) {
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
    if(persistent) {
        TORCH_CHECK(worker_limit>=0,"worker limit must be nonnegative");
        int count=tiled.m()*tiled.n();
        int workers=std::min(count,worker_limit?std::min(worker_limit,props->multiProcessorCount):props->multiProcessorCount);
        at::Tensor audit;
        if(audit_out) {
            audit=at::zeros({count+workers},x.options().dtype(at::kInt));*audit_out=audit;
            if(meta_out) *meta_out=at::tensor({int64_t(count),int64_t(workers),int64_t(1),int64_t(smem)},at::TensorOptions().dtype(at::kLong));
        }
        C10_CUDA_CHECK(cudaFuncSetAttribute(persistent_linear<K0>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
        persistent_linear<K0><<<workers,K0::kThreadCount,smem,at::cuda::getCurrentCUDAStream()>>>(
            params,count,audit.defined()?audit.data_ptr<int>():nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return;
    }
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


} // namespace wam_optimized_ffn
void optimized_ffn1_gelu_out(const at::Tensor& x,const at::Tensor& weight,
                            const at::Tensor& bias,at::Tensor out,bool persistent) {
    wam_optimized_ffn::launch_linear<wam_optimized_ffn::FusedGemm>(x,weight,bias,out,persistent);
}

std::vector<at::Tensor> audit_ffn1(const at::Tensor& x,const at::Tensor& w,const at::Tensor& b,int workers) {
    auto out=at::empty({x.size(0),w.size(0)},x.options());
    TORCH_CHECK(x.size(1)==3072 && w.size(0)==14336,"FFN1 audit only");
    at::Tensor audit,meta;
    wam_optimized_ffn::launch_linear<wam_optimized_ffn::FusedGemm>(x,w,b,out,true,&audit,workers,&meta);
    return {out,audit,meta};
}

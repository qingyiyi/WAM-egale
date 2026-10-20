// M37: ONE geometry-informed CUTLASS candidate, not a production replacement.
// SM80 public cuBLASLt macro geometry does not specify instruction/reduction order.
// Use the unmodified CUTLASS 3.5.1 mainloop, epilogue and serial split-K semaphore.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace wam_contract_v1 {
using B = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Epi = cutlass::epilogue::thread::LinearCombination<B, 8, float, float>;

template<int TM, int TN, int WM, int WN, int Stages>
using Gemm = cutlass::gemm::device::Gemm<
    B, Row, B, Col, B, Row, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, 64>,
    cutlass::gemm::GemmShape<WM, WN, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>, Epi,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages, 8, 8, true>;

template<class G>
torch::Tensor run(torch::Tensor x, torch::Tensor w, torch::Tensor bias, int splits) {
    const int m=x.size(0), n=w.size(0), k=x.size(1);
    auto out=torch::empty({m,n},x.options());
    typename G::Arguments args({m,n,k},
        {reinterpret_cast<B*>(x.data_ptr()),k},
        {reinterpret_cast<B*>(w.data_ptr()),k},
        {reinterpret_cast<B*>(bias.data_ptr()),0},
        {reinterpret_cast<B*>(out.data_ptr()),n}, {1.f,1.f}, splits);
    TORCH_CHECK(G::can_implement(args)==cutlass::Status::kSuccess,
                "geometry candidate cannot implement this input");
    // CUTLASS initializes its semaphore on the same stream. No host BLAS call.
    auto workspace=torch::empty({int64_t(G::get_workspace_size(args))},
                                x.options().dtype(at::kByte));
    G operation;
    auto stream=at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(operation.initialize(args,workspace.data_ptr(),stream)==cutlass::Status::kSuccess,
                "CUTLASS initialization failed");
    TORCH_CHECK(operation(stream)==cutlass::Status::kSuccess,"CUTLASS execution failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor forward(torch::Tensor x,torch::Tensor w,torch::Tensor bias) {
    for (const auto& t:{x,w,bias})
        TORCH_CHECK(t.is_cuda()&&t.scalar_type()==at::kBFloat16&&t.is_contiguous(),
                    "contiguous CUDA BF16 only");
    TORCH_CHECK(x.dim()==2&&w.dim()==2&&bias.dim()==1&&w.size(1)==x.size(1)
                &&bias.numel()==w.size(0)&&x.device()==w.device()&&x.device()==bias.device(),
                "linear input contract");
    TORCH_CHECK(w.size(0)==3072&&(x.size(1)==3072||x.size(1)==14336)
                &&(x.size(0)==32||x.size(0)==224),"four registered signatures only");
    c10::cuda::CUDAGuard guard(x.device());
    auto prop=at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major==8&&prop->minor==0,"SM80 only");
    if(x.size(0)==224)return run<Gemm<128,128,64,64,3>>(x,w,bias,2);
    if(x.size(1)==3072)return run<Gemm<64,64,32,32,5>>(x,w,bias,1);
    return run<Gemm<64,256,32,64,3>>(x,w,bias,7);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward",&wam_contract_v1::forward);
    m.attr("production_admitted")=false;
    m.attr("candidate")="single_public_geometry_serial_splitK_candidate";
}

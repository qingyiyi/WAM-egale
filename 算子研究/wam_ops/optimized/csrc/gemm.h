// BF16 bias GEMM: narrow copy of pinned Torch2.9 CUDABlas.cpp's cuBLASLt path.
// No ATen linear/addmm. cuBLASLt is a disclosed external GEMM primitive.
#pragma once
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

namespace wam_optimized_block {
inline void blas_check(cublasStatus_t s) {
    TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(s));
}
struct GemmDescriptors {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    ~GemmDescriptors() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (op) cublasLtMatmulDescDestroy(op);
    }
    template<class T> void attribute(cublasLtMatmulDescAttributes_t tag, T value) {
        blas_check(cublasLtMatmulDescSetAttribute(op, tag, &value, sizeof(T)));
    }
    template<class T> void pref(cublasLtMatmulPreferenceAttributes_t tag, T value) {
        blas_check(cublasLtMatmulPreferenceSetAttribute(preference, tag, &value, sizeof(T)));
    }
};
inline uint32_t alignment(const void* p) {
    uint32_t n = 256;
    while (reinterpret_cast<uintptr_t>(p) % n) n /= 2;
    return n;
}
inline at::Tensor linear(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias) {
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==x.scalar_type()
        && bias.scalar_type()==x.scalar_type() && x.is_contiguous() && w.is_contiguous()
        && bias.is_contiguous(), "native linear requires contiguous BF16");
    const int64_t k=w.size(1), n=w.size(0), m=x.numel()/k;
    TORCH_CHECK(x.size(-1)==k && bias.numel()==n, "linear dimensions");
    auto out=at::empty({m,n}, x.options());
    GemmDescriptors d;
    blas_check(cublasLtMatmulDescCreate(&d.op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    blas_check(cublasLtMatmulPreferenceCreate(&d.preference));
    d.attribute(CUBLASLT_MATMUL_DESC_TRANSA,CUBLAS_OP_T);
    d.attribute(CUBLASLT_MATMUL_DESC_TRANSB,CUBLAS_OP_N);
    d.attribute(CUBLASLT_MATMUL_DESC_EPILOGUE,CUBLASLT_EPILOGUE_BIAS);
    d.attribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER,bias.data_ptr());
    // Row-major Y=X W^T becomes column-major Y^T=W X^T.
    blas_check(cublasLtMatrixLayoutCreate(&d.a,CUDA_R_16BF,k,n,k));
    blas_check(cublasLtMatrixLayoutCreate(&d.b,CUDA_R_16BF,k,m,k));
    blas_check(cublasLtMatrixLayoutCreate(&d.c,CUDA_R_16BF,n,m,n));
    if (!at::globalContext().allowBF16ReductionCuBLAS())
        d.pref(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            uint32_t(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE|CUBLASLT_REDUCTION_SCHEME_NONE));
    if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value())
        d.attribute(CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,int32_t(
            at::cuda::getCurrentDeviceProperties()->multiProcessorCount-
            at::globalContext()._SMCarveout_EXPERIMENTAL().value()));
    const size_t bytes=at::cuda::getCUDABlasLtWorkspaceSize();
    void* workspace=at::cuda::getCUDABlasLtWorkspace();
    d.pref(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,bytes);
    d.pref(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,alignment(w.data_ptr()));
    d.pref(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,alignment(x.data_ptr()));
    d.pref(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,alignment(out.data_ptr()));
    // This D preference intentionally follows the pinned Torch bias pointer.
    d.pref(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,alignment(bias.data_ptr()));
    auto handle=at::cuda::getCurrentCUDABlasLtHandle();
    cublasLtMatmulHeuristicResult_t h{};int returned=0;
    blas_check(cublasLtMatmulAlgoGetHeuristic(handle,d.op,d.a,d.b,d.c,d.c,
        d.preference,1,&h,&returned));
    TORCH_CHECK(returned==1,"no matching Torch cuBLASLt algorithm; no silent fallback");
    const float alpha=1.f,beta=0.f;
    blas_check(cublasLtMatmul(handle,d.op,&alpha,w.data_ptr(),d.a,x.data_ptr(),d.b,
        &beta,out.data_ptr(),d.c,out.data_ptr(),d.c,&h.algo,workspace,bytes,
        at::cuda::getCurrentCUDAStream()));
    return out;
}
} // namespace wam_optimized_block

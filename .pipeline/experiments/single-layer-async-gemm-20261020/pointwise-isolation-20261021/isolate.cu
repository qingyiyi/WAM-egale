
// 同一数学体的有限真实输入诊断；不替换活动整层实现。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <cuda_bf16.h>
#include <cutlass/gemm/device/gemm.h>
#include "cutlass_gemm_mainloop.cuh"
#include "isolated_cutlass.cuh"
using B=__nv_bfloat16;using at::Tensor;
__device__ float gelu_value(float x){
 constexpr float beta=M_SQRT2*M_2_SQRTPI*float(.5),kappa=.044715;
 float cube=x*x*x,inner=beta*(x+kappa*cube);
 return .5f*x*(1.f+c10::cuda::compat::tanh(inner));
}
// 标量地址不意味着串行；全网格线程并行。mode1仅固定108个CTA。
__global__ void scalar_point(const B* x,const float* w,B* y,float* z,int n,bool table){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
  if(table)z[i]=__fadd_rn(w[i%18432],__bfloat162float(x[i]));
  else y[i]=__float2bfloat16_rn(gelu_value(__bfloat162float(x[i])));
 }
}
struct alignas(8) V4 {B v[4];};
__global__ void vector_point(const V4* x,const float4* w,V4* y,float4* z,int n,bool table){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n/4;i+=gridDim.x*blockDim.x){
  V4 a=x[i];
  if(table){float4 b=w[i%4608];float* p=&b.x;
   #pragma unroll
   for(int j=0;j<4;++j)p[j]=__fadd_rn(p[j],__bfloat162float(a.v[j]));
   z[i]=b;
  }else{
   #pragma unroll
   for(int j=0;j<4;++j)a.v[j]=__float2bfloat16_rn(gelu_value(__bfloat162float(a.v[j])));
   y[i]=a;
  }
 }
}
Tensor point(Tensor x,Tensor w,bool table,int mode){
 auto out=at::empty_like(x,table?x.options().dtype(at::kFloat):x.options());
 int n=x.numel();auto stream=at::cuda::getCurrentCUDAStream();
 if(mode==2)vector_point<<<(n+1023)/1024,256,0,stream>>>((V4*)x.data_ptr(),(float4*)w.data_ptr(),(V4*)out.data_ptr(),(float4*)out.data_ptr(),n,table);
 else scalar_point<<<mode==1?108:(n+255)/256,256,0,stream>>>((B*)x.data_ptr(),(float*)w.data_ptr(),(B*)out.data_ptr(),(float*)out.data_ptr(),n,table);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
template<int TM,bool SharedPartial>__global__ void ffn_kernel(const B*a,const B*w,const B*b,B*out){
 extern __shared__ __align__(32) unsigned char raw[];
 auto &s=*reinterpret_cast<typename CutlassTile<TM>::Storage*>(raw);
 B*prev=reinterpret_cast<B*>(raw+sizeof(s));
 cutlass_gemm_tile<TM,SharedPartial>(s,a,w,b,out,224,3072,14336,blockIdx.x,true,prev);
}
Tensor ffn(Tensor a,Tensor w,Tensor b,int mode){
 auto out=at::empty({224,3072},a.options());
 auto kernel=mode==0?(void*)ffn_kernel<64,false>:mode==1?(void*)ffn_kernel<128,false>:(void*)ffn_kernel<64,true>;
 int tm=mode==1?128:64;int smem=mode==1?sizeof(CutlassTile<128>::Storage):sizeof(CutlassTile<64>::Storage);
 if(mode==2)smem+=64*128*sizeof(B);
 C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 const B*ap=(B*)a.data_ptr(),*wp=(B*)w.data_ptr(),*bp=(B*)b.data_ptr();B*op=(B*)out.data_ptr();void*args[]={&ap,&wp,&bp,&op};
 C10_CUDA_CHECK(cudaLaunchKernel(kernel,dim3(((224+tm-1)/tm)*24),dim3(256),args,smem,at::cuda::getCurrentCUDAStream()));return out;
}
std::vector<int64_t> resources(int kind,int mode){
 void*k=kind==0?(mode==2?(void*)vector_point:(void*)scalar_point):(mode==0?(void*)ffn_kernel<64,false>:mode==1?(void*)ffn_kernel<128,false>:(void*)ffn_kernel<64,true>);
 cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));
 return {a.numRegs,(int64_t)a.sharedSizeBytes,(int64_t)a.localSizeBytes};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("point",&point);m.def("ffn",&ffn);m.def("resources",&resources);}

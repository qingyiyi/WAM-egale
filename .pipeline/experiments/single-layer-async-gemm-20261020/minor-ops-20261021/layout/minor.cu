
// 独立数学体：保持原算式/舍入，仅比较向量搬运、网格与RMS直接读取。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "norm.cuh"
#include "minor_norm.cuh"
using B=at::BFloat16;using at::Tensor;
template<int V>struct alignas(sizeof(B)*V) PackB {B v[V];};
template<int V>struct alignas(sizeof(float)*V) PackF {float v[V];};
template<int V>__global__ void point_kernel(const void*x,const B*y,const float*w,void*out,int n,int op,int param){
 for(int base=(blockIdx.x*blockDim.x+threadIdx.x)*V;base<n;base+=gridDim.x*blockDim.x*V){
  PackB<V> xb,yb,z;PackF<V> xf,scale,shift,zf;
  if(op==1)xf=*reinterpret_cast<const PackF<V>*>((const float*)x+base);
  else xb=*reinterpret_cast<const PackB<V>*>((const B*)x+base);
  if(op==0)shift=*reinterpret_cast<const PackF<V>*>(w+base%18432);
  if(op==1){int m=(base/3072*6+param)*3072+base%3072;shift=*reinterpret_cast<const PackF<V>*>(w+m);scale=*reinterpret_cast<const PackF<V>*>(w+m+3072);}
  if(op==2){yb=*reinterpret_cast<const PackB<V>*>(y+base);if(param>=0)scale=*reinterpret_cast<const PackF<V>*>(w+(base/3072*6+param)*3072+base%3072);}
#pragma unroll
  for(int j=0;j<V;++j){
   if(op==0)zf.v[j]=__fadd_rn(float(xb.v[j]),shift.v[j]);
   else if(op==1)z.v[j]=B(__fadd_rn(__fmul_rn(xf.v[j],__fadd_rn(1.f,scale.v[j])),shift.v[j]));
   else {float v=float(yb.v[j]);if(param>=0)v=__fmul_rn(v,scale.v[j]);z.v[j]=B(__fadd_rn(float(xb.v[j]),v));}
  }
  if(op==0)*reinterpret_cast<PackF<V>*>((float*)out+base)=zf;
  else *reinterpret_cast<PackB<V>*>((B*)out+base)=z;
 }
}
template<int V>__global__ void rope_kernel(const B*x,const c10::complex<float>*freq,B*out,int n,int q,int fb){
 for(int base=(blockIdx.x*blockDim.x+threadIdx.x)*V;base<n;base+=gridDim.x*blockDim.x*V){
  auto a=*reinterpret_cast<const PackB<V>*>(x+base);PackB<V>z;
#pragma unroll
  for(int j=0;j<V;j+=2){
   int pair=(base+j)/2,r=pair/64,b=r/(q*24),t=(r/24)%q,d=pair%64;
   auto f=freq[((fb==1?0:b)*q+t)*64+d];double u=double(a.v[j]),v=double(a.v[j+1]);
   float real=float(u*double(f.real())-v*double(f.imag()));float imag=float(u*double(f.imag())+v*double(f.real()));
   z.v[j]=B(real);z.v[j+1]=B(imag);
  }
  *reinterpret_cast<PackB<V>*>(out+base)=z;
 }
}
__global__ void rms_kernel(const B*x,const B*w,B*y,int rows,float eps){
 __shared__ float scratch[6];
 for(int row=blockIdx.x;row<rows;row+=gridDim.x)
  wam_optimized_norm::minor_rms_forward<B,float,true>(3072,eps,x+row*3072,w,nullptr,nullptr,nullptr,y+row*3072,scratch);
}
Tensor point(Tensor x,Tensor y,Tensor w,int op,int param,int mode){
 auto out=at::empty_like(x,op==0?x.options().dtype(at::kFloat):x.options().dtype(at::kBFloat16));
 int n=x.numel(),v=mode>=2?4:1,threads=mode==3?128:256,grid=mode==1?108:(n+threads*v-1)/(threads*v);auto stream=at::cuda::getCurrentCUDAStream();
 const B*yp=y.numel()?(B*)y.data_ptr():nullptr;const float*wp=w.numel()?(float*)w.data_ptr():nullptr;
 if(v==4)point_kernel<4><<<grid,threads,0,stream>>>(x.data_ptr(),yp,wp,out.data_ptr(),n,op,param);
 else point_kernel<1><<<grid,threads,0,stream>>>(x.data_ptr(),yp,wp,out.data_ptr(),n,op,param);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
Tensor rope(Tensor x,Tensor freq,int q,int mode){
 auto out=at::empty_like(x);int n=x.numel(),v=mode>=2?4:2,threads=mode==3?128:256,grid=mode==1?108:(n+threads*v-1)/(threads*v);auto st=at::cuda::getCurrentCUDAStream();
 if(v==4)rope_kernel<4><<<grid,threads,0,st>>>((B*)x.data_ptr(),freq.data_ptr<c10::complex<float>>(),(B*)out.data_ptr(),n,q,freq.size(0));
 else rope_kernel<2><<<grid,threads,0,st>>>((B*)x.data_ptr(),freq.data_ptr<c10::complex<float>>(),(B*)out.data_ptr(),n,q,freq.size(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
Tensor rms(Tensor x,Tensor w,float eps,int mode){
 auto out=at::empty_like(x);int rows=x.numel()/3072;int grid=mode==1?std::min(108,rows):rows;
 rms_kernel<<<grid,dim3(32,4),0,at::cuda::getCurrentCUDAStream()>>>((B*)x.data_ptr(),(B*)w.data_ptr(),(B*)out.data_ptr(),rows,eps);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("point",&point);m.def("rope",&rope);m.def("rms",&rms);}

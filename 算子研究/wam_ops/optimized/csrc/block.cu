// Complete sequential native WAM block. Tensor handles/allocators are PyTorch;
// mathematical work is own CUDA kernels, own Flash CUDA, or disclosed cuBLASLt.
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <map>
#include <functional>
#include <optional>
#include "norm.cuh"
#include "gemm.h"

namespace wam_optimized_sdpa {
at::Tensor sdpa(const at::Tensor&,const at::Tensor&,const at::Tensor&,std::optional<double>, bool);
}
// Existing own CUTLASS primitive; keeps the BF16 GEMM rounding before GELU.
void optimized_ffn1_gelu_out(const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor);
namespace wam_optimized_block {
using BF = at::BFloat16;
using at::Tensor;
inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
inline int grid(int64_t n) { return int((n+255)/256); }

template<class I,class O> __global__ void cast_kernel(const I* x,O* y,int64_t n) {
    for(int64_t i=blockIdx.x*int64_t(blockDim.x)+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) y[i]=O(x[i]);
}
Tensor fp32(const Tensor& x) {
    if(x.scalar_type()==at::kFloat) return x;
    TORCH_CHECK(x.scalar_type()==at::kBFloat16,"only BF16/FP32 casts");
    auto y=at::empty(x.sizes(),x.options().dtype(at::kFloat));
    cast_kernel<<<grid(x.numel()),256,0,stream()>>>(x.data_ptr<BF>(),y.data_ptr<float>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
Tensor bf16(const Tensor& x) {
    if(x.scalar_type()==at::kBFloat16) return x;
    auto y=at::empty(x.sizes(),x.options().dtype(at::kBFloat16));
    cast_kernel<<<grid(x.numel()),256,0,stream()>>>(x.data_ptr<float>(),y.data_ptr<BF>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
Tensor norm(const Tensor& x,const Tensor& gamma,const Tensor& beta,double eps,bool rms) {
    const int d=x.size(-1),m=x.numel()/d;
    TORCH_CHECK(d==3072 && x.is_contiguous(),"native norm currently D3072 contiguous");
    auto y=at::empty_like(x),mean=at::empty({m},x.options().dtype(at::kFloat)),rstd=at::empty_like(mean);
    const dim3 threads(32,4);const int smem=24;
    if(x.scalar_type()==at::kFloat) {
        TORCH_CHECK(!rms,"FP32 RMS not in current contract");
        wam_optimized_norm::vectorized_layer_norm_kernel<float,float,false><<<m,threads,smem,stream()>>>(
            d,float(eps),x.data_ptr<float>(),gamma.defined()?gamma.data_ptr<float>():nullptr,
            beta.defined()?beta.data_ptr<float>():nullptr,mean.data_ptr<float>(),rstd.data_ptr<float>(),y.data_ptr<float>());
    } else {
        TORCH_CHECK(rms && gamma.defined(),"BF16 branch is affine RMSNorm");
        wam_optimized_norm::vectorized_layer_norm_kernel<BF,float,true><<<m,threads,smem,stream()>>>(
            d,float(eps),x.data_ptr<BF>(),gamma.data_ptr<BF>(),nullptr,nullptr,rstd.data_ptr<float>(),y.data_ptr<BF>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
template<class T,class W> __global__ void table_kernel(const T* temb,const W* table,float* out,int64_t n,int q,int l,int d) {
    for(int64_t i=blockIdx.x*int64_t(blockDim.x)+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t row=i/(6*d),cell=i%(6*d),b=row/q,t=row%q;
        out[i]=__fadd_rn(float(table[cell]),float(temb[(b*l+(l==1?0:t))*6*d+cell]));
    }
}
Tensor table(const Tensor& temb,const Tensor& w,int q) {
    const int b=temb.size(0),l=temb.size(1),d=w.size(-1);
    TORCH_CHECK((l==1||l==q)&&temb.is_contiguous()&&w.is_contiguous(),"temb layout");
    auto out=at::empty({b*q,6,d},temb.options().dtype(at::kFloat));
    TORCH_CHECK(w.scalar_type()==at::kBFloat16||w.scalar_type()==at::kFloat,"table BF16/FP32 contract");
    if(w.scalar_type()==at::kFloat && temb.scalar_type()==at::kBFloat16)
        table_kernel<<<grid(out.numel()),256,0,stream()>>>(temb.data_ptr<BF>(),w.data_ptr<float>(),out.data_ptr<float>(),out.numel(),q,l,d);
    else if(w.scalar_type()==at::kFloat)
        table_kernel<<<grid(out.numel()),256,0,stream()>>>(temb.data_ptr<float>(),w.data_ptr<float>(),out.data_ptr<float>(),out.numel(),q,l,d);
    else if(temb.scalar_type()==at::kBFloat16)
        table_kernel<<<grid(out.numel()),256,0,stream()>>>(temb.data_ptr<BF>(),w.data_ptr<BF>(),out.data_ptr<float>(),out.numel(),q,l,d);
    else table_kernel<<<grid(out.numel()),256,0,stream()>>>(temb.data_ptr<float>(),w.data_ptr<BF>(),out.data_ptr<float>(),out.numel(),q,l,d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
__global__ void modulation_kernel(const float* x,const float* mod,BF* y,int64_t n,int d,int shift) {
    for(int64_t i=blockIdx.x*int64_t(blockDim.x)+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t s=(i/d*6+shift)*d+i%d;
        y[i]=BF(__fadd_rn(__fmul_rn(x[i],__fadd_rn(1.f,mod[s+d])),mod[s]));
    }
}
Tensor modulate(const Tensor& x,const Tensor& mod,int shift) {
    auto y=at::empty(x.sizes(),x.options().dtype(at::kBFloat16));
    modulation_kernel<<<grid(x.numel()),256,0,stream()>>>(x.data_ptr<float>(),mod.data_ptr<float>(),y.data_ptr<BF>(),x.numel(),x.size(-1),shift);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
__global__ void residual_kernel(const BF* x,const BF* a,const float* mod,BF* y,int64_t n,int d,int gate) {
    for(int64_t i=blockIdx.x*int64_t(blockDim.x)+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) {
        float v=float(a[i]);
        if(gate>=0) v=__fmul_rn(v,mod[(i/d*6+gate)*d+i%d]);
        y[i]=BF(__fadd_rn(float(x[i]),v));
    }
}
Tensor residual(const Tensor& x,const Tensor& a,const Tensor& mod,int gate) {
    auto y=at::empty_like(x);
    residual_kernel<<<grid(x.numel()),256,0,stream()>>>(x.data_ptr<BF>(),a.data_ptr<BF>(),
        mod.defined()?mod.data_ptr<float>():nullptr,y.data_ptr<BF>(),x.numel(),x.size(-1),gate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
__global__ void rope_kernel(const BF* x,const c10::complex<float>* f,BF* y,int64_t pairs,int q,int heads,int fb) {
    for(int64_t i=blockIdx.x*int64_t(blockDim.x)+threadIdx.x;i<pairs;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t r=i/64,b=r/(q*heads),t=(r/heads)%q,j=i%64;
        const auto z=f[((fb==1?0:b)*q+t)*64+j];
        const double a=double(x[2*i]),c=double(x[2*i+1]);
        y[2*i]=BF(a*double(z.real())-c*double(z.imag()));
        y[2*i+1]=BF(a*double(z.imag())+c*double(z.real()));
    }
}
Tensor rope(const Tensor& x,const Tensor& freq,int b,int q) {
    TORCH_CHECK(freq.scalar_type()==at::kComplexFloat && freq.is_contiguous()
        && freq.size(-1)==64,"native FP64 RoPE requires complex64 frequencies");
    auto y=at::empty_like(x);
    rope_kernel<<<grid(x.numel()/2),256,0,stream()>>>(x.data_ptr<BF>(),freq.data_ptr<c10::complex<float>>(),y.data_ptr<BF>(),x.numel()/2,q,24,freq.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
__global__ void gelu_kernel(const BF* x,BF* y,int64_t n) {
    for(int64_t i=blockIdx.x*int64_t(blockDim.x)+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) {
        const float a=float(x[i]);constexpr float beta=M_SQRT2*M_2_SQRTPI*float(.5),kappa=.044715;
        const float cube=a*a*a,inner=beta*(a+kappa*cube);
        y[i]=BF(.5f*a*(1.f+c10::cuda::compat::tanh(inner)));
    }
}
Tensor gelu(const Tensor& x) {
    auto y=at::empty_like(x);gelu_kernel<<<grid(x.numel()),256,0,stream()>>>(x.data_ptr<BF>(),y.data_ptr<BF>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

// For the actual pool2160 and Q16/Q128, an eviction has >128 used entries.
// Pinned Torch Sort.cu then uses stable CUB BlockRadixSort (4096/128x32).
__global__ void layout_kernel(bool* mask,int64_t* ids,bool* pred,int64_t* slots,
    int64_t* free,int64_t* info,int size,int q,int update,bool candidate,
    const int64_t* rows,int batch,int pool_batch) {
    using Sort=cub::BlockRadixSort<int64_t,128,32,int64_t>;
    using Scan=cub::BlockScan<int,128>;
    __shared__ typename Sort::TempStorage storage;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ int used,need,error;
    if(candidate) {
        // Stable physical order, like nonzero(mask), without one thread
        // serially fetching thousands of mask bytes from global memory.
        if(threadIdx.x==0) error=(size>4096);
        __syncthreads();
        for(int i=threadIdx.x;i<batch;i+=128)
            if(rows[i]<0||rows[i]>=pool_batch) atomicExch(&error,2);
        int flags[32],prefix[32],aggregate;
        #pragma unroll
        for(int j=0;j<32;++j) {
            int i=threadIdx.x*32+j;flags[j]=(i<size&&mask[i]);
        }
        Scan(scan_storage).ExclusiveSum(flags,prefix,aggregate);
        #pragma unroll
        for(int j=0;j<32;++j) if(flags[j]) slots[prefix[j]]=threadIdx.x*32+j;
        __syncthreads();
        if(threadIdx.x==0) {info[0]=aggregate;info[1]=error;}
        return;
    }
    if(threadIdx.x==0) {
        used=0;int nf=0;error=0;
        for(int i=0;i<size;++i) {
            if(mask[i]) slots[used++]=i;
            else if(nf<q) free[nf++]=i;
        }
        need=candidate?0:q-nf;
        if(need>0&&(size!=2160||used<=128||used>4096)) error=1;
        if(candidate) for(int i=0;i<batch;++i)
            if(rows[i]<0||rows[i]>=pool_batch) error=2;
    }
    __syncthreads();
    if(error) {if(threadIdx.x==0){info[0]=used;info[1]=error;}return;}
    if(need>0) {
        int64_t keys[32],values[32];
        #pragma unroll
        for(int i=0;i<32;++i) {
            int j=threadIdx.x*32+i;
            values[i]=j<used?slots[j]:0;
            keys[i]=j<used?ids[values[i]]:INT64_MAX;
        }
        Sort(storage).Sort(keys,values);
        #pragma unroll
        for(int i=0;i<32;++i) if(threadIdx.x*32+i<need) {
            mask[values[i]]=false;ids[values[i]]=-1;
        }
    }
    __syncthreads();
    if(threadIdx.x==0) {
        if(!candidate) {
            int nf=0;int64_t next=-1;
            for(int i=0;i<size;++i) {
                if(mask[i]) {if(ids[i]>next) next=ids[i];}
                else if(nf<q) free[nf++]=i;
            }
            for(int j=0;j<q;++j) {int i=free[j];mask[i]=true;ids[i]=next+1;pred[i]=(update==1);}
        }
        int n=0;for(int i=0;i<size;++i) if(mask[i]) slots[n++]=i;
        info[0]=n;info[1]=0;
    }
}
__global__ void private_kernel(const BF* k,const BF* v,BF* pk,BF* pv,const int64_t* free,int b,int q,int size,int d) {
    const int width=d/8;
    const auto* key=reinterpret_cast<const uint4*>(k);
    const auto* value=reinterpret_cast<const uint4*>(v);
    auto* pool_key=reinterpret_cast<uint4*>(pk);
    auto* pool_value=reinterpret_cast<uint4*>(pv);
    int64_t n=int64_t(b)*q*width;
    for(int64_t i=blockIdx.x*int64_t(blockDim.x)+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x) {
        int64_t dest=((i/(q*width))*size+free[(i/width)%q])*width+i%width;
        pool_key[dest]=key[i];pool_value[dest]=value[i];
    }
}
__global__ void pack_kernel(const BF* pool,const BF* current,BF* out,const int64_t* slots,const int64_t* rows,int b,int total,int n,int q,int size,int d,bool candidate) {
    const int width=d/8;
    const auto* history=reinterpret_cast<const uint4*>(pool);
    const auto* suffix=reinterpret_cast<const uint4*>(current);
    auto* packed=reinterpret_cast<uint4*>(out);
    int64_t count=int64_t(b)*total*width;
    for(int64_t i=blockIdx.x*int64_t(blockDim.x)+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        int batch=i/(total*width),t=(i/width)%total,j=i%width;
        if(candidate&&t>=n) packed[i]=suffix[(int64_t(batch)*q+t-n)*width+j];
        else packed[i]=history[(int64_t(candidate?rows[batch]:batch)*size+slots[t])*width+j];
    }
}
__global__ void restore_kernel(bool* mask,const int64_t* free,int q) {
    if(threadIdx.x<q) mask[free[threadIdx.x]]=false;
}
Tensor self_attention(const Tensor& q,const Tensor& k,const Tensor& v,
    const std::vector<Tensor>& cache,const Tensor& rows,int update,bool tiled) {
    if(cache.empty()) {
        TORCH_CHECK(!rows.defined(),"candidate path requires history");
        return wam_optimized_sdpa::sdpa(q,k,v,std::nullopt,tiled);
    }
    const int b=q.size(0),tokens=q.size(1),size=cache[0].size(1),d=3072;
    const bool candidate=rows.defined();
    TORCH_CHECK(candidate?update==0:cache[0].size(0)==b,"KV branch contract");
    TORCH_CHECK(cache.size()==5,"cache schema: K,V,id,mask,is_pred");
    auto opt=cache[2].options();auto slots=at::empty({size},opt),free=at::empty({tokens},opt),info=at::empty({2},opt);
    layout_kernel<<<1,128,0,stream()>>>(cache[3].data_ptr<bool>(),cache[2].data_ptr<int64_t>(),cache[4].data_ptr<bool>(),slots.data_ptr<int64_t>(),free.data_ptr<int64_t>(),info.data_ptr<int64_t>(),size,tokens,update,candidate,candidate?rows.data_ptr<int64_t>():nullptr,b,cache[0].size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto host=at::empty({2},at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
    C10_CUDA_CHECK(cudaMemcpyAsync(host.data_ptr(),info.data_ptr(),16,cudaMemcpyDeviceToHost,stream()));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream()));
    TORCH_CHECK(host.data_ptr<int64_t>()[1]==0,"native KV contract error ",host.data_ptr<int64_t>()[1]);
    const int n=host.data_ptr<int64_t>()[0],total=n+(candidate?tokens:0);
    if(!candidate) {
        private_kernel<<<grid(k.numel()/8),256,0,stream()>>>(k.data_ptr<BF>(),v.data_ptr<BF>(),cache[0].data_ptr<BF>(),cache[1].data_ptr<BF>(),free.data_ptr<int64_t>(),b,tokens,size,d);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto pk=at::empty({b,total,24,128},q.options()),pv=at::empty_like(pk);
    pack_kernel<<<grid(pk.numel()/8),256,0,stream()>>>(cache[0].data_ptr<BF>(),k.data_ptr<BF>(),pk.data_ptr<BF>(),slots.data_ptr<int64_t>(),candidate?rows.data_ptr<int64_t>():nullptr,b,total,n,tokens,size,d,candidate);
    pack_kernel<<<grid(pv.numel()/8),256,0,stream()>>>(cache[1].data_ptr<BF>(),v.data_ptr<BF>(),pv.data_ptr<BF>(),slots.data_ptr<int64_t>(),candidate?rows.data_ptr<int64_t>():nullptr,b,total,n,tokens,size,d,candidate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out=wam_optimized_sdpa::sdpa(q,pk,pv,std::nullopt,tiled);
    if(!candidate&&update==0) {
        restore_kernel<<<1,256,0,stream()>>>(cache[3].data_ptr<bool>(),free.data_ptr<int64_t>(),tokens);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// Stage launches share one CUDA stream: each consumer waits for all producer
// tiles. No cross-stage interleaving or device ready queue is claimed here.
using Records = std::map<std::string,Tensor>;
using Saver = std::function<void(const char*, const Tensor&)>;
Tensor self_subgraph(const std::vector<Tensor>& w, const Tensor& x, const Tensor& mod,
    const std::optional<Tensor>& freq, const std::vector<Tensor>& cache,
    const std::optional<Tensor>& candidate_rows, int update,
    const std::vector<double>& eps, bool tiled, const Saver& save) {
    const int b=x.size(0),q=x.size(1);
    auto n1=norm(fp32(x),{}, {},eps[0],false);save("norm1",n1);
    auto h1=modulate(n1,mod,0);save("modulated1",h1);
    auto sq=norm(linear(h1,w[3],w[4]),w[11],{},eps[3],true);
    auto sk=norm(linear(h1,w[5],w[6]),w[12],{},eps[4],true);
    auto sv=linear(h1,w[7],w[8]);save("self_q_norm",sq);save("self_k_norm",sk);save("self_v",sv);
    if(freq.has_value()) {sq=rope(sq,*freq,b,q);sk=rope(sk,*freq,b,q);}
    save("self_q_rope",sq);save("self_k_rope",sk);
    auto sa=self_attention(sq.view({b,q,24,128}),sk.view({b,q,24,128}),sv.view({b,q,24,128}),cache,candidate_rows.value_or(Tensor{}),update,tiled);save("self_attention",sa);
    auto so=linear(sa.view({b,q,3072}),w[9],w[10]);save("self_out",so);
    auto h2=residual(x,so,mod,2);save("residual1",h2);
    return h2;
}
Tensor cross_subgraph(const std::vector<Tensor>& w,const Tensor& h2,const Tensor& context,
    const std::vector<double>& eps,bool tiled,const Saver& save) {
    const int b=h2.size(0),q=h2.size(1);
    auto n2=w[1].numel()?bf16(norm(fp32(h2),fp32(w[1]),fp32(w[2]),eps[1],false)):h2;save("norm2",n2);
    auto cq=norm(linear(n2,w[13],w[14]),w[21],{},eps[5],true);
    auto ck=norm(linear(context,w[15],w[16]),w[22],{},eps[6],true);
    auto cv=linear(context,w[17],w[18]);save("cross_q",cq);save("cross_k",ck);save("cross_v",cv);
    auto ca=wam_optimized_sdpa::sdpa(cq.view({b,q,24,128}),ck.view({b,context.size(1),24,128}),cv.view({b,context.size(1),24,128}),std::nullopt,tiled);save("cross_attention",ca);
    auto co=linear(ca.view({b,q,3072}),w[19],w[20]);save("cross_out",co);
    auto h3=residual(h2,co,{},-1);save("residual2",h3);
    return h3;
}
Tensor ffn_subgraph(const std::vector<Tensor>& w,const Tensor& h3,const Tensor& mod,
    const std::vector<double>& eps,bool fused_ffn1,const Saver& save) {
    const int b=h3.size(0),q=h3.size(1);
    auto n3=norm(fp32(h3),{}, {},eps[2],false);save("norm3",n3);
    auto f0=modulate(n3,mod,3);save("modulated3",f0);
    Tensor act;
    if(fused_ffn1 && q==16) {
        // Only the previously verified action shape is admitted to this path.
        act=at::empty({b*q,w[23].size(0)},f0.options());
        optimized_ffn1_gelu_out(f0.view({b*q,3072}),w[23],w[24],act);
    } else {
        auto f1=linear(f0,w[23],w[24]);save("ffn1",f1);
        act=wam_optimized_block::gelu(f1);
    }
    save("gelu",act);
    auto f2=linear(act,w[25],w[26]);save("ffn2",f2);
    return residual(h3,f2,mod,5);
}

std::map<std::string,Tensor> forward(const std::vector<Tensor>& w,const Tensor& x,
    const Tensor& context,const Tensor& temb,const std::optional<Tensor>& freq,const std::vector<Tensor>& cache,
    const std::optional<Tensor>& candidate_rows,int update,const std::vector<double>& eps,bool debug,bool fused_ffn1,bool tiled) {
    TORCH_CHECK(w.size()==27&&eps.size()==7,"native parameter schema");
    TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()
        && x.dim()==3&&x.size(2)==3072&&(x.size(1)==16||x.size(1)==128)
        &&context.is_contiguous(),"actual BF16 block contract Q16/Q128 D3072");
    for(size_t i=0;i<w.size();++i) {
        const auto& t=w[i];
        const bool dtype=t.scalar_type()==at::kBFloat16 || (i<=2&&t.scalar_type()==at::kFloat);
        TORCH_CHECK(t.device()==x.device()&&dtype&&t.is_contiguous()&&!t.requires_grad(),
            "frozen canonical parameter index ",i," dtype ",t.scalar_type());
    }
    c10::cuda::CUDAGuard guard(x.device());
    const int b=x.size(0),q=x.size(1);std::map<std::string,Tensor> record;
    auto save=[&](const char* name,const Tensor& v){if(debug) record[name]=v;};
    auto mod=table(temb,w[0],q);save("modulation",mod);
    if(debug) record["execution_flags"]=at::tensor({int64_t(tiled),int64_t(fused_ffn1)},at::TensorOptions().dtype(at::kLong));
    auto h2=self_subgraph(w,x,mod,freq,cache,candidate_rows,update,eps,tiled,save);
    auto h3=cross_subgraph(w,h2,context,eps,tiled,save);
    record["output"]=ffn_subgraph(w,h3,mod,eps,fused_ffn1,save);return record;

}
} // namespace wam_block

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("sdpa", &wam_optimized_sdpa::sdpa,py::arg("q"),py::arg("k"),py::arg("v"),
          py::arg("scale")=py::none(),py::arg("tiled")=false);
    m.attr("independent_module")="wam_optimized_block_cuda";
    m.attr("persistent_scheduler")=false;
    m.def("forward",&wam_optimized_block::forward,py::arg("weights"),py::arg("x"),
        py::arg("context"),py::arg("temb"),py::arg("freq"),py::arg("cache"),
        py::arg("candidate_rows"),py::arg("update"),py::arg("eps"),
        py::arg("debug")=false,py::arg("fused_ffn1")=false,py::arg("tiled")=false);
    m.def("linear",&wam_optimized_block::linear);
    m.def("norm",[](const at::Tensor& x,const std::optional<at::Tensor>& g,
        const std::optional<at::Tensor>& b,double eps,bool rms) {
        return wam_optimized_block::norm(x,g.value_or(at::Tensor{}),b.value_or(at::Tensor{}),eps,rms);
    });
    m.def("gelu",&wam_optimized_block::gelu);
    m.def("fp32",&wam_optimized_block::fp32);
    m.def("bf16",&wam_optimized_block::bf16);
    m.attr("production_admitted")=false;
    m.attr("torch_math_operators")=false;
    m.attr("external_GEMM")="cuBLASLt; pinned Torch descriptor/heuristic/workspace contract";
}

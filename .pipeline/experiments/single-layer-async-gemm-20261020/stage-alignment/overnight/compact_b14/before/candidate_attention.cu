#define FLASH_NAMESPACE wam_candidate_tile
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "vendor/flash_attn/flash_fwd_launch_template.h"
#include "candidate_kernel.h"
#include <cmath>
using Params=wam_candidate_tile::GroupParams;
using Traits=Flash_fwd_kernel_traits<128,64,64,4,false,false,cutlass::bfloat16_t>;

template<int G> __global__ void grouped_kernel(KERNEL_PARAM_MODIFIER const Params original) {
#if defined(ARCH_SUPPORTS_FLASH)
    Params p=original;
    const int groups=(p.candidates+G-1)/G;
    const int cfg=blockIdx.y/groups, group=blockIdx.y%groups;
    const int count=min(G,p.candidates-group*G), first=cfg*p.candidates+group*G;
    p.group_base=first;p.group_cfg=cfg;
    p.q_ptr=reinterpret_cast<cutlass::bfloat16_t *>(p.q_ptr)+first*p.q_batch_stride;
    p.o_ptr=reinterpret_cast<cutlass::bfloat16_t *>(p.o_ptr)+first*p.o_batch_stride;
    p.softmax_lse_ptr=reinterpret_cast<float *>(p.softmax_lse_ptr)+first*24*16;
    p.q_batch_stride=0;p.o_batch_stride=0;
    p.k_batch_stride=0;p.v_batch_stride=0;
    p.b=1;p.seqlen_q=count*16;p.seqlen_k=p.prefix_length+count*16;
    p.seqlen_q_rounded=128;p.seqlen_k_rounded=(p.seqlen_k+127)/128*128;
    wam_candidate_tile::compute_group_attn_1rowblock<
        Traits,false,false,false,false,false,true,false,false>(p,0,blockIdx.z,0);
#endif
}

at::Tensor grouped(const at::Tensor& q,const at::Tensor& pk,const at::Tensor& pv,
                   const at::Tensor& ck,const at::Tensor& cv,const at::Tensor& slots,int group) {
    TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kBFloat16&&q.is_contiguous(),"contiguous CUDA BF16 Q");
    TORCH_CHECK(q.dim()==4&&q.size(1)==16&&q.size(2)==24&&q.size(3)==128,"actual Q16 H24 D128");
    const int nv=q.size(0)/2;
    TORCH_CHECK(q.size(0)==2*nv&&(nv==1||nv==3||nv==7),"CFG2 n1/3/7 only");
    TORCH_CHECK(group==1||group==2||group==4,"only baseline G1 and two registered groupings G2/G4");
    for(const auto& x:{q,pk,pv,ck,cv}) {
        TORCH_CHECK(x.device()==q.device()&&x.scalar_type()==q.scalar_type()&&!x.requires_grad(),"same device/dtype inference");
        TORCH_CHECK(x.dim()==4&&x.size(2)==24&&x.size(3)==128&&x.stride(3)==1&&x.stride(2)==128
                    &&x.stride(1)%8==0&&x.stride(0)%8==0&&reinterpret_cast<uintptr_t>(x.data_ptr())%16==0,"aligned KV layout");
    }
    TORCH_CHECK(pk.size(0)==2&&pv.sizes()==pk.sizes()&&ck.sizes()==q.sizes()&&cv.sizes()==q.sizes(),"shared CFG2 prefix and private suffix");
    TORCH_CHECK(slots.device()==q.device()&&slots.scalar_type()==at::kLong&&slots.dim()==1&&slots.is_contiguous()
                &&slots.numel()>0&&slots.numel()<=pk.size(1),"prevalidated actual slots");
    c10::cuda::CUDAGuard guard(q.device());
    const auto *props=at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major==8&&props->minor==0&&props->multiProcessorCount==108,"fixed SM80 A100108");
    auto out=at::empty_like(q),lse=at::empty({2*nv,24,16},q.options().dtype(at::kFloat));
    Params p{};
    p.q_ptr=q.data_ptr();p.o_ptr=out.data_ptr();p.softmax_lse_ptr=lse.data_ptr();
    p.q_batch_stride=q.stride(0);p.q_row_stride=q.stride(1);p.q_head_stride=q.stride(2);
    p.o_batch_stride=out.stride(0);p.o_row_stride=out.stride(1);p.o_head_stride=out.stride(2);
    p.k_ptr=pk.data_ptr();p.v_ptr=pv.data_ptr();p.k_row_stride=pk.stride(1);p.k_head_stride=pk.stride(2);
    p.v_row_stride=pv.stride(1);p.v_head_stride=pv.stride(2);
    p.prefix_k=pk.data_ptr();p.prefix_v=pv.data_ptr();p.current_k=ck.data_ptr();p.current_v=cv.data_ptr();
    p.pk_batch=pk.stride(0);p.pk_row=pk.stride(1);p.pk_head=pk.stride(2);
    p.pv_batch=pv.stride(0);p.pv_row=pv.stride(1);p.pv_head=pv.stride(2);
    p.ck_batch=ck.stride(0);p.ck_row=ck.stride(1);p.ck_head=ck.stride(2);
    p.cv_batch=cv.stride(0);p.cv_row=cv.stride(1);p.cv_head=cv.stride(2);
    p.valid_slots=slots.data_ptr<int64_t>();p.prefix_length=slots.numel();p.candidates=nv;
    p.b=2*nv;p.h=p.h_k=24;p.h_h_k_ratio=1;p.d=p.d_rounded=128;
    p.scale_softmax=float(1./std::sqrt(128.));p.scale_softmax_log2=p.scale_softmax*M_LOG2E;
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.window_size_left=p.window_size_right=-1;p.is_bf16=true;p.is_causal=false;p.is_seqlens_k_cumulative=true;p.num_splits=1;
    const dim3 grid(1,2*((nv+group-1)/group),24);
    #define RUN(G) C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_kernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,Traits::kSmemSize)); grouped_kernel<G><<<grid,128,Traits::kSmemSize,at::cuda::getCurrentCUDAStream()>>>(p)
    if(group==1){RUN(1);}else if(group==2){RUN(2);}else{RUN(4);}
    #undef RUN
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("grouped",&grouped);m.attr("production_admitted")=false;}

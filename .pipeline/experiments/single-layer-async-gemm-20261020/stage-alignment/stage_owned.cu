// Research-only complete SINGLE WAM block in ONE cooperative persistent kernel.
// SM80: no TMA/WGMMA, host GEMM, dynamic parallelism, or cross-CTA shared memory.
// Numerical admission is intentionally separate: WMMA != cuBLASLt bitwise order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda/atomic>
#include <map>
#include <optional>
#include "norm.cuh"  // pinned Torch Welford arithmetic, read-only helper
#include <cutlass/gemm/device/gemm.h>
#include "cutlass_gemm_mainloop.cuh"
#define FLASH_NAMESPACE wam_whole_fa2
#include "fa2_device_generated.cuh"

namespace wam_whole_layer_v1 {
namespace cg = cooperative_groups;
namespace wm = nvcuda::wmma;
using B = __nv_bfloat16;
using at::Tensor;
constexpr int D=3072, H=24, HD=128, F=14336, THREADS=256;
constexpr int MAX_STAGES=40;
using FaStandard=Flash_fwd_kernel_traits<128,128,64,4,false,false,cutlass::bfloat16_t>;
using FaSplit=Flash_fwd_kernel_traits<128,64,128,4,false,false,cutlass::bfloat16_t>;
constexpr int FA_SMEM=FaStandard::kSmemSize>FaSplit::kSmemSize?FaStandard::kSmemSize:FaSplit::kSmemSize;
enum Buf {MOD,N1,M1,SQ0,SK0,SV,SQN,SKN,SQR,SKR,SA,SO,R1,N2,
          CQ0,CK0,CV,CQ,CK,CA,CO,R2,N3,M3,F1,ACT,F2,OUT,NB};
enum Point {TABLE,MODULATE,RESIDUAL,ROPE,GELU,INSERT,RESTORE};
// Warp 0 claims CTA tasks. Warps 3..6 compute; small-M GEMM uses warps 1/2
// as loaders. Large-M GEMM loads within its compute group, then all eight
// warps cooperate in the epilogue. Norm/small-M still use warp 7 to store.
struct Params {
    const void* w[27]; void* a[NB];
    const B *x,*context; const void* temb;
    const c10::complex<float>* freq;
    B *pk,*pv; int64_t *ids,*rows,*slots,*free,*info;
    bool *mask,*pred;
    int *queues,*visits,*stats,*ffn_counts; int64_t* stage_ticks;
    int b,q,c,l,fb,pool,pool_b,update,max_tasks;
    bool temb_float,cache,candidate,affine,audit,bf16_splits,tile_pipeline;
    float eps[7];
    float *fa_lse,*fa_lse_partial,*fa_out_partial;
    int64_t* fa_meta; int sms; float attn_scale,attn_scale_log2;
};
template<int TN,int TK>struct GemmStorage {B x[2][64*(TK+8)],w[2][TN*(TK+8)];float y[64*TN],partial_y[64*TN];};
using GemmShared=GemmStorage<64,32>;
#include "cutlass_gemm.cuh"
struct NormShared {alignas(16) B input[D];float x[D],y[D],reduce[6];};
struct AttnShared {
    B q[16*128],k[64*128],v[64*128],p[16*64];
    float score[16*64],o[16*128],pv[16*128];
    float m[16],l[16],alpha[16];
};
union alignas(32) TileShared {GemmShared gemm;CutlassGemmStorage cutlass_gemm;NormShared norm;AttnShared attn;float point[512];char fa2_workspace[FA_SMEM];};
struct Shared {TileShared tile;int task;wam_whole_fa2::WholeParams fa2;};
__device__ inline int warp(){return threadIdx.x/32;}
__device__ inline int lane(){return threadIdx.x%32;}
__device__ inline bool compute(){return warp()>=3&&warp()<7;}
__device__ inline int ct(){return int(threadIdx.x)-96;}
__device__ inline float value(B x){return __bfloat162float(x);}
__device__ inline B bf(float x){return __float2bfloat16_rn(x);}
__device__ inline float* fp(const Params& p,int i){return (float*)p.a[i];}
__device__ inline B* bp(const Params& p,int i){return (B*)p.a[i];}
__device__ inline void group_sync(){asm volatile("bar.sync 1, 128;" ::: "memory");}

// Coarse stage barriers encode the real dependency DAG. All resident CTAs must
// reach each barrier. No consumer spins while its producer is unscheduled.
__device__ void begin(const Params& p,Shared& s,cg::grid_group& grid,int st,int n){
    if(grid.thread_rank()==0){p.queues[st]=0;p.stats[st*3]=n;p.stats[st*3+1]=0;p.stats[st*3+2]=0;}
    grid.sync();
}
__device__ int claim(const Params& p,Shared& s,int st,int n){
    if(threadIdx.x==0)s.task=atomicAdd(p.queues+st,1);
    __syncthreads();
    if(s.task<n&&threadIdx.x==0&&p.audit){
        if(atomicAdd(p.visits+st*p.max_tasks+s.task,1)!=0)atomicAdd(p.stats+st*3+2,1);
    }
    return s.task;
}
__device__ void done(const Params& p,int st){
    __syncthreads();if(threadIdx.x==0&&p.audit)atomicAdd(p.stats+st*3+1,1);
}

// Small-M: 64x64 output / K32 with separate loader warps. Action M>=128:
// 64x128 / K64. Context M>=1024: compile-selected 64/128 x128 / K64.
// Every path retains increasing-K and the recorded BF16/bias boundaries.
#include "gemm_pipeline.cuh"
__device__ void linear(const Params& p,Shared& s,cg::grid_group& g,int st,
                      const B* a,int wi,B* out,int m,int n,int k){
    const int tn=m>=128?128:64;
    const int tm=gemm_tile_rows(m);
    int ntiles=((m+tm-1)/tm)*((n+tn-1)/tn);begin(p,s,g,st,ntiles);
    for(int t;(t=claim(p,s,st,ntiles))<ntiles;){
        gemm_tile(s,a,(const B*)p.w[wi],(const B*)p.w[wi+1],out,m,n,k,t,p.bf16_splits);done(p,st);
    }
    g.sync();
}

#include "norm_device_generated.cuh"

// Same vector-4 per-thread input order and Welford tree as the pinned Torch
// helper. Logical compute coordinates are explicitly remapped to four warps;
// its group barrier has exactly 128 participants, never a partial CTA barrier.
__device__ void norm_tile(Shared& sh,const B* x,void* y,const void* gamma,
                         const float* beta,float eps,bool rms,bool output_float,int row){
    auto& s=sh.tile.norm;
    if(warp()==1){
        for(int i=lane();i<D/8;i+=32)copy16_async(s.input+i*8,x+row*D+i*8,true);
        asm volatile("cp.async.commit_group;":::"memory");
        asm volatile("cp.async.wait_group 0;":::"memory");
    }
    __syncthreads();
    if(compute())for(int v=ct();v<D/4;v+=128){
        #pragma unroll
        for(int j=0;j<4;++j)s.x[v*4+j]=value(s.input[v*4+j]);
    }
    __syncthreads();
    if(compute() && !rms){
        whole_norm_forward<float,float,false>(D,eps,s.x,
            static_cast<const float*>(gamma),beta,nullptr,nullptr,s.y,s.reduce);
    }else if(compute()){
        using namespace wam_optimized_norm;
        int tid=ct(),cw=tid/32,cl=tid%32;WelfordDataLN wd;
        for(int v=tid;v<D/4;v+=128)for(int j=0;j<4;++j)
            wd=rms?cuWelfordOnlineSum<float,true>(s.x[v*4+j],wd):cuWelfordOnlineSum<float,false>(s.x[v*4+j],wd);
        for(int offset=16;offset;offset>>=1){
            WelfordDataLN b{__shfl_down_sync(0xffffffff,wd.mean,offset),
                __shfl_down_sync(0xffffffff,wd.sigma2,offset),__shfl_down_sync(0xffffffff,wd.count,offset)};
            wd=rms?cuWelfordCombine<true>(wd,b):cuWelfordCombine<false>(wd,b);
        }
        for(int offset=2;offset;offset>>=1){
            if(cl==0&&cw>=offset&&cw<2*offset){int j=cw-offset;
                s.reduce[2*j]=wd.mean;s.reduce[2*j+1]=wd.sigma2;s.reduce[4+j]=wd.count;}
            group_sync();
            if(cl==0&&cw<offset){WelfordDataLN b{s.reduce[2*cw],s.reduce[2*cw+1],s.reduce[4+cw]};
                wd=rms?cuWelfordCombine<true>(wd,b):cuWelfordCombine<false>(wd,b);}
            group_sync();
        }
        if(tid==0){s.reduce[0]=wd.mean;s.reduce[1]=wd.sigma2/float(D);}
        group_sync();
        float rstd=c10::cuda::compat::rsqrt(s.reduce[1]+eps),mean=s.reduce[0];
        for(int v=tid;v<D/4;v+=128)for(int j=0;j<4;++j){
            int i=v*4+j;float z=rms?rstd*s.x[i]:rstd*(s.x[i]-mean);
            if(gamma)z=(rms?value(((const B*)gamma)[i]):((const float*)gamma)[i])*z;
            if(beta)z=z+beta[i];s.y[i]=z;
        }
    }
    __syncthreads();
    if(warp()==7){
        if(output_float){
            for(int i=lane();i<D/4;i+=32)
                reinterpret_cast<float4*>((float*)y+row*D)[i]=reinterpret_cast<const float4*>(s.y)[i];
        }else{
            for(int i=lane();i<D/8;i+=32){
                uint32_t packed[4];
                #pragma unroll
                for(int j=0;j<4;++j){
                    const unsigned lo=__bfloat16_as_ushort(bf(s.y[i*8+j*2]));
                    const unsigned hi=__bfloat16_as_ushort(bf(s.y[i*8+j*2+1]));
                    packed[j]=lo|(hi<<16);
                }
                reinterpret_cast<uint4*>((B*)y+row*D)[i]=make_uint4(packed[0],packed[1],packed[2],packed[3]);
            }
        }
    }
    __syncthreads();
}
__device__ void norm(const Params& p,Shared& s,cg::grid_group& g,int st,
                    const B* x,void* y,int rows,int gi,int bi,float eps,bool rms,bool out_float){
    begin(p,s,g,st,rows);for(int t;(t=claim(p,s,st,rows))<rows;){
        norm_tile(s,x,y,gi>=0?p.w[gi]:nullptr,bi>=0?(const float*)p.w[bi]:nullptr,eps,rms,out_float,t);done(p,st);
    }g.sync();
}

__device__ void layout(const Params& p){
    if(blockIdx.x!=0||threadIdx.x!=0)return;
    p.info[0]=0;p.info[1]=0;if(!p.cache)return;
    if(p.pool>4096){p.info[1]=1;return;}
    if(p.candidate){
        for(int b=0;b<p.b;++b)if(p.rows[b]<0||p.rows[b]>=p.pool_b)p.info[1]=2;
    }else{
        int free=0,used=0;for(int i=0;i<p.pool;++i){if(p.mask[i])++used;else++free;}
        int need=max(0,p.q-free);
        if(need&&(p.pool!=2160||used<=128)){p.info[1]=3;return;}
        // Stable sort equivalent for the needed smallest IDs; ties use physical
        // slot order. It is intentionally a slow first correctness path.
        for(int e=0;e<need;++e){int chosen=-1;int64_t low=INT64_MAX;
            for(int i=0;i<p.pool;++i)if(p.mask[i]&&(chosen<0||p.ids[i]<low)){chosen=i;low=p.ids[i];}
            p.mask[chosen]=false;p.ids[chosen]=-1;
        }
        int n=0;int64_t high=-1;
        for(int i=0;i<p.pool;++i){if(p.mask[i])high=max(high,p.ids[i]);else if(n<p.q)p.free[n++]=i;}
        if(n!=p.q){p.info[1]=4;return;}
        for(int j=0;j<p.q;++j){int i=p.free[j];p.mask[i]=true;p.ids[i]=high+1;p.pred[i]=(p.update==1);}
    }
    int n=0;for(int i=0;i<p.pool;++i)if(p.mask[i])p.slots[n++]=i;p.info[0]=n;
}

__device__ void point(const Params& p,Shared& s,cg::grid_group& g,int st,Point op,
                     int dst,int a,int b,int param,int64_t count){
    constexpr int tile_elements=2048;
    int tiles=(count+tile_elements-1)/tile_elements;begin(p,s,g,st,tiles);
    for(int task;(task=claim(p,s,st,tiles))<tiles;){
        if(compute())for(int j=ct();j<tile_elements;j+=128){
            int64_t i=int64_t(task)*tile_elements+j;float z=0.f;
            if(i<count){
                if(op==TABLE){int row=i/(6*D),cell=i%(6*D),batch=row/p.q,t=row%p.q;
                    int64_t j=(int64_t(batch)*p.l+(p.l==1?0:t))*6*D+cell;
                    float te=p.temb_float?((const float*)p.temb)[j]:value(((const B*)p.temb)[j]);
                    z=__fadd_rn(((const float*)p.w[0])[cell],te);
                }else if(op==MODULATE){int64_t j=(i/D*6+param)*D+i%D;
                    z=__fadd_rn(__fmul_rn(fp(p,a)[i],__fadd_rn(1.f,fp(p,MOD)[j+D])),fp(p,MOD)[j]);
                }else if(op==RESIDUAL){float v=value(bp(p,b)[i]);
                    if(param>=0)v=__fmul_rn(v,fp(p,MOD)[(i/D*6+param)*D+i%D]);
                    z=__fadd_rn(value(a<0?p.x[i]:bp(p,a)[i]),v);
                }else if(op==ROPE){int64_t r=i/64,batch=r/(p.q*H),t=(r/H)%p.q,j=i%64;
                    auto f=p.freq[((p.fb==1?0:batch)*p.q+t)*64+j];
                    double x=double(value(bp(p,a)[2*i])),y=double(value(bp(p,a)[2*i+1]));
                    const float real=float(x*double(f.real())-y*double(f.imag()));
                    const float imag=float(x*double(f.imag())+y*double(f.real()));
                    bp(p,dst)[2*i]=bf(real);bp(p,dst)[2*i+1]=bf(imag);
                }else if(op==GELU){float x=value(bp(p,a)[i]);
                    constexpr float beta=M_SQRT2*M_2_SQRTPI*float(.5),kappa=.044715;
                    float cube=x*x*x,inner=beta*(x+kappa*cube);z=.5f*x*(1.f+c10::cuda::compat::tanh(inner));
                }
                if(op==TABLE)fp(p,dst)[i]=z;
                else if(op!=ROPE)bp(p,dst)[i]=bf(z);
            }
        }
        done(p,st);
    }g.sync();
}
__device__ void insert_cache(const Params& p,Shared& s,cg::grid_group& g,int st){
    int count=(p.cache&&!p.candidate&&p.info[1]==0)?p.b*p.q*D:0;
    constexpr int tile_elements=2048;
    int tiles=(count+tile_elements-1)/tile_elements;begin(p,s,g,st,tiles);
    for(int t;(t=claim(p,s,st,tiles))<tiles;){
        if(warp()==1||warp()==2)for(int i=t*tile_elements+lane();i<min(count,(t+1)*tile_elements);i+=32){
            int64_t dest=((int64_t(i/(p.q*D))*p.pool+p.free[(i/D)%p.q])*D+i%D);
            if(warp()==1)p.pk[dest]=bp(p,SKR)[i];else p.pv[dest]=bp(p,SV)[i];
        }done(p,st);
    }g.sync();
}
__device__ const B* self_kv(const Params& p,bool val,int batch,int token,int head,int dim){
    if(!p.cache)return bp(p,val?SV:SKR)+(batch*p.q+token)*D+head*HD+dim;
    if(p.candidate&&token>=p.info[0])return bp(p,val?SV:SKR)+(batch*p.q+token-p.info[0])*D+head*HD+dim;
    int pb=p.candidate?int(p.rows[batch]):batch;
    return (val?p.pv:p.pk)+(int64_t(pb)*p.pool+p.slots[token])*D+head*HD+dim;
}

// Query tiles always stay within one (candidate, head) instance. B14 is NOT a
// Q112 sequence. Candidate private suffix visibility is preserved explicitly.
#include "fa2_stage.cuh"
#include "ffn_pipeline.cuh"


__device__ inline int64_t diagnostic_ns(){
    unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;
}
template<int Stage> __device__ void execute_stage(const Params& p,Shared& s,cg::grid_group& g){
    const int m=p.b*p.q,mc=p.b*p.c;
    if constexpr(Stage==0){point(p,s,g,0,TABLE,MOD,0,0,0,int64_t(m)*6*D);}
    if constexpr(Stage==1){norm(p,s,g,1,p.x,p.a[N1],m,-1,-1,p.eps[0],false,true);}
    if constexpr(Stage==2){point(p,s,g,2,MODULATE,M1,N1,0,0,int64_t(m)*D);}
    if constexpr(Stage==3){linear(p,s,g,3,bp(p,M1),3,bp(p,SQ0),m,D,D);}
    if constexpr(Stage==4){linear(p,s,g,4,bp(p,M1),5,bp(p,SK0),m,D,D);}
    if constexpr(Stage==5){linear(p,s,g,5,bp(p,M1),7,bp(p,SV),m,D,D);}
    if constexpr(Stage==6){norm(p,s,g,6,bp(p,SQ0),p.a[SQN],m,11,-1,p.eps[3],true,false);}
    if constexpr(Stage==7){norm(p,s,g,7,bp(p,SK0),p.a[SKN],m,12,-1,p.eps[4],true,false);}
    if constexpr(Stage==8){point(p,s,g,8,ROPE,SQR,SQN,0,0,int64_t(m)*D/2);}
    if constexpr(Stage==9){point(p,s,g,9,ROPE,SKR,SKN,0,0,int64_t(m)*D/2);}
    if constexpr(Stage==10){insert_cache(p,s,g,10);}
    if constexpr(Stage==11){attention(p,s,g,11,false);}
    if constexpr(Stage==12){linear(p,s,g,12,bp(p,SA),9,bp(p,SO),m,D,D);}
    if constexpr(Stage==13){point(p,s,g,13,RESIDUAL,R1,-1,SO,2,int64_t(m)*D);}
    if constexpr(Stage==14){norm(p,s,g,14,bp(p,R1),p.a[N2],m,1,2,p.eps[1],false,false);}
    if constexpr(Stage==15){linear(p,s,g,15,bp(p,N2),13,bp(p,CQ0),m,D,D);}
    if constexpr(Stage==16){linear(p,s,g,16,p.context,15,bp(p,CK0),mc,D,D);}
    if constexpr(Stage==17){linear(p,s,g,17,p.context,17,bp(p,CV),mc,D,D);}
    if constexpr(Stage==18){norm(p,s,g,18,bp(p,CQ0),p.a[CQ],m,21,-1,p.eps[5],true,false);}
    if constexpr(Stage==19){norm(p,s,g,19,bp(p,CK0),p.a[CK],mc,22,-1,p.eps[6],true,false);}
    if constexpr(Stage==20){attention(p,s,g,20,true);}
    if constexpr(Stage==21){linear(p,s,g,21,bp(p,CA),19,bp(p,CO),m,D,D);}
    if constexpr(Stage==22){point(p,s,g,22,RESIDUAL,R2,R1,CO,-1,int64_t(m)*D);}
    if constexpr(Stage==23){norm(p,s,g,23,bp(p,R2),p.a[N3],m,-1,-1,p.eps[2],false,true);}
    if constexpr(Stage==24){point(p,s,g,24,MODULATE,M3,N3,0,3,int64_t(m)*D);}
    if constexpr(Stage==25){linear(p,s,g,25,bp(p,M3),23,bp(p,F1),m,F,D);}
    if constexpr(Stage==26){point(p,s,g,26,GELU,ACT,F1,0,0,int64_t(m)*F);}
    if constexpr(Stage==27){linear(p,s,g,27,bp(p,ACT),25,bp(p,F2),m,D,F);}
    if constexpr(Stage==28){point(p,s,g,28,RESIDUAL,OUT,R2,F2,5,int64_t(m)*D);}
    if constexpr(Stage==29){layout(p);g.sync();}
}
template<int Stage> __device__ void timed_stage(const Params& p,Shared& s,cg::grid_group& g){
    if(g.thread_rank()==0)p.stage_ticks[2*Stage]=diagnostic_ns();
    execute_stage<Stage>(p,s,g);
    if(g.thread_rank()==0)p.stage_ticks[2*Stage+1]=diagnostic_ns();
}
template<int Stage> __global__ void one_stage(const __grid_constant__ Params p){
    extern __shared__ __align__(32) unsigned char memory[];
    auto& s=*reinterpret_cast<Shared*>(memory);auto g=cg::this_grid();
    execute_stage<Stage>(p,s,g);
}
__global__ void whole_layer(const __grid_constant__ Params p){
    extern __shared__ __align__(32) unsigned char memory[];
    auto& s=*reinterpret_cast<Shared*>(memory);auto g=cg::this_grid();
    for(int i=g.thread_rank();i<MAX_STAGES*3;i+=g.size())p.stats[i]=0;
    timed_stage<29>(p,s,g);
    timed_stage<0>(p,s,g);
    timed_stage<1>(p,s,g);
    timed_stage<2>(p,s,g);
    timed_stage<3>(p,s,g);
    timed_stage<4>(p,s,g);
    timed_stage<5>(p,s,g);
    timed_stage<6>(p,s,g);
    timed_stage<7>(p,s,g);
    timed_stage<8>(p,s,g);
    timed_stage<9>(p,s,g);
    timed_stage<10>(p,s,g);
    timed_stage<11>(p,s,g);
    timed_stage<12>(p,s,g);
    timed_stage<13>(p,s,g);
    timed_stage<14>(p,s,g);
    timed_stage<15>(p,s,g);
    timed_stage<16>(p,s,g);
    timed_stage<17>(p,s,g);
    timed_stage<18>(p,s,g);
    timed_stage<19>(p,s,g);
    timed_stage<20>(p,s,g);
    timed_stage<21>(p,s,g);
    timed_stage<22>(p,s,g);
    timed_stage<23>(p,s,g);
    timed_stage<24>(p,s,g);
    timed_stage<25>(p,s,g);
    timed_stage<26>(p,s,g);
    timed_stage<27>(p,s,g);
    timed_stage<28>(p,s,g);
    g.sync();
}
__global__ void marker_kernel(int64_t* times,int index){
    if(threadIdx.x==0)times[index]=diagnostic_ns();
}
void marker(Tensor times,int index){
    marker_kernel<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>(times.data_ptr<int64_t>(),index);
}

std::map<std::string,Tensor> forward(const std::vector<Tensor>& w,const Tensor& x,
    const Tensor& context,const Tensor& temb,const std::optional<Tensor>& freq,
    const std::vector<Tensor>& cache,const std::optional<Tensor>& rows,int update,
    const std::vector<double>& eps,bool audit,bool bf16_splits,bool tile_pipeline,int selected_stage,const std::vector<Tensor>& preload,const std::vector<Tensor>& meta){
    TORCH_CHECK(w.size()==27&&eps.size()==7,"whole-layer weight schema");
    TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&x.dim()==3&&x.size(2)==D,"BF16 [B,Q,3072]");
    TORCH_CHECK(uintptr_t(x.data_ptr())%16==0,"input 16-byte alignment");
    TORCH_CHECK((x.size(0)==2&&(x.size(1)==16||x.size(1)==128))||(x.size(0)==14&&x.size(1)==16),"nine actual signatures only: B2 Q16/Q128 or B14 Q16");
    TORCH_CHECK(context.is_contiguous()&&context.scalar_type()==at::kBFloat16&&context.size(0)==x.size(0)&&context.size(2)==D,"context");
    TORCH_CHECK(context.device()==x.device()&&context.size(1)>0&&uintptr_t(context.data_ptr())%16==0,"context device/alignment");
    TORCH_CHECK(temb.is_contiguous()&&(temb.scalar_type()==at::kFloat||temb.scalar_type()==at::kBFloat16)&&temb.size(0)==x.size(0)&&temb.size(-2)==6&&temb.size(-1)==D&&(temb.size(1)==1||temb.size(1)==x.size(1)),"temb");
    TORCH_CHECK(temb.device()==x.device(),"temb device");
    TORCH_CHECK(update>=0&&update<=2,"update mode");c10::cuda::CUDAGuard guard(x.device());
    Params p{};p.b=x.size(0);p.q=x.size(1);p.c=context.size(1);p.l=temb.size(1);p.update=update;p.audit=audit;p.bf16_splits=bf16_splits;p.tile_pipeline=tile_pipeline;
    p.x=(const B*)x.data_ptr();p.context=(const B*)context.data_ptr();p.temb=temb.data_ptr();p.temb_float=temb.scalar_type()==at::kFloat;
    for(int i=0;i<27;++i){TORCH_CHECK(w[i].device()==x.device()&&w[i].is_contiguous()&&!w[i].requires_grad(),"weight device/layout/grad");
        TORCH_CHECK(w[i].scalar_type()==(i<=2?at::kFloat:at::kBFloat16)||w[i].numel()==0,"explicit canonical dtype index ",i);
        TORCH_CHECK(w[i].numel()==0||uintptr_t(w[i].data_ptr())%16==0,"weight alignment");p.w[i]=w[i].data_ptr();}
    for(int i=0;i<7;++i)p.eps[i]=eps[i];p.affine=w[1].numel()!=0;
    if(freq.has_value()){TORCH_CHECK(freq->device()==x.device()&&freq->scalar_type()==at::kComplexFloat&&freq->is_contiguous()
        &&(freq->dim()==3||(freq->dim()==4&&freq->size(2)==1))&&freq->size(1)==p.q&&freq->size(-1)==64
        &&(freq->size(0)==1||freq->size(0)==p.b),"RoPE [B,Q,1,64] or [B,Q,64]");
        p.freq=freq->data_ptr<c10::complex<float>>();p.fb=freq->size(0);}
    if(!cache.empty()){
        TORCH_CHECK(cache.size()==5&&cache[0].is_contiguous()&&cache[1].is_contiguous(),"cache schema");
        for(const auto& t:cache)TORCH_CHECK(t.device()==x.device()&&t.is_contiguous(),"cache device/layout");
        TORCH_CHECK(cache[0].dim()==4&&cache[0].size(2)==H&&cache[0].size(3)==HD&&cache[0].sizes()==cache[1].sizes()
            &&cache[0].scalar_type()==at::kBFloat16&&cache[1].scalar_type()==at::kBFloat16
            &&uintptr_t(cache[0].data_ptr())%16==0&&uintptr_t(cache[1].data_ptr())%16==0,"cache values/alignment");
        p.cache=true;p.pk=(B*)cache[0].data_ptr();p.pv=(B*)cache[1].data_ptr();p.ids=cache[2].data_ptr<int64_t>();p.mask=cache[3].data_ptr<bool>();p.pred=cache[4].data_ptr<bool>();
        p.pool=cache[0].size(1);p.pool_b=cache[0].size(0);
        TORCH_CHECK(cache[2].numel()==p.pool&&cache[3].numel()==p.pool&&cache[4].numel()==p.pool,"cache metadata size");
        TORCH_CHECK(p.pool<=4096&&p.pool>=p.q,"finite pool contract");
        p.candidate=rows.has_value();if(p.candidate){TORCH_CHECK(update==0&&rows->device()==x.device()&&rows->numel()==p.b&&rows->is_contiguous(),"candidate read-only contract");p.rows=rows->data_ptr<int64_t>();}
        else TORCH_CHECK(p.pool_b==p.b,"noncandidate pool batch");
    }else TORCH_CHECK(!rows.has_value(),"candidate requires history");
    std::vector<Tensor> b(NB);auto bo=x.options(),fo=bo.dtype(at::kFloat),lo=bo.dtype(at::kLong),io=bo.dtype(at::kInt);
    for(int i=0;i<NB;++i){if(!preload.empty()){b[i]=preload[i];p.a[i]=b[i].data_ptr();continue;}int r=(i==CK0||i==CK||i==CV)?p.b*p.c:p.b*p.q;
        if(i==MOD)b[i]=at::empty({r,6,D},fo);
        else b[i]=at::empty({r,(i==F1||i==ACT)?F:D},(i==N1||i==N3)?fo:bo);
        p.a[i]=b[i].data_ptr();}
    if(!p.freq){b[SQR]=b[SQN];b[SKR]=b[SKN];p.a[SQR]=p.a[SQN];p.a[SKR]=p.a[SKN];}
    if(!p.affine){b[N2]=b[R1];p.a[N2]=p.a[R1];}
    auto slots=at::empty({std::max(p.pool,1)},lo),free=at::empty({p.q},lo),info=at::empty({2},lo);
    if(!meta.empty()){slots=meta[0];info=meta[1];}
    p.slots=slots.data_ptr<int64_t>();p.free=free.data_ptr<int64_t>();p.info=info.data_ptr<int64_t>();
    p.max_tasks=std::max({(p.b*p.q*6*D+127)/128,(p.b*p.q*F+127)/128,((p.b*p.c+63)/64)*48});
    auto ticks=at::empty({30,2},lo);p.stage_ticks=ticks.data_ptr<int64_t>();
    auto queues=at::empty({MAX_STAGES},io),stats=at::empty({MAX_STAGES,3},io);
    auto visits=at::empty({audit?MAX_STAGES*p.max_tasks:1},io);
    p.queues=queues.data_ptr<int>();p.stats=stats.data_ptr<int>();p.visits=visits.data_ptr<int>();
    auto ffn_counts=at::empty({((p.b*p.q+63)/64)*2+4},io);p.ffn_counts=ffn_counts.data_ptr<int>();
    int device=x.get_device(),sm=0,cooperative=0,active=0;
    p.sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    p.attn_scale=float(1.0/std::sqrt(128.0));
    p.attn_scale_log2=float(double(p.attn_scale)*M_LOG2E);
    int max_splits=(p.b*H*((p.q+63)/64)>=0.8f*p.sms*2)?1:
        std::min(128,(std::max(p.pool+p.q,p.c)+127)/128);
    auto lse=at::empty({p.b,H,p.q},fo);
    auto partial_lse=at::empty({max_splits,p.b,H,p.q},fo);
    auto partial_out=at::empty({max_splits,p.b,H,p.q,HD},fo);
    auto fa_meta=at::empty({2,4},lo);
    p.fa_lse=lse.data_ptr<float>();p.fa_lse_partial=partial_lse.data_ptr<float>();
    p.fa_out_partial=partial_out.data_ptr<float>();p.fa_meta=fa_meta.data_ptr<int64_t>();
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device));TORCH_CHECK(cooperative,"cooperative launch required");
    void* kernel=(void*)whole_layer;
    switch(selected_stage){case 0:kernel=(void*)one_stage<0>;break;case 1:kernel=(void*)one_stage<1>;break;case 2:kernel=(void*)one_stage<2>;break;case 3:kernel=(void*)one_stage<3>;break;case 4:kernel=(void*)one_stage<4>;break;case 5:kernel=(void*)one_stage<5>;break;case 6:kernel=(void*)one_stage<6>;break;case 7:kernel=(void*)one_stage<7>;break;case 8:kernel=(void*)one_stage<8>;break;case 9:kernel=(void*)one_stage<9>;break;case 10:kernel=(void*)one_stage<10>;break;case 11:kernel=(void*)one_stage<11>;break;case 12:kernel=(void*)one_stage<12>;break;case 13:kernel=(void*)one_stage<13>;break;case 14:kernel=(void*)one_stage<14>;break;case 15:kernel=(void*)one_stage<15>;break;case 16:kernel=(void*)one_stage<16>;break;case 17:kernel=(void*)one_stage<17>;break;case 18:kernel=(void*)one_stage<18>;break;case 19:kernel=(void*)one_stage<19>;break;case 20:kernel=(void*)one_stage<20>;break;case 21:kernel=(void*)one_stage<21>;break;case 22:kernel=(void*)one_stage<22>;break;case 23:kernel=(void*)one_stage<23>;break;case 24:kernel=(void*)one_stage<24>;break;case 25:kernel=(void*)one_stage<25>;break;case 26:kernel=(void*)one_stage<26>;break;case 27:kernel=(void*)one_stage<27>;break;case 28:kernel=(void*)one_stage<28>;break;case 29:kernel=(void*)one_stage<29>;break;}

    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,THREADS,sizeof(Shared)));
    TORCH_CHECK(active>0,"stage cannot reside");
    // Same 108 resident-CTA dispatch as the full kernel. This is not a tile/grid sweep.
    int blocks=sm;void* args[]={&p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel,dim3(blocks),dim3(THREADS),args,sizeof(Shared),at::cuda::getCurrentCUDAStream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::map<std::string,Tensor> result;
    const char* names[NB]={"modulation","norm1","modulated1","self_q_linear","self_k_linear","self_v","self_q_norm","self_k_norm","self_q_rope","self_k_rope","self_attention","self_out","residual1","norm2","cross_q_linear","cross_k_linear","cross_v","cross_q","cross_k","cross_attention","cross_out","residual2","norm3","modulated3","ffn1","gelu","ffn2","output"};
    for(int i=0;i<NB;++i)result[names[i]]=b[i];result["output"]=b[OUT].view(x.sizes());
    result["stage_ticks"]=ticks;result["tile_stats"]=stats;result["cache_info"]=info;
    // CPU resource metadata is not part of the mathematical GPU work.
    cudaFuncAttributes attr;C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
    result["resources"]=at::tensor({int64_t(blocks),int64_t(active),int64_t(sizeof(Shared)),int64_t(attr.numRegs),int64_t(attr.localSizeBytes),int64_t(THREADS)},at::TensorOptions().dtype(at::kLong));
    result["attention_meta"]=fa_meta;
    if(tile_pipeline)result["ffn_schedule"]=ffn_counts;
    result["logical_slots"]=slots; // Only cache_info[0] entries are initialized.
    return result;
}
} // namespace wam_whole_layer_v1
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("forward",&wam_whole_layer_v1::forward);
    m.def("marker",&wam_whole_layer_v1::marker);
    m.attr("production_admitted")=false;
    m.attr("computation_launches")=1;
    m.attr("context_tile_rows")=WAM_CONTEXT_TILE_M;
    m.attr("point_tile_elements")=2048;
    m.attr("cooperative_epilogue")=true;
    m.attr("scheduling")="cooperative stage DAG; warp0 queue, warp1 activation, warp2 weight/KV, warp3-6 compute, warp7 store";
}

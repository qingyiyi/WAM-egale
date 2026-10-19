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
#include <map>
#include <optional>
#include "norm.cuh"  // pinned Torch Welford arithmetic, read-only helper

namespace wam_whole_layer_v1 {
namespace cg = cooperative_groups;
namespace wm = nvcuda::wmma;
using B = __nv_bfloat16;
using at::Tensor;
constexpr int D=3072, H=24, HD=128, F=14336, THREADS=256;
constexpr int MAX_STAGES=40;
enum Buf {MOD,N1,M1,SQ0,SK0,SV,SQN,SKN,SQR,SKR,SA,SO,R1,N2,
          CQ0,CK0,CV,CQ,CK,CA,CO,R2,N3,M3,F1,ACT,F2,OUT,NB};
enum Point {TABLE,MODULATE,RESIDUAL,ROPE,GELU,INSERT,RESTORE};
// Warp 0: queue/metadata; 1: activation/Q; 2: weights/KV;
// 3..6: Tensor Core/normalization/pointwise; 7: output/epilogue publication.
struct Params {
    const void* w[27]; void* a[NB];
    const B *x,*context; const void* temb;
    const c10::complex<float>* freq;
    B *pk,*pv; int64_t *ids,*rows,*slots,*free,*info;
    bool *mask,*pred;
    int *queues,*visits,*stats;
    int b,q,c,l,fb,pool,pool_b,update,max_tasks;
    bool temb_float,cache,candidate,affine,audit;
    float eps[7];
};
struct GemmShared { B x[64*32],w[64*32]; float y[64*64]; };
struct NormShared {float x[D],y[D],reduce[6];};
struct AttnShared {
    B q[16*128],k[64*128],v[64*128],p[16*64];
    float score[16*64],o[16*128],pv[16*128];
    float m[16],l[16],alpha[16];
};
union alignas(32) TileShared {GemmShared gemm;NormShared norm;AttnShared attn;float point[512];};
struct Shared {TileShared tile;int task;};
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

// Four compute warps each own a 32x32 portion of a 64x64 tile.
// Two loader warps stage A and W separately; the store warp alone adds bias
// and performs the public BF16 rounding. K is consumed in increasing order.
__device__ void gemm_tile(Shared& shared,const B* a,const B* w,const B* bias,
                         B* out,int m,int n,int k,int task){
    auto& s=shared.tile.gemm;
    const int cols=(n+63)/64,rm=(task/cols)*64,cn=(task%cols)*64;
    wm::fragment<wm::accumulator,16,16,16,float> acc[4];
    if(compute())for(int j=0;j<4;++j)wm::fill_fragment(acc[j],0.f);
    for(int z=0;z<k;z+=32){
        if(warp()==1)for(int i=lane();i<64*32/8;i+=32){
            int r=i/4,c=(i%4)*8;
            ((uint4*)s.x)[i]=(rm+r<m)?*(const uint4*)(a+(rm+r)*k+z+c):make_uint4(0,0,0,0);
        }
        if(warp()==2)for(int i=lane();i<64*32/8;i+=32){
            int r=i/4,c=(i%4)*8;
            ((uint4*)s.w)[i]=(cn+r<n)?*(const uint4*)(w+(cn+r)*k+z+c):make_uint4(0,0,0,0);
        }
        __syncthreads();
        if(compute()){
            int cw=warp()-3,m0=(cw/2)*32,n0=(cw%2)*32;
            #pragma unroll
            for(int t=0;t<2;++t){
                wm::fragment<wm::matrix_a,16,16,16,B,wm::row_major> af[2];
                wm::fragment<wm::matrix_b,16,16,16,B,wm::col_major> wf[2];
                #pragma unroll
                for(int j=0;j<2;++j){
                    wm::load_matrix_sync(af[j],s.x+(m0+j*16)*32+t*16,32);
                    wm::load_matrix_sync(wf[j],s.w+(n0+j*16)*32+t*16,32);
                }
                #pragma unroll
                for(int i=0;i<2;++i)for(int j=0;j<2;++j)
                    wm::mma_sync(acc[i*2+j],af[i],wf[j],acc[i*2+j]);
            }
        }
        __syncthreads();
    }
    if(compute()){
        int cw=warp()-3,m0=(cw/2)*32,n0=(cw%2)*32;
        for(int i=0;i<2;++i)for(int j=0;j<2;++j)
            wm::store_matrix_sync(s.y+(m0+i*16)*64+n0+j*16,acc[i*2+j],64,wm::mem_row_major);
    }
    __syncthreads();
    if(warp()==7)for(int i=lane();i<4096;i+=32){
        int r=rm+i/64,c=cn+i%64;
        if(r<m&&c<n)out[r*n+c]=bf(__fadd_rn(s.y[i],value(bias[c])));
    }
    __syncthreads();
}
__device__ void linear(const Params& p,Shared& s,cg::grid_group& g,int st,
                      const B* a,int wi,B* out,int m,int n,int k){
    int ntiles=((m+63)/64)*((n+63)/64);begin(p,s,g,st,ntiles);
    for(int t;(t=claim(p,s,st,ntiles))<ntiles;){
        gemm_tile(s,a,(const B*)p.w[wi],(const B*)p.w[wi+1],out,m,n,k,t);done(p,st);
    }
    g.sync();
}

// Same vector-4 per-thread input order and Welford tree as the pinned Torch
// helper. Logical compute coordinates are explicitly remapped to four warps;
// its group barrier has exactly 128 participants, never a partial CTA barrier.
__device__ void norm_tile(Shared& sh,const B* x,void* y,const void* gamma,
                         const float* beta,float eps,bool rms,bool output_float,int row){
    auto& s=sh.tile.norm;
    if(warp()==1)for(int i=lane();i<D;i+=32)s.x[i]=value(x[row*D+i]);
    __syncthreads();
    if(compute()){
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
    if(warp()==7)for(int i=lane();i<D;i+=32){
        if(output_float)((float*)y)[row*D+i]=s.y[i];else((B*)y)[row*D+i]=bf(s.y[i]);
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
    int tiles=(count+127)/128;begin(p,s,g,st,tiles);
    for(int task;(task=claim(p,s,st,tiles))<tiles;){
        if(compute()){
            int64_t i=int64_t(task)*128+ct();float z=0.f;
            if(i<count){
                if(op==TABLE){int row=i/(6*D),cell=i%(6*D),batch=row/p.q,t=row%p.q;
                    int64_t j=(int64_t(batch)*p.l+(p.l==1?0:t))*6*D+cell;
                    float te=p.temb_float?((const float*)p.temb)[j]:value(((const B*)p.temb)[j]);
                    z=__fadd_rn(((const float*)p.w[0])[cell],te);
                }else if(op==MODULATE){int64_t j=(i/D*6+param)*D+i%D;
                    z=__fadd_rn(__fmul_rn(fp(p,a)[i],__fadd_rn(1.f,fp(p,MOD)[j+D])),fp(p,MOD)[j]);
                }else if(op==RESIDUAL){float v=value(bp(p,b)[i]);
                    if(param>=0)v=__fmul_rn(v,fp(p,MOD)[(i/D*6+param)*D+i%D]);
                    z=__fadd_rn(value(bp(p,a)[i]),v);
                }else if(op==ROPE){int64_t r=i/64,batch=r/(p.q*H),t=(r/H)%p.q,j=i%64;
                    auto f=p.freq[((p.fb==1?0:batch)*p.q+t)*64+j];
                    double x=double(value(bp(p,a)[2*i])),y=double(value(bp(p,a)[2*i+1]));
                    s.tile.point[ct()]=float(x*double(f.real())-y*double(f.imag()));
                    s.tile.point[128+ct()]=float(x*double(f.imag())+y*double(f.real()));
                }else if(op==GELU){float x=value(bp(p,a)[i]);
                    constexpr float beta=M_SQRT2*M_2_SQRTPI*float(.5),kappa=.044715;
                    float cube=x*x*x,inner=beta*(x+kappa*cube);z=.5f*x*(1.f+c10::cuda::compat::tanh(inner));
                }
            }
            if(op!=ROPE)s.tile.point[ct()]=z;
        }
        __syncthreads();
        if(warp()==7)for(int j=lane();j<128;j+=32){int64_t i=int64_t(task)*128+j;
            if(i<count){
                if(op==TABLE)fp(p,dst)[i]=s.tile.point[j];
                else if(op==ROPE){bp(p,dst)[2*i]=bf(s.tile.point[j]);bp(p,dst)[2*i+1]=bf(s.tile.point[128+j]);}
                else bp(p,dst)[i]=bf(s.tile.point[j]);
            }
        }
        done(p,st);
    }g.sync();
}
__device__ void insert_cache(const Params& p,Shared& s,cg::grid_group& g,int st){
    int count=(p.cache&&!p.candidate&&p.info[1]==0)?p.b*p.q*D:0;
    int tiles=(count+127)/128;begin(p,s,g,st,tiles);
    for(int t;(t=claim(p,s,st,tiles))<tiles;){
        if(warp()==1||warp()==2)for(int i=t*128+lane();i<min(count,(t+1)*128);i+=32){
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
__device__ void attention_tile(const Params& p,Shared& shared,bool cross,int task){
    auto& s=shared.tile.attn;int qt=(p.q+15)/16;
    int qr=(task%qt)*16,head=(task/qt)%H,batch=task/(qt*H);
    int n=cross?p.c:(p.cache?int(p.info[0])+(p.candidate?p.q:0):p.q);
    if(p.info[1])n=0;
    if(warp()==1)for(int i=lane();i<16*HD/8;i+=32){int r=qr+i/16,d=(i%16)*8;
        ((uint4*)s.q)[i]=r<p.q?*(const uint4*)(bp(p,cross?CQ:SQR)+(batch*p.q+r)*D+head*HD+d):make_uint4(0,0,0,0);
    }
    if(warp()==7){
        for(int i=lane();i<16*HD;i+=32)s.o[i]=0;
        for(int r=lane();r<16;r+=32){s.m[r]=-INFINITY;s.l[r]=0;}
    }
    __syncthreads();
    for(int off=0;off<n;off+=64){
        if(warp()==2)for(int i=lane();i<64*HD/8;i+=32){int t=off+i/16,d=(i%16)*8;
            if(t>=n){((uint4*)s.k)[i]=make_uint4(0,0,0,0);((uint4*)s.v)[i]=make_uint4(0,0,0,0);}
            else if(cross){int64_t z=(int64_t(batch)*p.c+t)*D+head*HD+d;
                ((uint4*)s.k)[i]=*(const uint4*)(bp(p,CK)+z);((uint4*)s.v)[i]=*(const uint4*)(bp(p,CV)+z);}
            else{((uint4*)s.k)[i]=*(const uint4*)self_kv(p,false,batch,t,head,d);
                 ((uint4*)s.v)[i]=*(const uint4*)self_kv(p,true,batch,t,head,d);}
        }
        __syncthreads();
        if(compute()){
            int cw=warp()-3;wm::fragment<wm::accumulator,16,16,16,float> acc;wm::fill_fragment(acc,0.f);
            for(int d=0;d<128;d+=16){
                wm::fragment<wm::matrix_a,16,16,16,B,wm::row_major> a;
                wm::fragment<wm::matrix_b,16,16,16,B,wm::col_major> b;
                wm::load_matrix_sync(a,s.q+d,128);wm::load_matrix_sync(b,s.k+cw*16*128+d,128);
                wm::mma_sync(acc,a,b,acc);
            }
            wm::store_matrix_sync(s.score+cw*16,acc,64,wm::mem_row_major);
        }
        __syncthreads();
        if(compute()&&ct()<16){int r=ct();float maxv=-INFINITY;
            for(int j=0;j<64&&off+j<n;++j)maxv=fmaxf(maxv,s.score[r*64+j]*0.08838834764831845f);
            float m=fmaxf(s.m[r],maxv),alpha=exp2f((s.m[r]-m)*1.4426950408889634f),den=0;
            for(int j=0;j<64;++j){float z=off+j<n?exp2f((s.score[r*64+j]*0.08838834764831845f-m)*1.4426950408889634f):0;
                den+=z;s.p[r*64+j]=bf(z);}
            s.l[r]=s.l[r]*alpha+den;s.m[r]=m;s.alpha[r]=alpha;
        }
        __syncthreads();
        if(compute()){
            int cw=warp()-3;wm::fragment<wm::accumulator,16,16,16,float> acc[2];
            wm::fill_fragment(acc[0],0.f);wm::fill_fragment(acc[1],0.f);
            for(int k=0;k<64;k+=16){
                wm::fragment<wm::matrix_a,16,16,16,B,wm::row_major> a;wm::load_matrix_sync(a,s.p+k,64);
                for(int j=0;j<2;++j){wm::fragment<wm::matrix_b,16,16,16,B,wm::row_major> b;
                    wm::load_matrix_sync(b,s.v+k*128+cw*32+j*16,128);wm::mma_sync(acc[j],a,b,acc[j]);}
            }
            for(int j=0;j<2;++j)wm::store_matrix_sync(s.pv+cw*32+j*16,acc[j],128,wm::mem_row_major);
        }
        __syncthreads();
        if(warp()==7)for(int i=lane();i<16*HD;i+=32)s.o[i]=s.o[i]*s.alpha[i/HD]+s.pv[i];
        __syncthreads();
    }
    if(warp()==7)for(int i=lane();i<16*HD;i+=32){int r=qr+i/HD;
        if(r<p.q)bp(p,cross?CA:SA)[(batch*p.q+r)*D+head*HD+i%HD]=bf(n?s.o[i]/s.l[i/HD]:NAN);
    }
    __syncthreads();
}
__device__ void attention(const Params& p,Shared& s,cg::grid_group& g,int st,bool cross){
    int n=p.b*H*((p.q+15)/16);begin(p,s,g,st,n);
    for(int t;(t=claim(p,s,st,n))<n;){attention_tile(p,s,cross,t);done(p,st);}g.sync();
}

__global__ void whole_layer(Params p){
    extern __shared__ __align__(32) unsigned char memory[];
    Shared& s=*reinterpret_cast<Shared*>(memory);auto g=cg::this_grid();
    // Workspace audit initialization is inside this same computational launch.
    for(int i=g.thread_rank();i<MAX_STAGES*p.max_tasks;i+=g.size())if(p.audit)p.visits[i]=0;
    for(int i=g.thread_rank();i<MAX_STAGES*3;i+=g.size())p.stats[i]=0;
    layout(p);g.sync();int st=0,m=p.b*p.q,mc=p.b*p.c;
    point(p,s,g,st++,TABLE,MOD,0,0,0,int64_t(m)*6*D);
    norm(p,s,g,st++,p.x,p.a[N1],m,-1,-1,p.eps[0],false,true);
    point(p,s,g,st++,MODULATE,M1,N1,0,0,int64_t(m)*D);
    linear(p,s,g,st++,bp(p,M1),3,bp(p,SQ0),m,D,D);
    linear(p,s,g,st++,bp(p,M1),5,bp(p,SK0),m,D,D);
    linear(p,s,g,st++,bp(p,M1),7,bp(p,SV),m,D,D);
    norm(p,s,g,st++,bp(p,SQ0),p.a[SQN],m,11,-1,p.eps[3],true,false);
    norm(p,s,g,st++,bp(p,SK0),p.a[SKN],m,12,-1,p.eps[4],true,false);
    if(p.freq){point(p,s,g,st++,ROPE,SQR,SQN,0,0,int64_t(m)*D/2);
               point(p,s,g,st++,ROPE,SKR,SKN,0,0,int64_t(m)*D/2);}
    insert_cache(p,s,g,st++);attention(p,s,g,st++,false);
    linear(p,s,g,st++,bp(p,SA),9,bp(p,SO),m,D,D);
    // Input x is an external tensor, passed as an alias only in this invocation.
    Params r=p;r.a[SQ0]=(void*)p.x;
    point(r,s,g,st++,RESIDUAL,R1,SQ0,SO,2,int64_t(m)*D);
    if(p.affine)norm(p,s,g,st++,bp(p,R1),p.a[N2],m,1,2,p.eps[1],false,false);
    linear(p,s,g,st++,bp(p,N2),13,bp(p,CQ0),m,D,D);
    linear(p,s,g,st++,p.context,15,bp(p,CK0),mc,D,D);
    linear(p,s,g,st++,p.context,17,bp(p,CV),mc,D,D);
    norm(p,s,g,st++,bp(p,CQ0),p.a[CQ],m,21,-1,p.eps[5],true,false);
    norm(p,s,g,st++,bp(p,CK0),p.a[CK],mc,22,-1,p.eps[6],true,false);
    attention(p,s,g,st++,true);linear(p,s,g,st++,bp(p,CA),19,bp(p,CO),m,D,D);
    point(p,s,g,st++,RESIDUAL,R2,R1,CO,-1,int64_t(m)*D);
    norm(p,s,g,st++,bp(p,R2),p.a[N3],m,-1,-1,p.eps[2],false,true);
    point(p,s,g,st++,MODULATE,M3,N3,0,3,int64_t(m)*D);
    linear(p,s,g,st++,bp(p,M3),23,bp(p,F1),m,F,D);
    point(p,s,g,st++,GELU,ACT,F1,0,0,int64_t(m)*F);
    linear(p,s,g,st++,bp(p,ACT),25,bp(p,F2),m,D,F);
    point(p,s,g,st++,RESIDUAL,OUT,R2,F2,5,int64_t(m)*D);
    if(p.cache&&!p.candidate&&p.info[1]==0&&p.update==0&&blockIdx.x==0&&warp()==0)
        for(int j=lane();j<p.q;j+=32)p.mask[p.free[j]]=false;
    g.sync();
}

std::map<std::string,Tensor> forward(const std::vector<Tensor>& w,const Tensor& x,
    const Tensor& context,const Tensor& temb,const std::optional<Tensor>& freq,
    const std::vector<Tensor>& cache,const std::optional<Tensor>& rows,int update,
    const std::vector<double>& eps,bool audit){
    TORCH_CHECK(w.size()==27&&eps.size()==7,"whole-layer weight schema");
    TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&x.dim()==3&&x.size(2)==D,"BF16 [B,Q,3072]");
    TORCH_CHECK((x.size(0)==2||x.size(0)==14)&&(x.size(1)==16||x.size(1)==128),"finite actual shape contract");
    TORCH_CHECK(context.is_contiguous()&&context.scalar_type()==at::kBFloat16&&context.size(0)==x.size(0)&&context.size(2)==D,"context");
    TORCH_CHECK(context.device()==x.device()&&context.size(1)>0&&uintptr_t(context.data_ptr())%16==0,"context device/alignment");
    TORCH_CHECK(temb.is_contiguous()&&(temb.scalar_type()==at::kFloat||temb.scalar_type()==at::kBFloat16)&&temb.size(0)==x.size(0)&&temb.size(-2)==6&&temb.size(-1)==D&&(temb.size(1)==1||temb.size(1)==x.size(1)),"temb");
    TORCH_CHECK(temb.device()==x.device(),"temb device");
    TORCH_CHECK(update>=0&&update<=2,"update mode");c10::cuda::CUDAGuard guard(x.device());
    Params p{};p.b=x.size(0);p.q=x.size(1);p.c=context.size(1);p.l=temb.size(1);p.update=update;p.audit=audit;
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
    for(int i=0;i<NB;++i){int r=(i==CK0||i==CK||i==CV)?p.b*p.c:p.b*p.q;
        if(i==MOD)b[i]=at::empty({r,6,D},fo);
        else b[i]=at::empty({r,(i==F1||i==ACT)?F:D},(i==N1||i==N3)?fo:bo);
        p.a[i]=b[i].data_ptr();}
    if(!p.freq){b[SQR]=b[SQN];b[SKR]=b[SKN];p.a[SQR]=p.a[SQN];p.a[SKR]=p.a[SKN];}
    if(!p.affine){b[N2]=b[R1];p.a[N2]=p.a[R1];}
    auto slots=at::empty({std::max(p.pool,1)},lo),free=at::empty({p.q},lo),info=at::empty({2},lo);
    p.slots=slots.data_ptr<int64_t>();p.free=free.data_ptr<int64_t>();p.info=info.data_ptr<int64_t>();
    p.max_tasks=std::max({(p.b*p.q*6*D+127)/128,(p.b*p.q*F+127)/128,((p.b*p.c+63)/64)*48});
    auto queues=at::empty({MAX_STAGES},io),stats=at::empty({MAX_STAGES,3},io);
    auto visits=at::empty({audit?MAX_STAGES*p.max_tasks:1},io);
    p.queues=queues.data_ptr<int>();p.stats=stats.data_ptr<int>();p.visits=visits.data_ptr<int>();
    int device=x.get_device(),sm=0,cooperative=0,active=0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device));TORCH_CHECK(cooperative,"cooperative launch required");
    C10_CUDA_CHECK(cudaFuncSetAttribute(whole_layer,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,whole_layer,THREADS,sizeof(Shared)));
    TORCH_CHECK(active>0,"whole-layer resources cannot be resident");
    int blocks=sm*active;void* args[]={&p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)whole_layer,dim3(blocks),dim3(THREADS),args,sizeof(Shared),at::cuda::getCurrentCUDAStream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::map<std::string,Tensor> result;
    const char* names[NB]={"modulation","norm1","modulated1","self_q_linear","self_k_linear","self_v","self_q_norm","self_k_norm","self_q_rope","self_k_rope","self_attention","self_out","residual1","norm2","cross_q_linear","cross_k_linear","cross_v","cross_q","cross_k","cross_attention","cross_out","residual2","norm3","modulated3","ffn1","gelu","ffn2","output"};
    for(int i=0;i<NB;++i)result[names[i]]=b[i];result["output"]=b[OUT].view(x.sizes());
    result["tile_stats"]=stats;result["cache_info"]=info;
    // CPU resource metadata is not part of the mathematical GPU work.
    cudaFuncAttributes attr;C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,whole_layer));
    result["resources"]=at::tensor({int64_t(blocks),int64_t(active),int64_t(sizeof(Shared)),int64_t(attr.numRegs),int64_t(attr.localSizeBytes),int64_t(THREADS)},at::TensorOptions().dtype(at::kLong));
    return result;
}
} // namespace wam_whole_layer_v1
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("forward",&wam_whole_layer_v1::forward);
    m.attr("production_admitted")=false;
    m.attr("computation_launches")=1;
    m.attr("scheduling")="cooperative stage DAG; warp0 queue, warp1 activation, warp2 weight/KV, warp3-6 compute, warp7 store";
}

// A row-group may enter FFN2 only after every FFN1/GELU column tile is ready.
// All CTAs are resident. No CTA blocks on an unclaimed producer: the dispatcher
// takes a ready consumer, otherwise a producer, and only idles once all producer
// tickets have been issued. Global buffers retain original BF16 boundaries.
#pragma once

__device__ int ffn_bounded_ticket(int* cursor,int limit){
    cuda::atomic_ref<int,cuda::thread_scope_device> value(*cursor);
    int next=value.load(cuda::memory_order_relaxed);
    while(next<limit){
        if(value.compare_exchange_weak(next,next+1,cuda::memory_order_relaxed))return next;
    }
    return -1;
}

__device__ void ffn_gelu_tile(Shared& s,const B* input,B* output,
                             int m,int n,int task,int tn){
    const int cols=(n+tn-1)/tn,rm=(task/cols)*64,cn=(task%cols)*tn;
    // Both GEMM layouts expose their FP32 output scratch at the same typed view.
    float* scratch=tn==128?s.tile.cutlass_gemm.result:s.tile.gemm.y;
    if(compute())for(int i=ct();i<64*tn;i+=128){
        int r=rm+i/tn,c=cn+i%tn;
        if(r<m&&c<n){
            const float x=value(input[r*n+c]);
            constexpr float beta=M_SQRT2*M_2_SQRTPI*float(.5),kappa=.044715;
            const float cube=x*x*x,inner=beta*(x+kappa*cube);
            scratch[i]=.5f*x*(1.f+c10::cuda::compat::tanh(inner));
        }
    }
    __syncthreads();
    if(warp()==7)for(int i=lane();i<64*tn;i+=32){
        int r=rm+i/tn,c=cn+i%tn;
        if(r<m&&c<n)output[r*n+c]=bf(scratch[i]);
    }
    // Publish every writer's global data before the row's release counter.
    __syncthreads();__threadfence();__syncthreads();
}

__device__ void ffn_visit(const Params& p,int stage,int task){
    if(threadIdx.x==0&&p.audit)
        if(atomicAdd(p.visits+stage*p.max_tasks+task,1)!=0)
            atomicAdd(p.stats+stage*3+2,1);
}

__device__ void ffn_subgraph(const Params& p,Shared& s,cg::grid_group& grid,
                            int stage,int m){
    using Atomic=cuda::atomic_ref<int,cuda::thread_scope_device>;
    const int tn=m>=128?128:64,groups=(m+63)/64;
    const int columns1=(F+tn-1)/tn,columns2=(D+tn-1)/tn;
    const int producers=groups*columns1,consumers=groups*columns2;
    const int completed_producers=groups*2,completed_tasks=groups*2+1;
    const int early_consumers=groups*2+2,idle_polls=groups*2+3;
    if(grid.thread_rank()==0){
        p.queues[stage]=0;
        for(int i=0;i<groups*2+4;++i)p.ffn_counts[i]=0;
        for(int i=0;i<3;++i){
            p.stats[(stage+i)*3]=i==2?consumers:producers;
            p.stats[(stage+i)*3+1]=0;p.stats[(stage+i)*3+2]=0;
        }
    }
    grid.sync();
    while(true){
        if(threadIdx.x==0){
            int ticket=-1;
            // Prefer ready consumers, but never wait for a particular row.
            for(int i=0;i<groups;++i){
                const int row=(int(blockIdx.x)+i)%groups;
                if(Atomic(p.ffn_counts[2*row]).load(cuda::memory_order_acquire)==columns1){
                    int column=ffn_bounded_ticket(p.ffn_counts+2*row+1,columns2);
                    if(column>=0){
                        ticket=producers+row*columns2+column;
                        if(Atomic(p.ffn_counts[completed_producers]).load(cuda::memory_order_relaxed)<producers)
                            atomicAdd(p.ffn_counts+early_consumers,1);
                        break;
                    }
                }
            }
            if(ticket<0){
                int producer=ffn_bounded_ticket(p.queues+stage,producers);
                if(producer>=0)ticket=producer;
                else if(Atomic(p.ffn_counts[completed_tasks]).load(cuda::memory_order_acquire)<producers+consumers){
                    ticket=-2;atomicAdd(p.ffn_counts+idle_polls,1);__nanosleep(1000);
                }
            }
            s.task=ticket;
        }
        __syncthreads();const int ticket=s.task;__syncthreads();
        if(ticket==-1)break;
        if(ticket==-2)continue;
        if(ticket<producers){
            ffn_visit(p,stage,ticket);
            gemm_tile(s,bp(p,M3),(const B*)p.w[23],(const B*)p.w[24],
                      bp(p,F1),m,F,D,ticket,p.bf16_splits);
            done(p,stage);ffn_visit(p,stage+1,ticket);
            ffn_gelu_tile(s,bp(p,F1),bp(p,ACT),m,F,ticket,tn);
            done(p,stage+1);
            if(threadIdx.x==0){
                Atomic(p.ffn_counts[completed_producers]).fetch_add(1,cuda::memory_order_release);
                Atomic(p.ffn_counts[2*(ticket/columns1)]).fetch_add(1,cuda::memory_order_release);
            }
        }else{
            const int task=ticket-producers;ffn_visit(p,stage+2,task);
            gemm_tile(s,bp(p,ACT),(const B*)p.w[25],(const B*)p.w[26],
                      bp(p,F2),m,D,F,task,p.bf16_splits);
            done(p,stage+2);
        }
        __syncthreads();__threadfence();__syncthreads();
        if(threadIdx.x==0)Atomic(p.ffn_counts[completed_tasks]).fetch_add(1,cuda::memory_order_release);
        __syncthreads();
    }
    grid.sync();
}

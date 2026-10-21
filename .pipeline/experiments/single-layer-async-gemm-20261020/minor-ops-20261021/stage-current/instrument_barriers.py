
from pathlib import Path
B=Path(__file__).resolve().parent
p=B/'stage_simple.cu';s=p.read_text();s=s.replace('int64_t* stage_cycles;', 'int64_t* stage_cycles; int64_t* barrier_ticks;')
a=s.index('__device__ void begin_stage(');z=s.index('__device__ int claim_tile(',a)
s=s[:a]+r"""
__device__ __forceinline__ int64_t barrier_ns(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ __forceinline__ void barrier_stamp(const Params&p,int st,int slot){
 if(threadIdx.x==0)p.barrier_ticks[(st*p.sms+blockIdx.x)*4+slot]=barrier_ns();
}
__device__ void begin_stage(const Params&p,cg::grid_group&grid,int stage){
 barrier_stamp(p,stage,0);
 if(grid.thread_rank()==0)p.queues[stage]=0;
 grid.sync();barrier_stamp(p,stage,1);
}
__device__ void end_stage(const Params&p,cg::grid_group&grid,int stage){
 barrier_stamp(p,stage,2);grid.sync();barrier_stamp(p,stage,3);
}
"""+s[z:]
a=s.index('__device__ void linear(');z=s.index('// All threads read',a) if '// All threads read' in s[a:] else s.index('__device__ inline int64_t diagnostic_ns',a)
s=s[:a]+s[a:z].replace('g.sync();','end_stage(p,g,st);')+s[z:]
needle='p.sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;';assert s.count(needle)==1
s=s.replace(needle,needle+'\n    auto barriers=at::empty({MAX_STAGES,p.sms,4},lo);p.barrier_ticks=barriers.data_ptr<int64_t>();')
s=s.replace('result["stage_ticks"]=ticks;', 'result["barrier_ticks"]=barriers;result["stage_ticks"]=ticks;');assert 'result["barrier_ticks"]' in s
p.write_text(s)
# Attention has a second, distinct combine stage (30/31); keep its identity.
p=B/'fa2_stage.cuh';s=p.read_text();assert s.count('    g.sync();')==2;s=s.replace('    g.sync();','    end_stage(p,g,st);',1);i=s.rindex('    g.sync();');s=s[:i]+s[i:].replace('    g.sync();','    end_stage(p,g,combine_stage);',1);p.write_text(s)
print('Instrumented CTA leaders at begin/end barriers; no altered scheduling or arithmetic')

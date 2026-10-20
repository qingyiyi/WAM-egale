"""Derivative with stage clocks only; excluded from ordinary latency samples."""
from pathlib import Path
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
B=Path('/root/cxz/ops/persistent-tile-20261019/evidence/async-gemm-20261020/performance-followup')
B.mkdir(exist_ok=False)
for name in ['generated','build','tmp','source-aligned']:(B/name).mkdir(exist_ok=True)
import shutil
for n in ['csrc/whole_layer.cu','csrc/gemm_pipeline.cuh','whole_layer.py','norm_codegen.py','build_whole_layer.py']:
 p=B/'source-aligned'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,p)
for p in R.glob('wam_whole_fa2_cuda*.so'):shutil.copyfile(p,B/'source-aligned'/p.name)
s=(R/'csrc/whole_layer.cu').read_text()
s=s.replace('int *queues,*visits,*stats;','int *queues,*visits,*stats; int64_t* stage_ticks;')
s=s.replace('__device__ inline int warp(){','__device__ inline int64_t diagnostic_ns(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}\n__device__ inline int warp(){')
s=s.replace('    grid.sync();\n}', '    grid.sync();if(grid.thread_rank()==0)p.stage_ticks[st*2]=diagnostic_ns();\n}',1)
def body(text,name):
 a=text.index('__device__ void '+name+'(');i=text.index('{',a);depth=1;j=i+1
 while depth:
  depth+=(text[j]=='{')-(text[j]=='}');j+=1
 return a,j
for name in ['linear','norm','point','insert_cache']:
 a,b=body(s,name);v=s[a:b];assert v.count('g.sync();')==1
 v=v.replace('g.sync();','g.sync();if(g.thread_rank()==0)p.stage_ticks[st*2+1]=diagnostic_ns();');s=s[:a]+v+s[b:]
s=s.replace('    layout(p);g.sync();int st=0,m=p.b*p.q,mc=p.b*p.c;','    for(int i=g.thread_rank();i<(MAX_STAGES+1)*2;i+=g.size())p.stage_ticks[i]=0;\n    layout(p);g.sync();if(g.thread_rank()==0)p.stage_ticks[MAX_STAGES*2]=diagnostic_ns();int st=0,m=p.b*p.q,mc=p.b*p.c;')
s=s.replace('    auto queues=at::empty({MAX_STAGES},io)', '    auto ticks=at::empty({MAX_STAGES+1,2},lo);p.stage_ticks=ticks.data_ptr<int64_t>();\n    auto queues=at::empty({MAX_STAGES},io)')
s=s.replace('    result["tile_stats"]=stats;', '    result["stage_ticks"]=ticks;result["tile_stats"]=stats;')
a=s.index('__global__ void whole_layer(');b=s.index('\nstd::map<std::string,Tensor> forward',a);v=s[a:b];assert v.count('    g.sync();')==1;v=v.replace('    g.sync();','    g.sync();if(g.thread_rank()==0)p.stage_ticks[MAX_STAGES*2+1]=diagnostic_ns();');s=s[:a]+v+s[b:]
s=s.replace('"fa2_stage.cuh"','"profile_fa2_stage.cuh"').replace('wam_whole_layer_v1','wam_whole_layer_profile')
(B/'profile.cu').write_text(s)
a=(R/'csrc/fa2_stage.cuh').read_text();first='    g.sync();';assert a.count(first)==2
a=a.replace(first,'    g.sync();if(g.thread_rank()==0)p.stage_ticks[st*2+1]=diagnostic_ns();',1)
pos=a.rindex(first);a=a[:pos]+a[pos:].replace(first,'    g.sync();if(g.thread_rank()==0)p.stage_ticks[combine_stage*2+1]=diagnostic_ns();',1)
(B/'profile_fa2_stage.cuh').write_text(a)

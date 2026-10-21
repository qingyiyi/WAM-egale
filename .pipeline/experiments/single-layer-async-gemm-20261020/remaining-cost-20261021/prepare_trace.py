"""Instrument a separate copy; preserve current mathematical functions and call order."""
from pathlib import Path
import hashlib,json,re
B=Path(__file__).resolve().parent;R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
s=(R/'csrc/whole_layer.cu').read_text();sha=hashlib.sha256(s.encode()).hexdigest();assert sha=='df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a'
s=s.replace('wam_whole_layer_v1','wam_cost_trace').replace('int *queues;','int *queues; int64_t* diag_ticks; int64_t* diag_barriers;')
helper='''__device__ inline int64_t diagnostic_ns() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ inline void diagnostic_sync(const Params& p, cg::grid_group& g, int st, int part) {
    const int offset=((st*p.sms+int(blockIdx.x))*3+part)*2;
    if(threadIdx.x==0)p.diag_barriers[offset]=diagnostic_ns();
    g.sync();
    if(threadIdx.x==0)p.diag_barriers[offset+1]=diagnostic_ns();
}
'''
pos=s.index('// 每个阶段有一个全局计数器');s=s[:pos]+helper+s[pos:]
s=s.replace('    grid.sync();\n}', '    diagnostic_sync(p,grid,stage,0);\n}',1)
def transform_function(text,name,slots):
 a=text.index('void '+name+'(');a=text.index('{',a);depth=1;z=a+1
 while depth:
  if text[z]=='{':depth+=1
  elif text[z]=='}':depth-=1
  z+=1
 body=text[a:z];assert body.count('g.sync();')==len(slots),(name,body.count('g.sync();'),slots)
 for slot in slots:body=body.replace('g.sync();',f'diagnostic_sync(p,g,st,{slot});',1)
 return text[:a]+body+text[z:]
for name,slots in [('linear',[1]),('context_linear',[1]),('action_ffn_linear',[1]),('action_ffn1_tail',[1,2]),('action_ffn2_parallel',[1,2]),('norm',[1,1]),('point',[1]),('insert_cache',[1])]:s=transform_function(s,name,slots)
f=(R/'csrc/fa2_stage.cuh').read_text();assert f.count('g.sync();')==2;f=f.replace('g.sync();','diagnostic_sync(p,g,st,1);',1).replace('g.sync();','diagnostic_sync(p,g,combine_stage,1);',1);(B/'fa2_stage.cuh').write_text(f)
a=s.index('__global__ void whole_layer');z=s.index('// Host只绑定',a);body=s[a:z]
pattern=r'(?m)^([ \t]*)(point|norm|linear|context_linear|action_ffn1_tail|action_ffn2_parallel|insert_cache|attention)\((.*?)\);'
count=0
def mark(m):
 global count
 if 'st++' not in m.group(0):return m.group(0)
 count+=1;indent=m.group(1);call=m.group(0).strip()
 return indent+'{ const int ds=st; if(g.thread_rank()==0)p.diag_ticks[2*ds]=diagnostic_ns();\n'+indent+'  '+call+'\n'+indent+'  if(g.thread_rank()==0)p.diag_ticks[2*ds+1]=diagnostic_ns(); }'
body=re.sub(pattern,mark,body,flags=re.S);assert count==31,count
body=body.replace('    layout(p, s);\n    g.sync();','    if(g.thread_rank()==0)p.diag_ticks[58]=diagnostic_ns();\n    layout(p, s);\n    diagnostic_sync(p,g,29,0);\n    if(g.thread_rank()==0)p.diag_ticks[59]=diagnostic_ns();')
# The final grid fence is separate from each stage's fence.
assert body.count('    g.sync();')==1;body=body.replace('    g.sync();','    diagnostic_sync(p,g,29,1);');s=s[:a]+body+s[z:]
s=s.replace('    p.sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;', '    p.sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;\n    auto dt=at::empty({30,2},lo),db=at::zeros({32,p.sms,3,2},lo);\n    p.diag_ticks=dt.data_ptr<int64_t>();p.diag_barriers=db.data_ptr<int64_t>();')
s=s.replace('    result["cache_info"] = info;','    result["diag_ticks"]=dt;result["diag_barriers"]=db;\n    result["cache_info"] = info;')
# Device marker for original stage intervals, with the same prior prefix.
pos=s.index('// Host只绑定');s=s[:pos]+'''__global__ void marker_kernel(int64_t* t,int i){if(threadIdx.x==0)t[i]=diagnostic_ns();}
void marker(Tensor t,int i){marker_kernel<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int64_t>(),i);}
'''+s[pos:];s=s.replace('m.def("forward", &wam_cost_trace::forward);','m.def("forward", &wam_cost_trace::forward);m.def("marker", &wam_cost_trace::marker);')
(B/'cost_trace.cu').write_text(s);(B/'TRACE-SOURCE-MANIFEST.json').write_text(json.dumps(dict(active_source_SHA256=sha,diagnostic_source_SHA256=hashlib.sha256(s.encode()).hexdigest(),marked_calls=count,active_source_modified=False,extra_zero_kernel='diagnostic barrier tape only; included in perturbation measurement'),indent=2)+'\n')

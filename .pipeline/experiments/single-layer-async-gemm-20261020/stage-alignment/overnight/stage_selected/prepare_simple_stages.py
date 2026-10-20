"""外部测量副本：当前简易版的独立阶段与整层内阶段，不改活动源码。"""
from pathlib import Path
import ast
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
# Explicit stage descriptions from the reviewed existing diagnostic preparation.
tree=ast.parse((B/'prepare.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='stages' for t in n.targets):stages=ast.literal_eval(n.value)
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in n.targets) and isinstance(n.value,ast.Constant):kernel=ast.literal_eval(n.value)
s=(R/'csrc/whole_layer.cu').read_text().replace('wam_whole_layer_v1','wam_stage_simple')
s=s.replace('int *queues;', 'int *queues; int64_t* stage_ticks; int64_t* stage_cycles;')
kernel=kernel.replace('    for(int i=g.thread_rank();i<MAX_STAGES*3;i+=g.size())p.stats[i]=0;\n','')
kernel=kernel.replace('layout(p)', 'layout(p,s)')
kernel=kernel.replace('if(g.thread_rank()==0)p.stage_ticks[2*Stage]=diagnostic_ns();',
    'if(g.thread_rank()==0){p.stage_ticks[2*Stage]=diagnostic_ns();p.stage_cycles[2*Stage]=clock64();}')
kernel=kernel.replace('if(g.thread_rank()==0)p.stage_ticks[2*Stage+1]=diagnostic_ns();',
    'if(g.thread_rank()==0){p.stage_cycles[2*Stage+1]=clock64();p.stage_ticks[2*Stage+1]=diagnostic_ns();}')
a=kernel.index('template<int Stage> __global__ void one_stage')
z=kernel.index('__global__ void whole_layer',a)
kernel=kernel[:a]+kernel[a:z].replace('execute_stage<Stage>(p,s,g);','timed_stage<Stage>(p,s,g);')+kernel[z:]
kernel=kernel.replace('STAGE_BRANCHES','\n'.join(f'if constexpr(Stage=={i}){{{stmt}}}' for i,(_,stmt,_) in enumerate(stages)))
kernel=kernel.replace('WHOLE_STAGES','\n'.join(f'timed_stage<{i}>(p,s,g);' for i in range(29)))
kernel=kernel.replace('layout(p)', 'layout(p,s)')
a=s.index('// All threads read');z=s.index('// Host只绑定',a);s=s[:a]+kernel+'\n'+s[z:]
s=s.replace('int update, const std::vector<double> &eps) {','int update, const std::vector<double> &eps, int selected_stage, const std::vector<Tensor>& preload, const std::vector<Tensor>& meta) {')
s=s.replace('    for (int i = 0; i < NB; ++i) {\n        const int rows', '    for (int i = 0; i < NB; ++i) {\n        if(!preload.empty()){b[i]=preload[i];p.a[i]=b[i].data_ptr();continue;}\n        const int rows')
s=s.replace('    p.slots = slots.data_ptr<int64_t>();','    if(!meta.empty()){slots=meta[0];info=meta[1];}\n    auto ticks=at::empty({30,2},lo),cycles=at::empty({30,2},lo);p.stage_ticks=ticks.data_ptr<int64_t>();p.stage_cycles=cycles.data_ptr<int64_t>();\n    p.slots = slots.data_ptr<int64_t>();')
a=s.index('    int active = 0;');z=s.index('    // debug只暴露',a)
dispatch='void* kernel=(void*)whole_layer;switch(selected_stage){'+''.join(f'case {i}:kernel=(void*)one_stage<{i}>;break;' for i in range(30))+'}\n'
s=s[:a]+'    int active=0;'+dispatch+'''
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,THREADS,sizeof(Shared)));
    TORCH_CHECK(active>0,"stage cannot reside");
    int blocks=p.sms*active;void* args[]={&p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel,dim3(blocks),dim3(THREADS),args,sizeof(Shared),at::cuda::getCurrentCUDAStream()));
'''+s[z:]
s=s.replace('cudaFuncGetAttributes(&attr, whole_layer)','cudaFuncGetAttributes(&attr, kernel)')
s=s.replace('    result["cache_info"] = info;','    result["stage_ticks"]=ticks;result["stage_cycles"]=cycles;\n    result["cache_info"] = info;')
s=s.replace('m.def("forward", &wam_stage_simple::forward);','m.def("forward", &wam_stage_simple::forward);m.def("marker", &wam_stage_simple::marker);')
(B/'stage_simple.cu').write_text(s)
print('Prepared current simplified mathematical stage replay')

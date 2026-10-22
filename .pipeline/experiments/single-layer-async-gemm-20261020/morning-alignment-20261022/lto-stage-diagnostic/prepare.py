"""当前活动单层的外部诊断副本；数学与原控制流保留。"""
from pathlib import Path
import hashlib,json,re
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
j=json.loads((B/'JOB.json').read_text());original=(R/'csrc/whole_layer.cu').read_text()
assert hashlib.sha256(original.encode()).hexdigest()==j['source_sha256']
s=original.replace('wam_whole_layer_v1','wam_resource_control')
s=s.replace('int *queues;', 'int *queues; int64_t *stage_ticks; int64_t *stage_cycles;')
clock='__device__ inline int64_t diagnostic_ns() {\n    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;\n}\n'
pos=s.index('// All threads read');s=s[:pos]+clock+s[pos:]
a=s.index('__global__ void whole_layer');z=s.index('// Host只绑定',a);body=s[a:z]
# 真实B14 affine/freq均存在；备选B2 FFN分支映射到同一个阶段编号。
lines=[x.strip() for x in body.splitlines() if 'st++' in x and x.strip().endswith(';')]
calls={};all_calls=[];stage=0
for call in lines:
 if call.startswith('linear') and 'bp(p, F1)' in call:sid=25
 elif call.startswith('linear') and 'bp(p, F2)' in call:sid=27
 else:sid=stage;calls[sid]=call;stage+=1
 all_calls.append((sid,call))
assert stage==29 and set(calls)==set(range(29)),calls

def marked(call,i):
 return '{ if(g.thread_rank()==0){p.stage_ticks[2*%d]=diagnostic_ns();p.stage_cycles[2*%d]=clock64();} '%(i,i)+call+' if(g.thread_rank()==0){p.stage_cycles[2*%d+1]=clock64();p.stage_ticks[2*%d+1]=diagnostic_ns();} }'%(i,i)
for sid,call in all_calls:
 assert body.count(call)==1,(sid,call)
 body=body.replace(call,marked(call,sid))
s=s[:a]+body+s[z:]
selected=[i for i in j['matrix']['stages'] if i!=10]
kernel='template<int Stage> __global__ void one_stage(const __grid_constant__ Params p) { extern __shared__ __align__(32) unsigned char memory[]; auto &s=*reinterpret_cast<Shared*>(memory); auto g=cg::this_grid(); int st=Stage,m=p.b*p.q,mc=p.b*p.c;\n'
for sid in selected:kernel+='if constexpr(Stage==%d) '%sid+marked(calls[sid],sid)+'\n'
kernel+='}\n'+'__global__ void marker_kernel(int64_t* times,int index){if(threadIdx.x==0) times[index]=diagnostic_ns();}\nvoid marker(Tensor times,int index){marker_kernel<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>(times.data_ptr<int64_t>(),index);}\n'
pos=s.index('// Host只绑定');s=s[:pos]+kernel+s[pos:]
old='bool debug) {'
assert s.count(old)==1
s=s.replace(old,'int selected_stage, const std::vector<Tensor>& preload, const std::vector<Tensor>& meta) {')
s=s.replace('    for (int i = 0; i < NB; ++i) {\n        const int rows','    for (int i = 0; i < NB; ++i) {\n        if(!preload.empty()){b[i]=preload[i];p.a[i]=b[i].data_ptr();continue;}\n        const int rows')
s=s.replace('    p.slots = slots.data_ptr<int64_t>();','    if(!meta.empty()){slots=meta[0];info=meta[1];}\n    auto ticks=at::zeros({30,2},lo),cycles=at::zeros({30,2},lo);\n    p.stage_ticks=ticks.data_ptr<int64_t>();p.stage_cycles=cycles.data_ptr<int64_t>();\n    p.slots = slots.data_ptr<int64_t>();')
a=s.index('    int active = 0;');z=s.index('    // debug只暴露',a)
launch='    int active=0; void* kernel=(void*)whole_layer; switch(selected_stage){'+''.join('case %d:kernel=(void*)one_stage<%d>;break;'%(i,i) for i in selected)+'}\n'
launch+='    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));\n    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,THREADS,sizeof(Shared)));\n    TORCH_CHECK(active>0,"no resident block");\n    int blocks=p.sms*active; void* args[]={&p};\n    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel,dim3(blocks),dim3(THREADS),args,sizeof(Shared),at::cuda::getCurrentCUDAStream()));\n    std::map<std::string, Tensor> result;\n'
s=s[:a]+launch+s[z:]
s=s.replace('cudaFuncGetAttributes(&attr, whole_layer)','cudaFuncGetAttributes(&attr, kernel)')
s=s.replace('result["cache_info"] = info;', 'result["cache_info"]=info;result["stage_ticks"]=ticks;result["stage_cycles"]=cycles;')
a=s.index('PYBIND11_MODULE(')
s=s[:a]+'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&wam_resource_control::forward);m.def("marker",&wam_resource_control::marker);}\n'
(B/'stage_resource.cu').write_text(s)
(B/'SOURCE-MANIFEST.json').write_text(json.dumps({'source':str(R/'csrc/whole_layer.cu'),'source_SHA256':hashlib.sha256(original.encode()).hexdigest(),'diagnostic_SHA256':hashlib.sha256(s.encode()).hexdigest(),'timing_boundary':'original stage callsite: includes internal synchronization and device entry/return, not pure computation','stage_calls':calls,'active_source_modified':False},indent=2)+'\n')
print('Prepared same-control-flow stage diagnostic',len(calls))

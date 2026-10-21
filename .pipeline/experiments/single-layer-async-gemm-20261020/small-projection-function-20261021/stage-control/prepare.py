"""Current mathematical functions, original full body, ten isolated launchers.

The external timestamps include the stage device-function entry/exit. This
diagnostic copy is never installed as the active mathematical implementation.
"""
from pathlib import Path
import hashlib
import json

B = Path(__file__).resolve().parent
R = Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
source = R / 'csrc/whole_layer.cu'
manifest=json.loads((B/'INPUT-SOURCE-MANIFEST.json').read_text())
for item in manifest['source_files']:
    assert hashlib.sha256((R/item['relative']).read_bytes()).hexdigest()==item['sha256'], item['relative']
original = source.read_text()
assert hashlib.sha256(original.encode()).hexdigest() == 'e91e09eb1cc569a90e96c0ffe9bdbaf38aa312bd86b492a0e9723b954ed46d49'
s = original.replace('wam_whole_layer_v1', 'wam_resource_control')
s = s.replace('int *queues;', 'int *queues; int64_t *stage_ticks; int64_t *stage_cycles;')
clock = '''__device__ inline int64_t diagnostic_ns() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
'''
pos = s.index('// All threads read')
s = s[:pos] + clock + s[pos:]


def marked(call, stage):
    return ('{\n if(g.thread_rank()==0){p.stage_ticks[2*' + str(stage) +
            ']=diagnostic_ns();p.stage_cycles[2*' + str(stage) + ']=clock64();}\n ' + call +
            '\n if(g.thread_rank()==0){p.stage_cycles[2*' + str(stage) + '+1]=clock64();'
            'p.stage_ticks[2*' + str(stage) + '+1]=diagnostic_ns();}\n}')


calls = {3: 'linear(p, s, g, st++, bp(p, M1), 3, bp(p, SQ0), m, D, D);', 4: 'linear(p, s, g, st++, bp(p, M1), 5, bp(p, SK0), m, D, D);', 5: 'linear(p, s, g, st++, bp(p, M1), 7, bp(p, SV), m, D, D);', 12: 'linear(p, s, g, st++, bp(p, SA), 9, bp(p, SO), m, D, D);', 15: 'linear(p, s, g, st++, bp(p, N2), 13, bp(p, CQ0), m, D, D);', 16: 'context_linear(p, s, g, st++, 15, bp(p, CK0), mc);', 17: 'context_linear(p, s, g, st++, 17, bp(p, CV), mc);', 21: 'linear(p, s, g, st++, bp(p, CA), 19, bp(p, CO), m, D, D);', 25: 'action_ffn1_tail(p, s, g, st++);', 27: 'action_ffn2_parallel(p, s, g, st++);'}
for stage, call in calls.items():
    assert s.count(call) == 1
    s = s.replace(call, marked(call, stage))

kernel = '''template<int Stage> __global__ void one_stage(const __grid_constant__ Params p) {
    extern __shared__ __align__(32) unsigned char memory[];
    auto &s=*reinterpret_cast<Shared*>(memory); auto g=cg::this_grid();
    int st=Stage, m=p.b*p.q, mc=p.b*p.c;
'''
for stage, call in calls.items():
    kernel += f'    if constexpr(Stage=={stage}) ' + marked(call, stage) + '\n'
kernel += '''}
__global__ void marker_kernel(int64_t* times, int index) {
    if(threadIdx.x==0) times[index]=diagnostic_ns();
}
void marker(Tensor times, int index) {
    marker_kernel<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>(times.data_ptr<int64_t>(),index);
}
'''
pos = s.index('// Host只绑定')
s = s[:pos] + kernel + s[pos:]
s = s.replace('int update, const std::vector<double> &eps) {',
              'int update, const std::vector<double> &eps, int selected_stage, '
              'const std::vector<Tensor>& preload, const std::vector<Tensor>& meta) {')
s = s.replace('    for (int i = 0; i < NB; ++i) {\n        const int rows',
              '    for (int i = 0; i < NB; ++i) {\n'
              '        if(!preload.empty()){b[i]=preload[i];p.a[i]=b[i].data_ptr();continue;}\n        const int rows')
s = s.replace('    p.slots = slots.data_ptr<int64_t>();',
              '    if(!meta.empty()){slots=meta[0];info=meta[1];}\n'
              '    auto ticks=at::empty({30,2},lo),cycles=at::empty({30,2},lo);\n'
              '    p.stage_ticks=ticks.data_ptr<int64_t>();p.stage_cycles=cycles.data_ptr<int64_t>();\n'
              '    p.slots = slots.data_ptr<int64_t>();')
start = s.index('    int active = 0;')
end = s.index('    // debug只暴露', start)
s = s[:start] + '''    int active=0; void* kernel=(void*)whole_layer;
    switch(selected_stage) {
''' + ''.join(f'    case {stage}:kernel=(void*)one_stage<{stage}>;break;\n' for stage in calls) + '''    }
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,THREADS,sizeof(Shared)));
    TORCH_CHECK(active>0,"no resident block");
    int blocks=p.sms*active; void* args[]={&p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel,dim3(blocks),dim3(THREADS),args,sizeof(Shared),at::cuda::getCurrentCUDAStream()));
''' + s[end:]
s = s.replace('cudaFuncGetAttributes(&attr, whole_layer)', 'cudaFuncGetAttributes(&attr, kernel)')
s = s.replace('result["cache_info"] = info;', 'result["cache_info"] = info; result["stage_ticks"]=ticks; result["stage_cycles"]=cycles;')
s = s.replace('m.def("forward", &wam_resource_control::forward);',
              'm.def("forward", &wam_resource_control::forward);m.def("marker", &wam_resource_control::marker);')
(B/'stage_resource.cu').write_text(s)
(B/'SOURCE-MANIFEST.json').write_text(json.dumps(dict(source=str(source),
    source_SHA256=hashlib.sha256(original.encode()).hexdigest(),
    diagnostic_SHA256=hashlib.sha256(s.encode()).hexdigest(),
    timing_boundary='call site includes device-function prologue/epilogue; different from prior inner-function timestamps',
    active_source_modified=False), indent=2) + '\n')

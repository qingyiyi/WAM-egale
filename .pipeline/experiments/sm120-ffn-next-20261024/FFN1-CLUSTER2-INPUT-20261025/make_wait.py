from pathlib import Path
import re,json
P=Path(__file__).resolve().parent
s=(P/'whole_layer_profile.cu').read_text()
s=s.replace('struct Params {','struct Params {\n    int64_t *ffn1_wait_trace;',1)
helper='''__device__ __forceinline__ uint64_t ffn_wait_clock() {
    uint64_t now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); return now;
}
__device__ __forceinline__ void ffn_timed_wait(uint64_t *bar,unsigned phase,uint64_t &sum) {
    uint64_t begin=ffn_wait_clock(); projection_wait(bar,phase); sum+=ffn_wait_clock()-begin;
}
'''
a=s.index('__device__ __forceinline__ void action_ffn1_produce(');b=s.index('// SM120 FFN2：',a)
v=s[a:b]
v=v.replace('    int waits = 0, releases = 0, finished = 0;', '    uint64_t wait_ns[6]={0,0,0,0,0,0};\n    const uint64_t loop_start=ffn_wait_clock();\n    int waits = 0, releases = 0, finished = 0;',1)
def repl(m):
    args=m[1];which=0 if args.startswith('&pipe.ready') else 1 if args.startswith('&pipe.peer_ready') else 2 if args.startswith('&pipe.empty') else 3
    return 'ffn_timed_wait('+args+',wait_ns['+str(which)+']);'
v=re.sub(r'projection_wait\(([^;]+)\);',repl,v)
v=v.replace('                    asm volatile("cp.async.bulk.shared::cluster', '                    const uint64_t issue_start=ffn_wait_clock();\n                    asm volatile("cp.async.bulk.shared::cluster',1)
v=v.replace('                           "r"(projection_smem(remote_ready)) : "memory");', '                           "r"(projection_smem(remote_ready)) : "memory");\n                    wait_ns[4]+=ffn_wait_clock()-issue_start;',1)
v=v.replace('    if (lane == 0 && p.ffn1_role_trace) {','''    wait_ns[5]=ffn_wait_clock()-loop_start;
    if (lane == 0) {
        auto dst=p.ffn1_wait_trace+(int(blockIdx.x)*8+w)*6;
        #pragma unroll
        for (int i=0;i<6;++i) dst[i]=wait_ns[i];
    }
    if (lane == 0 && p.ffn1_role_trace) {''',1)
s=s[:a]+helper+v+s[b:]
s=s.replace('    auto ffn1_role_trace =', '    auto ffn1_wait_trace=at::empty({at::cuda::getCurrentDeviceProperties()->multiProcessorCount,8,6},lo);\n    p.ffn1_wait_trace=ffn1_wait_trace.data_ptr<int64_t>();\n    auto ffn1_role_trace =',1)
s=s.replace('    result["stage_ticks"] = stage_ticks;', '    result["ffn1_wait_trace"] = ffn1_wait_trace;\n    result["stage_ticks"] = stage_ticks;',1)
(P/'whole_layer_wait.cu').write_text(s)
build=(P/'build_candidate.py').read_text().replace('wam_sm120_ffn1_cluster_input','wam_sm120_ffn1_cluster_input_wait').replace("parent/'whole_layer.cu'", "parent/'whole_layer_wait.cu'")
(P/'build_wait.py').write_text(build)
(P/'WAIT-PLAN.md').write_text('''# 仅等待位置观察器

共享A与其独立加载对照的完整层退化已定位FFN1；补A交接等待，与旧B观察分列。
在原100分钟/9000调用预算内补globaltimer观察器，不改数学/调度、不追加设计。
记录每warp READY、目的credit、EMPTY、result等待，以及bulk指令提交和整个角色循环。
记录是软件观察器等待区间，包含竞争与插桩扰动；不是copy纯带宽或独立可相加的全层成本。
三真实fixture，捕获27tensor/七段/状态核验，观察器与共享A正常整层计时分列，最多1000调用。
若观察器改变输出或资源/扰动过大，只保留定位限制，不据此强行解释根因。
''')
print('wait observer generated')

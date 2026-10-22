"""Generate measurement-only copy; production source/binary remain unchanged."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
R=Path('/rjs/cxz/论文探索/算子研究/wam_ops/optimized')
s=(R/'csrc/whole_layer.cu').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='32bb924ebeb36363942e7377fb36bff6748a0e282e2e49246711d157120c9c55'
def sub(a,b):
 global s
 assert s.count(a)==1,(a[:100],s.count(a))
 s=s.replace(a,b)
sub('    int64_t *ffn_trace;', '    int64_t *diag;\n    bool profile_on, ffn_only;\n    int64_t *ffn_trace;')
sub('template <int TN, int TK> struct GemmStorage {', '''// Diagnostic only: device-global nanosecond timestamps. No new synchronization.
__device__ __forceinline__ void diag_stamp(const Params &p, int at, int writer=0) {
    if (p.profile_on && threadIdx.x == writer) {
        int64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        p.diag[at] = t;
    }
}
constexpr int F1BASE=0, F2BASE=120*8, CTABASE=120*8+96*12;
template <int TN, int TK> struct GemmStorage {''')
# Tile record FFN1: start, before MMA, after MMA/barrier, after epilogue, after publish, CTA id.
sub('        const int rm = (tile / 56) * 128, cn = (tile % 56) * 256;', '''        diag_stamp(p, F1BASE+tile*8);
        if(p.profile_on && threadIdx.x==0) p.diag[F1BASE+tile*8+5]=blockIdx.x;
        const int rm = (tile / 56) * 128, cn = (tile % 56) * 256;''')
sub('        mma(D / 32, acc, ai, bi, acc);\n        __syncthreads();', '''        diag_stamp(p, F1BASE+tile*8+1);
        mma(D / 32, acc, ai, bi, acc);
        __syncthreads();
        diag_stamp(p, F1BASE+tile*8+2);''')
sub('        ffn_publish(p, 0, tile / 28, tile / 28 == 3 ? 36 : 28);', '''        diag_stamp(p, F1BASE+tile*8+3);
        ffn_publish(p, 0, tile / 28, tile / 28 == 3 ? 36 : 28);
        diag_stamp(p, F1BASE+tile*8+4);''')
sub('                                               const __nv_bfloat16 *w, int tile, int partition) {', '                                               const __nv_bfloat16 *w, int tile, int partition, const Params &p, int task) {')
sub('        mma(7168 / 64, acc, ai, bi, acc);', '''        diag_stamp(p,F2BASE+task*12+8,96);
        mma(7168 / 64, acc, ai, bi, acc);
        diag_stamp(p,F2BASE+task*12+9,96);''')
# FFN2 columns: dispatch, before partition, after partition, after wait, after commit, after publish, CTA, reserved, MMA begin/end, reserved.
sub('    action_ffn1_produce(p, s);\n    while (true) {', '''    diag_stamp(p,CTABASE+blockIdx.x*4);
    action_ffn1_produce(p, s);
    diag_stamp(p,CTABASE+blockIdx.x*4+1);
    while (true) {''')
sub('        const int group = task / 24, tile = (group / 2) * 24 + task % 24;', '''        diag_stamp(p,F2BASE+task*12);
        if(p.profile_on && threadIdx.x==0) p.diag[F2BASE+task*12+6]=blockIdx.x;
        const int group = task / 24, tile = (group / 2) * 24 + task % 24;''')
sub('        ffn2_partition(workspace, bp(p, ACT), (const B *)p.w[25], tile, group & 1);', '''        diag_stamp(p,F2BASE+task*12+1);
        ffn2_partition(workspace, bp(p, ACT), (const B *)p.w[25], tile, group & 1,p,task);
        diag_stamp(p,F2BASE+task*12+2);''')
sub('        ffn2_commit(p, workspace, tile, group & 1);\n        ffn_publish(p, 8, group, 24);', '''        diag_stamp(p,F2BASE+task*12+3);
        ffn2_commit(p, workspace, tile, group & 1);
        diag_stamp(p,F2BASE+task*12+4);
        ffn_publish(p, 8, group, 24);
        diag_stamp(p,F2BASE+task*12+5);''')
sub('    g.sync(); // FFN所有输出已完成；取代FFN1出口和FFN2两处grid同步。', '''    diag_stamp(p,CTABASE+blockIdx.x*4+2);
    g.sync(); // original FFN exit barrier
    diag_stamp(p,CTABASE+blockIdx.x*4+3);''')
sub('    // 缓存只整理本次可见slot；B14不会把候选的private KV写入公共历史。', '''    if (p.ffn_only) {
        g.sync(); // FFN counters initialized before independent replay
        action_ffn_pipeline(p,s,g);
        return;
    }
    // 缓存只整理本次可见slot；B14不会把候选的private KV写入公共历史。''')
sub('        const std::vector<Tensor> &context_kv) {','        const std::vector<Tensor> &context_kv, bool profile_on, const std::vector<Tensor> &ffn_inputs) {')
sub('    p.debug_modulation = debug;', '    p.debug_modulation = debug;\n    p.profile_on = profile_on;\n    p.ffn_only = !ffn_inputs.empty();')
sub('    // slot表和任务计数属于执行所需状态；逐tile visits/stats审计已移出核心。', '''    if(p.ffn_only) {
        b[M3]=ffn_inputs[0]; p.a[M3]=b[M3].data_ptr();
        b[R2]=ffn_inputs[1]; p.a[R2]=b[R2].data_ptr();
    }
    auto diag=at::zeros({profile_on ? CTABASE+108*4 : 0},lo);
    p.diag=profile_on ? diag.data_ptr<int64_t>() : nullptr;
    // slot表和任务计数属于执行所需状态；逐tile visits/stats审计已移出核心。''')
sub('    if (!debug)\n        return result;', '''    if(profile_on) result["ffn_diag"]=diag;
    if (!debug)
        return result;''')
sub('          py::arg("context_kv") = std::vector<at::Tensor>{});', '''          py::arg("context_kv") = std::vector<at::Tensor>{},
          py::arg("profile_on") = false, py::arg("ffn_inputs") = std::vector<at::Tensor>{});''')

# Tail tasks have a different shape but identical timestamp meaning.
sub('    const int rm = 128 + (tail % 3) * 32, cn = (52 + tail / 3) * 256;', '    const int tile = 108 + tail;\n    diag_stamp(p,F1BASE+tile*8);\n    if(p.profile_on && threadIdx.x==0) p.diag[F1BASE+tile*8+5]=blockIdx.x;\n    const int rm = 128 + (tail % 3) * 32, cn = (52 + tail / 3) * 256;')
sub('        mma(D / 32, acc, ai, bi, acc);\n        asm volatile("bar.sync 2, 128;" ::: "memory");', '        diag_stamp(p,F1BASE+tile*8+1);\n        mma(D / 32, acc, ai, bi, acc);\n        asm volatile("bar.sync 2, 128;" ::: "memory");\n        diag_stamp(p,F1BASE+tile*8+2);')
sub('    ffn_publish(p, 0, 3, 36);', '    diag_stamp(p,F1BASE+tile*8+3);\n    ffn_publish(p, 0, 3, 36);\n    diag_stamp(p,F1BASE+tile*8+4);')

(B/'profile.cu').write_text(s)
build=(R/'build_whole_layer.py').read_text()
build=build.replace('root=Path(__file__).resolve().parent',"root=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')")
build=build.replace("name='wam_whole_fa2_cuda',sources=[str(root/'csrc/whole_layer.cu')]", "name='wam_ffn_diagnostic_cuda',sources=[str(Path(__file__).parent/'profile.cu')]")
build="import sys\nsys.path.insert(0,'/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')\n"+build
(B/'build_profile.py').write_text(build)
(B/'SOURCE-MANIFEST.json').write_text(json.dumps({'production_sha256':'32bb924ebeb36363942e7377fb36bff6748a0e282e2e49246711d157120c9c55','diagnostic_sha256':hashlib.sha256(s.encode()).hexdigest(),'production_changed':False,'measurement_scope':'CTA leader milestones; FFN2 MMA warp96 timestamps; no new intra-kernel barriers in full-layer route; standalone entry has initial counter barrier','layout':{'F1':[120,8],'F2':[96,12],'CTA':[108,4]}},indent=2)+'\n')

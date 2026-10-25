from pathlib import Path
import json,re
P=Path(__file__).resolve().parent
R=P.parent.parent
O=R/'workspace/算子研究/wam_ops/optimized'
s=(P/'whole_layer.cu').read_text()
s=s.replace('struct Params {','struct Params {\n    float *ffn_numeric;',1)
s=s.replace('    auto ffn_trace =', '    auto ffn_numeric = at::empty({debug ? 7*2*224*3072 : 0}, fo);\n    p.ffn_numeric = debug ? ffn_numeric.data_ptr<float>() : nullptr;\n    auto ffn_trace =',1)
s=s.replace('    result["ffn_pipeline_timestamps"] =', '    result["ffn_numeric"] = ffn_numeric.view({7,2,224,3072});\n    result["ffn_pipeline_timestamps"] =',1)
b=s.index('__device__ __forceinline__ void ffn2_commit64(');e=s.index('using DeviceCounter',b)
x=s[b:e]
x=x.replace('            if (partition > 0) {', '            if(p.ffn_numeric) *reinterpret_cast<float4*>(p.ffn_numeric+((partition*2)*224+row)*3072+col)=z;\n            if (partition > 0) {',1)
x=x.replace('            if (!second || p.debug_modulation)', '            if(p.ffn_numeric) {const float2 nl=__bfloat1622float2(rounded.lo),nh=__bfloat1622float2(rounded.hi); *reinterpret_cast<float4*>(p.ffn_numeric+((partition*2+1)*224+row)*3072+col)={nl.x,nl.y,nh.x,nh.y};}\n            if (!second || p.debug_modulation)',1)
s=s[:b]+x+s[e:]
# Insert a profiling-only pointer into Params. New normal-only members do not
# affect this derivation; the normal source is never overwritten.
assert s.count('struct Params {') == 1
s=s.replace('struct Params {','struct Params {\n    int64_t *stage_ticks;',1)
helper='''
__device__ __forceinline__ void stage_tick(const Params &p, int stage, int edge) {
    if (threadIdx.x == 0) {
        unsigned long long now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        p.stage_ticks[(stage * gridDim.x + blockIdx.x) * 2 + edge] = now;
    }
}
'''
s=s.replace('__device__ __forceinline__ void action_ffn_stages(',helper+'\n__device__ __forceinline__ void action_ffn_stages(',1)
s=s.replace('    action_ffn1_produce(p, s);\n#if __CUDA_ARCH__ >= 1200\n    g.sync();', '    stage_tick(p, 15, 0);\n    action_ffn1_produce(p, s);\n#if __CUDA_ARCH__ >= 1200\n    g.sync();\n    stage_tick(p, 15, 1);', 1)
s=s.replace('    action_ffn2_splitk(p, s);\n    g.sync();', '    stage_tick(p, 16, 0);\n    action_ffn2_splitk(p, s);\n    g.sync();\n    stage_tick(p, 16, 1);', 1)
begin=s.index('__global__ void whole_layer(');end=s.index('// Host只绑定',begin)
head,body,tail=s[:begin],s[begin:end],s[end:]
regions=[
 ('layout','    layout(p, s);\n    g.sync();'),
 ('LN1+modulation','        norm_modulate(p, s, g, p.x, p.a[N1], bp(p, M1), p.eps[0], 0);'),
 ('self QKV','        self_qkv_projection_group(p, s, g, st, bp(p, M1));'),
 ('self Q RMS+RoPE','        norm_rope(p, s, g, bp(p, SQ0), bp(p, SQN), bp(p, SQR), 11, p.eps[3]);'),
 ('self K RMS+RoPE','        norm_rope(p, s, g, bp(p, SK0), bp(p, SKN), bp(p, SKR), 12, p.eps[4]);'),
 ('self attention','    attention(p, s, g, st++, false);'),
 ('self out','    linear(p, s, g, st++, bp(p, SA), 9, bp(p, SO), m, D, D);'),
 ('residual1','    point(p, s, g, st++, RESIDUAL, R1, -1, SO, 2, int64_t(m) * D);'),
 ('LN2','    if (p.affine)\n        norm(p, s, g, st++, bp(p, R1), p.a[N2], m, 1, 2, p.eps[1], false, false);'),
 ('cross Q','    linear(p, s, g, st++, bp(p, N2), 13, bp(p, CQ0), m, D, D);'),
 ('cross Q RMS','    norm(p, s, g, st++, bp(p, CQ0), p.a[CQ], m, 21, -1, p.eps[5], true, false);'),
 ('cross attention','    attention(p, s, g, st++, true);'),
 ('cross out','    linear(p, s, g, st++, bp(p, CA), 19, bp(p, CO), m, D, D);'),
 ('residual2','    point(p, s, g, st++, RESIDUAL, R2, R1, CO, -1, int64_t(m) * D);'),
 ('LN3+modulation','        norm_modulate(p, s, g, bp(p, R2), p.a[N3], bp(p, M3), p.eps[2], 3);'),
]
for i,(name,code) in enumerate(regions):
    assert body.count(code)==1,(i,name)
    body=body.replace(code,f'    stage_tick(p, {i}, 0);\n'+code+f'\n    stage_tick(p, {i}, 1);',1)
where=body.rfind('    g.sync();')
body=body[:where]+body[where:].replace('    g.sync();','    stage_tick(p, 17, 0);\n    g.sync();\n    stage_tick(p, 17, 1);',1)
tail=tail.replace('    void *launch_args[] = {&p};',
    '    auto stage_ticks = at::empty({18, blocks, 2}, x.options().dtype(at::kLong));\n'
    '    p.stage_ticks = stage_ticks.data_ptr<int64_t>();\n    void *launch_args[] = {&p};',1)
tail=tail.replace('    if (!debug)\n        return result;',
    '    result["stage_ticks"] = stage_ticks;\n    if (!debug)\n        return result;',1)
(P/'whole_layer_profile.cu').write_text(head+body+tail)
build=(P/'build_candidate.py').read_text()
build=re.sub(r"setup\(name='[^']+'", "setup(name='wam_sm120_ffn1_cluster_weight_profile'", build, count=1)
build=re.sub(r"name='wam_sm120_[^']+_cuda'", "name='wam_sm120_ffn1_cluster_weight_profile_cuda'", build, count=1)
build=build.replace("parent/'whole_layer.cu'", "parent/'whole_layer_profile.cu'")
assert "name='wam_sm120_ffn1_cluster_weight_profile_cuda'" in build
(P/'build_whole_profile.py').write_text(build)
(P/'STAGE-REGIONS.json').write_text(json.dumps([{'id':i,'name':n} for i,(n,_) in enumerate(regions)]+[{'id':15,'name':'FFN1+GELU'},{'id':16,'name':'FFN2+gate+residual3'},{'id':17,'name':'final exit'}],indent=2))
print('18 full-block regions; observer adds timestamps only; split barrier is in normal source')

bp=P/'build_whole_profile.py'
build=bp.read_text().replace('root=Path(__file__).resolve().parent',f'root=Path({str(O)!r})')
build=build.replace("str(root/'csrc/whole_layer_profile.cu')",repr(str(P/'whole_layer_profile.cu')))
bp.write_text(build)

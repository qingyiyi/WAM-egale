from pathlib import Path
import json,shutil,sys,difflib,hashlib
E=Path(__file__).resolve().parent;name=sys.argv[1];assert name in ['L1','L13']
src=E/'worktree/算子研究/wam_ops/optimized';dst=E/f'variants/{name}/算子研究/wam_ops/optimized';shutil.copytree(src,dst)
p=dst/'csrc/whole_layer.cu';old=p.read_text();s=old
s=s.replace('// Norm tile为完整3072维行。','__device__ __forceinline__ float4 modulation4(const Params &p, int64_t element, int param);\n\n// Norm tile为完整3072维行。',1)
s=s.replace('float eps, bool rms, bool output_float, int row) {','float eps, bool rms, bool output_float, int row,\n                          const Params *mod = nullptr, int mod_param = 0, B *mod_out = nullptr) {',1)
a=s.index('        if (output_float) {',s.index('void norm_tile'))
s=s[:a]+'''        if (mod) {
            // Row's FP32 Welford output is ready in shared. Keep the exact
            // FP32 add/multiply order and final BF16 conversion of MODULATE.
            for (int v = int(threadIdx.x); v < D / 4; v += THREADS) {
                const int64_t element = int64_t(row) * D + v * 4;
                const float4 z = reinterpret_cast<const float4 *>(s.y)[v];
                if (mod->debug_modulation)
                    reinterpret_cast<float4 *>((float *)y + row * D)[v] = z;
                const float4 shift = modulation4(*mod, element, mod_param);
                const float4 scale = modulation4(*mod, element, mod_param + 1);
                float out[4];
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    out[c] = __fadd_rn(__fmul_rn(((const float *)&z)[c],
                               __fadd_rn(1.f, ((const float *)&scale)[c])),
                               ((const float *)&shift)[c]);
                auto *dest = reinterpret_cast<__nv_bfloat162 *>(mod_out + element);
                dest[0] = __floats2bfloat162_rn(out[0], out[1]);
                dest[1] = __floats2bfloat162_rn(out[2], out[3]);
            }
        } else '''+s[a+8:]
pos=s.index('// All threads read the same immutable launch descriptors.')
s=s[:pos]+'''// One owner computes the complete LN row then its modulation. Only the
// FP32 global intermediate and the intervening grid barrier disappear.
__device__ void norm_modulate(const Params &p, Shared &s, cg::grid_group &g,
                               const B *x, void *audit, B *out, float eps, int param) {
    for (int row = int(blockIdx.x); row < p.b * p.q; row += int(gridDim.x))
        norm_tile(s, x, audit, nullptr, nullptr, eps, false, true, row, &p, param, out);
    g.sync();
}

'''+s[pos:]
for which in ([1,3] if name=='L13' else [1]):
 x='p.x' if which==1 else 'bp(p, R2)';idx=0 if which==1 else 2;param=0 if which==1 else 3
 prev=f'''    norm(p, s, g, st++, {x}, p.a[N{which}], m, -1, -1, p.eps[{idx}], false, true);
    point(p, s, g, st++, MODULATE, M{which}, N{which}, 0, {param}, int64_t(m) * D);'''
 new=f'''    if (p.candidate && m == 224) {{
        norm_modulate(p, s, g, {x}, p.a[N{which}], bp(p, M{which}), p.eps[{idx}], {param});
        st += 2; // Preserve all later queue IDs.
    }} else {{
{prev}
    }}'''
 assert prev in s;s=s.replace(prev,new)
needle='''        else if (i == F1 && p.candidate && p.b * p.q == 224 && !debug)'''
condition='(i == N1 || i == N3)' if name=='L13' else 'i == N1'
s=s.replace(needle,f'''        else if ({condition} && p.candidate && p.b * p.q == 224 && !debug)
            b[i] = at::empty({{0}}, fo);
'''+needle)
p.write_text(s);d=E/f'T3-{name}-build';d.mkdir();(d/'CHANGE.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='A/whole_layer.cu',tofile=name+'/whole_layer.cu')))
R='/root/cxz/ops/local-warp-pipeline-20261023';PY='/root/cxz/envs/va/bin/python'
j=dict(goal=f'Independent anchor-A {name} norm modulation fusion build',gpu=False,cap_seconds=600,command=[PY,f'{R}/variants/{name}/算子研究/wam_ops/optimized/build_whole_layer.py','build_ext','--build-lib',f'{R}/{d.name}','--build-temp',f'{R}/{d.name}/build/obj'],source_sha256=hashlib.sha256(s.encode()).hexdigest());(d/'JOB.json').write_text(json.dumps(j,indent=2)+'\n');print(name,j['source_sha256'])

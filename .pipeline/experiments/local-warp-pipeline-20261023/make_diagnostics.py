from pathlib import Path
import shutil,json,hashlib
E=Path(__file__).resolve().parent;R='/root/cxz/ops/local-warp-pipeline-20261023';PY='/root/cxz/envs/va/bin/python'
for label in ['A','BC']:
 src=(E/'worktree' if label=='A' else E/'variants/BC')/'算子研究/wam_ops/optimized'
 dst=E/f'variants/{label}_diag/算子研究/wam_ops/optimized';shutil.copytree(src,dst)
 p=dst/'csrc/whole_layer.cu';s=p.read_text()
 s=s.replace('    int sms;','    B *partial_debug;\n    int64_t *local_trace;\n    int sms;')
 s=s.replace('            if (!second || p.debug_modulation)\n', '            if (!second && p.debug_modulation)\n                *reinterpret_cast<B4 *>(p.partial_debug + row * D + col) = rounded;\n            if (!second || p.debug_modulation)\n')
 # Full per-task trace, kept exclusively in this diagnostic DSO.
 if label=='BC':
  s=s.replace('        if (!p.ffn_prepare_overlap) {','''        if (p.debug_modulation && threadIdx.x == 0) p.local_trace[task*8] = ffn_timestamp();
        if (!p.ffn_prepare_overlap) {''')
  s=s.replace('            ffn2_prepare(p, prepared, tile, group & 1);\n            __syncthreads();','''            ffn2_prepare(p, prepared, tile, group & 1);
            if (p.debug_modulation && threadIdx.x == 0) p.local_trace[task*8+1] = ffn_timestamp();
            __syncthreads();''')
  s=s.replace('''        if (compute())
            ffn2_partition(workspace, bp(p, ACT), (const B *)p.w[25], tile, group & 1);
        else if (p.ffn_prepare_overlap)
            ffn2_prepare(p, prepared, tile, group & 1);''','''        if (compute()) {
            if (p.debug_modulation && threadIdx.x == 96) p.local_trace[task*8+2] = ffn_timestamp();
            ffn2_partition(workspace, bp(p, ACT), (const B *)p.w[25], tile, group & 1);
            if (p.debug_modulation && threadIdx.x == 96) p.local_trace[task*8+3] = ffn_timestamp();
        } else if (p.ffn_prepare_overlap) {
            ffn2_prepare(p, prepared, tile, group & 1);
            if (p.debug_modulation && threadIdx.x == 0) p.local_trace[task*8+1] = ffn_timestamp();
        }''')
  s=s.replace('''        if (group & 1) {
            if (threadIdx.x == 0)''','''        if (p.debug_modulation && threadIdx.x == 0) p.local_trace[task*8+4] = ffn_timestamp();
        if (group & 1) {
            if (threadIdx.x == 0)''')
  s=s.replace('        ffn2_commit(p, workspace, tile, group & 1, prepared);','''        if (p.debug_modulation && threadIdx.x == 0) p.local_trace[task*8+5] = ffn_timestamp();
        ffn2_commit(p, workspace, tile, group & 1, prepared);
        if (p.debug_modulation && threadIdx.x == 0) p.local_trace[task*8+6] = ffn_timestamp();''')
  s=s.replace('        ffn_publish(p, 8, group, 24);','''        ffn_publish(p, 8, group, 24);
        if (p.debug_modulation && threadIdx.x == 0) p.local_trace[task*8+7] = ffn_timestamp();''')
 s=s.replace('    p.slots = slots.data_ptr<int64_t>();','''    auto partial_debug = at::empty({debug ? p.b * p.q : 0, D}, bo);
    auto local_trace = at::zeros({debug ? 96 : 0,8}, lo);
    p.partial_debug = debug ? (B *)partial_debug.data_ptr() : nullptr;
    p.local_trace = debug ? local_trace.data_ptr<int64_t>() : nullptr;
    p.slots = slots.data_ptr<int64_t>();''')
 s=s.replace('    result["ffn_pipeline_state"] =','''    result["first_partial"] = partial_debug;
    result["local_trace"] = local_trace;
    result["ffn_pipeline_state"] =''')
 p.write_text(s);d=E/f'T2-{label}-diag-build';d.mkdir()
 j=dict(goal='Independent diagnostic copy: first BF16 partial and software local intervals only, not adoption timing',gpu=False,cap_seconds=600,command=[PY,f'{R}/variants/{label}_diag/算子研究/wam_ops/optimized/build_whole_layer.py','build_ext','--build-lib',f'{R}/{d.name}','--build-temp',f'{R}/{d.name}/build/obj'],source_sha256=hashlib.sha256(s.encode()).hexdigest());(d/'JOB.json').write_text(json.dumps(j,indent=2)+'\n')

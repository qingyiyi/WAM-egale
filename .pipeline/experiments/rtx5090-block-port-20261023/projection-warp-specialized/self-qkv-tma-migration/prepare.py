"""Prepare one Self-QKV backend change; retain selected A as independent code/DSO."""
from pathlib import Path
import difflib
import json
import shutil

P = Path(__file__).resolve().parent
S = P.parent
R = S.parent
O = R / 'workspace/算子研究/wam_ops/optimized'
files = ['csrc/whole_layer.cu', 'csrc/cutlass_gemm.cuh', 'csrc/gemm_pipeline.cuh',
         'whole_layer.py', 'build_whole_layer.py',
         'wam_whole_warp_specialized_cuda.cpython-312-x86_64-linux-gnu.so']
checks = {}
for name in files:
    source = O / name
    checks[name] = source.read_bytes() == (S / 'SELECTED-source' / name).read_bytes()
    dest = P / 'A-source' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
assert all(checks.values()), checks
shutil.copy2(O / files[-1], P / 'A.so')
(P / 'REFERENCE-BYTE-CHECK.json').write_text(json.dumps(checks, indent=2))

original = (O / 'csrc/whole_layer.cu').read_text()
s = original
replacements = [
    ('CUtensorMap projection_maps[3][2]', 'CUtensorMap projection_maps[6][2]'),
    ('debug ? 3 * 96 * 4 : 0', 'debug ? 6 * 96 * 4 : 0'),
    ('debug ? 3 * 96 * 8 * 3 : 0', 'debug ? 6 * 96 * 8 * 3 : 0'),
    ('const int input[3] = {SA, N2, CA}, weight[3] = {9, 13, 19};',
     'const int input[6] = {SA, N2, CA, M1, M1, M1}, weight[6] = {9, 13, 19, 3, 5, 7};'),
    ('for (int projection = 0; projection < 3; ++projection)',
     'for (int projection = 0; projection < 6; ++projection)'),
    ('projection_trace.view({3, 96, 4})', 'projection_trace.view({6, 96, 4})'),
    ('projection_role_trace.view({3,96,8,3})', 'projection_role_trace.view({6,96,8,3})'),
]
for before, after in replacements:
    assert s.count(before) == 1, before
    s = s.replace(before, after, 1)
before = '''    constexpr int per_projection = 96;
    begin_stage(p, g, st);'''
after = '''    constexpr int per_projection = 96;
#if __CUDA_ARCH__ >= 1200
    // 发布前一阶段M1写入，后续TMA async proxy读取同一global输入。
    asm volatile("fence.proxy.async.global;" ::: "memory");
#endif
    begin_stage(p, g, st);'''
assert s.count(before) == 1
s = s.replace(before, after, 1)
before = '''        self_qkv8_tile(s.tile.cutlass_gemm.self_qkv8, a, (const B *)p.w[wi],
                      (const B *)p.w[wi + 1], out, task % per_projection);'''
after = '''#if __CUDA_ARCH__ >= 1200
        // Q/K/V沿用288项队列，各自权重、输出和数值分段保持独立。
        projection_tma_tile(p, s.tile.projection_tma, 3 + projection,
                            (const B *)p.w[wi + 1], out, task % per_projection);
#else
        self_qkv8_tile(s.tile.cutlass_gemm.self_qkv8, a, (const B *)p.w[wi],
                      (const B *)p.w[wi + 1], out, task % per_projection);
#endif'''
assert s.count(before) == 1
s = s.replace(before, after, 1)
s = s.replace('// CTA共8个warp，职责按阶段变化：小投影/Context/FFN2由warp3–6做MMA；',
              '// CTA共8个warp：SM120 action投影W0做TMA、W1–3做epilogue、W4–7做MMA；')
(P / 'whole_layer.cu').write_text(s)
(P / 'CHANGE.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True), s.splitlines(True), fromfile='A/whole_layer.cu',
    tofile='Self-QKV-TMA/whole_layer.cu')))

build = (S / 'build_candidate.py').read_text().replace('wam_whole_warp_specialized', 'wam_whole_qkv_tma')
(P / 'build_candidate.py').write_text(build)
profile = (S / 'make_profile.py').read_text()
profile = profile.replace("R=Path(__file__).resolve().parent.parent\nP=R/'projection-warp-specialized'",
                          'P=Path(__file__).resolve().parent\nR=P.parent.parent')
profile = profile.replace('wam_whole_warp_specialized', 'wam_whole_qkv_tma')
(P / 'make_profile.py').write_text(profile)
exec(compile(profile, str(P / 'make_profile.py'), 'exec'), {'__file__': str(P / 'make_profile.py')})

measure = (S / 'measure.py').read_text()
measure = measure.replace("default='wam_whole_splitk_cuda'", "default='wam_whole_warp_specialized_cuda'")
measure = measure.replace("P.parent/'eager-stage-comparison/build/guarded'", "P.parent.parent/'eager-stage-comparison/build/guarded'")
measure = measure.replace("        old_tma = load_extension(P/'TMA-normal.so', 'wam_whole_tma_continuous_cuda')\n", '')
measure = measure.replace(
    "'Own':lambda:own()['output'],'A':lambda:own(referenceA)['output'],\n                           'TMA':lambda:own(old_tma)['output']}",
    "'Own':lambda:own()['output'],'A':lambda:own(referenceA)['output']}")
measure = measure.replace('for _ in range(3)]', 'for _ in range(6)]')
measure = measure.replace("                row = dict(projection_tma_counts=tma_counts,", """                role_counts = debug['projection_role_counts'].tolist()
                if role_counts != [[([[48,48,96]]+[[10,10,10]]*3+[[48,48,10]]*4) for _ in range(96)] for _ in range(6)]:
                    raise RuntimeError('six projections warp-role count gate failed')
                row = dict(projection_role_count_pass=True, projection_tma_counts=tma_counts,""")
(P / 'measure.py').write_text(measure)

public = (S / 'precision_public.py').read_text().replace('for _ in range(3)]', 'for _ in range(6)]')
(P / 'precision_public.py').write_text(public)
gate = public.replace('from tools.time_whole_layer import capture, device_tree, resource',
                      'from tools.time_whole_layer import capture, device_tree, resource, load_extension')
gate = gate.replace('        extension = whole_api.extension()',
                    '        extension = load_extension(args.binary, args.module_name)\n        whole_api.extension = lambda: extension')
gate = gate.replace("loader='actual whole_api.extension; not replaced'",
                    "loader='independent candidate DSO through same WholeLayerBlock API; explicit loader injection'")
gate = gate.replace('The public source must already select the declared module/binary.',
                    'The frozen public source remains unchanged; inject one independent candidate module.')
gate = gate.replace('does not patch the loader or bypass the public cached-context implementation.',
                    'uses explicit loader injection and the real public cached-context implementation.')
(P / 'precision_candidate.py').write_text(gate)

definitions = json.loads((S / 'STAGE-DEFINITIONS.json').read_text())
definitions[2]['warps'] = 'W0 TMA；W1–3 epilogue；W4–7 MMA（每warp32×64）'
definitions[2]['work'] = 'Q/K/V共享288输出任务；各96×64×128；每tile十段K=9×320+192（48×K64）'
definitions[2]['grid'] = '2；tile进入/退出3次CTA同步，热循环mbarrier'
for i in (6, 9, 12):
    definitions[i]['warps'] = 'W0 TMA；W1–3 epilogue；W4–7 MMA（每warp32×64）'
(P / 'STAGE-DEFINITIONS.json').write_text(json.dumps(definitions, ensure_ascii=False, indent=2))
report = (S / 'report.py').read_text().replace("('Torch', 'Guarded', 'A', 'TMA', 'Own')", "('Torch', 'Guarded', 'A', 'Own')")
report = report.replace('冻结SPLITK-M64-FFN2', '冻结WARP-SPECIALIZED-4MMA（Self QKV旧八MMA）')
report = report.replace('五臂Torch／Guarded／A／原TMA／候选', '四臂Torch／Guarded／A／候选')
report = report.replace('12轮五臂预热', '12轮四臂预热')
(P / 'report.py').write_text(report)
print('PREPARED: one backend change, independent A and candidate; no GPU/no SHA')

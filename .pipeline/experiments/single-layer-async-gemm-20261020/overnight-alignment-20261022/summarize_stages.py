"""Summarize frozen diagnostic output; never convert marker gaps to pure barriers."""
from pathlib import Path
import csv
import json
import statistics as st

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'stage-control-final-r2'
rows = []
controls = []
for proc in (1, 2):
    data = json.loads((BASE / f'RESULT-{proc}.json').read_text())
    assert data['status'] == 'COMPLETE_CURRENT_SOURCE_PREFIX_DIAGNOSTIC'
    for case in data['rows']:
        assert case['cache_check']['pass_all'] and case['inputs_unchanged'] and case['RNG_unchanged']
        for stage in case['stages']:
            sid = stage['stage']
            embedded = st.median((r['ticks_ns'][str(sid)][1] - r['ticks_ns'][str(sid)][0]) / 1000 for r in case['embedded'])
            for cond in stage['conditions']:
                assert all(x['bitwise_pass'] for x in cond['precision'].values())
                for checks in cond['prefix_equal'].values():
                    assert all(checks.values())
                values = {a: st.median(x['us'] for x in cond['samples'] if x['arm'] == a)
                          for a in sorted({x['arm'] for x in cond['samples']})}
                rows.append(dict(process=proc, KV=case['case'], stage=sid, name=stage['name'],
                    condition=cond['condition'], native_us=values['A_native'],
                    own_independent_full_resource_us=values['B_owned'],
                    own_embedded_us=embedded, guarded_us=values.get('C_guarded', ''),
                    shared_us=values.get('D_shared', ''), input_bitwise_pass=True,
                    output_bitwise_pass=True, resources=json.dumps(cond['owned_resources'])))
        timings = case['instrumentation_control']
        med = {a: st.median(x['ms'] for x in timings if x['arm'] == a) for a in ('active', 'diagnostic')}
        controls.append(dict(process=proc, KV=case['case'], **med,
                             overhead_percent=100*(med['diagnostic']/med['active']-1)))
with (BASE / 'PHASES.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(BASE / 'INSTRUMENTATION-CONTROL.json').write_text(json.dumps(controls, indent=2)+'\n')
lines = ['# 最终单层逐阶段测量', '',
'OBSERVED：两个独立进程、3个真实B14 KV输入、28阶段。共有输入/输出/中间量、有效KV及RNG有限逐位核验通过；每进程3756次完整调用和125772次stage调用。原始JSON保留。', '',
'下表均为同一原版前缀的输入，两进程中位数的均值，单位µs。self_attention对照取guarded强路径，其余取原版。自有独立stage沿用整层108CTA/100096B shared资源，不是自由资源最优standalone。', '',
'|KV|阶段|原版/强对照|自有独立|自有整层嵌入|', '|---|---|---:|---:|---:|']
for case in ('short','middle','long'):
    for name in dict.fromkeys(r['name'] for r in rows):
        subset = [r for r in rows if r['KV']==case and r['name']==name and r['condition']=='same_real_native_prefix']
        native = st.mean(float(r['guarded_us'] or r['native_us']) for r in subset)
        independent = st.mean(r['own_independent_full_resource_us'] for r in subset)
        embedded = st.mean(r['own_embedded_us'] for r in subset)
        lines.append(f'|{case}|{name}|{native:.3f}|{independent:.3f}|{embedded:.3f}|')
lines += ['', '计时边界：独立stage为外部GPU marker区间，包含kernel边界；整层为阶段内部marker，包含原stage内部同步/分派/device调用。两者相减不是纯全局同步等待。各阶段中位数不能直接相加代替完整层计时。', '',
f"插桩完整层相对活动二进制开销范围：{min(x['overhead_percent'] for x in controls):.3f}%–{max(x['overhead_percent'] for x in controls):.3f}%。插桩寄存器/local资源与非插桩略有不同，正式整层性能只引用final-whole-comparison。", '',
'HYPOTHESIS：剩余差距含阶段几何/固定资源/函数组织；Cross-K RMS与GELU的独立stage也明显偏慢，不支持将全部差距归因于跨阶段全局等待。resource-envelope-control的资源干预支持“固定整层资源限制是部分原因”，但降低shared或增加CTA并非普遍改善。', '',
'UNKNOWN：当前版本动态local流量/实际achieved occupancy/SM stall归因等待本机NCU；静态localSizeBytes不能换算成实测spill字节。', '',
'RECOMMENDATION：先以当前最终版本NCU检验内存和函数边界，再决定下一轮最小改动。保留单层限制和强B14基线，不用旧慢dense路径宣布完全对齐；不开始跨阶段ready-DAG。', '']
(BASE / 'RESULT.md').write_text('\n'.join(lines))
print(json.dumps(controls, indent=2))
print('rows', len(rows))

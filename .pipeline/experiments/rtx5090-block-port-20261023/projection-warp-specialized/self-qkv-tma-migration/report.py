"""Report only this candidate's two completed real-fixture measurements.

The normal whole-layer measurements determine retention. Instrumented regions
are diagnostics: their sum is not substituted for normal block time.
"""
import argparse
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
CASES = ('short', 'middle', 'long')
ARMS = ('Torch', 'Guarded', 'A', 'Own')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant')
    parser.add_argument('--retained', default='GRAIN-M64-FFN2（本候选待判定）')
    args = parser.parse_args()
    out = ROOT / 'reports' / args.variant
    out.mkdir(parents=True, exist_ok=True)
    processes = [json.loads((ROOT / f'results/measure{i}-data/RESULT.json').read_text())
                 for i in (1, 2)]
    if any(p['status'] != 'COMPLETE_EXACT_EAGER' for p in processes):
        raise ValueError('two complete exact eager runs are required; retain partial records')
    for process in processes:
        if process['loaded_modules']['A']['name'] == process['loaded_modules']['candidate']['name']:
            raise ValueError('reference and candidate extension identities collide')
    definitions = json.loads((ROOT / 'STAGE-DEFINITIONS.json').read_text())
    if len(definitions) != 17:
        raise ValueError('expected the complete 17-region block table')
    rows = {case: [next(r for r in p['rows'] if r['case'] == case) for p in processes]
            for case in CASES}
    whole = {case: {arm: st.mean(r['median_ms'][arm] for r in rows[case]) for arm in ARMS}
             for case in CASES}
    stages = {}
    for case in CASES:
        row = rows[case][0]
        own = [st.median(x[i] for x in row['regions_ms']) for i in range(17)]
        torch = {str(x['region']): x['median_ms'] for x in row['torch_stages']}
        normal_profile = {label: st.median(x['event_ms'] for x in row['instrumentation_samples']
                                          if x['arm'] == label) for label in ('normal', 'profile')}
        stages[case] = dict(own_ms=own, torch_ms=torch,
                            profile_overhead_pct=100 * (normal_profile['profile'] / normal_profile['normal'] - 1),
                            observed_span_sum_ms=sum(own),
                            span_share_pct=[100 * x / sum(own) for x in own])

    md = f'''# 当前完整block阶段表

评估变体：**{args.variant}**。当前实际保留版本：**{args.retained}**。

本表全部来自本候选本轮测量，无历史值混填。A为冻结WARP-SPECIALIZED-4MMA（Self QKV旧八MMA）独立二进制。四臂Torch／Guarded／A／候选均eager、无CUDA Graph；同三份真实B14/Q16/block0输入，KV288／1152／2160，共同使用等价预计算Cross K/V。

加速比=原Torch时间÷候选时间，大于1表示候选更快。自有阶段是整层kernel内globaltimer区间（含阶段原有同步），Torch阶段是相同数学区域独立eager CUDA事件。二者是瓶颈诊断口径，不是纯GEMM吞吐对照。FFN局部重叠必须合并计时，不相加推断整层性能。

|阶段|实际tile/工作量|CTA内warp分工|grid同步|短KV比值|中KV比值|长KV比值|自有区间占比：短/中/长|
|---|---|---|---|---:|---:|---:|---|
'''
    for item in definitions:
        index = item['region']
        ratios = [stages[c]['torch_ms'].get(str(index), None) for c in CASES]
        speed = ['N/A' if numerator is None else f'{numerator / stages[c]["own_ms"][index]:.3f}×'
                 for c, numerator in zip(CASES, ratios)]
        shares = '/'.join(f'{stages[c]["span_share_pct"][index]:.1f}%' for c in CASES)
        md += (f'|{index + 1}. {item["name"]}|{item["work"]}|{item["warps"]}|{item["grid"]}|'
               + '|'.join(speed) + f'|{shares}|\n')

    md += '''
布局和退出没有Torch同义独立区域，因此为N/A。阶段占比分母是插桩区间总和，不能冒充无插桩整层精确可加分解。预计算Cross K/V是共同warm基线，冷构建单列。

## 无插桩完整层

|KV|Torch/候选|Torch/A|Torch/Guarded|A/候选|候选相对A延迟变化|Guarded/候选|
|---|---:|---:|---:|---:|---:|---:|
'''
    for case in CASES:
        w = whole[case]
        md += (f'|{case}|{w["Torch"]/w["Own"]:.4f}×|{w["Torch"]/w["A"]:.4f}×|'
               f'{w["Torch"]/w["Guarded"]:.4f}×|{w["A"]/w["Own"]:.4f}×|'
               f'{100*(w["Own"]/w["A"]-1):+.3f}%|{w["Guarded"]/w["Own"]:.4f}×|\n')
    md += '\n两个独立进程，每档12轮四臂预热、16轮交错×5完整block；上表为两进程各自中位数的均值。原始轮次保留。\n\n'
    md += '|进程|KV|A毫秒|候选毫秒|候选相对A延迟变化|\n|---|---|---:|---:|---:|\n'
    for process in processes:
        for row in process['rows']:
            w = row['median_ms']
            md += f'|{process["process"]}|{row["case"]}|{w["A"]:.6f}|{w["Own"]:.6f}|{100*(w["Own"]/w["A"]-1):+.3f}%|\n'
    md += '\n## 诊断边界\n\n'
    md += '候选插桩相对正常整层扰动（短/中/长）：' + '/'.join(f'{stages[c]["profile_overhead_pct"]:+.2f}%' for c in CASES) + '。\n\n'
    md += '冷Cross K/V构建原Torch耗时（ms，短/中/长）：' + '/'.join(f'{stages[c]["torch_ms"]["cold_crossKV"]:.6f}' for c in CASES) + '；不计warm完整层。\n\n'
    md += '原Torch调制表独立耗时（ms）：' + '/'.join(f'{stages[c]["torch_ms"]["table"]:.6f}' for c in CASES) + '；自有按需融合，不任意摊入其他阶段。\n\n'
    md += 'Guarded仅更换Self Attention，其余Torch阶段共同复用。Self Attention的Guarded/候选为：' + '/'.join(f'{stages[c]["torch_ms"]["guarded_self"]/stages[c]["own_ms"][5]:.3f}×' for c in CASES) + '。\n\n'
    md += '精度与输入/KV/RNG通过范围仅为这三份真实单层输入；不代表30层、在线接受路径或机器人质量。四科学权限不改变。\n'
    summary = dict(status='COMPLETE_EXACT_EAGER', variant=args.variant,
                   currently_retained=args.retained, whole_ms=whole, stages=stages,
                   process_medians=[dict(process=p['process'], rows=[dict(case=r['case'], median_ms=r['median_ms']) for r in p['rows']]) for p in processes],
                   processes=[dict(process=p['process'], calls=p['calls'], lifecycle_seconds=p['lifecycle_seconds']) for p in processes],
                   measurements='all eager/no graph; normal whole-layer: two independent processes; diagnostic stages: first process',
                   definitions=definitions)
    (out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / 'STAGE-TABLE.md').write_text(md)
    print(md)


if __name__ == '__main__':
    main()

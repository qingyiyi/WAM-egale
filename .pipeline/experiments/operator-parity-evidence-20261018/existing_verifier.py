"""Read existing finite verifier evidence only; no GPU imports, execution or hashes."""
import csv
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path('/rjs/cxz/论文探索')
OUT=Path(__file__).resolve().parent
RAW=ROOT/'投机采样/results/runtime'


def load(path):return json.loads(path.read_text())


def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    baseline=RAW/'p0-wam-static-prompt-baseline-v2-20261016-r1'
    guarded=RAW/'p0-wam-candidate-tile-guarded-v2-20261016-r1'
    b=load(baseline/'BASELINE-RESULT.json');g=load(guarded/'GUARDED-RESULT.json')
    issues=[];rows=[]
    def require(ok,what):
        if not ok:issues.append(what)
    require(load(baseline/'SUPERVISOR-RESULT.json')['all_owned_live_exited'],'static baseline cleanup')
    require(load(guarded/'SUPERVISOR-RESULT.json')['all_owned_live_exited'],'guarded cleanup')
    require(len(b['rows'])==6 and len(b['admission'])==36,'six real queries and36 eager admissions')
    require(b['all_six_queries_all_six_arms_complete'],'static six query/arm coverage')
    require(b['all_valid_KV_RNG_inputs_slots_versions_unchanged'],'static valid state checks')
    require(len(g['actual_queries'])==6 and len(g['actual_rows'])==180,'guarded six queries/30layers')
    for item in g['actual_queries']:
        require(item['full_frozen_output']['bitwise'] and item['full_frozen_output']['finite'],str(item)+'guarded output')
        require(item['all_valid_KV_slots_RNG_and_versions_unchanged'],'guarded valid state checks')
    for point in b['rows']:
        for arm, measured in point['timing'].items():
            values=[sample['cuda_ms'] for sample in measured['samples']]
            walls=[sample['wall_ms'] for sample in measured['samples']]
            require(len(values)==6,'six balanced timing rounds')
            require(math.isclose(statistics.median(values),measured['median_cuda_ms'],rel_tol=1e-12),'event median')
            require(math.isclose(statistics.median(walls),measured['median_wall_ms'],rel_tol=1e-12),'wall median')
            require(all(s['same_shape_original_bitwise_and_finite'] for s in measured['samples']),'timed output parity')
            require(point['first_actual_replay'][arm]['bitwise'],'first actual Graph replay parity')
            rows.append(dict(length=point['length'],total_KV=point['total_KV'],query=point['query'],arm=arm,
                cuda_median_ms=measured['median_cuda_ms'],wall_median_ms=measured['median_wall_ms'],
                min_cuda_ms=min(values),max_cuda_ms=max(values),samples=len(values),
                same_shape_original_bitwise_and_finite=True,
                source_pointer=f'{baseline}/BASELINE-RESULT.json#/rows/{len(rows)//6}/timing/{arm}'))
    with (OUT/'EXISTING-VERIFIER-PERFORMANCE.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fills=[{k:v for k,v in item.items() if k!='shapes'} for item in b['prefills']]
    result=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        CPU_audit_issues=issues,baseline=str(baseline),guarded=str(guarded),
        performance_cells=len(rows),raw_samples=sum(r['samples'] for r in rows),prefills=fills,
        guarded_finite_queries=6,guarded_actual_layer_sites=180,
        new_GPU_jobs=0,new_hashes=0,independent_test_complete=False,formal_performance_permission=False)
    (OUT/'EXISTING-VERIFIER-SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 既有verifier证据的补充CPU核对','',f'更新：{result["at"]}。没有新GPU重放；数值一致性来自原GPU记录，本轮仅独立重算原始时间样本。','',
        '此报告与原standalone SDPA验收分开。这里使用现有shared-prefix与guarded G2路径、同shape静态prompt缓存，不能给standalone继承逐位证书。','',
        '| KV | query | B2未缓存/缓存 ms | B14 shared未缓存/缓存 ms | B14 guarded未缓存/缓存 ms |',
        '|---|---|---|---|---|']
    for point in b['rows']:
        t=point['timing']
        pairs=[]
        for prefix in ('n1_dense','n7_native','n7_guarded'):
            pairs.append(f'{t[prefix+"_uncached"]["median_cuda_ms"]:.4f} / {t[prefix+"_cached"]["median_cuda_ms"]:.4f}')
        lines.append(f'| {point["total_KV"]} | {point["query"]} | '+ ' | '.join(pairs)+' |')
    lines += ['','## 数值与资源边界','',
        '- 六个真实冻结query、全部30层：guarded路径六query完整输出逐位/有限，180层位点真正调用guarded，记录有效KV、slots、RNG和版本不变。它仍只是有限fixture证据。',
        '- 静态baseline有36次eager准入、36次首次Graph replay、216次计时样本；原记录输出均与同shape原Torch逐位一致。两套实验的自有进程实际退出。',
        '- 表是单个完整30层model query的静态Graph延迟，不含整个候选树、draft/pack/accept、有效提交g、完整policy或机器人E2E。',
        '- B2与B14分别使用原shape投影；缓存普通收益单列，不能将cached B14与uncached B2比较后包装为MegaTransformer协同。',
        '- prompt首填不在热Graph内：B2 payload383778816B，B14 payload2686451712B，均保存61个投影；实际首填开销见SUMMARY，不忽略构建和缓存容量。',
        '- 只覆盖1/7状态、不含3状态；没有任意新输入/真实cache生命周期、完整block逐层误差分布或单query内有效提交成本证据。',
        '- 原未guarded G2的失败不因后续几何guard改写；当前新算子矩阵仍要独立测性能与误差。四项质量/性能权限不变。','',
        '完整36单元CSV：[EXISTING-VERIFIER-PERFORMANCE.csv](EXISTING-VERIFIER-PERFORMANCE.csv)；机器摘要：[EXISTING-VERIFIER-SUMMARY.json](EXISTING-VERIFIER-SUMMARY.json)。','',
        f'原始证据：`{baseline}/BASELINE-RESULT.json`、`{guarded}/GUARDED-RESULT.json`及各SUPERVISOR-RESULT.json。','',
        'CPU复核问题：'+('无。' if not issues else repr(issues))]
    (OUT/'EXISTING-VERIFIER-REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

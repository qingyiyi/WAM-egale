"""CPU/std-library aggregation of every registered operator point, without hashes."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path('/rjs/cxz/论文探索')
OUT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low]+(position-low)*(values[high]-values[low])


def csvwrite(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ['status'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    job = Path(sys.argv[1]).resolve(strict=True)
    issues, fixtures, accuracy, performance, rounds, quantiles, oracles, processes = [], [], [], [], [], [], [], []
    event_samples = wall_samples = 0

    def require(value, explanation):
        if not value:
            issues.append(explanation)

    for number in (1, 2):
        path = job/f'process{number}'/'RESULT.json'
        if not path.exists():
            issues.append(f'进程{number}尚无RESULT.json')
            continue
        record = load(path)
        process_complete = record['status'] == 'COMPLETE_REAL_B2_VIDEO_ACTION_AND_B14_OPERATOR_BLOCK_MATRIX'
        resource_path = path.with_name('RESOURCE-ADMISSION.json')
        resource = load(resource_path) if resource_path.exists() else {}
        resource_counter_qualified = resource.get('GPU_utilization_counter_qualified', resource.get('confirmed_idle_seconds', 0)>=10)
        processes.append(dict(process=number, source=str(path), status=record['status'],
                              error=record.get('error'), points=len(record['points']),
                              target_forwards=record.get('target_forwards'),
                              elapsed_seconds=record.get('elapsed_seconds'),
                              peak_allocated_bytes=record.get('peak_allocated_bytes')))
        require(len(record['points']) == 54, f'进程{number}完整54点覆盖')
        require(record['status'] == 'COMPLETE_REAL_B2_VIDEO_ACTION_AND_B14_OPERATOR_BLOCK_MATRIX',
                f'进程{number}未完整结束')
        for oracle in record.get('full_query_oracles', []):
            oracles.append(dict(process=number, **oracle))
            require(oracle['byte_equal'] and oracle['finite'], f'进程{number}完整冻结B14 query不一致')
        require(len(record.get('full_query_oracles', [])) == 6, f'进程{number}完整query oracle覆盖')
        identities = set()
        for index, point in enumerate(record['points']):
            identity = {key: point[key] for key in ('length', 'batch', 'kind', 'query_role', 'query_index', 'layer')}
            identity['process'] = number
            identity['process_complete'] = process_complete
            identity['timing_window_qualified'] = process_complete and resource_counter_qualified
            identity['source_pointer'] = f'{path}#/points/{index}'
            key = tuple(identity[k] for k in ('length', 'batch', 'kind', 'query_role', 'layer'))
            require(key not in identities, f'进程{number}重复point：{key}')
            identities.add(key)
            require(point['valid_KV_preserved'] and point['valid_metadata_preserved'], f'{key}有效KV或元数据变化')
            self_attention = point['attention']['self']
            fixtures.append(dict(**identity, source=point['source'],
                timestep_min=point['timestep_min'], timestep_max=point['timestep_max'],
                prefix_tokens=point['prefix_tokens'],
                self_q_shape=str(self_attention['q_shape']), self_k_shape=str(self_attention['k_shape']),
                cross_k_shape=str(point['attention']['cross']['k_shape']),
                valid_KV_preserved=point['valid_KV_preserved'], valid_metadata_preserved=point['valid_metadata_preserved']))
            sections = [('self', point['attention']['self']), ('cross', point['attention']['cross']), ('block', point['block'])]
            for part, section in sections:
                for engine, stats in section['accuracy'].items():
                    accuracy.append(dict(**identity, part=part, engine=engine, **stats))
                    require(stats['finite'], f'{key}/{part}/{engine}非有限值')
                    if engine == 'torch':
                        require(stats['byte_equal'], f'{key}/{part}原Torch重复不逐位相等')
                for mode in ('eager',) if part == 'block' else ('eager', 'graph_core'):
                    measurements = section[mode]
                    torch_time = measurements['torch']['median_ms']
                    shared_time = measurements.get('shared', {}).get('median_ms')
                    for engine, measured in measurements.items():
                        samples, walls = measured['raw_event_ms'], measured['raw_wall_ms']
                        expected = 20 if part == 'block' else 50
                        require(len(samples) == len(walls) == 7, f'{key}/{part}/{mode}/{engine}轮数')
                        require(all(len(row) == expected for row in samples+walls), f'{key}/{part}/{mode}/{engine}样本数')
                        medians = [statistics.median(row) for row in samples]
                        require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                    for a, b in zip(medians, measured['round_medians_ms'])), f'{key}/{part}/{mode}/{engine}轮中位数')
                        median = statistics.median(medians)
                        require(math.isclose(median, measured['median_ms'], rel_tol=1e-12, abs_tol=1e-12), f'{key}/{part}/{mode}/{engine}总中位数')
                        require(math.isclose(median/torch_time, measured['over_torch'], rel_tol=1e-12, abs_tol=1e-12), f'{key}/{part}/{mode}/{engine}Torch比例')
                        event_samples += sum(map(len, samples)); wall_samples += sum(map(len, walls))
                        performance.append(dict(**identity, part=part, mode=mode, engine=engine,
                            torch_ms=torch_time, engine_ms=median, engine_over_torch=median/torch_time,
                            torch_over_engine=torch_time/median, latency_reduction_pct=(1-median/torch_time)*100,
                            shared_ms=shared_time, engine_over_shared=median/shared_time if shared_time else None,
                            synchronized_wall_ms=statistics.median(statistics.median(row) for row in walls),
                            slower_than_torch_rounds=sum(a>b for a,b in zip(medians,measurements['torch']['round_medians_ms']))))
                        for rid, value in enumerate(medians):
                            baseline = measurements['torch']['round_medians_ms'][rid]
                            rounds.append(dict(**identity, part=part, mode=mode, engine=engine,
                                               round=rid+1, engine_ms=value, torch_ms=baseline,
                                               engine_over_torch=value/baseline))
                        for label, raw in (('cuda_event',samples),('synchronized_wall',walls)):
                            flat = [value for row in raw for value in row]
                            require(all(math.isfinite(value) and value>0 for value in flat), f'{key}/{part}/{mode}/{engine}/{label}非正/非有限')
                            quantiles.append(dict(**identity, part=part, mode=mode, engine=engine, measurement=label,
                                samples=len(flat), min_ms=min(flat), p05_ms=quantile(flat,.05), p50_ms=quantile(flat,.5),
                                p90_ms=quantile(flat,.9), p99_ms=quantile(flat,.99), max_ms=max(flat)))

    terminal = load(job/'TERMINAL.json') if (job/'TERMINAL.json').exists() else None
    require(terminal is not None and terminal['status']=='COMPLETE_TWO_PROCESS_REAL_INPUT_OPERATOR_MATRIX', '双进程launcher尚未完整终态')
    if terminal:
        require(terminal['owned_children_exited'], '自有子进程未清理')
    summary = dict(updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        job=str(job), source_version='operator_kv_matrix_v1', terminal=terminal, processes=processes,
        issues=issues, audit_complete=not issues, points=len(fixtures), correctness_cells=len(accuracy),
        performance_cells=len(performance), rounds=len(rounds), event_samples=event_samples, wall_samples=wall_samples,
        all_timing_windows_qualified=bool(fixtures) and all(r['timing_window_qualified'] for r in fixtures),
        oracles=oracles, performance_boundary='prepared attention and real single block, NOT full verifier/chunk/episode E2E',
        numerical_boundary='finite real selected queries/layers only, not a universal BF16 or robotic quality certificate',
        engineering_quality_screen=False, independent_test_complete=False, noninferiority_established=False, formal_performance_permission=False)
    groups = []
    for kind, batch in (('video',2),('action',2),('action',14)):
        for length in ('short','middle','long'):
            for part, mode in (('self','eager'),('self','graph_core'),('cross','graph_core'),('block','eager')):
                selected = [r for r in performance if (r['kind'],r['batch'],r['length'],r['part'],r['mode'])==(kind,batch,length,part,mode)]
                for engine in sorted({r['engine'] for r in selected}-{'torch'}):
                    rows = [r for r in selected if r['engine']==engine]
                    correct = [r for r in accuracy if (r['kind'],r['batch'],r['length'],r['part'],r['engine'])==(kind,batch,length,part,engine)]
                    ratios = [r['engine_over_torch'] for r in rows]
                    shared = [r['engine_over_shared'] for r in rows if r['engine_over_shared'] is not None]
                    groups.append(dict(kind=kind,batch=batch,length=length,part=part,mode=mode,engine=engine,
                        cells=len(rows),median_engine_over_torch=statistics.median(ratios),min_engine_over_torch=min(ratios),max_engine_over_torch=max(ratios),
                        median_engine_over_shared=statistics.median(shared) if shared else None,
                        slower_cells=sum(r>1 for r in ratios),
                        max_abs=max(r['max_abs'] for r in correct), max_rel_l2=max(r['rel_l2'] for r in correct),
                        allclose_02_passed=sum(r['allclose_02'] for r in correct),byte_equal=sum(r['byte_equal'] for r in correct),
                        accuracy_cells=len(correct)))
    summary['grouped']=groups
    for name, rows in [('FIXTURES.csv',fixtures),('ACCURACY.csv',accuracy),('PERFORMANCE.csv',performance),('ROUND-METRICS.csv',rounds),('SAMPLE-QUANTILES.csv',quantiles),('GROUPED.csv',groups)]:
        csvwrite(name,rows)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# 原始与verifier规模短中长KV算子对照', '', f'更新：{summary["updated"]}。输入/原始样本：`{job}`。', '',
        f'CPU复核：{"完整" if not issues else "PARTIAL，见缺口"}；实际{len(processes)}个进程、{len(fixtures)}点、{len(accuracy)}数值单元、{len(performance)}性能单元、{event_samples} Event样本。', '',
        '计时资源状态：'+('本登记窗口核查通过；物理宿主/CPU独占仍未证明。' if summary['all_timing_windows_qualified'] else 'EXPLORATORY：存在未完整结束或GPU利用率计数未满足登记空闲条件的进程；时间仅作探索性结果，不作为已确认无争用的正式性能。') , '',
        '每个KV、输入规模使用真实first/last query与层0/8/29。B14仍为7个独立Q16、CFG2；不是跨候选Q112。video实际K与action K分开列在FIXTURES。', '',
        '表中是同组12个点（两进程×两query×三层）的比例中位数及完整范围，不是单一模型或任务加速比。大于1表示比Torch慢；B14必须同时看相对shared列。', '',
        '| 输入 | KV档 | 部分/计时 | 实现 | 实现/Torch中位数 [min,max] | 实现/shared中位数 | .02通过/检查 | 逐字节相同 | 最大绝对误差 |',
        '|---|---|---|---|---|---|---|---|---|']
    for group in groups:
        value=group
        ratio=f'{value["median_engine_over_torch"]:.4f} [{value["min_engine_over_torch"]:.4f},{value["max_engine_over_torch"]:.4f}]'
        shared='NA' if value['median_engine_over_shared'] is None else f'{value["median_engine_over_shared"]:.4f}'
        lines.append(f'| B{value["batch"]} {value["kind"]} | {value["length"]} | {value["part"]}/{value["mode"]} | {value["engine"]} | {ratio} | {shared} | {value["allclose_02_passed"]}/{value["accuracy_cells"]} | {value["byte_equal"]}/{value["accuracy_cells"]} | {value["max_abs"]:.8g} |')
    lines += ['', '## 证据边界', '',
        '- 全部点/样本/慢轮保留在CSV与runtime原JSON；分位数用type7线性插值。eager与Graph-core独立，Graph每replay32调用、按32摊分。',
        '- 完整block包括原norm、projection、FFN和所列attention替换；不是30层verifier、scheduler、动作或机器人闭环证书。',
        '- .02为原绝对加相对容差；更紧容差与ULP是诊断统计，不改旧门。容差通过不等于逐位相等。',
        '- captured B14 oracle仅证明冻结query在原Torch路径复现；不证明替换算子后完整模型/接受路径不变。有效KV检查只覆盖这些真实block重放。',
        '- timing_window_qualified综合进程完整终态及GPU利用率准入；process_complete只代表本轮覆盖完成。若NVML非零而可见进程/显存为空，按用户明确窗口先采输入，延迟标探索性，不猜测计数根因。其他版本、设备、任意输入仍未覆盖。',
        '- 四项研究权限保持false，未跑TF正式45chunk、独立TEST或任务E2E；不能据算子结果声称完整投机或机器人加速。', '', '## 完整数据', '',
        '- [每点输入与实际形状](FIXTURES.csv)、[全部精度](ACCURACY.csv)、[全部性能](PERFORMANCE.csv)',
        '- [逐轮](ROUND-METRICS.csv)、[样本分布](SAMPLE-QUANTILES.csv)、[分组总表](GROUPED.csv)、[机器可读摘要](SUMMARY.json)', '', '## 缺口', '']
    lines += ['- '+issue for issue in issues] if issues else ['- 本登记矩阵已完整；未覆盖的完整模型/机器人契约仍不外推。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key:summary[key] for key in ('updated','audit_complete','points','correctness_cells','performance_cells','event_samples','issues')},ensure_ascii=False))


if __name__=='__main__':
    main()

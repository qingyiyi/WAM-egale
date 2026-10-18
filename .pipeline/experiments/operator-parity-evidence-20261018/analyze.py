"""仅以标准库复核既有结果；不导入Torch、不创建CUDA上下文、不计算hash。"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import tarfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path('/rjs/cxz/论文探索')
OUT = Path(__file__).resolve().parent
WORKTREE = ROOT / 'wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922'
PACKAGE = WORKTREE / 'operator_lab/wam_transformer_tile'
RAW = WORKTREE / 'outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012'
LABELS = ['短video', '短action', 'tile0长video', 'tile0长action', 'tile1长video', 'tile1长action']


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def load(path):
    return json.loads(path.read_text())


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def main():
    start_wall, started = time.monotonic(), stamp()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    reports = [load(RAW / f'release-r{i}.json') for i in (1, 2)]
    prior = load(RAW / 'final-acceptance-summary.json')
    resource = load(RAW / 'final-resource-and-boundary-audit.json')
    issues = []

    def require(condition, context):
        if not condition:
            issues.append(context)

    def close(a, b, context):
        require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), context)

    correctness, performance, rounds, quantiles, fixtures = [], [], [], [], []
    event_count = wall_count = 0
    timing_cells = []
    for process, report in enumerate(reports, 1):
        require(len(report['results']) == 6 and report['rounds'] == 7, f'进程{process}输入/轮数')
        require(report['atol'] == .02 and report['rtol'] == .02, f'进程{process}原容差')
        for index, result in enumerate(report['results']):
            fixture_id = f'F{index + 1}'
            identity = dict(process=process, fixture_id=fixture_id, label=LABELS[index])
            fixture_path = Path(result['fixture'])
            require(fixture_path.is_file(), f'真实输入缺失：{fixture_path}')
            if process == 1:
                fixtures.append(dict(fixture_id=fixture_id, label=LABELS[index], fixture_path=str(fixture_path),
                                     fixture_exists=fixture_path.is_file(),
                                     fixture_bytes=fixture_path.stat().st_size if fixture_path.is_file() else None,
                                     q_shape=str(result['attention']['self']['q_shape']),
                                     self_k_shape=str(result['attention']['self']['k_shape']),
                                     cross_k_shape=str(result['attention']['cross']['k_shape']),
                                     weight_path=str(fixture_path.with_name('block0_weights.pt')),
                                     weights_exist=fixture_path.with_name('block0_weights.pt').is_file()))
            else:
                require(result['fixture'] == reports[0]['results'][index]['fixture'], f'进程间fixture身份{index}')
            for category, records in [('fixture_oracle', result['fixture_oracle']), ('block_stage', result['stages'])]:
                for check_name, values in records.items():
                    correctness.append(dict(**identity, category=category, check=check_name, **values,
                                            source_pointer=f'/results/{index}/{"stages" if category == "block_stage" else category}/{check_name}'))
            for part, attention in result['attention'].items():
                correctness.append(dict(**identity, category='standalone_attention', check=part,
                                        **attention['correctness'], source_pointer=f'/results/{index}/attention/{part}/correctness'))
                require(attention['torch_backend'] == 1, f'进程{process}/{fixture_id}/{part}参考backend')
                for name, passed in attention['contract_checks'].items():
                    correctness.append(dict(**identity, category='api_contract', check=f'{part}/{name}', passed=passed,
                                            source_pointer=f'/results/{index}/attention/{part}/contract_checks/{name}'))
                for mode in ('eager', 'cuda_core'):
                    timing_cells.append((identity, part, mode, attention[mode], f'/results/{index}/attention/{part}/{mode}'))
            timing_cells.append((identity, 'block', 'full_block', result['full_block'], f'/results/{index}/full_block'))
            flags = [r['passed'] for r in correctness if r['process'] == process and r['fixture_id'] == fixture_id]
            require(len(flags) == 48, f'进程{process}/{fixture_id}检查项数量')
            require(all(flags) == result['correctness_passed'], f'进程{process}/{fixture_id}正确性聚合')

    for identity, part, mode, timing, pointer in timing_cells:
        expected_count = 20 if mode == 'full_block' else 50
        for engine in ('torch', 'cuda'):
            data = timing[engine]
            samples = data['raw_cuda_event_ms']
            walls = data['raw_synchronized_wall_ms']
            require(len(samples) == len(walls) == 7, f'{identity}/{part}/{mode}/{engine}轮数')
            require(all(len(row) == expected_count for row in samples + walls), f'{identity}/{part}/{mode}/{engine}样本数')
            recomputed = [statistics.median(row) for row in samples]
            for a, b in zip(recomputed, data['round_medians_ms']):
                close(a, b, f'{identity}/{part}/{mode}/{engine}轮中位数')
            close(statistics.median(recomputed), data['median_ms'], f'{identity}/{part}/{mode}/{engine}最终中位数')
            close(statistics.median(statistics.median(row) for row in walls), data['synchronized_wall_ms'],
                  f'{identity}/{part}/{mode}/{engine}墙钟中位数')
            event_count += sum(map(len, samples))
            wall_count += sum(map(len, walls))
            for kind, raw_values in [('cuda_event', samples), ('synchronized_wall', walls)]:
                flat = [v for row in raw_values for v in row]
                require(all(math.isfinite(v) and v > 0 for v in flat), f'{identity}/{part}/{mode}/{engine}/{kind}有限正值')
                quantiles.append(dict(**identity, part=part, mode=mode, engine=engine, measurement=kind,
                                      n=len(flat), min_ms=min(flat), p05_ms=percentile(flat, .05),
                                      p50_ms=percentile(flat, .5), p90_ms=percentile(flat, .9),
                                      p99_ms=percentile(flat, .99), max_ms=max(flat),
                                      original_gate=False, source_pointer=pointer + '/' + engine))
        ratio = timing['cuda']['median_ms'] / timing['torch']['median_ms']
        close(ratio, timing['cuda_over_torch'], f'{identity}/{part}/{mode}比例')
        wins = 0
        for rid in range(7):
            a, b = [timing[x]['round_medians_ms'][rid] for x in ('torch', 'cuda')]
            wins += b <= a
            rounds.append(dict(**identity, part=part, mode=mode, round=rid + 1,
                               torch_event_median_ms=a, cuda_event_median_ms=b, cuda_over_torch=b/a,
                               cuda_not_slower=b <= a, source_pointer=pointer))
        require(wins == timing['cuda_wins_rounds'], f'{identity}/{part}/{mode}胜出轮数')
        performance.append(dict(**identity, part=part, mode=mode,
                                torch_event_median_ms=timing['torch']['median_ms'],
                                cuda_event_median_ms=timing['cuda']['median_ms'], cuda_over_torch=ratio,
                                latency_reduction_pct=(1-ratio)*100, torch_over_cuda=1/ratio,
                                torch_sync_wall_ms=timing['torch']['synchronized_wall_ms'],
                                cuda_sync_wall_ms=timing['cuda']['synchronized_wall_ms'],
                                cuda_wins_rounds=wins, rounds=7, original_gate_passed=ratio <= 1,
                                source_pointer=pointer))

    require(len(correctness) == 576 and all(row['passed'] for row in correctness), '576项原正确性记录聚合')
    require(len(performance) == 60 and all(row['original_gate_passed'] for row in performance), '60项原性能门')
    require(event_count == wall_count == 36960, '原始Event/同步墙钟样本总数')
    for r in performance:
        expected = next(x for x in prior['all_fixture_metrics'] if x['process'] == r['process'] and x['fixture_index'] == int(r['fixture_id'][1:])-1)
        key = 'full_block' if r['part'] == 'block' else r['part'] + ('_core' if r['mode'] == 'cuda_core' else '_eager')
        close(r['cuda_over_torch'], expected[key], f'原最终摘要比例/{r["process"]}/{r["fixture_id"]}/{key}')

    compared = []
    with tarfile.open(RAW / 'pre-cleanup-validated-package.tar.gz', 'r:gz') as archive:
        for file in sorted(PACKAGE.rglob('*')):
            if not file.is_file() or file.suffix not in {'.py', '.cu', '.cuh', '.h', '.hpp', '.cpp', '.so'}:
                continue
            relative = file.relative_to(PACKAGE).as_posix()
            if '__pycache__' in relative or relative.startswith('build/'):
                continue
            member = 'wam_transformer_tile/' + relative
            try:
                with archive.extractfile(member) as stream:
                    archived = stream.read()
                equal = file.read_bytes() == archived
                state = 'EQUAL' if equal else 'DIFFERENT'
            except KeyError:
                state = 'ARCHIVE_MEMBER_MISSING'
            compared.append(dict(path=relative, archive_member=member, state=state, current_bytes=file.stat().st_size))
            require(state == 'EQUAL', '归档对应正式文件不一致：' + relative)
    dump('SOURCE-COMPARISON.json', dict(method='exact_bytes_no_hash', archive=str(RAW/'pre-cleanup-validated-package.tar.gz'),
                                        package=str(PACKAGE), checked=len(compared), files=compared,
                                        note='只比较现存实现/头文件/测试/正式包二进制；归档retired build副本不是active binary；不声称文档/元数据全部相同。'))

    coverage = [
        ('真实fixture', 'OBSERVED', '6份真实输入，两次进程使用相同六份；不是576份独立输入'),
        ('硬件与精度', 'OBSERVED', 'A100-SXM4-40GB/SM80，BF16，B2/H24/D128，contiguous BSHD'),
        ('self attention形状', 'OBSERVED', 'Q128/K128、Q16/K144、Q128/K2160、Q16/K2160'),
        ('cross attention形状', 'OBSERVED', 'Q128或16、K512'),
        ('原容差', 'OBSERVED', 'atol=rtol=.02；未独立重算未保存的元素级误差'),
        ('真实block0与12阶段', 'OBSERVED', '逐stage比较；完整block包含原PyTorch norm/linear/FFN与attention替换'),
        ('API契约', 'OBSERVED', 'CPU/FP32/维度/KV形状/布局/空输入/NaN scale/autograd拒绝；自定义scale和非默认stream'),
        ('数值重复', 'OBSERVED', '两进程记录的max_abs/rel_l2/torch.equal一致；不是额外独立输入'),
        ('逐元素严格逐位', 'NOT_ESTABLISHED', '多数输出torch.equal为false；其语义也不是原始字节比较'),
        ('中等KV1152/任意尾块', 'NOT_COVERED', '最终六fixture未覆盖；不能由K2160结果外推'),
        ('B14与1/3/7候选', 'NOT_COVERED_BY_THIS_ACCEPTANCE', '现有报告是B2 dense；后续原型必须独立核对'),
        ('共享prefix/候选私有KV', 'NOT_COVERED_BY_THIS_ACCEPTANCE', '没有这份基线证书可继承'),
        ('更新cache分支', 'NOT_COVERED', 'fixture要求update_cache=0；只精确验证记录的self K/V'),
        ('30层累计与CFG方向', 'NOT_COVERED', '单block0不是全模型链'),
        ('投机接受分数/翻转/路径', 'NOT_COVERED', '原allclose不能替代接受阈值与状态契约'),
        ('完整scheduler/chunk/decoded action/RNG', 'NOT_COVERED', '没有此实现完整替换的对应证据'),
        ('LIBERO成功率/trace/task E2E', 'NOT_COVERED', '不将投机方法既有任务结果签给此算子'),
        ('FSDP整模型', 'NOT_COVERED', '参照是Torch SDPA FlashAttention，并非FSDP运行时'),
        ('资源空闲', 'OBSERVED_LIMITED', '原资源gate两次通过，无计时中管理API/profiler；物理宿主独占未证明'),
        ('任意调用/任意轮均更快', 'FALSE', '逐轮存在回退，原gate是median-of-round-medians'),
        ('所有形状稳定加速/统计置信界', 'NOT_ESTABLISHED', '两进程不足；长video完整block margin约0.1%'),
        ('其他硬件/版本/FP16/训练梯度', 'NOT_COVERED_OR_UNSUPPORTED', '本地固定SM80、BF16、推理契约'),
        ('元素级误差分位数/ULP/更紧allclose', 'UNKNOWN', '原JSON没有完整输出张量或所需统计；不能离线补造'),
        ('后续G1/G2/guard/cache原型', 'SEPARATE_EVIDENCE', '本包只审旧已验收实现，不能继承其通过结论'),
    ]
    write_csv('COVERAGE.csv', [dict(subject=a, status=b, detail=c) for a,b,c in coverage])
    write_csv('FIXTURES.csv', fixtures)
    write_csv('CORRECTNESS.csv', correctness)
    write_csv('PERFORMANCE.csv', performance)
    write_csv('ROUND-METRICS.csv', rounds)
    write_csv('SAMPLE-QUANTILES.csv', quantiles)

    categories = {}
    for cat in {r['category'] for r in correctness}:
        items = [r for r in correctness if r['category'] == cat]
        numeric = [r for r in items if 'max_abs' in r]
        categories[cat] = dict(checks=len(items), passed=sum(bool(r['passed']) for r in items),
                               torch_equal_count=sum(bool(r.get('bitwise')) for r in numeric),
                               max_abs=max((r['max_abs'] for r in numeric), default=None),
                               max_rel_l2=max((r['rel_l2'] for r in numeric), default=None))
    numerical_same = all(reports[0]['results'][i][category] == reports[1]['results'][i][category]
                         for i in range(6) for category in ('fixture_oracle', 'stages')) and all(
        reports[0]['results'][i]['attention'][part]['correctness'] == reports[1]['results'][i]['attention'][part]['correctness']
        for i in range(6) for part in ('self', 'cross'))
    summary = dict(status='COMPLETE_EXISTING_DATA_AUDIT' if not issues else 'AUDIT_ISSUES_FOUND',
                   started_at=started, finished_at=stamp(), cpu_process_seconds=time.process_time(),
                   analysis_wall_seconds=time.monotonic()-start_wall, gpu_jobs_started=0, gpu_hours=0,
                   new_tensor_replay=False, independent_tensor_error_recompute=False, new_hashes_computed=False,
                   source_reports=[str(RAW/f'release-r{i}.json') for i in (1,2)],
                   original_acceptance_summary=str(RAW/'final-acceptance-summary.json'),
                   original_resource_audit=str(RAW/'final-resource-and-boundary-audit.json'),
                   independent_processes=2, distinct_real_fixtures=6, rounds_per_process=7,
                   correctness_checks=len(correctness), correctness_categories=categories,
                   attention_performance_gates=48, full_block_performance_gates=12,
                   event_samples=event_count, synchronized_wall_samples=wall_count,
                   performance_cells=len(performance), round_cells=len(rounds),
                   round_cells_cuda_slower=sum(not r['cuda_not_slower'] for r in rounds),
                   block_round_cells_cuda_slower=sum(not r['cuda_not_slower'] for r in rounds if r['part']=='block'),
                   numerical_metrics_identical_across_processes=numerical_same,
                   archived_implementation_files_compared=len(compared),
                   archived_implementation_all_equal=all(r['state']=='EQUAL' for r in compared),
                   original_resource=resource, issues=issues,
                   historical_failures_retained=True,
                   current_robot_quality_and_formal_permission_changed=False,
                   note='数值为旧GPU运行保存的统计，CPU仅复核聚合；性能所有原始值参与重算，不删轮、不重设门。')
    dump('SUMMARY.json', summary)

    lines = [
        '# 原算子与Torch对照：完整既有证据复核', '', f'复核完成：{summary["finished_at"]}。状态：`{summary["status"]}`。', '',
        '结论：已验收的独立SDPA/单block实现，在六份真实输入与原容差、原中位数性能门范围内通过；不能写成逐位等价、任意形状稳定加速或整模型/机器人任务已验证。', '',
        '本轮为CPU离线复核，没有导入Torch、创建CUDA上下文、运行GPU测试或计算新hash；没有改原tile/HW、原FAILED和质量权限。', '',
        '## 身份与范围', '',
        '- Torch 2.9.0+cu126，Torch commit `0fabc3ba44823f257e70ce397d989c8de5e362c1`；FlashAttention commit `979702c87a8713a8e0a5e9fee122b90d2ef13be5`。',
        '- GPU：NVIDIA A100-SXM4-40GB，108 SM；自有扩展nvcc12.4，Torch wheel CUDA12.6，不声称二者SASS相同。',
        '- BF16、B2/H24/D128、连续BSHD、无mask/noncausal/无dropout/推理；cross K512；fixture为block0、update_cache=0。',
        f'- 现存{len(compared)}项实现/头文件/测试/正式包.so与10/12验收归档对应成员逐字节一致。退役build副本不作active binary。细目见[SOURCE-COMPARISON.json](SOURCE-COMPARISON.json)。',
        '- 六份真实输入，两独立进程，各7轮；正确性在fixture/阶段上检查，不是576条独立机器人轨迹。', '',
        '## 数值数据', '',
        '所有记录的576项原检查通过。拆分为168项fixture oracle、24项独立attention输出、144项block阶段、240项API/stream/scale契约。原保存的检查标志参与CPU聚合；未重新运行tensor比较。', '',
        '`allclose`：逐元素满足 `abs(actual-ref) <= 0.02 + 0.02*abs(ref)`。这不是统一2%相对误差，也不要求最大绝对误差低于0.02。`rel_l2 = ||actual-ref||₂ / ||ref||₂` 是整体相对范数，不是最坏元素相对误差。', '',
        '| 输入 | self max_abs / rel_L2 | cross max_abs / rel_L2 | block最终 max_abs / rel_L2 | 最终torch.equal |',
        '|---|---:|---:|---:|---|',
    ]
    for i, result in enumerate(reports[0]['results']):
        sa, ca, final = result['attention']['self']['correctness'], result['attention']['cross']['correctness'], result['stages']['output']
        lines.append(f'| F{i+1} {LABELS[i]} | {sa["max_abs"]:.9g} / {sa["rel_l2"]:.7g} | {ca["max_abs"]:.9g} / {ca["rel_l2"]:.7g} | {final["max_abs"]:.9g} / {final["rel_l2"]:.7g} | {final["bitwise"]} |')
    lines += ['', '两进程上述数值统计完全相同。原Torch replay与fixture oracle的168项torch.equal均true；这说明参考重放复现原fixture，不表示自有算子所有输出逐位一致。自有block最终输出仅2/12项torch.equal为true（同一份F4在两个进程）。记录中的bitwise由torch.equal产生，并非原始字节核验。', '',
              '最坏中间阶段是F6的ffn_output：max_abs=0.125，rel_L2约4.18265e-4；在原相对+绝对容差下通过。最终block max_abs最大0.015625，rel_L2最大约9.22510e-5。不能只展示最终小误差而省略中间放大。', '',
              '[全部576检查](CORRECTNESS.csv)包含12阶段、原oracle及每项契约；无原始输出全张量，元素误差P99/ULP/更紧容差结果UNKNOWN。', '',
              '## 性能数据', '',
              '门限保持原 `median_of_round_medians(cuda) / median_of_round_medians(torch) <= 1`。每轮attention各50次，block各20次；graph attention每个graph含32次调用，原始值已按32归一化。同步墙钟也保留，但原gate使用CUDA Event中位数，不是墙钟或P99 gate。', '',
              '| 输入 | self eager比r1/r2 | self graph-core比r1/r2 | cross eager比r1/r2 | cross graph-core比r1/r2 | 完整block比r1/r2 |',
              '|---|---:|---:|---:|---:|---:|']
    for i in range(6):
        cells = []
        for part, mode in [('self','eager'),('self','cuda_core'),('cross','eager'),('cross','cuda_core'),('block','full_block')]:
            pair=[next(x['cuda_over_torch'] for x in performance if x['process']==p and x['fixture_id']==f'F{i+1}' and x['part']==part and x['mode']==mode) for p in (1,2)]
            cells.append('/'.join(f'{x:.6f}' for x in pair))
        lines.append('| F'+str(i+1)+' '+LABELS[i]+' | '+' | '.join(cells)+' |')
    lines += ['', '全部48项attention门与12项block门通过。eager延迟降低17.31%–49.06%，graph-core降低0.36%–8.59%，完整block降低0.1060%–4.6062%；这三个边界分别报告，不能把eager收益全当纯kernel收益。', '',
              '| 输入 | r1 Torch / 自有完整block(ms) | r2 Torch / 自有完整block(ms) | 自有不慢于Torch的轮数 r1/r2 |',
              '|---|---:|---:|---:|']
    for i in range(6):
        pair=[r for r in performance if r['fixture_id']==f'F{i+1}' and r['part']=='block']
        lines.append(f'| F{i+1} {LABELS[i]} | '+ ' | '.join(f'{r["torch_event_median_ms"]:.6f} / {r["cuda_event_median_ms"]:.6f}' for r in pair)+f' | {pair[0]["cuda_wins_rounds"]}/7；{pair[1]["cuda_wins_rounds"]}/7 |')
    lines += ['', f'全部420个“输入×进程×类型×轮”的比较有{summary["round_cells_cuda_slower"]}个轮中位数自有更慢；其中完整block {summary["block_round_cells_cuda_slower"]}/84轮更慢。原gate通过并不意味着每轮或每次调用都更快。未删除任何慢轮或离群值。', '',
              '例如F3长video完整block，r1 Torch/自有为3.160064/3.106304ms，r2为1.909248/1.907200ms；跨进程绝对值波动明显，r2收益仅约0.107%。原稀疏资源gate通过、无计时中profiler/管理API，但物理宿主完全独占未证明。两次进程不足以给出普适稳定加速的统计保证。', '',
              '[全部60单元](PERFORMANCE.csv)、[全部420逐轮值](ROUND-METRICS.csv)、[240组原始样本分位数](SAMPLE-QUANTILES.csv)。分位数采用线性插值type7，基于同一单元/引擎的7轮样本，作为描述性分布，不改原gate，不把重复样本当独立实验。', '',
              '## 覆盖缺口与建议', '',
              '[完整覆盖矩阵](COVERAGE.csv)。当前证据未覆盖中KV1152/更多合法尾块、B14共享prefix与私有KV、update_cache其他分支、30层、CFG方向、接受翻转、scheduler/RNG/完整chunk、最终动作与LIBERO闭环、FSDP整模型。后续G1/G2/guard/cache是独立研究实现，不继承本证书。', '',
              '推荐补测顺序：先在独立namespace登记真实短中长输入与合法边界；再测真实完整block/30层累计及全部有效KV；之后检查接受分数/路径/scheduler/RNG/最终动作。独占正式性能与闭环质量另按原门放行，不打断当前已登记GPU任务，不自动追加或放宽门限。', '',
              '原生接口不支持的mask/dropout/FP32/梯度等应按契约拒绝，不能把“拒绝输入通过”写成支持这些输入的数值等价。更紧容差可以前瞻登记，但不能重写旧原门。', '',
              '## 原始证据与失败保留', '',
              f'- [r1原raw]({RAW / "release-r1.json"})',
              f'- [r2原raw]({RAW / "release-r2.json"})',
              f'- [原完整审计]({RAW / "actual-release-report-audit.json"})',
              f'- [原验收摘要]({RAW / "final-acceptance-summary.json"})',
              f'- [资源边界]({RAW / "final-resource-and-boundary-audit.json"})',
              f'- [原报告]({ROOT / ".pipeline/research/tile-sdpa/PYTORCH29_FINAL_20261012.md"})',
              f'- [上一版本失败]({ROOT / ".pipeline/research/tile-sdpa/PYTORCH29_RELEASE_FIRST_20261012.md"})：长video full-block +0.4013%原FAILED保留。不同版本不混进当前60门，也不能删除。', '',
              '本次导出只引用原raw，不复制大JSON或模型tensor。新数值实验为NOT_RUN；四项质量/正式性能权限及全局默认均不变。', '',
              '## CPU审计完整性', '', f'问题清单：{json.dumps(issues,ensure_ascii=False)}。汇总见[SUMMARY.json](SUMMARY.json)。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('OUTPUT_BYTES', sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()))


if __name__ == '__main__':
    main()

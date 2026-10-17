"""CPU/std-library-only description of saved pre-treatment first-two-S scores.

No production imports, model/tensor loads, digests, threshold fitting or tests.
Run only after the parent has retrieved the final registered panel receipts.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path

UNKNOWN = 'UNKNOWN'
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
FIXED = [(1, 1, 29, 'risk'), (2, 1, 36, 'control'), (3, 3, 26, 'risk'),
         (4, 3, 36, 'control'), (5, 4, 26, 'risk'), (6, 4, 36, 'control')]
ROUTE_FIELDS = ('first_costs', 'second_costs', 'threshold', 'actual_parent',
    'actual_child', 'actual_depth', 'sigma', 'next_sigma', 'timestep', 'frame_st_id')
CAP = 8 * 2**20


def read(path):
    if path.stat().st_size > 64 * 2**20:
        raise ValueError('oversized receipt: ' + str(path))
    return json.loads(path.read_text(), parse_constant=lambda value:
        (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(value):
    require(type(value) in (float, int) and math.isfinite(value), 'finite saved scalar required')
    return float(value)


def describe(values):
    if not values:
        return dict(n=0, sum=0.0, mean=UNKNOWN, min=UNKNOWN, max=UNKNOWN)
    return dict(n=len(values), sum=math.fsum(values), mean=math.fsum(values)/len(values),
        min=min(values), max=max(values))


def summarize(trees):
    edges = [v for tree in trees for v in tree['accepted_saved_scores']]
    margins = [tree['threshold']-v for tree in trees for v in tree['accepted_saved_scores']]
    normalized = [v/tree['threshold'] for tree in trees for v in tree['accepted_saved_scores']]
    return dict(tree_count=len(trees), accepted_depth_counts={str(k): v for k, v in
            sorted(Counter(tree['actual_depth'] for tree in trees).items())},
        accepted_edge_count=len(edges), accepted_saved_score=describe(edges),
        accepted_path_score_sum=describe([tree['accepted_path_score_sum'] for tree in trees]),
        accepted_path_score_max=describe([tree['accepted_path_score_max'] for tree in trees
            if tree['accepted_path_score_max'] != UNKNOWN]),
        accepted_edge_margin=describe(margins), accepted_score_over_threshold=describe(normalized),
        accepted_normalized_margin=describe([1-v for v in normalized]),
        root_sigma=describe([tree['sigma'] for tree in trees]),
        all_six_saved_scores=describe([v for tree in trees for v in tree['all_six_saved_scores']]))


def first_two(counter, spec, arm):
    for key in ('pair_id', 'task', 'init'):
        require(counter[key] == spec[key], 'COUNTER identity differs: ' + key)
    require(counter['arm'] == arm and counter.get('permissions') == PERMISSIONS,
        'COUNTER arm/permissions differ')
    rows = counter['policy_records']
    require(isinstance(rows, list) and len(rows) >= 2, 'two complete policy records missing')
    first = rows[:2]
    for ordinal, row in enumerate(first):
        require(row['kind'] == 'policy' and row['role'] == 'S' and row['ok'] is True
            and row['successful_policy_ordinal'] == ordinal, 'pre-treatment S0/S1 differs')
        require(row['audit']['status'] == 'PASS_COMPLETE_ROLE_REQUEST', 'first S role audit failed')
        require(isinstance(row['acceptance'], list) and row['acceptance'], 'saved routes missing')
    return first


def tree_row(route, ordinal, index, threshold):
    missing = [key for key in ROUTE_FIELDS if key not in route]
    require(not missing, 'saved route fields missing: ' + ','.join(missing))
    first = route['first_costs']; second = route['second_costs']
    require(len(first) == 2 and len(second) == 2 and all(len(v) == 2 for v in second),
        'frozen two-by-two tree score inventory differs')
    all_scores = [number(v) for v in first] + [number(v) for row in second for v in row]
    require(min(all_scores) >= 0, 'negative saved score')
    tau = number(route['threshold'])
    require(tau == threshold and tau > 0, 'registered threshold changed')
    parent, child, depth = [route[k] for k in ('actual_parent', 'actual_child', 'actual_depth')]
    require(type(parent) is int and parent in (0, 1) and type(child) is int and child in (0, 1)
        and type(depth) is int and depth in (0, 1, 2), 'actual route indices/depth invalid')
    a, b = number(first[parent]), number(second[parent][child])
    inferred = 0 if a > tau else 1 if b > tau else 2
    require(depth == inferred, 'saved actual_depth disagrees with saved selected gate scores')
    accepted = [] if depth == 0 else [a] if depth == 1 else [a, b]
    return dict(policy_ordinal=ordinal, tree_index=index, actual_depth=depth,
        actual_parent=parent, actual_child=child, threshold=tau,
        sigma=number(route['sigma']), next_sigma=number(route['next_sigma']),
        timestep=number(route['timestep']), frame_st_id=route['frame_st_id'],
        second_edge_sigma=UNKNOWN, all_six_saved_scores=all_scores,
        selected_first_saved_score=a, selected_second_saved_score=b,
        accepted_saved_scores=accepted, accepted_path_score_sum=math.fsum(accepted),
        accepted_path_score_max=max(accepted) if accepted else UNKNOWN,
        accepted_margins=[tau-v for v in accepted],
        accepted_normalized_margins=[1-v/tau for v in accepted],
        maximum_admissible_depth=route.get('maximum_admissible_depth', UNKNOWN),
        score_origin='saved ObservedAPI verifier score; no velocity norm recomputation',
        scope='accepted draft edges only; excludes target bonus/fallback scheduler updates')


def pair_row(job, spec, threshold):
    result = dict(**{k: spec[k] for k in ('pair_id', 'task', 'init', 'stratum')},
        status=UNKNOWN, chosen_arm='C', statistical_unit='one registered task/init pair',
        complete_outcomes={}, sources=[], issues=[], summary=UNKNOWN, chunks=[], trees=[])
    try:
        arms, audits = {}, {}
        for arm in ('C', 'P'):
            out = job/'episodes'/('pair-%02d' % spec['pair_id'])/arm
            docs = {}
            for name in ('COUNTER.json', 'AUDIT.json', 'EPISODE-TERMINAL.json'):
                path = out/name; docs[name] = read(path)
                stat = path.stat()
                result['sources'].append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
            terminal = docs['EPISODE-TERMINAL.json']
            require(all(terminal[k] == spec[k] for k in ('pair_id', 'task', 'init'))
                and terminal['arm'] == arm and terminal['complete_episode'] is True
                and terminal['quality_terminal'] is True
                and terminal['status'] == 'COMPLETE_RECORDED_FULL_EPISODE'
                and terminal['permissions'] == PERMISSIONS, 'complete registered terminal missing')
            require(type(terminal['success']) is bool, 'raw success/failure must be preserved')
            result['complete_outcomes'][arm] = dict(success=terminal['success'], env_steps=terminal['env_steps'])
            arms[arm] = first_two(docs['COUNTER.json'], spec, arm)
            audit = docs['AUDIT.json']
            require(audit['pair_id'] == spec['pair_id'] and audit['arm'] == arm
                and audit['permissions'] == PERMISSIONS
                and audit['status'] == 'PASS_COMPLETE_EPISODE_COMPONENTS',
                'same-arm complete component audit missing')
            audits[arm] = audit['first2S']
        require(sorted(v['status'] for v in audits.values()) == ['PASS_EXACT', 'REFERENCE_STORED'],
            'saved first-two-S component comparison not qualified')
        result['first_two_S_component_receipts'] = audits
        for ordinal in range(2):
            c, p = arms['C'][ordinal]['acceptance'], arms['P'][ordinal]['acceptance']
            require(len(c) == len(p), 'C/P first-two-S tree count differs')
            for index, (a, b) in enumerate(zip(c, p)):
                require(all(k in a and k in b and a[k] == b[k] for k in ROUTE_FIELDS),
                    'saved pre-treatment C/P route fields differ at %s/%s' % (ordinal, index))
            trees = [tree_row(route, ordinal, index, threshold) for index, route in enumerate(c)]
            result['chunks'].append(dict(policy_ordinal=ordinal, **summarize(trees)))
            result['trees'].extend(trees)
        result['summary'] = summarize(result['trees'])
        result['status'] = 'DESCRIPTIVE_SAVED_PRETREATMENT_TWO_S_AVAILABLE'
        result['saved_C_P_route_fields_equal'] = True
        result['new_strict_numerical_certificate'] = False
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result['status'] = 'UNKNOWN_MISSING_OR_INCONSISTENT_EVIDENCE'
        result['issues'].append(repr(exc))
    return result


def write_new(directory, result):
    directory.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    require(len(raw) <= CAP, 'registered eight-MiB analysis cap exceeded')
    lines = ['# 处理前两段 S 的已保存接受代理诊断', '',
        '状态：'+result['status']+'。同一 task/init 的 C/P 只计一次，固定取 C；6 个描述单位，风险/控制各3。',
        '只描述已保存 score 与实际选择路径，不重算 velocity 范数，不拟合阈值或分类器，不声称根因、预测泛化或质量放行。', '',
        '|pair|task/init|分组|树数|depth计数|已接受边cost和|最大cost/τ|最小τ−cost|',
        '|---|---|---|---|---|---|---|---|']
    for row in result['pairs']:
        s = row['summary']
        values = [row['pair_id'], '%s/%s' % (row['task'], row['init']), row['stratum']]
        values += [s['tree_count'], s['accepted_depth_counts'], s['accepted_saved_score']['sum'],
            s['accepted_score_over_threshold']['max'], s['accepted_edge_margin']['min']] if isinstance(s, dict) else [UNKNOWN]*5
        lines.append('|'+ '|'.join(str(v) for v in values)+'|')
    lines += ['', '同任务风险−控制的描述差值见 JSON matched_task_differences；树/边不当作独立 episode。',
        'cost和只含 actual_depth 指示的已接受 draft 边；depth0无已接受边，sum=0、max/margin=UNKNOWN。',
        'margin=τ−saved_score；归一化 margin=1−saved_score/τ，正值表示离门限的已保存余量。',
        'sigma/next_sigma 是原树根记录；第二边独立 sigma 未保存则 UNKNOWN。bonus/fallback 的 g 不由此推断。',
        '若缺日志/配对组件失败，相关 pair 标 UNKNOWN，不能从可用子集得出完整 panel 结论。',
        'first2S 组件收据及保存字段对账不构成新的独立质量或全字段数值证书；四权限全部 false。', '']
    md = '\n'.join(lines).encode()
    require(len(raw)+len(md) <= CAP, 'combined analysis cap exceeded')
    # Exclusive outputs: neither old evidence nor a prior analysis is overwritten.
    with (directory/'pre-treatment-score-analysis.json').open('xb') as stream: stream.write(raw)
    with (directory/'pre-treatment-score-analysis.md').open('xb') as stream: stream.write(md)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='fresh local analysis result directory')
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    plan = read(job/'plan.json')
    require([(s['pair_id'], s['task'], s['init'], s['stratum']) for s in plan['pairs']] == FIXED,
        'six pre-registered task/init/strata differ')
    require(plan['frozen_permissions'] == PERMISSIONS, 'four frozen false permissions required')
    require(plan['threshold'] == 0.007839680526450805, 'frozen registered threshold differs')
    rows = [pair_row(job, spec, number(plan['threshold'])) for spec in plan['pairs']]
    differences = []
    for task in (1, 3, 4):
        risk, control = [next(r for r in rows if r['task'] == task and r['stratum'] == group)
            for group in ('risk', 'control')]
        diff = dict(task=task, risk_pair=risk['pair_id'], control_pair=control['pair_id'],
            status=UNKNOWN, values=UNKNOWN)
        if all(isinstance(r['summary'], dict) for r in (risk, control)):
            paths = [('accepted_saved_score', 'mean'), ('accepted_path_score_sum', 'mean'),
                ('accepted_score_over_threshold', 'max'), ('accepted_edge_margin', 'min')]
            diff['values'] = {a+'.'+b: risk['summary'][a][b]-control['summary'][a][b]
                if type(risk['summary'][a][b]) in (int, float) and type(control['summary'][a][b]) in (int, float)
                else UNKNOWN for a, b in paths}
            diff['status'] = 'DESCRIPTIVE_RISK_MINUS_CONTROL_ONLY'
        differences.append(diff)
    result = dict(schema='tf-pf2-pretreatment-saved-score-analysis/v1',
        created_at=datetime.now().astimezone().isoformat(), source_job=str(job),
        status='COMPLETE_DESCRIPTIVE_SIX_PAIR_ANALYSIS' if all(isinstance(r['summary'], dict) for r in rows)
            else 'PARTIAL_DESCRIPTIVE_EVIDENCE_UNKNOWN',
        chosen_arm_rule='always C; compare saved C/P first-two-S fields and inspect existing component receipts',
        unique_task_init_units=6, risk_units=3, control_units=3, pre_treatment_policy_ordinals=[0, 1],
        primary_unit='task/init; trees/edges are repeated correlated observations',
        threshold=plan['threshold'], threshold_fitted=False, classifier_trained=False,
        velocity_norm_recomputed=False, causal_or_predictive_claim=False,
        new_strict_numerical_certificate=False, performance_claim=False,
        permissions=PERMISSIONS, pairs=rows, matched_task_differences=differences,
        group_inference='UNKNOWN; no p-values, classification accuracy or generalization claim with three task-matched comparisons')
    write_new(args.out, result)


if __name__ == '__main__':
    main()

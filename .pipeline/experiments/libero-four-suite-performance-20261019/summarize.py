"""Audit all 6000 completed outcomes; retain exposure and timing limitations."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARMS = ('T', 'C', 'P')
SUITES = ('long', 'goal', 'spatial', 'object')


def load(path):
    return json.loads(path.read_text())


def number(value):
    return type(value) in (float, int) and math.isfinite(value)


def stats(values):
    values = sorted(values)
    def quantile(q):
        if not values:
            return None
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return dict(n=len(values), sum=sum(values), mean=sum(values)/len(values) if values else None,
                p50=quantile(.5), p90=quantile(.9), p95=quantile(.95), p99=quantile(.99),
                max=values[-1] if values else None, quantile_method='linear type7')


def normalized(suite, source):
    timing = source.get('timing', {})
    infer = source.get('client_infer_timing', {})
    gpu = source.get('GPU_identity', {})
    gpu = gpu if isinstance(gpu, dict) else {}
    infer = infer if isinstance(infer, dict) else {}
    timing = timing if isinstance(timing, dict) else {}
    return dict(suite=suite, case_id=source['case_id'], task=source['task'], init=source['init'],
        arm=source['arm'], success=source['success'], steps=source['actual_command_steps'],
        penalty=source['failure800_penalty'], chunks=source['policy_chunks'],
        episode_sec=source.get('client_episode_wall_sec'),
        policy_sec=timing.get('full_policy_elapsed_sec'), reset_sec=infer.get('reset_infer_wall_sec'),
        cache_sec=infer.get('cache_infer_wall_sec'), action_sec=timing.get('action_total_sec'),
        round_sec=timing.get('sum_T_round_sec'), g=timing.get('sum_g'),
        coverage=bool(timing.get('round_coverage_complete') and timing.get('action_measurement_complete')
                      and source.get('closed_and_drained')),
        gpu_uuid=gpu.get('uuid', '').removeprefix('GPU-'), gpu_name=gpu.get('name'),
        execution_source=source.get('source_job'), source=source['source'],
        strict_components_pass=source['strict_components_pass'], exposure=source['exposure'],
        complete=source['complete'], registered_timing_is_formal=False)


def get_rows():
    rows = []
    sources = []
    old = ROOT / '投机采样/results/runtime/p0-long-final-evidence-return-local-cpu1-v1-20261018-r1/remote-originals/report/EPISODES.csv'
    sources.append(str(old.relative_to(ROOT)))
    for r in csv.DictReader(old.open()):
        case = int(r['case_id'])
        def val(key):
            try:
                v = float(r[key])
                return v if math.isfinite(v) else None
            except (ValueError, KeyError):
                return None
        rows.append(dict(suite='long',case_id=case,task=case%10,init=case//10,
            arm=r['arm'],success=r['success']=='True',steps=int(r['env_steps']),
            penalty=int(r['failure800_penalty']),chunks=int(r['policy_chunks']),
            episode_sec=val('elapsed_sec'),policy_sec=val('policy_wall_sec'),
            reset_sec=val('reset_wall_sec'),cache_sec=val('cache_wall_sec'),
            action_sec=val('action_total_sec'),round_sec=val('sum_T_round_sec'),g=val('sum_g'),
            coverage=r['round_coverage_complete']=='True' and r['evidence_drained']=='True',
            gpu_uuid=r['GPU_UUID'].removeprefix('GPU-'),gpu_name=r['GPU_name'],
            execution_source=r['hardware_execution_source'],source=r['source'],
            strict_components_pass=r['audit_status']=='PASS_COMPLETE_EPISODE_COMPONENTS',
            exposure='FIT_CAL_DEV_EXPOSURE_NOT_INDEPENDENT_TEST',complete=True,
            registered_timing_is_formal=False))
    for suite in ('goal','spatial','object'):
        p = (OUT/'remote-evidence/OBJECT-REPORT.json' if suite=='object' else
             ROOT/f'.pipeline/experiments/libero-four-suite-morning-20261019/raw/{suite}/reports/final/RESULT.json')
        sources.append(str(p.relative_to(ROOT)))
        data=load(p)
        assert len(data['rows'])==1500 and all(x['complete'] for x in data['rows'])
        rows += [normalized(suite,r) for r in data['rows']]
    expected={(s,c,a) for s in SUITES for c in range(500) for a in ARMS}
    assert len(rows)==6000 and {(r['suite'],r['case_id'],r['arm']) for r in rows}==expected
    assert all(r['penalty']==(r['steps'] if r['success'] else 800) for r in rows)
    assert all(r['task']==r['case_id']%10 and r['init']==r['case_id']//10 for r in rows)
    return rows, sources


def cost(rows):
    good=[r for r in rows if r['coverage'] and all(number(r[k]) for k in
          ('episode_sec','policy_sec','action_sec','round_sec','g')) and r['g']>0]
    total=lambda key:sum(r[key] for r in good)
    result=dict(total_episodes=len(rows),covered_episodes=len(good),
        coverage_complete=len(good)==len(rows),scope='shared evaluation costs; no isolated performance claim')
    if good:
        result.update(episode_sum_sec=total('episode_sec'),policy_sum_sec=total('policy_sec'),
            action_sum_sec=total('action_sec'),round_sum_sec=total('round_sec'),sum_g=total('g'),
            round_sec_per_g=total('round_sec')/total('g'),action_sec_per_g=total('action_sec')/total('g'),
            policy_sec_per_chunk=total('policy_sec')/sum(r['chunks'] for r in good))
    return result


def ratios(pairs):
    out=dict(n=len(pairs),scope='same GPU identity only; concurrency/CPU/clock not matched; EXPLORATORY',
             formal_speedup='UNKNOWN')
    for key in ('episode_sec','policy_sec'):
        good=[(a,b) for a,b in pairs if number(a[key]) and number(b[key]) and a[key]>0 and b[key]>0]
        out[key]=dict(n=len(good),target_sum=sum(a[key] for a,b in good),
            candidate_sum=sum(b[key] for a,b in good))
        v=out[key]
        v['ratio_of_sums']=v['target_sum']/v['candidate_sum'] if v['candidate_sum'] else None
    a,b=cost([x for x,y in pairs]),cost([y for x,y in pairs])
    out['effective_update']=dict(target=a,candidate=b,
        ratio=(a['round_sec_per_g']/b['round_sec_per_g']
               if a.get('round_sec_per_g') and b.get('round_sec_per_g') else None))
    return out


def main():
    rows, sources=get_rows()
    results=dict(at_UTC=datetime.now(timezone.utc).isoformat(),sources=sources,
        complete_rows=6000,unique_task_init_arm_rows=6000,SUITES={},
        engineering_quality_screen=False,independent_test_complete=False,
        noninferiority_established=False,formal_performance_permission=False,
        timing_scope='reset-to-terminal warm episode excludes initial model/env construction; shared run',
        infrastructure_attempt_cost='Original partial attempts retained in original ledgers; not silently included in complete6000 sums')
    for suite in SUITES:
        ss=[r for r in rows if r['suite']==suite];index={(r['case_id'],r['arm']):r for r in ss}
        arms={}
        for arm in ARMS:
            aa=[r for r in ss if r['arm']==arm];success=[r['steps'] for r in aa if r['success']]
            arms[arm]=dict(N=500,successes=len(success),SR=len(success)/500,
                failures=500-len(success),all_actual_steps=stats([r['steps'] for r in aa]),
                success_only_steps=stats(success),failure800_penalty=stats([r['penalty'] for r in aa]),
                policy_chunks=sum(r['chunks'] for r in aa),cost=cost(aa),
                strict_component_pass=sum(r['strict_components_pass'] for r in aa),
                exposure=dict(Counter(r['exposure'] for r in aa)),
                GPU_counts=dict(Counter(r['gpu_name'] for r in aa)))
        comparisons={}
        for arm in ('C','P'):
            pairs=[(index[c,'T'],index[c,arm]) for c in range(500)]
            both=[(a,b) for a,b in pairs if a['success'] and b['success']]
            same=[(a,b) for a,b in pairs if a['gpu_uuid'] and a['gpu_uuid']==b['gpu_uuid']]
            groups=defaultdict(list)
            for a,b in same:groups[(a['gpu_name'],a['gpu_uuid'],a['execution_source'],b['execution_source'])].append((a,b))
            comparisons[arm]=dict(lost_success=sum(a['success'] and not b['success'] for a,b in pairs),
                rescued=sum(not a['success'] and b['success'] for a,b in pairs),both_success=len(both),
                both_failure=sum(not a['success'] and not b['success'] for a,b in pairs),
                SR_difference=arms[arm]['SR']-arms['T']['SR'],
                successful_length_ratio=stats([b['steps']/a['steps'] for a,b in both]),
                successful_length_ratio_over1p30=sum(b['steps']/a['steps']>1.30 for a,b in both),
                same_GPU_pairs=len(same),same_GPU_both_success_pairs=sum(a['success'] and b['success'] for a,b in same),
                all_same_GPU_timing=ratios(same) if len({a['gpu_name'] for a,b in same})==1 else
                    dict(status='NOT_POOLED_DIFFERENT_GPU_MODELS'),
                execution_strata=[dict(hardware=k[0],uuid=k[1],target_source=k[2],candidate_source=k[3],
                    all_complete=ratios(v),both_success=ratios([(a,b) for a,b in v if a['success'] and b['success']]))
                    for k,v in groups.items()])
        results['SUITES'][suite]=dict(by_arm=arms,comparisons=comparisons,
            task_success={str(t):{a:sum(r['success'] for r in ss if r['task']==t and r['arm']==a)
                                      for a in ARMS} for t in range(10)})
    (OUT/'RESULT.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'EPISODES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['# 四套LIBERO三臂完整质量与现有成本','',
      '6000完整episode，四套各500组、各臂500。T=Target，C=连续TF，P=TF+PF2。只复用已完成终态，无新增GPU/机器人任务。',
      '', '|套件|Target成功率|TF成功率|TF+PF2成功率|','|---|---:|---:|---:|']
    for s,d in results['SUITES'].items():
        a=d['by_arm'];lines.append(f"|{s}|{a['T']['successes']}/500 ({a['T']['SR']:.1%})|{a['C']['successes']}/500 ({a['C']['SR']:.1%})|{a['P']['successes']}/500 ({a['P']['SR']:.1%})|")
    lines+=['','|套件|方法|双方成功N|长度比P50/P90/max|>1.30|丢失/救回|',
            '|---|---|---:|---|---:|---|']
    for s,d in results['SUITES'].items():
        for a,c in d['comparisons'].items():
            q=c['successful_length_ratio'];lines.append(f"|{s}|{a}/T|{q['n']}|{q['p50']:.4f}/{q['p90']:.4f}/{q['max']:.4f}|{c['successful_length_ratio_over1p30']}|{c['lost_success']}/{c['rescued']}|")
    lines+=['','长度type7，仅双方成功作为补充；全体成功率、实际步数及失败800罚分在RESULT.json中完整保留。',
      '', '|套件|臂|实际命令均值|成功局命令均值|失败800罚分均值|', '|---|---|---:|---:|---:|']
    for s,d in results['SUITES'].items():
        for a,v in d['by_arm'].items():lines.append(f"|{s}|{a}|{v['all_actual_steps']['mean']:.3f}|{v['success_only_steps']['mean']:.3f}|{v['failure800_penalty']['mean']:.3f}|")
    lines+=['','## 耗时边界与正式性能缺口','',
      '现有耗时来自并行质量运行，不是独占benchmark。每个case双方GPU UUID匹配不保证worker并发数、CPU、时钟和运行批次相同；不同执行来源分层。Long含PCIe与SXM，禁止合成一个正式速度数。',
      'RESULT.json保存每源的全体完整局、双方成功局的ΣTarget时间/Σcandidate时间，以及(ΣT_round/Σg)的比值。均为EXPLORATORY；不是target-call reduction，也不是正式推理或任务E2E加速。',
      '旧partial/准备/失败启动的成本保持原账本，complete6000时间和式不冒充所有资源成本；初次模型/环境构建排除，不能称cold E2E。',
      '四项旧科学字段仍false。独立TEST、非劣、严格首两S配对及组件FAILED未由本CPU统计重新签证。正式TF入口沿登记门处理，不能把重命名作业作为放行。',
      '', '下一步按READINESS.md核验完整可恢复输入与独占计时依赖；没有新性能结果时标UNKNOWN。']
    (OUT/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print('AUDITED 6000 unique complete episodes; no formal speedup inferred')


if __name__=='__main__':main()

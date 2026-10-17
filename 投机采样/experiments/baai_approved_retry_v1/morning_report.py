"""CPU-only 09:15 evidence snapshot, never an evaluation or performance run."""
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
import csv
import json
import math
import os
import signal
import time
import traceback

ARMS = ('T', 'C', 'P')
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
                   noninferiority_established=False, formal_performance_permission=False)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def quantiles(values):
    values = sorted(values)
    def at(q):
        x = (len(values) - 1) * q
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return dict(n=len(values), p50=at(.5), p90=at(.9), max=values[-1],
                algorithm='linear interpolation type7') if values else dict(n=0, p50=None, p90=None, max=None)


def build(registration, out):
    normal = Path(registration['normal_job'])
    p = read(normal / 'plan.json')
    roots = [Path(r['physical_source_job']) for r in p['historical_source_roots']] + [normal]
    retry = Path(registration['retry_job'])
    if retry.exists():
        roots.append(retry)
    unique = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    raw = []
    outcomes = {}
    for root in unique:
        for f in sorted(root.glob('episodes/case-*/*/EPISODE-TERMINAL.json')):
            t = read(f)
            row = dict(case_id=t['case_id'], arm=t['arm'], source=str(f),
                status=t['status'], complete_episode=t.get('complete_episode') is True,
                success=t.get('success'), env_steps=t.get('env_steps'),
                failure800_penalty=t.get('failure800_penalty'), elapsed_sec=t.get('elapsed_sec'),
                policy_chunks=t.get('policy_chunks'), termination_reason=t.get('termination_reason'),
                evidence_drained=(f.parent / 'FINISH_READY.json').exists(),
                audit_status=read(f.parent / 'AUDIT.json').get('status')
                             if (f.parent / 'AUDIT.json').exists() else 'PENDING_OR_UNKNOWN')
            timing = t.get('client_infer_timing', {})
            row['policy_wall_sec'] = timing.get('policy_infer_wall_sec')
            row['reset_wall_sec'] = timing.get('reset_infer_wall_sec')
            row['cache_wall_sec'] = timing.get('cache_infer_wall_sec')
            raw.append(row)
            if row['complete_episode']:
                assert type(row['success']) is bool and t['horizon'] == 800
                assert row['env_steps'] > 0 and (row['success'] or t['horizon_reached'])
                assert row['failure800_penalty'] == (row['env_steps'] if row['success'] else 800)
                key = (row['case_id'], row['arm'])
                assert key not in outcomes, 'duplicate complete outcome; do not select a better attempt'
                outcomes[key] = row
    common = sorted(cid for cid in range(500) if all((cid, arm) in outcomes for arm in ARMS))
    def group(arm, ids):
        rows = [outcomes[cid, arm] for cid in ids if (cid, arm) in outcomes]
        return dict(n=len(rows), successes=sum(r['success'] for r in rows),
            failures=sum(not r['success'] for r in rows),
            success_rate=sum(r['success'] for r in rows) / len(rows) if rows else None,
            actual_commands=sum(r['env_steps'] for r in rows),
            failure800_penalty_sum=sum(r['failure800_penalty'] for r in rows),
            failure800_penalty_mean=sum(r['failure800_penalty'] for r in rows) / len(rows) if rows else None,
            complete_episode_elapsed=quantiles([r['elapsed_sec'] for r in rows if isinstance(r['elapsed_sec'], (int, float))]),
            policy_chunks=sum(r['policy_chunks'] for r in rows),
            policy_wall_sec=sum(r['policy_wall_sec'] for r in rows if isinstance(r['policy_wall_sec'], (int, float))),
            undrained_at_snapshot=sum(not r['evidence_drained'] for r in rows),
            audit_status_counts=dict(Counter(r['audit_status'] for r in rows)))
    def paired(first, second):
        rows = [(outcomes[cid, first], outcomes[cid, second]) for cid in common]
        return dict(n=len(rows), lost=sum(a['success'] and not b['success'] for a,b in rows),
            rescued=sum(not a['success'] and b['success'] for a,b in rows),
            both_success=sum(a['success'] and b['success'] for a,b in rows),
            both_failure=sum(not a['success'] and not b['success'] for a,b in rows),
            success_rate_difference_second_minus_first=sum(int(b['success'])-int(a['success']) for a,b in rows)/len(rows) if rows else None,
            both_success_step_ratio=quantiles([b['env_steps']/a['env_steps'] for a,b in rows if a['success'] and b['success']]),
            strict_initial_pair_requalification=False)
    all_summary = {arm: group(arm, range(500)) for arm in ARMS}
    common_summary = {arm: group(arm, common) for arm in ARMS}
    pairs = {a+'_to_'+b: paired(a,b) for a,b in [('T','C'),('T','P'),('C','P')]}
    pending = [dict(case_id=cid, task=cid%10, init=cid//10,
                    missing_arms=[arm for arm in ARMS if (cid,arm) not in outcomes])
               for cid in range(500) if cid not in common]
    for cid in common:
        assert p['cases'][cid]['task'] == cid%10 and p['cases'][cid]['init'] == cid//10
    result = dict(schema='long-first-round-morning-evidence/v1',
        status='COMPLETE_DESCRIPTIVE_500_CASE_PANEL' if len(common)==500 else 'PARTIAL_DESCRIPTIVE_PANEL',
        observed_at=datetime.now(timezone.utc).isoformat(), benchmark='libero_10', round=1,
        evaluation_seed=p['evaluation_seed'], planned_cases=500, planned_episodes=1500,
        complete_common_cases=len(common), complete_common_episodes=3*len(common),
        complete_raw_episode_count=len(outcomes), partial_attempts=sum(not r['complete_episode'] for r in raw),
        common_case_ids=common, per_arm_common_panel=common_summary, per_arm_all_complete=all_summary,
        paired_outcomes=pairs,
        task_strata={str(task):{arm:group(arm,[cid for cid in common if cid%10==task]) for arm in ARMS} for task in range(10)},
        missing_cases=pending, source_roots=[str(root) for root in unique],
        original_absolute_work_stop=p['absolute_work_stop'], original_cleanup=p['absolute_cleanup_deadline'],
        no_timing_claim=True, formal_E2E_speedup='UNKNOWN_NOT_MEASURED_UNDER_EXCLUSIVE_REGISTERED_CONDITIONS',
        caveats=['partial common panel may not be task balanced',
                 'full 50-init panel includes training/dev exposure; not independent TEST',
                 'cross-hardware and shared-resource wall times cannot be pooled as formal speedup',
                 'natural failures retained; infrastructure partials separate; no automatic PASS',
                 'three-round means/standard deviations unavailable because only first round is authorized'],
        permissions=PERMISSIONS)
    out.mkdir(exist_ok=False)
    write(out/'RESULT.json',result)
    write(out/'SOURCE-MANIFEST.json',dict(roots=result['source_roots'],attempts=raw,
          no_SHA=True,old_originals_rewritten=False,permissions=PERMISSIONS))
    fields=['case_id','arm','success','env_steps','failure800_penalty','policy_chunks','elapsed_sec',
            'policy_wall_sec','reset_wall_sec','cache_wall_sec','evidence_drained','audit_status','source']
    with (out/'EPISODES.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for key in sorted(outcomes):writer.writerow(outcomes[key])
    lines=['# Long10 第一轮晨间汇总','',
        f"状态：{result['status']}；共同覆盖 {len(common)}/500 组，{3*len(common)}/1500 局。",'',
        '| 方法 | 完整局 | 成功 | 失败 | 成功率 | 实际命令总数 | 800罚分均值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r=common_summary[arm];rate=f"{100*r['success_rate']:.3f}%" if r['success_rate'] is not None else 'UNKNOWN'
        lines.append(f"|{arm}|{r['n']}|{r['successes']}|{r['failures']}|{rate}|{r['actual_commands']}|{r['failure800_penalty_mean']}|")
    lines += ['', 'T=Target；C=冻结TF_ON连续投机；P=同权重PF2（SST）。全部使用同等512-token BF16 reset bank。','',
        '| 配对：前者→后者 | 丢失成功 | 救回 | 双方成功 | 双方失败 | 成功子集长度P50/P90/max |',
        '|---|---:|---:|---:|---:|---|']
    for name,r in pairs.items():
        q=r['both_success_step_ratio'];lines.append(f"|{name}|{r['lost']}|{r['rescued']}|{r['both_success']}|{r['both_failure']}|{q['p50']}/{q['p90']}/{q['max']}|")
    lines += ['', '正确完整执行但失败的局均在分母及800罚分中；基础设施PARTIAL另列。未结束、未开始及未drain不伪装完成。',
        '成功判据来自原LIBERO环境 step 的 _check_success；horizon以env.env.timestep的800为准，实际动作数单列。',
        '本表是同case面板描述，未重新签发跨旧新硬件的逐值/严格初态证书。原组件FAILED保留。',
        '正式E2E加速比：UNKNOWN。共享并发耗时按硬件/并发层另列，不以调用数替代速度。',
        '只有第一轮结果；三轮均值/标准差尚不可报告。含训练/DEV暴露，四项权限继续false。','',
        '原始逐局来源见EPISODES.csv和SOURCE-MANIFEST.json；缺项、每任务分层及探索性耗时见RESULT.json。','']
    (out/'LONG-RESULT.md').write_text('\n'.join(lines))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--immediate',action='store_true')
    args=parser.parse_args();job=args.job.resolve(strict=True)
    registration=read(job/'REGISTRATION.json');workspace=(Path.home()/'cxz').resolve(strict=True)
    assert job.is_relative_to(workspace)
    os.sched_setaffinity(0,{43})
    target=datetime.fromisoformat(registration['report_at']).timestamp()
    assert target<datetime.fromisoformat(registration['cleanup_at']).timestamp()
    interrupted=False
    def stop(*_):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    write(job/'STARTED.json',dict(pid=os.getpid(),at=time.time(),CPU=[43],GPU=0))
    try:
        while not args.immediate and time.time()<target:
            assert not interrupted and not (job/'STOP_REQUESTED').exists()
            time.sleep(min(30,target-time.time()))
        assert time.time()<datetime.fromisoformat(registration['cleanup_at']).timestamp()
        result=build(registration,job/'report')
        write(job/'TERMINAL.json',dict(status='CPU_REPORT_COMPLETE',report_status=result['status'],
              complete_common_cases=result['complete_common_cases'],at=time.time(),permissions=PERMISSIONS))
    except BaseException as exc:
        write(job/'TERMINAL.json',dict(status='REPORT_FAILED_OR_STOPPED',error=repr(exc),
              traceback=traceback.format_exc(),at=time.time(),permissions=PERMISSIONS))
        raise
    finally:
        write(job/'CLEANUP.json',dict(status='CPU_ONLY_REPORTER_EXITS',GPU_processes_started=0,
              model_tests=0,SHA_jobs=0,at=time.time()))


if __name__=='__main__':
    main()

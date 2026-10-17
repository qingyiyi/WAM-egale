"""Independent descriptive report v2; frozen producer/report-v1 artifacts stay unchanged.

Expected coverage is 500 registered case IDs per arm. Complete robot outcomes
remain separate from component closure, incomplete attempts and unstarted rows.
No torch, CUDA, model calls, digest, tests, implicit reruns or quality certificate.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import time

FALSE=dict(engineering_quality_screen=False,independent_test_complete=False,
    noninferiority_established=False,formal_performance_permission=False)
UNKNOWN='UNKNOWN'

def read(path):return json.loads(Path(path).read_text())
def finite(x):return type(x) in (int,float) and math.isfinite(x) and x>=0
def stats(values):
    values=sorted(values)
    def q(p):
        if not values:return UNKNOWN
        x=(len(values)-1)*p;a=math.floor(x);b=math.ceil(x)
        return values[a]+(values[b]-values[a])*(x-a)
    return dict(n=len(values),sum=sum(values),mean=sum(values)/len(values) if values else UNKNOWN,
        p50=q(.5),p90=q(.9),max=values[-1] if values else UNKNOWN,quantile_method='linear index=(N-1)*q')
def safe(path):
    try:return read(path),None
    except (OSError,ValueError,TypeError) as e:return None,type(e).__name__

def exposure(init):
    return ('TARGET_FIT' if init<=17 else 'TARGET_CAL' if init<=19 else
        'PARENT_FIT' if init<=21 else 'PARENT_CAL' if init==22 else
        'CURRENT_FIT' if init<=24 else 'CURRENT_CAL' if init==25 else
        'UNSEEN_BY_NEW_SUITE_TRAINING_NOT_CERTIFIED_TEST')


def row(plan,spec):
    out=Path((spec.get('reuse_source') or {}).get('out',spec['out']))
    source_job=Path((spec.get('reuse_source') or {}).get('job',plan['job']))
    attempted=out.exists()
    docs={n:safe(out/n) for n in ('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json')}
    r=dict(suite=plan['suite'],case_id=spec['case_id'],task=spec['task'],init=spec['init'],arm=spec['arm'],
        source=str(out),source_job=str(source_job),reused_T_POOL=bool(spec.get('reuse_source')),
        exposure=spec.get('data_exposure',exposure(spec['init'])),
        registered_phase_plan_available=plan.get('_registered_phase_plan_available',True),
        complete=False,attempted=attempted,episode_status='PARTIAL_NO_COMPLETE_TERMINAL' if attempted else 'NOT_STARTED',
        success=UNKNOWN,actual_command_steps=UNKNOWN,failure800_penalty=UNKNOWN,policy_chunks=UNKNOWN,
        target_forward_attempts=UNKNOWN,target_forward_returned=UNKNOWN,target_forward_failed=UNKNOWN,
        cost_scope='instrumented shared quality run; exploratory, not formal E2E',
        errors=[dict(file=k,error=e) for k,(_,e) in docs.items() if e and (out/k).exists()],permissions=FALSE)
    t,c,a,f=[docs[n][0] for n in docs]
    if t is not None:
        r['attempted']=True
        try:
            assert t['schema']=='tf-suite-full-episode-terminal/v1' and t['suite']==plan['suite']
            assert t['case_id']==spec['case_id'] and t['arm']==spec['arm'] and t['permissions']==FALSE
            assert t['task']==spec['task'] and t['init']==spec['init']
            r['source_row_id']=t.get('row_id',UNKNOWN)
            if t['complete_episode'] is True:
                assert t['horizon']==800 and type(t['success']) is bool and (t['success'] or t['horizon_reached'] is True)
                assert type(t['env_steps']) is int and 0<t['env_steps']<=800
                assert t['failure800_penalty']==(t['env_steps'] if t['success'] else 800)
                r.update(complete=True,episode_status='COMPLETE_RECORDED_FULL_EPISODE',success=t['success'],
                    actual_command_steps=t['env_steps'],failure800_penalty=t['failure800_penalty'],
                    policy_chunks=t['policy_chunks'],client_episode_wall_sec=t['elapsed_sec'],
                    client_total_wall_sec=t['client_total_wall_sec'],client_infer_timing=t.get('client_infer_timing',UNKNOWN),
                    original_elapsed_boundary=t.get('original_elapsed_boundary',UNKNOWN),
                    original_termination_reason=t.get('termination_reason',UNKNOWN))
            else:
                r.update(episode_status=t.get('status','PARTIAL_CLIENT_EXCEPTION'),partial_not_algorithm_failure=True,
                    partial_error=t.get('error',UNKNOWN),partial_traceback=t.get('traceback',UNKNOWN),
                    partial_raw_artifact_error=t.get('raw_artifact_error'),
                    actual_command_steps=t.get('env_steps',UNKNOWN),
                    recorded_robot_outcome_if_already_returned=t.get('recorded_robot_outcome_if_already_returned'),
                    client_total_wall_sec=t.get('client_total_wall_sec',UNKNOWN))
        except (AssertionError,KeyError,TypeError) as e:
            r['errors'].append(dict(file='EPISODE-TERMINAL.json',error=repr(e)))
            r.update(episode_status='PARTIAL_INVALID_CLIENT_TERMINAL',partial_not_algorithm_failure=True,
                partial_error=t.get('error',repr(e)) if isinstance(t,dict) else repr(e),
                raw_terminal_status=t.get('status',UNKNOWN) if isinstance(t,dict) else UNKNOWN)
    if c is not None:
        r['attempted']=True
        try:
            assert c['schema']=='tf-suite-episode-counter/v1' and c['case_id']==spec['case_id']
            assert c['suite']==plan['suite'] and c['arm']==spec['arm'] and c['permissions']==FALSE
            for key in ('target_forward_attempts','target_forward_returned','target_forward_failed'):
                assert type(c[key]) is int and 0<=c[key]<=4000
            assert c['target_forward_returned']+c['target_forward_failed']<=c['target_forward_attempts']
            r.update(target_forward_attempts=c['target_forward_attempts'],
                target_forward_returned=c['target_forward_returned'],target_forward_failed=c['target_forward_failed'],
                forward_receipt_scope='actual COUNTER receipt; finality requires its complete/drained closure')
            r['timing']=c.get('timing',UNKNOWN)
            r['counters']=dict(phase_counters=c.get('phase_counters',UNKNOWN),
                typed_forward_calls=c.get('typed_forward_calls',UNKNOWN),total_actual_g=c.get('total_actual_g',UNKNOWN))
            r['GPU_identity']=next((q['GPU_identity'] for q in c.get('requests',[])
                if isinstance(q,dict) and isinstance(q.get('GPU_identity'),dict)),UNKNOWN)
            r['GPU_identity_scope']='first actual request; remaining request/device observations stay in original COUNTER'
            r['routes']=dict(source=str(out/'COUNTER.json'),field='policy_records[*].acceptance',
                policy_record_count=len(c.get('policy_records',[])))
        except (AssertionError,KeyError,TypeError) as e:r['errors'].append(dict(file='COUNTER.json',error=repr(e)))
    valid_audit=bool(isinstance(a,dict) and a.get('schema')=='tf-suite-episode-audit/v1'
        and a.get('case_id')==spec['case_id'] and a.get('arm')==spec['arm'] and a.get('permissions')==FALSE)
    valid_finish=bool(isinstance(f,dict) and f.get('case_id')==spec['case_id']
        and f.get('arm')==spec['arm'] and f.get('permissions')==FALSE)
    r['component_status']=a.get('status',UNKNOWN) if isinstance(a,dict) else UNKNOWN
    r['finish_status']=f.get('status',UNKNOWN) if isinstance(f,dict) else UNKNOWN
    r['closed_and_drained']=bool(valid_finish and f.get('full_episode_terminal') is True and f.get('async_drained') is True)
    r['strict_components_pass']=bool(valid_audit and r['closed_and_drained']
        and a.get('status')=='PASS_COMPLETE_EPISODE_COMPONENTS' and r['finish_status']=='PASS')
    r['strict_pair_qualification']='UNKNOWN_NOT_RECOMPUTED_BY_THIS_DESCRIPTIVE_REPORT'
    if not r['registered_phase_plan_available'] and r['complete']:
        r['unregistered_robot_outcome_observation']=dict(success=r['success'],
            actual_command_steps=r['actual_command_steps'],failure800_penalty=r['failure800_penalty'])
        r.update(complete=False,success=UNKNOWN,failure800_penalty=UNKNOWN,
            episode_status='PARTIAL_UNREGISTERED_PHASE_MANIFEST',
            partial_error='actual terminal exists but its registered phase manifest is absent')
    if not r['complete'] and r['attempted']:
        r['partial_not_algorithm_failure']=True
        r['partial_exception_receipts']=[]
        related=[out/'SERVER-REQUEST-FAILED.json']
        source_row=spec['case_id'] if spec.get('reuse_source') else spec.get('row_id')
        if type(source_row) is int:
            related.append(source_job/f'ROW-{source_row:04d}-RESULT.json')
        for receipt in related:
            original,receipt_error=safe(receipt)
            if isinstance(original,dict) and original.get('error') is not None:
                r['partial_exception_receipts'].append(dict(source=str(receipt),
                    status=original.get('status',UNKNOWN),error=original['error'],
                    traceback=original.get('traceback',UNKNOWN)))
                if not r.get('partial_error') or r['partial_error']==UNKNOWN:
                    r['partial_error']=original['error']
        r.setdefault('partial_error',UNKNOWN)
        r['partial_original_stderr_source']=str(out/'client.log') if (out/'client.log').is_file() else UNKNOWN
        r['episode_status']='PARTIAL_NO_COMPLETE_TERMINAL' if r['episode_status']=='NOT_STARTED' else r['episode_status']
        r['episode_attempt_forward_total']=r['target_forward_attempts']
        r['attempt_forward_finality']='UNKNOWN_UNLESS_FINAL_CLOSED_RECEIPT_PROVES_COMPLETE_ATTEMPT'
        if not finite(r['target_forward_attempts']):
            r['episode_attempt_forward_total']=UNKNOWN
            r['registered_per_episode_attempt_upper_bound']=4000
            # A worker pulse is cumulative, not an exact episode counter.
            pulse,pulse_error=safe(source_job/'server-worker0/LIVE-COUNTER.json')
            ep=pulse.get('episode') if isinstance(pulse,dict) else None
            if isinstance(ep,dict) and ep.get('out')==str(out):
                r['partial_worker_pulse']=dict(source=str(source_job/'server-worker0/LIVE-COUNTER.json'),
                    cumulative_worker_attempts=pulse.get('target_forward_attempts',UNKNOWN),
                    episode_attempts=UNKNOWN,reason='cumulative pulse cannot supply missing episode start counter')
    elif not r['attempted']:
        r['episode_status']='NOT_STARTED'
    return r


def aggregate(rows):
    expected=500
    complete=[r for r in rows if r['complete']]
    partial=[r for r in rows if not r['complete'] and r['attempted']]
    not_started=[r for r in rows if not r['complete'] and not r['attempted']]
    good_cost=[r for r in complete if isinstance(r.get('timing'),dict) and r['closed_and_drained']
        and r['timing'].get('round_coverage_complete') is True
        and r['timing'].get('action_measurement_complete') is True
        and all(finite(r['timing'].get(k)) for k in
            ('action_total_sec','action_g','full_policy_elapsed_sec','sum_T_round_sec','sum_g'))
        and r['timing']['action_g']>0 and r['timing']['sum_g']>0
        and type(r['policy_chunks']) is int and r['policy_chunks']>0]
    def sumfield(k):return sum(r['timing'][k] for r in good_cost)
    coverage_complete=len(rows)==expected and len(complete)==expected and len(good_cost)==expected
    g=sumfield('action_g');rg=sumfield('sum_g');chunks=sum(r['policy_chunks'] for r in good_cost)
    known_forward=[r for r in rows if type(r['target_forward_attempts']) is int]
    all_forward_exact=coverage_complete and all(r['target_forward_returned']+r['target_forward_failed']
        ==r['target_forward_attempts'] for r in rows)
    return dict(expected_complete_episodes=expected,planned_rows=expected,manifest_or_expected_rows=len(rows),
        complete=len(complete),partial=len(partial),NOT_STARTED=len(not_started),
        missing_or_partial=expected-len(complete),successes=sum(r['success'] for r in complete),
        success_rate=sum(r['success'] for r in complete)/expected if len(complete)==expected else UNKNOWN,
        observed_complete_subset_success_rate=sum(r['success'] for r in complete)/len(complete) if complete else UNKNOWN,
        observed_subset_is_not_full_suite_SR=len(complete)!=expected,
        raw_success_is_not_strict_certificate=True,strict_component_complete=sum(r['strict_components_pass'] for r in complete),
        actual_steps=stats([r['actual_command_steps'] for r in complete]),
        all_complete_failure800_penalty=stats([r['failure800_penalty'] for r in complete]),
        partial_actual_steps_observed=stats([r['actual_command_steps'] for r in partial if finite(r.get('actual_command_steps'))]),
        attempted_partial_count=len(partial),attempt_forward_receipts_observed=len(known_forward),
        observed_attempt_forward_subtotal=sum(r['target_forward_attempts'] for r in known_forward),
        observed_attempt_forward_subtotal_is_not_an_exact_full_panel_total=not all_forward_exact,
        attempt_forward_unknown_row_ids=[dict(case_id=r['case_id'],arm=r['arm']) for r in rows
            if r['attempted'] and type(r['target_forward_attempts']) is not int],
        total_attempt_forward_exact=sum(r['target_forward_attempts'] for r in rows) if all_forward_exact else UNKNOWN,
        cost_covered_complete=len(good_cost),cost_coverage_complete=coverage_complete,
        cost_gap_case_ids=[r['case_id'] for r in rows if r not in good_cost],
        sum_action_sec=sumfield('action_total_sec') if coverage_complete else UNKNOWN,
        sum_action_g=g if coverage_complete else UNKNOWN,
        C_update=sumfield('action_total_sec')/g if coverage_complete else UNKNOWN,
        sum_T_round_sec=sumfield('sum_T_round_sec') if coverage_complete else UNKNOWN,
        sum_g=rg if coverage_complete else UNKNOWN,
        cost_per_legal_round_commit=sumfield('sum_T_round_sec')/rg if coverage_complete else UNKNOWN,
        C_chunk=sumfield('full_policy_elapsed_sec')/chunks if coverage_complete else UNKNOWN,
        observed_complete_cost_subtotal=dict(complete_episodes=len(good_cost),
            sum_action_sec=sumfield('action_total_sec') if good_cost else UNKNOWN,
            sum_action_g=g if good_cost else UNKNOWN,
            sum_T_round_sec=sumfield('sum_T_round_sec') if good_cost else UNKNOWN,
            sum_g=rg if good_cost else UNKNOWN,
            full_policy_elapsed_sec=sumfield('full_policy_elapsed_sec') if good_cost else UNKNOWN,
            policy_chunks=chunks,
            scope='only rows with original complete cost coverage; not total panel cost, not speedup'),
        all_complete_client_wall=stats([r['client_episode_wall_sec'] for r in complete if finite(r.get('client_episode_wall_sec'))]),
        formal_E2E_speedup=UNKNOWN,formal_performance_permission=False)


def comparison(rows,left,right):
    idx={(r['case_id'],r['arm']):r for r in rows if r['complete']}
    pairs=[(idx[(c,left)],idx[(c,right)]) for c in range(500) if (c,left) in idx and (c,right) in idx]
    both=[(a,b) for a,b in pairs if a['success'] and b['success']]
    return dict(reference=left,candidate=right,complete_pairs=len(pairs),
        rescued=sum(not a['success'] and b['success'] for a,b in pairs),
        lost_success=sum(a['success'] and not b['success'] for a,b in pairs),
        both_success=len(both),both_failure=sum(not a['success'] and not b['success'] for a,b in pairs),
        paired_SR_difference=sum(int(b['success'])-int(a['success']) for a,b in pairs)/len(pairs) if pairs else UNKNOWN,
        task_weighted_SR_difference=UNKNOWN,paired_interval=UNKNOWN,
        common_success_steps_ratio=stats([b['actual_command_steps']/a['actual_command_steps'] for a,b in both]),
        paired_failure800_penalty_difference=stats_signed([b['failure800_penalty']-a['failure800_penalty'] for a,b in pairs]),
        exploratory_common_success_client_time_ratio=stats([b['client_episode_wall_sec']/a['client_episode_wall_sec'] for a,b in both
            if finite(a.get('client_episode_wall_sec')) and a['client_episode_wall_sec']>0 and finite(b.get('client_episode_wall_sec'))]),
        formal_E2E_speedup=UNKNOWN,independent_test_complete=False)

def stats_signed(v):return dict(n=len(v),sum=sum(v),mean=sum(v)/len(v) if v else UNKNOWN)

def expected_rows(campaign,registration):
    """Materialize the approved fixed 500x3 expectation, never new execution."""
    full='libero_'+registration['suite'].removeprefix('libero_')
    index,plans={},[]
    for name in ('FINAL_0','FINAL_1'):
        file=campaign/'stages'/name/'plan.json'
        if not file.exists():continue
        p=read(file)
        assert p['schema']=='tf-suite-pipeline/v1' and p['phase']=='FINAL_EVAL' and p['suite']==full
        assert p['frozen_permissions']==FALSE
        plans.append(str(file))
        for spec in p['rows']:
            c,a=spec['case_id'],spec['arm']
            assert type(c) is int and 0<=c<500 and a in ('T','C','P')
            assert spec['task']==c%10 and spec['init']==c//10 and (c,a) not in index
            index[c,a]=(p,spec)
    pool_file=campaign/'stages/T_POOL/plan.json'
    if pool_file.exists():
        p=read(pool_file)
        assert p['schema']=='tf-suite-pipeline/v1' and p['phase']=='T_POOL' and p['suite']==full
        assert p['frozen_permissions']==FALSE
        plans.append(str(pool_file))
        for spec in p['rows']:
            c=spec['case_id']
            assert type(c) is int and 0<=c<200 and spec['arm']=='T'
            assert spec['task']==c%10 and spec['init']==c//10
            if (c,'T') not in index:
                effective=dict(spec,reuse_source=dict(job=p['job'],out=spec['out']))
                index[c,'T']=(p,effective)
    values=[]
    for c in range(500):
        for a in ('T','C','P'):
            if (c,a) in index:
                plan,spec=index[c,a]
            else:
                base=campaign/'stages'/('FINAL_0' if c<250 else 'FINAL_1')
                plan=dict(job=str(base),suite=full,phase='FINAL_EVAL',_registered_phase_plan_available=False)
                spec=dict(case_id=c,task=c%10,init=c//10,arm=a,
                    out=str(base/'episodes'/f'case-{c:04d}'/a),data_exposure=exposure(c//10))
            values.append(row(plan,spec))
    assert len(values)==1500 and len({(r['case_id'],r['arm']) for r in values})==1500
    return values,plans


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--campaign',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    x=a.parse_args();start=time.time();x.out.mkdir(parents=True,exist_ok=False)
    try:
        p=read(x.campaign/'registration.json')
        assert p['schema']=='tf-suite-campaign/v1' and p['permissions']==FALSE
        rows,plans=expected_rows(x.campaign,p)
        result=dict(schema='tf-suite-descriptive-report/v2',
            status='COMPLETE_RAW_DESCRIPTIVE_COVERAGE' if all(r['complete'] for r in rows) else 'PARTIAL_DESCRIPTIVE_COVERAGE',
            expected_case_ids=500,expected_complete_episodes_per_arm=500,expected_rows=1500,
            suite=p['suite'],source_producer_commit=p.get('source_code_commit',UNKNOWN),
            report_version='report_v2.py; independent of frozen producer and original report-v1',source_plans=plans,
            by_arm={a:aggregate([r for r in rows if r['arm']==a]) for a in ('T','C','P')},
            unseen_training_init26to49={a:dict(expected_episodes=240,
                complete=sum(r['complete'] for r in rows if r['arm']==a and r['init']>=26),
                rows=[dict(case_id=r['case_id'],complete=r['complete'],episode_status=r['episode_status'],
                    success=r['success'],actual_command_steps=r['actual_command_steps'],failure800_penalty=r['failure800_penalty'])
                    for r in rows if r['arm']==a and r['init']>=26],
                not_an_independent_TEST=True) for a in ('T','C','P')},
            comparisons=[comparison(rows,l,r) for l,r in [('T','C'),('T','P'),('C','P')]],rows=rows,
            CPU_report_compute_wall_sec=time.time()-start,permissions=FALSE,noninferiority='NOT_ESTABLISHED',
            strict_initial_physics_camera_and_RNG_pairing='UNKNOWN_NOT_RECOMPUTED; original evidence remains authoritative',
            formal_E2E_speedup='UNKNOWN_REQUIRES_REGISTERED_QUALITY_AND_ISOLATED_COMPLETE_EPISODE_MEASUREMENT',
            raw_component_failures_retained=True,
            partial_attempt_cost='observed COUNTER attempts retained; missing attempts UNKNOWN; no replacement by zero',
            cost_rule='main arm cost numeric only for all500 complete episodes with complete original covered/drained costs',
            cold_start='see bank producers and per-worker MODEL-SETUP-OBSERVATION; excluded from client episode timer')
        with (x.out/'RESULT.json').open('x') as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        fields=['suite','case_id','task','init','arm','exposure','source','reused_T_POOL','episode_status',
            'attempted','complete','success','actual_command_steps','failure800_penalty','policy_chunks',
            'client_episode_wall_sec','target_forward_attempts','target_forward_returned','target_forward_failed',
            'partial_error','component_status','finish_status','strict_components_pass']
        with (x.out/'EPISODES.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
        text=['# 套件三臂完整任务描述汇总 v2','',
            f"状态：{result['status']}。每臂期望500局。四项权限保持false；含训练与校准来源，不是独立TEST。",'',
            '|臂|完整/500|partial|未启动|成功|实际完整命令总数|完整局800罚分均值|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for arm,v in result['by_arm'].items():
            text.append(f"|{arm}|{v['complete']}/500|{v['partial']}|{v['NOT_STARTED']}|{v['successes']}|{v['actual_steps']['sum']}|{v['all_complete_failure800_penalty']['mean']}|")
        text+=['','partial保留真实异常、已观察命令及forward receipt，不能当机器人失败或用0补未知成本。',
            '主要C_update、ΣT_round/Σg和C_chunk仅在该臂500局成本覆盖全部完整时出数值；观察子集成本另列subtotal。',
            '计时来自共享资源和审计质量运行，仅探索性成本；正式E2E加速UNKNOWN。物理、双相机和RNG严格配对未由本描述报告重新签证。',
            '全成败、partial与未启动、组件FAILED、救回/新增失败和共同成功长度比见RESULT.json。旧producer、旧报告和失败记录不变。']
        (x.out/'RESULT.md').write_text('\n'.join(text)+'\n')
        print(result['status'],p['suite'])
    except BaseException as exc:
        failure=dict(schema='tf-suite-descriptive-report-failed/v2',status='FAILED_CPU_REPORT',
            error=repr(exc),CPU_report_wall_sec=time.time()-start,permissions=FALSE,
            no_producer_modified=True,no_automatic_retry=True)
        with (x.out/'FAILED.json').open('x') as f:
            json.dump(failure,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        raise


if __name__=='__main__':main()

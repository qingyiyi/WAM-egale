"""Read only actual complete/partial suite artifacts; no certificate upgrade."""
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

def row(plan,spec):
    out=Path((spec.get('reuse_source') or {}).get('out',spec['out']))
    docs={n:safe(out/n) for n in ('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json')}
    r=dict(suite=plan['suite'],case_id=spec['case_id'],task=spec['task'],init=spec['init'],arm=spec['arm'],
        source=str(out),reused_T_POOL=bool(spec.get('reuse_source')),exposure=spec['data_exposure'],
        complete=False,success=UNKNOWN,actual_command_steps=UNKNOWN,failure800_penalty=UNKNOWN,policy_chunks=UNKNOWN,
        cost_scope='instrumented shared quality run; exploratory, not formal E2E',
        errors=[dict(file=k,error=e) for k,(_,e) in docs.items() if e],permissions=FALSE)
    t,c,a,f=[docs[n][0] for n in docs]
    if t is not None:
        try:
            assert t['schema']=='tf-suite-full-episode-terminal/v1' and t['suite']==plan['suite']
            assert t['case_id']==spec['case_id'] and t['arm']==spec['arm'] and t['horizon']==800 and t['permissions']==FALSE
            assert t['task']==spec['task'] and t['init']==spec['init']
            if t['complete_episode'] is True:
                assert type(t['success']) is bool and (t['success'] or t['horizon_reached'] is True)
                assert type(t['env_steps']) is int and 0<t['env_steps']<=800
                assert t['failure800_penalty']==(t['env_steps'] if t['success'] else 800)
                r.update(complete=True,success=t['success'],actual_command_steps=t['env_steps'],
                    failure800_penalty=t['failure800_penalty'],policy_chunks=t['policy_chunks'],
                    client_episode_wall_sec=t['elapsed_sec'],client_total_wall_sec=t['client_total_wall_sec'],
                    client_infer_timing=t.get('client_infer_timing',UNKNOWN))
            else:r.update(partial_error=t.get('error'),partial_not_algorithm_failure=True)
        except (AssertionError,KeyError,TypeError) as e:r['errors'].append(dict(file='EPISODE-TERMINAL.json',error=repr(e)))
    if c is not None:
        try:
            assert c['schema']=='tf-suite-episode-counter/v1' and c['case_id']==spec['case_id']
            assert c['suite']==plan['suite'] and c['arm']==spec['arm'] and c['permissions']==FALSE
            r['timing']=c['timing'];r['target_forward_attempts']=c['target_forward_attempts']
            r['target_forward_returned']=c['target_forward_returned'];r['target_forward_failed']=c['target_forward_failed']
            r['counters']=c.get('counters',UNKNOWN);r['GPU_identity']=c.get('GPU_identity',UNKNOWN)
            r['routes']=c.get('routes',UNKNOWN)
        except (AssertionError,KeyError,TypeError) as e:r['errors'].append(dict(file='COUNTER.json',error=repr(e)))
    r['component_status']=a.get('status',UNKNOWN) if a else UNKNOWN
    r['finish_status']=f.get('status',UNKNOWN) if f else UNKNOWN
    r['closed_and_drained']=bool(f and f.get('full_episode_terminal') is True and f.get('async_drained') is True)
    r['strict_components_pass']=bool(a and a.get('status')=='PASS_COMPLETE_EPISODE_COMPONENTS' and r['finish_status']=='PASS')
    return r

def aggregate(rows):
    complete=[r for r in rows if r['complete']]
    good_cost=[r for r in complete if r.get('timing') and r['closed_and_drained']
        and all(finite(r['timing'].get(k)) for k in ('action_total_sec','action_g','full_policy_elapsed_sec','sum_T_round_sec','sum_g'))]
    def sumfield(k):return sum(r['timing'][k] for r in good_cost)
    g=sumfield('action_g');rg=sumfield('sum_g');chunks=sum(r['policy_chunks'] for r in good_cost)
    return dict(planned_rows=len(rows),complete=len(complete),missing_or_partial=len(rows)-len(complete),
        successes=sum(r['success'] for r in complete),success_rate=sum(r['success'] for r in complete)/len(complete) if complete else UNKNOWN,
        raw_success_is_not_strict_certificate=True,strict_component_complete=sum(r['strict_components_pass'] for r in complete),
        actual_steps=stats([r['actual_command_steps'] for r in complete]),
        all_complete_failure800_penalty=stats([r['failure800_penalty'] for r in complete]),
        attempted_partial_count=sum(not r['complete'] and bool(r.get('partial_error')) for r in rows),
        cost_covered_complete=len(good_cost),sum_action_sec=sumfield('action_total_sec'),sum_action_g=g,
        C_update=sumfield('action_total_sec')/g if g>0 else UNKNOWN,
        sum_T_round_sec=sumfield('sum_T_round_sec'),sum_g=rg,
        cost_per_legal_round_commit=sumfield('sum_T_round_sec')/rg if rg>0 else UNKNOWN,
        C_chunk=sumfield('full_policy_elapsed_sec')/chunks if chunks>0 else UNKNOWN,
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

def main():
    a=argparse.ArgumentParser();a.add_argument('--campaign',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    x=a.parse_args();start=time.time();x.out.mkdir(parents=True,exist_ok=False)
    p=read(x.campaign/'registration.json');rows=[];plans=[]
    for name in ('FINAL_0','FINAL_1'):
        f=x.campaign/'stages'/name/'plan.json'
        if f.exists():
            plan=read(f);plans.append(str(f));rows.extend(row(plan,s) for s in plan['rows'])
    if not plans:
        f=x.campaign/'stages/T_POOL/plan.json'
        if f.exists():
            plan=read(f);plans.append(str(f));rows.extend(row(plan,s) for s in plan['rows'])
    result=dict(status='COMPLETE_RAW_DESCRIPTIVE_COVERAGE' if len(rows)==1500 and all(r['complete'] for r in rows) else 'PARTIAL_DESCRIPTIVE_COVERAGE',
        suite=p['suite'],source_plans=plans,by_arm={a:aggregate([r for r in rows if r['arm']==a]) for a in ('T','C','P')},
        unseen_training_init26to49={a:aggregate([r for r in rows if r['arm']==a and r['init']>=26]) for a in ('T','C','P')},
        comparisons=[comparison(rows,l,r) for l,r in [('T','C'),('T','P'),('C','P')]],rows=rows,
        CPU_report_wall_sec=time.time()-start,permissions=FALSE,noninferiority='NOT_ESTABLISHED',
        formal_E2E_speedup='UNKNOWN_REQUIRES_REGISTERED_QUALITY_AND_ISOLATED_COMPLETE_EPISODE_MEASUREMENT',
        raw_component_failures_retained=True,partial_attempt_cost='see original stage counter and live-counter; not silently folded into success-only speed',
        cold_start='see bank producers and per-worker MODEL-SETUP-OBSERVATION; not included in client episode timer')
    with (x.out/'RESULT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    fields=['suite','case_id','task','init','arm','exposure','source','reused_T_POOL','complete','success','actual_command_steps',
        'failure800_penalty','policy_chunks','client_episode_wall_sec','component_status','finish_status','strict_components_pass']
    with (x.out/'EPISODES.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    text=['# 套件三臂完整任务描述汇总','',f"状态：{result['status']}。四项权限保持false；含FIT/CAL/DEV来源，不是独立TEST。",'',
        '|臂|完整/计划|成功|实际命令总数|800失败罚分均值|', '|---|---:|---:|---:|---:|']
    for arm,s in result['by_arm'].items():text.append(f"|{arm}|{s['complete']}/{s['planned_rows']}|{s['successes']}|{s['actual_steps']['sum']}|{s['all_complete_failure800_penalty']['mean']}|")
    text+=['','计时来自并行且有审计的质量运行，仅探索性成本；正式E2E加速比UNKNOWN。全成败、缺项、组件FAILED、paired救回/新增失败及共同成功长度分布见RESULT.json，不用成功子集隐藏失败。']
    (x.out/'RESULT.md').write_text('\n'.join(text)+'\n')
    print(result['status'],p['suite'])

if __name__=='__main__':main()

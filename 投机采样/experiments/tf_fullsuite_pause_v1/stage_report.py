"""Local descriptive audit of every complete episode in the frozen migration tar."""
from __future__ import annotations
import argparse,csv,json,math,os,signal,tarfile,time
from pathlib import Path

PERMISSIONS=dict(engineering_quality_screen=False,independent_test_complete=False,noninferiority_established=False,formal_performance_permission=False)
def q(values):
    values=sorted(values)
    def at(f):
        if not values:return 'UNKNOWN'
        x=(len(values)-1)*f;a=math.floor(x);b=math.ceil(x);return values[a]+(values[b]-values[a])*(x-a)
    return dict(n=len(values),sum=sum(values),mean=sum(values)/len(values) if values else 'UNKNOWN',p50=at(.5),p90=at(.9),max=max(values) if values else 'UNKNOWN',method='linear: (N-1)*q')
def write(f,v):
    with f.open('x') as out:json.dump(v,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')
def main():
    a=argparse.ArgumentParser();a.add_argument('--backup',type=Path,required=True);a.add_argument('--job',type=Path,required=True);x=a.parse_args()
    backup=json.loads((x.backup/'RESULT.json').read_text());assert backup['status']=='COMPLETE_LOCAL_TAR_BACKUP_AND_ENTRY_AUDIT'
    assert len(os.sched_getaffinity(0))==1 and os.getenv('CUDA_VISIBLE_DEVICES')=='' and not x.job.exists()
    job=x.job.resolve();job.mkdir();start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU report 600s cap')));signal.alarm(600)
    write(job/'REGISTRATION.json',dict(goal='Describe all frozen stage189 complete three-arm cases with no selection or gate change',backup=str(x.backup.resolve()),GPU=0,CPU=1,wall_cap_sec=600,storage_cap_bytes=8*2**20,source_deletion=False,permissions=PERMISSIONS))
    try:
        rows=[];case_observations=[]
        with tarfile.open(backup['archive'],'r:') as tar:
            members={m.name:m for m in tar.getmembers()}
            def rd(name):
                m=members[name];assert m.size<=64*2**20
                return json.loads(tar.extractfile(m).read())
            prefix='runtime/p0-autodl-migration-stage-pause-v1-20261017-r1/'
            coverage=rd(prefix+'MIGRATION-COVERAGE.json');assert len(coverage['complete_cases'])==189 and not coverage['partial_cases_no_automatic_retry']
            for entry in coverage['complete_cases']:
                cid=entry['case_id'];base='runtime/'+Path(entry['source_job']).name
                casepath=base+f'/CASE-{cid:04d}-RESULT.json'
                if casepath in members:
                    native=rd(casepath);case_observations.append(dict(case_id=cid,native_status=native['status'],observations=native.get('observations',[])))
                for arm in ('T','C','P'):
                    ep=base+f'/episodes/case-{cid:04d}/{arm}/'
                    t,c,a,f=[rd(ep+n) for n in ('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json')]
                    assert all(v['case_id']==cid and v['arm']==arm and v['permissions']==PERMISSIONS for v in (t,c,a,f))
                    assert t['complete_episode'] and t['quality_terminal'] and t['horizon']==800 and f['full_episode_terminal'] and f['async_drained']
                    assert t['failure800_penalty']==(t['env_steps'] if t['success'] else 800)
                    timing=c['timing'];covered=timing.get('action_measurement_complete') is True and c.get('active_target_forwards',0)==0
                    roundcovered=timing.get('round_coverage_complete') is True
                    row=dict(case_id=cid,task=entry['task'],init=entry['init'],arm=arm,source=ep,success=t['success'],actual_env_steps=t['env_steps'],failure800_penalty=t['failure800_penalty'],policy_chunks=t['policy_chunks'],S_chunks=c['S_chunks'],T_chunks=c['T_chunks'],
                        target_forward_attempts=c['target_forward_attempts'],audit_status=a['status'],finish_status=f['status'],client_episode_wall_sec=t['elapsed_sec'],client_total_wall_sec=t['client_total_wall_sec'],
                        action_measurement_complete=covered,round_coverage_complete=roundcovered,
                        action_total_sec=timing.get('action_total_sec','UNKNOWN'),action_g=timing.get('action_g','UNKNOWN'),full_policy_elapsed_sec=timing.get('full_policy_elapsed_sec','UNKNOWN'),
                        sum_T_round_sec=timing.get('sum_T_round_sec','UNKNOWN') if roundcovered else 'UNKNOWN',sum_g=timing.get('sum_g','UNKNOWN') if roundcovered else 'UNKNOWN',
                        physical_GPU_UUID=(c.get('policy_records') or [{}])[0].get('GPU_identity','UNKNOWN'))
                    rows.append(row)
        arms={}
        for arm in ('T','C','P'):
            rr=[v for v in rows if v['arm']==arm];good=sum(v['success'] for v in rr)
            complete=all(v['action_measurement_complete'] and all(type(v[k]) in (float,int) for k in ('action_total_sec','action_g','full_policy_elapsed_sec')) for v in rr)
            roundcomplete=all(v['round_coverage_complete'] and type(v['sum_T_round_sec']) in (float,int) and type(v['sum_g']) is int for v in rr)
            arms[arm]=dict(episodes=len(rr),success=good,failure=len(rr)-good,success_rate=good/len(rr),actual_steps=q([v['actual_env_steps'] for v in rr]),failure800_penalty=q([v['failure800_penalty'] for v in rr]),
                policy_chunks=sum(v['policy_chunks'] for v in rr),S_chunks=sum(v['S_chunks'] for v in rr),T_chunks=sum(v['T_chunks'] for v in rr),target_forward_attempts=sum(v['target_forward_attempts'] for v in rr),
                component_audit_failed=sum(v['audit_status'].startswith('FAILED') or v['finish_status'].startswith('FAILED') for v in rr),client_episode_wall_sec=q([v['client_episode_wall_sec'] for v in rr]),
                exploratory_C_update=sum(v['action_total_sec'] for v in rr)/sum(v['action_g'] for v in rr) if complete and sum(v['action_g'] for v in rr)>0 else 'UNKNOWN',
                exploratory_C_chunk=sum(v['full_policy_elapsed_sec'] for v in rr)/sum(v['policy_chunks'] for v in rr) if complete else 'UNKNOWN',
                sum_T_round_over_sum_g=sum(v['sum_T_round_sec'] for v in rr)/sum(v['sum_g'] for v in rr) if roundcomplete and sum(v['sum_g'] for v in rr)>0 else 'UNKNOWN',
                timer_scope='shared PCIe concurrent instrumented quality evaluation; not formal speedup; failure costs retained')
        by={(r['case_id'],r['arm']):r for r in rows};pairs={}
        for first,second in (('T','C'),('T','P'),('C','P')):
            xrows=[(by[c['case_id'],first],by[c['case_id'],second]) for c in coverage['complete_cases']]
            both=[(a,b) for a,b in xrows if a['success'] and b['success']]
            pairs[first+'_'+second]=dict(pairs=len(xrows),first_success_second_fail=sum(a['success'] and not b['success'] for a,b in xrows),first_fail_second_success=sum(not a['success'] and b['success'] for a,b in xrows),both_success=len(both),both_fail=sum(not a['success'] and not b['success'] for a,b in xrows),
                length_ratio_second_over_first=q([b['actual_env_steps']/a['actual_env_steps'] for a,b in both]),all_case_penalty_difference_second_minus_first=sum(b['failure800_penalty']-a['failure800_penalty'] for a,b in xrows),
                strict_pair_qualification='NOT_REQUALIFIED; existing physics/RNG/camera/component FAILED remain; matched case IDs are descriptive only')
        tasks={str(task):{arm:dict(episodes=sum(v['task']==task and v['arm']==arm for v in rows),success=sum(v['task']==task and v['arm']==arm and v['success'] for v in rows)) for arm in ('T','C','P')} for task in range(10)}
        with (job/'EPISODES.csv').open('x',newline='') as out:
            w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        write(job/'NATIVE-CASE-OBSERVATIONS.json',case_observations)
        result=dict(status='COMPLETE_DESCRIPTIVE_FROZEN_STAGE_REPORT',complete_cases=189,complete_episodes=567,unstarted_cases=311,partial_cases=0,arms=arms,comparisons=pairs,task_breakdown=tasks,
            independent_test=False,source_backup=backup['archive'],no_new_robot_execution=True,no_source_modified=True,previous_failures_preserved=True,permissions=PERMISSIONS)
        write(job/'RESULT.json',result)
        text=['# Long阶段189组三臂：描述结果','','固定暂停清单全部纳入。不是500组全量完成、独立TEST、质量通过或正式性能结果；旧FAILED和初态配对缺口保留。','', '|方法|成功/完整局|实际命令步均值|800失败罚分均值|单局墙钟P50/P90秒（并行探索）|','|---|---|---|---|---|']
        for arm,label in [('T','Target'),('C','TF连续'),('P','TF＋PF2')]:
            d=arms[arm];w=d['client_episode_wall_sec'];text.append(f"|{label}|{d['success']}/{d['episodes']}|{d['actual_steps']['mean']:.2f}|{d['failure800_penalty']['mean']:.2f}|{w['p50']:.1f}/{w['p90']:.1f}|")
        text+=['','ΣT_round/Σg仅在全部round计时覆盖时计算，缺失标UNKNOWN。调用次数不代表延迟；共享并行墙钟不能作为正式E2E加速。','', '逐局及成对成败/长度/任务分层见RESULT.json、EPISODES.csv。当前四项权限全部false。']
        (job/'RESULT.md').write_text('\n'.join(text)+'\n')
        assert sum(f.stat().st_size for f in job.rglob('*') if f.is_file())<=8*2**20
        print(json.dumps(dict(status=result['status'],arms=arms,comparisons=pairs),ensure_ascii=False),flush=True)
    except BaseException as e:write(job/'FAILED.json',dict(error=repr(e),no_source_modified=True));raise
    finally:signal.alarm(0);write(job/'CLEANUP.json',dict(GPU_created=0,local_CPU_only=True,wall_sec=time.time()-start))

if __name__=='__main__':raise SystemExit(main())

"""Snapshot real isolated timing terminals; preserve failures and missing phases."""
from datetime import datetime,timezone
import json
from pathlib import Path

BASE=Path('/root/cxz/performance/libero-four-suite-20261019/isolated')


def read(p):return json.loads(p.read_text())


def main():
    records=[];phases=[]
    jobs=read(BASE/'JOBS.json')+(read(BASE/'LONG-JOBS.json') if (BASE/'LONG-JOBS.json').exists() else [])
    for item in jobs:
        job=Path(item['job']);plan=read(job/'plan.json');islong=item['suite']=='long'
        specs=([dict(case_id=c,arm=a,task=c%10,init=49) for c in plan['timing_case_ids'] for a in ('T','C','P')]
               if islong else plan['rows'])
        phase=dict(**item,planned_rows=30,complete_rows=0,terminal=None,guard=None)
        terminal=job/('BENCHMARK-PHASE-RESULT.json' if islong else 'SUPERVISOR-RESULT.json')
        if terminal.exists():
            d=read(terminal);phase['terminal']=d['status'];phase['cleanup']=d.get('cleanup')
        supervisor=job/'SUPERVISOR-RESULT.json'
        if supervisor.exists():
            original=read(supervisor)
            phase['runner_elapsed_wall_sec']=original.get('elapsed_wall_sec')
            phase['original_supervisor_status']=original.get('status')
            phase['original_exclusive_GPU_window_verified']=original.get('exclusive_GPU_window_verified')
        if (job/'RUN-STARTED.json').exists():
            phase['run_started_at']=read(job/'RUN-STARTED.json').get('at')
        if (job/'GUARD-FINAL.json').exists():
            g=read(job/'GUARD-FINAL.json');phase['guard']=g.get('status')
            phase['guard_terminal_scalars']={k:v for k,v in g.items() if not isinstance(v,(dict,list))}
        if (job/'MIGRATED-ENTRY-FAILED.json').exists():
            phase['CPU_admission_failure']=read(job/'MIGRATED-ENTRY-FAILED.json')
        setup=job/('server-worker1' if islong else 'server-worker0')/'MODEL-SETUP-OBSERVATION.json'
        if setup.exists():phase['shared_model_setup_observation']=read(setup)
        for spec in specs:
            out=job/'episodes'/f'case-{spec["case_id"]:04d}'/spec['arm']
            r=dict(suite=item['suite'],repetition=item['repetition'],case_id=spec['case_id'],
                task=spec['task'],init=spec['init'],arm=spec['arm'],source=str(out),
                complete=False,status='NOT_STARTED',registered_GPU=item['GPU'])
            p=out/'EPISODE-TERMINAL.json'
            if p.exists():
                t=read(p);r.update(status=t.get('status'),complete=t.get('complete_episode') is True,
                    success=t.get('success'),steps=t.get('env_steps'),penalty=t.get('failure800_penalty'),
                    policy_chunks=t.get('policy_chunks'),episode_sec=t.get('elapsed_sec'),
                    client_total_wall_sec=t.get('client_total_wall_sec'),client_infer_timing=t.get('client_infer_timing'),
                    termination_reason=t.get('termination_reason'),permissions=t.get('permissions'),
                    error=t.get('error'))
            if (out/'COUNTER.json').exists():
                c=read(out/'COUNTER.json');r.update(timing=c.get('timing'),
                    forward_attempts=c.get('target_forward_attempts'),S_chunks=c.get('S_chunks'),T_chunks=c.get('T_chunks'),
                    GPU_identity=next((q['GPU_identity'] for q in c.get('policy_records',[]) if 'GPU_identity' in q),None),
                    policy_timing=[dict(policy_ordinal=q.get('successful_policy_ordinal'),role=q.get('role'),
                        frame_before=q.get('frame_before'),timing=q.get('timing'),
                        GPU_identity=q.get('GPU_identity')) for q in c.get('policy_records',[])])
            if (out/'AUDIT.json').exists():r['component_status']=read(out/'AUDIT.json').get('status')
            if (out/'FINISH_READY.json').exists():
                f=read(out/'FINISH_READY.json');r.update(finish_status=f.get('status'),async_drained=f.get('async_drained'))
            if r['complete']:
                assert r['penalty']==(r['steps'] if r['success'] else 800)
                phase['complete_rows']+=1
            records.append(r)
        phases.append(phase)
    result=dict(at_UTC=datetime.now(timezone.utc).isoformat(),phases=phases,rows=records,
        complete_rows=sum(r['complete'] for r in records),expected_rows=360,
        source_registration=read(BASE/'REGISTRATION.json'),new_robot_runs_or_model_forwards=0,
        old_quality_fields_unchanged=True)
    result['queues']={}
    for name in ('GPU0-QUEUE-STATUS.json','GPU0-UNSTARTED-CONTINUATION.json',
                 'GPU1-QUEUE-STATUS.json','LONG-QUEUE-STATUS.json'):
        if (BASE/name).exists():
            q=read(BASE/name);result['queues'][name]={k:q.get(k) for k in
                ('status','active','complete_rows','error','original_budget_and_deadline_unchanged')}
    p=BASE/'COLLECTED-TIMING.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(result,indent=2)+'\n');q.replace(p)
    print(json.dumps(dict(complete=result['complete_rows'],expected=360,
        phase_counts={x['suite']+'-'+str(x['repetition']):x['complete_rows'] for x in phases})))


if __name__=='__main__':main()

"""Continue only pre-registered never-started Object repetitions 2/3.

Repetition1's CPU admission failure is immutable and is never retried. No new
budget or deadline. The failed original queue remains stopped and archived.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/root/cxz/performance/libero-four-suite-20261019/isolated')
OPS=Path('/root/cxz/ops/whole-layer-warp-20261020')
APP=Path('/root/cxz/WAM-egale/投机采样/experiments')


def read(p):return json.loads(p.read_text())
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def contexts():
    s=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    return sorted(int(v) for v in s.splitlines() if v.strip().isdigit())


def main():
    os.sched_setaffinity(0,{20});reg=read(BASE/'REGISTRATION.json')
    stop=datetime.fromisoformat(reg['absolute_work_stop']).timestamp()
    old=read(BASE/'GPU0-QUEUE-FAILED-ORIGINAL.json')
    assert old['status']=='STOPPED_OR_PARTIAL_ISOLATED_QUEUE'
    first=BASE/'object-repeat1'
    assert not list(first.glob('episodes/*/*/EPISODE-TERMINAL.json')) and not (first/'server-worker0').exists()
    old['completed'].append(dict(suite='object',repetition=1,GPU=0,job=str(first),
        status='CPU_ADMISSION_FAILED_NOT_RETRIED',complete_rows=0,new_teacher_forwards=0,
        source_failure=str(first/'MIGRATED-ENTRY-FAILED.json'),
        guard=read(first/'GUARD-FINAL.json')['status']))
    state=dict(status='WAITING_FOR_BOUNDED_OPERATOR_RELEASE',completed=old['completed'],active=None,
        original_failure_retained=True,original_budget_and_deadline_unchanged=True,
        no_retries=True,expected_original_panel_rows=360,maximum_complete_after_skipping_failed_phase=330)
    path=BASE/'GPU0-UNSTARTED-CONTINUATION.json';write(path,state)
    try:
        while True:
            assert time.time()+1320<stop,'original complete-episode reserve'
            q=read(OPS/'GPU-QUEUE-STATUS.json')
            if q.get('status') in ('FINITE_OPERATOR_RESEARCH_COMPLETE_NOT_PRODUCTION_ADMITTED','STOPPED_FAILED_OR_PARTIAL'):
                assert all(x.get('root_reaped') for x in q.get('jobs',[])),'operator cleanup not closed'
                if not contexts():break
            time.sleep(10)
        jobs=[x for x in read(BASE/'JOBS.json') if x['suite']=='object' and x['repetition'] in (2,3)]
        for item in jobs:
            job=Path(item['job']);p=read(job/'plan.json')
            assert not (job/'LAUNCH-RESULT.json').exists() and not (job/'RUN-STARTED.json').exists(),'never-started only'
            assert time.time()+1320<stop and not contexts(),'original reserve/GPU guard'
            assert sum(q.stat().st_size for q in BASE.rglob('*') if q.is_file())<=reg['storage_cap_bytes']
            state.update(status='RUNNING_PRE_REGISTERED_UNSTARTED_OBJECT',active=item,host_context=None,
                at=datetime.now(timezone.utc).isoformat());write(path,state)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (job/'NEW-OBJECT-NATIVE-LAUNCH.log').open('x') as f:
                r=subprocess.run([p['va_python'],'-B',str(APP/'object_capture_recovery_v3/launch.py'),
                    '--job',str(job)],cwd=APP,env=env,stdout=f,stderr=f,timeout=180)
            assert r.returncode==0,'Object launcher failure; no retry'
            contamination=[]
            while not (job/'GUARD-FINAL.json').exists():
                pids=contexts()
                if (job/'server-worker0/READY.json').exists() and state['host_context'] is None and len(pids)==1:
                    state['host_context']=pids[0]
                if len(pids)>1 or (state['host_context'] is not None and pids and pids!=[state['host_context']]):
                    contamination.append(dict(at=time.time(),pids=pids));(job/'STOP_AFTER_EPISODE').touch(exist_ok=True)
                if (job/'status.json').exists():state['complete_rows']=read(job/'status.json').get('complete_rows')
                state['checked_at']=datetime.now(timezone.utc).isoformat();write(path,state)
                assert time.time()<datetime.fromisoformat(reg['absolute_cleanup_deadline']).timestamp()+10
                time.sleep(5)
            # Guard may close before the runner's final file is flushed. Wait
            # only for its actual receipt; do not manufacture a phase terminal.
            until=min(time.time()+30,datetime.fromisoformat(reg['absolute_cleanup_deadline']).timestamp())
            while not (job/'SUPERVISOR-RESULT.json').exists() and time.time()<until:time.sleep(1)
            assert (job/'SUPERVISOR-RESULT.json').exists(),'missing true supervisor terminal; no retry'
            r=read(job/'SUPERVISOR-RESULT.json');close=read(job/'GUARD-FINAL.json')
            state['completed'].append(dict(**item,status=r['status'],complete_rows=r['complete_rows'],
                guard=close.get('status'),contamination=contamination))
            state['active']=None;write(path,state)
            assert r['status']=='COMPLETE_RECORDED_SUITE_PHASE' and not contamination and not contexts()
        state['status']='UNSTARTED_OBJECT_COMPLETE_ORIGINAL_PANEL_PARTIAL'
    except BaseException as exc:
        state.update(status='STOPPED_OR_PARTIAL_UNSTARTED_OBJECT',error=repr(exc))
    finally:write(path,state)


if __name__=='__main__':main()

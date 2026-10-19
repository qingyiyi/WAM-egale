"""One existing native suite worker per GPU, finite queue, no retries."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/root/cxz/performance/libero-four-suite-20261019/isolated')
APP=Path('/root/cxz/WAM-egale/投机采样/experiments')


def read(path):return json.loads(path.read_text())
def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def nvml_pids(gpu):
    p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if p.returncode:raise RuntimeError('GPU process observation failed')
    return sorted(int(x) for x in p.stdout.splitlines() if x.strip().isdigit())


def main():
    gpu=int(sys.argv[1]);os.sched_setaffinity(0,{20+gpu})
    reg=read(BASE/'REGISTRATION.json');work=datetime.fromisoformat(reg['absolute_work_stop']).timestamp()
    jobs=[x for x in read(BASE/'JOBS.json') if x['GPU']==gpu]
    state=dict(status='RUNNING_ISOLATED_ENGINEERING_QUEUE',GPU=gpu,completed=[],active=None,
               permissions=reg['permissions'],formal_quality_certificate=False)
    path=BASE/f'GPU{gpu}-QUEUE-STATUS.json'
    write(BASE/f'GPU{gpu}-QUEUE-OWNED.json',dict(pid=os.getpid(),birth_ticks=int(
          Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])))
    try:
        for item in jobs:
            job=Path(item['job']);plan=read(job/'plan.json')
            assert time.time()+1320<work,'insufficient complete-episode reserve'
            assert not nvml_pids(gpu),'GPU has another compute context; no launch'
            size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
            assert size<=reg['storage_cap_bytes'],'new measurement storage cap'
            state['active']=item;state['host_context_pid']=None;write(path,state)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            entry='object_capture_recovery_v3' if item['suite']=='object' else 'baai_suite_pipeline_v1'
            cmd=[plan['va_python'],'-B',str(APP/entry/'launch.py'),'--job',str(job)]
            with (job/'NEW-MEASUREMENT-LAUNCH.log').open('x') as f:
                launched=subprocess.run(cmd,cwd=APP,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
            assert launched.returncode==0,'native suite launch failed; no retry'
            contamination=[]
            while not (job/'GUARD-FINAL.json').exists():
                pids=nvml_pids(gpu)
                ready=(job/'server-worker0/READY.json').exists()
                if ready and state['host_context_pid'] is None and len(pids)==1:
                    state['host_context_pid']=pids[0]
                if len(pids)>1 or (state['host_context_pid'] is not None and pids
                                   and pids!=[state['host_context_pid']]):
                    contamination.append(dict(at=time.time(),pids=pids))
                    # Preserve the current full episode.  Do not signal a foreign process.
                    (job/'STOP_AFTER_EPISODE').touch(exist_ok=True)
                state['last_NVML_pids']=pids;state['last_observed_at']=datetime.now(timezone.utc).isoformat()
                pulse=job/'status.json'
                if pulse.exists():
                    d=read(pulse);state['complete_rows']=d.get('complete_rows');state['current']=d.get('current')
                write(path,state)
                if time.time()>datetime.fromisoformat(reg['absolute_cleanup_deadline']).timestamp()+10:
                    raise TimeoutError('native independent guard deadline; inspect owned receipts')
                time.sleep(5)
            until=min(time.time()+30,datetime.fromisoformat(reg['absolute_cleanup_deadline']).timestamp())
            while not (job/'SUPERVISOR-RESULT.json').exists() and time.time()<until:time.sleep(1)
            assert (job/'SUPERVISOR-RESULT.json').exists(),'entry failed before true supervisor terminal; no retry'
            result=read(job/'SUPERVISOR-RESULT.json');close=read(job/'GUARD-FINAL.json')
            record=dict(**item,status=result['status'],complete_rows=result['complete_rows'],
                        contamination=contamination,guard=close.get('status'),
                        cleanup=result.get('cleanup'))
            state['completed'].append(record);state['active']=None;write(path,state)
            assert not contamination,'GPU contention invalidated this timing attempt; no retry'
            assert result['status']=='COMPLETE_RECORDED_SUITE_PHASE','partial phase; no retry'
            assert not nvml_pids(gpu),'owned context not released after native cleanup'
        state['status']='COMPLETE_PREPARED_ISOLATED_QUEUE'
    except BaseException as error:
        state.update(status='STOPPED_OR_PARTIAL_ISOLATED_QUEUE',error=repr(error))
        raise
    finally:write(path,state)


if __name__=='__main__':main()

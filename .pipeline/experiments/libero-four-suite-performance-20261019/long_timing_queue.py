"""Wait for the isolated Spatial queue, then run the registered Long measurements."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

BASE=Path('/root/cxz/performance/libero-four-suite-20261019/isolated')


def read(p):return json.loads(p.read_text())
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def pids():
    p=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                     capture_output=True,text=True,timeout=10);p.check_returncode()
    return sorted(int(x) for x in p.stdout.splitlines() if x.strip().isdigit())


def main():
    os.sched_setaffinity(0,{22});reg=read(BASE/'REGISTRATION.json')
    work=datetime.fromisoformat(reg['absolute_work_stop']).timestamp()
    state=dict(status='WAITING_FOR_SPATIAL_ISOLATED_QUEUE',completed=[],active=None,GPU=1)
    out=BASE/'LONG-QUEUE-STATUS.json'
    write(BASE/'LONG-QUEUE-OWNED.json',dict(pid=os.getpid(),birth_ticks=int(
        Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])))
    try:
        while True:
            assert time.time()+4860<work,'cannot reserve a full Long case and cleanup'
            prior=read(BASE/'GPU1-QUEUE-STATUS.json');write(out,state)
            if prior['status']=='COMPLETE_PREPARED_ISOLATED_QUEUE':break
            if prior['status']=='STOPPED_OR_PARTIAL_ISOLATED_QUEUE':raise RuntimeError('Spatial queue partial; no automatic continuation')
            time.sleep(5)
        state['status']='RUNNING_ISOLATED_LONG_ENGINEERING_MEASUREMENT'
        for item in read(BASE/'LONG-JOBS.json'):
            assert time.time()+4860<work,'complete Long case reserve unavailable'
            assert not pids(),'GPU1 compute context not empty after Spatial'
            job=Path(item['job']);p=read(job/'plan.json');state['active']=item;state['complete_rows']=0;write(out,state)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            cmd=[p['va_python'],'-B',str(BASE.parent/'long_timing_entry.py'),'launch','--job',str(job)]
            with (job/'NEW-MEASUREMENT-LAUNCH.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=job,env=env,stdout=log,stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL,start_new_session=True)
            write(job/'LAUNCH-PARENT-OWNED.json',dict(pid=child.pid,birth_ticks=int(
                Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])))
            until=time.time()+60
            while not (job/'LAUNCH-RESULT.json').exists():
                assert child.poll() is None and time.time()<until,'Long launcher failed before owned receipt'
                time.sleep(.1)
            expected=None;contamination=[]
            while not (job/'GUARD-FINAL.json').exists():
                current=pids()
                if (job/'server-worker1/READY.json').exists() and expected is None and len(current)==1:expected=current[0]
                if len(current)>1 or (expected is not None and current and current!=[expected]):
                    contamination.append(dict(at=time.time(),pids=current));(job/'STOP_AFTER_EPISODE').touch(exist_ok=True)
                state.update(NVML_pids=current,last_observed_at=datetime.now(timezone.utc).isoformat(),
                    complete_rows=len(list((job/'episodes').glob('case-*/*/EPISODE-TERMINAL.json'))))
                write(out,state)
                assert time.time()<datetime.fromisoformat(reg['absolute_cleanup_deadline']).timestamp()+10,'independent guard cleanup deadline'
                time.sleep(5)
            child.wait(timeout=30)
            assert child.returncode==0,'Long owned parent exit failed'
            result=read(job/'BENCHMARK-PHASE-RESULT.json')
            state['completed'].append(dict(**item,status=result['status'],complete_rows=result['complete_rows'],
                legacy_500_supervisor_status=result['legacy_500_supervisor_status'],contamination=contamination))
            state['active']=None;write(out,state)
            assert result['status']=='COMPLETE_REGISTERED_ISOLATED_PANEL' and not contamination,'partial/contaminated Long phase; no retry'
            assert not pids(),'Long owned context not released'
        state['status']='COMPLETE_ISOLATED_LONG_ENGINEERING_MEASUREMENT'
    except BaseException as error:
        state.update(status='STOPPED_OR_PARTIAL_ISOLATED_LONG',error=repr(error));raise
    finally:write(out,state)


if __name__=='__main__':main()

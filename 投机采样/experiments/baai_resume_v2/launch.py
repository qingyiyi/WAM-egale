"""Distinct library-separated full-episode continuation; original clocks and no retries."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN.parent))
from baai_resume_v1.contract import install
protocol=install()
load,read,write,require,PERMISSIONS=protocol.load,protocol.read,protocol.write,protocol.require,protocol.PERMISSIONS
from tf_fullsuite_eval_v2.runner import process,matches



def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);x=a.parse_args();job=x.job.resolve(strict=True)
    p=load(job);require(not (job/'LAUNCH-STARTED.json').exists(),'one lifecycle only')
    write(job/'LAUNCH-STARTED.json',dict(**process(os.getpid()),at=datetime.now(timezone.utc).isoformat(),source_commit=p['source_code_commit']))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    children=[];guard=None;runner=None
    def spawn(role,script,cores):
        previous=os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0,set(cores))
            with (job/(role+'.log')).open('xb') as log:
                child=subprocess.Popen([p['va_python'],'-B',str(script),'--job',str(job)],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True,start_new_session=True)
        finally:os.sched_setaffinity(0,previous)
        pin=dict(**process(child.pid),owner_job=str(job),role=role)
        require(pin['pgid']==pin['pid'] and pin['uid']==os.getuid(),'fresh birth proof required')
        write(job/(f'OWNED-{child.pid}.json' if role=='runner' else 'GUARD-PROCESS.json'),pin)
        children.append((child,pin));return child,pin
    failure=None
    try:
        guard,gpin=spawn('guard',OWN.parent/'tf_fullsuite_eval_v2/guard.py',[44])
        end=time.monotonic()+20
        while not (job/'GUARD-STARTED.json').exists():
            require(guard.poll() is None and time.monotonic()<end,'actual absolute guard failed')
            time.sleep(.1)
        runner,rpin=spawn('runner',OWN/'runner.py',range(16,32))
        write(job/'LAUNCH-RESULT.json',dict(status='SUPERVISED_SHARD_RUNNER_AND_GUARD_STARTED',runner=rpin,guard=gpin,permissions=PERMISSIONS))
        runner.wait(timeout=max(1,datetime.fromisoformat(p['absolute_cleanup_deadline']).timestamp()-time.time()+15))
        if (job/'ALL-OWNED-CLEANUP.json').exists():guard.wait(timeout=30)
        else:
            failure=dict(reason='RUNNER_ENTRY_OR_FINALIZATION_WITHOUT_CLEANUP',runner_exit=runner.returncode)
            if matches(gpin):os.killpg(gpin['pgid'],signal.SIGTERM)
            guard.wait(timeout=10)
    except BaseException as exc:
        failure=dict(error=repr(exc))
        for child,pin in reversed(children):
            if child.poll() is None and matches(pin):os.killpg(pin['pgid'],signal.SIGTERM)
    finally:
        write(job/'LAUNCH-TERMINAL.json',dict(status='SUPERVISED_CHILDREN_EXITED' if all(c.poll() is not None for c,_ in children) else 'PARTIAL_OWNED_REMAIN_OR_UNKNOWN',
            failure=failure,children=[dict(pin=pin,returncode=c.poll(),observed=process(pin['pid'])) for c,pin in children],
            no_retry=True,permissions=PERMISSIONS))
    return 0 if failure is None else 2

if __name__=='__main__':raise SystemExit(main())

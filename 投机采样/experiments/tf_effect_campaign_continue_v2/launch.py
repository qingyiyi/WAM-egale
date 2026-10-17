"""Explicit one-shot launcher for the independently registered continuation."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OWN=Path(__file__).resolve().parent;EXP=OWN.parent;ROOT=OWN.parents[2]
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from tf_effect_campaign_continue_v2.protocol import load,write,require,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    require(os.getenv('CUDA_VISIBLE_DEVICES')=='','launcher must not allocate CUDA')
    job=args.job.resolve();p=load(job)
    require(not any((job/n).exists() for n in ('RUN-STARTED.json','ROOT-LAUNCH.json','FAILED.json','COLLECTION.json','LAUNCH-FAILED.json')),
        'only one fresh explicit continuation launch')
    spec=importlib.util.spec_from_file_location('_pending_continuation_ownership',EXP/'single_worker_chc_gpu_v4/ownership.py')
    ownership=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ownership;spec.loader.exec_module(ownership)
    root=clock=None;pin=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        os.sched_setaffinity(0,set(p['CPU_affinity']))
        command=[p['reference_plan']['va_python'],'-B',str(OWN/'run.py'),'--job',str(job)]
        with (job/'supervisor.log').open('x') as log:
            root=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        until=time.monotonic()+40
        while not (job/'RUN-STARTED.json').exists() and root.poll() is None and time.monotonic()<until:time.sleep(.1)
        require(root.poll() is None and (job/'RUN-STARTED.json').exists(),'continuation supervisor failed before clock admission')
        observed=ownership.read_process(root.pid)
        require(observed.argv==tuple(command) and observed.pgid==root.pid and observed.uid==os.getuid(),'exact spawned own root required')
        pin=asdict(observed);pin['argv']=list(pin['argv'])
        write(job/'ROOT-LAUNCH.json',dict(status='EXPLICIT_DETACHED_PENDING_ONLY_LAUNCH',pin=pin,command=command,
            launcher_source=str(Path(__file__).resolve()),launcher_source_sha256=sha(__file__),start_new_session=True))
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        command=[sys.executable,'-B',str(OWN/'clock.py'),'--job',str(job)]
        with (job/'clock.log').open('x') as log:
            clock=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        until=time.monotonic()+15
        while not (job/'CLOCK-GUARD.json').exists() and clock.poll() is None and time.monotonic()<until:time.sleep(.1)
        require((job/'CLOCK-GUARD.json').exists() and clock.poll() is None,'independent continuation clock failed to attach')
        cp=asdict(ownership.read_process(clock.pid));cp['argv']=list(cp['argv']);write(job/'CLOCK-LAUNCH.json',dict(pin=cp))
        print(json.dumps(dict(status='RUNNING_PENDING_ONLY_CONTINUATION_WITH_CLOCK',supervisor=root.pid,clock=clock.pid,
            job=str(job),pending=len(p['pending_keys']),GPU_cleanup_by=p['absolute_cleanup_deadline']),ensure_ascii=False))
    except BaseException as exc:
        if root is not None and root.poll() is None and pin is not None:
            current=ownership.read_process(root.pid)
            if all(getattr(current,k)==pin[k] for k in ('pid','uid','start_ticks','pgid')) and list(current.argv)==pin['argv']:
                os.kill(root.pid,signal.SIGTERM)
                try:root.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
        if not (job/'LAUNCH-FAILED.json').exists():write(job/'LAUNCH-FAILED.json',dict(status='FAILED_PENDING_ONLY_LAUNCH',error=repr(exc),no_foreign_signals=True))
        raise


if __name__=='__main__':main()

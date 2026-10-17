"""One detached supervisor plus independent CPU watchdog; no automatic retry."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import time

OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN.parent))
from task4_direction_completion_v1.protocol import load,read,write
from task4_direction_completion_v1.clock import base

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    job=args.job.resolve();p=load(job)
    if (job/'ROOT-LAUNCH.json').exists():raise ValueError('one launch only')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    python=p['va_python'];cmd=[python,str(OWN/'run.py'),'--job',str(job)]
    with (job/'supervisor.log').open('x') as log:
        proc=subprocess.Popen(cmd,env=env,cwd=OWN,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{12,13,14,15}))
    # Read the stable exact argv after Python exec, before any worker admission.
    stop=time.monotonic()+15
    while True:
        pin=base.ownership.read_process(proc.pid)
        if list(pin.argv)==cmd:break
        if proc.poll() is not None or time.monotonic()>stop:raise RuntimeError('root failed to exec before guard')
        time.sleep(.02)
    write(job/'ROOT-LAUNCH.json',dict(pin=asdict(pin),command=cmd,start_new_session=True,
        source=str(OWN/'launch.py'),CPU_affinity=[12,13,14,15]))
    end=time.monotonic()+30
    while not (job/'RUN-STARTED.json').exists():
        if proc.poll() is not None or time.monotonic()>end:raise RuntimeError('supervisor did not reach guard barrier')
        time.sleep(.1)
    command=[python,str(OWN/'clock.py'),'--job',str(job),'--attach']
    with (job/'clock.log').open('x') as log:
        guard=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=''),cwd=OWN,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            preexec_fn=lambda:os.sched_setaffinity(0,{0,1}))
    limit=time.monotonic()+20
    while not (job/'CLOCK-GUARD.json').exists():
        if guard.poll() is not None or proc.poll() is not None or time.monotonic()>limit:
            raise RuntimeError('independent clock not attached; supervisor barrier will stop before GPU admission')
        time.sleep(.1)
    print({'status':'LAUNCHED_EXPLICIT_DIAGNOSTIC_WITH_INDEPENDENT_CLOCK',
        'job':str(job),'supervisor_pid':proc.pid,'guard_pid':guard.pid,'episodes':2})

if __name__=='__main__':main()

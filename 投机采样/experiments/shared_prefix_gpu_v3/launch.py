"""One-shot detached controller launcher; never skips a fresh CPU gate."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
BASE = Path(__file__).resolve().parent.parent/'single_worker_chc_gpu_v4'
import sys
if str(BASE) not in sys.path:sys.path.append(str(BASE))

from task_contract import OWN, ENABLED, APPROVED, enabled_plan, write, stamp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--mode',choices=['cpu','gpu','auto'],required=True)
    a=ap.parse_args();job=Path(a.job).resolve();p=enabled_plan(job,cpu_gate=a.mode=='gpu')
    if any((job/name).exists() for name in ('FAILED.json','FINAL_RESULT.json','STOP_REQUESTED')):
        raise ValueError('terminal/stopped job cannot be launched again')
    controller=job/('CPU-CONTROLLER.json' if a.mode in ('cpu','auto') else 'GPU-CONTROLLER.json')
    marker=job/('LAUNCHER-'+a.mode+'.json')
    if controller.exists() or marker.exists():raise ValueError('controller launch cannot be repeated')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env[ENABLED]=env[APPROVED]='1'
    cmd=[p['model_python'],str(OWN/'supervisor.py'),'--job',str(job),'--mode',a.mode]
    # The child inherits IGN; setsid alone does not protect against all HUP cases.
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (job/(a.mode.upper()+'-supervisor.log')).open('xb') as log:
        child=subprocess.Popen(cmd,cwd=p['worktree'],env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(marker,dict(pid=child.pid,process_group=child.pid,command=cmd,timestamp=stamp(),
            independent_session=True,SIGHUP_ignored=True,stdin_DEVNULL=True,stdout_stderr_file=True,
            CPU_gate_required=a.mode=='gpu',fresh_CPU_gate_required_before_GPU=True,
            CPU_then_GPU_automatic_handoff=a.mode=='auto',resource_poll_interval_sec=180 if a.mode!='cpu' else None,
            GPU_hard_minutes=36,only_one_GPU_attempt=True),exclusive=True)
    print(child.pid)


if __name__=='__main__':main()

"""Bound final CPU gate to the immutable fresh plan; does not initialize CUDA."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from task_contract import OWN, plan, write, digest, VA, CLIENT, ROOT, WORKTREE, stamp

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    args=ap.parse_args();job=args.job.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only preflight')
    p=plan(job)
    out=job/'preflight';out.mkdir()
    started=time.monotonic()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    with (out/'CPU-tests.log').open('xb') as f:
        r=subprocess.run([VA,str(OWN/'test_cpu.py'),'--out',str(out/'CPU-tests.json')],
                         stdout=f,stderr=subprocess.STDOUT,env=env,cwd=WORKTREE,timeout=180)
    if r.returncode:raise RuntimeError('CPU integration failed; log preserved')
    tests=json.loads((out/'CPU-tests.json').read_text())
    if tests['status']!='PASS' or tests['CUDA_initialized'] or tests['CUDA_attempts']:
        raise RuntimeError('CPU pass/no CUDA entry required')
    for port in p['ports']:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    # Separate actual client environment imports and CLI, without EGL init.
    code="import sys;sys.path.insert(0,"+repr(str(ROOT/'投机采样/scripts'))+");import p0_libero_multichunk_client as c;import numpy;import libero.libero.benchmark;print('PASS_CLIENT_CPU_IMPORTS')"
    with (out/'client-import.log').open('xb') as f:
        c=subprocess.run([CLIENT,'-c',code],stdout=f,stderr=subprocess.STDOUT,
                         env={**env,'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl'},timeout=60)
    if c.returncode:raise RuntimeError('client CPU import failed')
    # CPU plan/source is checked again after tests, before creating gate.
    plan(job)
    write(out/'CPU-GATE.json',dict(status='PASS',timestamp=stamp(),plan_sha256=digest(job/'plan.json'),
          CPU_tests=tests['tests_run'],client_imports=True,ports_available=True,
          original30_layer_production_loader=True,CUDA_initialized=False,
          elapsed_seconds=time.monotonic()-started,private_only=True,
          reused_GPU_full_chunk_probe=p['reused_numerical_probe'],
          scope='interfaces/lifecycle/resource gates; identical original-model/client code and frozen GPU full-chunk bitwise acceptance reused'),exclusive=True,durable=True)
    print(json.dumps(dict(status='PASS',CPU_tests=tests['tests_run'],job=str(job)),ensure_ascii=False))

if __name__=='__main__':main()

"""Wait for real components then run one fresh bounded bank producer, no retry."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]

def write(p,data):
    with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--source-commit',required=True)
    x=a.parse_args();j=x.job.resolve();j.mkdir(parents=True,exist_ok=False);origin=time.time();deadline=origin+7200
    write(j/'REGISTRATION.json',dict(goal='produce durable Object full512 bank after complete runtime asset admission',
        source_commit=x.source_commit,origin_epoch=origin,deadline_epoch=deadline,wall_cap_sec=7200,
        GPU=0,CPU=7,CPU_affinity=sorted(os.sched_getaffinity(0)),storage_cap_bytes=256*2**20,
        pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),
        no_automatic_retry=True,no_SHA=True,cleanup='bounded child CPU producer; owned PID birth check and reap'))
    require=lambda ok:None if ok else (_ for _ in ()).throw(ValueError('bounded CPU7 producer admission'))
    require(os.getenv('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=7)
    child=None;pin=None
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('owned bank waiter terminated')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bank waiter total lifecycle')))
    signal.alarm(7200)
    try:
        model=ROOT/'models/community-libero-20261017-v1/object/runtime'
        ready=model.parent/'ASSET-READY.json'
        while not ready.is_file():
            require(time.time()+120<deadline);time.sleep(5)
        remaining=int(deadline-time.time()-30);require(remaining>=60)
        argv=[str(ROOT.parent/'miniconda3/envs/va/bin/python'),'-B',str(Path(__file__).with_name('bank_worker_v3.py')),
            '--job',str(j/'producer'),'--source-commit',x.source_commit,'--suites','object',
            '--prompts',str(ROOT/'投机采样/results/runtime/p0-tf-suite-prompts-cpu1-v1-20261017-r1/PROMPTS.json'),
            '--shared-components',str(ROOT/'models/lingbot-va-posttrain-libero-long-modelscope-20260922'),
            '--asset-root',str(ROOT/'models/community-libero-20261017-v1'),
            '--native-negative-bank',str(ROOT/'投机采样/results/runtime/p0-reset-text-bank-v1-20261015-cpu7-r2/bank.pt'),
            '--threads','7','--wall-cap',str(remaining)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='7',MKL_NUM_THREADS='7',OPENBLAS_NUM_THREADS='7',PYTHONDONTWRITEBYTECODE='1')
        with (j/'producer.log').open('xb') as log:child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,env=env,cwd=ROOT,start_new_session=True)
        pin=dict(pid=child.pid,birth_ticks=int(Path('/proc/'+str(child.pid)+'/stat').read_text().rpartition(') ')[2].split()[19]),pgid=os.getpgid(child.pid))
        write(j/'OWNED-CPU-PRODUCER.json',dict(**pin,argv=argv))
        rc=child.wait(timeout=max(1,deadline-time.time()-10))
        if rc:raise RuntimeError('durable bank producer failed, rc='+str(rc))
        write(j/'RESULT.json',dict(status='COMPLETE_DURABLE_OBJECT_BANK',wall_sec=time.time()-origin,
            bank=str(j/'producer/object/bank.pt'),producer_returncode=rc,GPU=0))
        return 0
    except BaseException as e:
        write(j/'FAILED.json',dict(status='PARTIAL_OBJECT_BANK_PREPARATION',error=repr(e),wall_sec=time.time()-origin));raise
    finally:
        signal.alarm(0)
        if child is not None and child.poll() is None:
            try:
                live=int(Path('/proc/'+str(pin['pid'])+'/stat').read_text().rpartition(') ')[2].split()[19])
                if live==pin['birth_ticks'] and os.getpgid(pin['pid'])==pin['pgid']==pin['pid']:
                    os.killpg(pin['pgid'],signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(pin['pgid'],signal.SIGKILL);child.wait(timeout=5)
            except ProcessLookupError:pass
        write(j/'CLEANUP.json',dict(owned_child_returncode=None if child is None else child.poll(),GPU=0,exits_next=True))

if __name__=='__main__':raise SystemExit(main())

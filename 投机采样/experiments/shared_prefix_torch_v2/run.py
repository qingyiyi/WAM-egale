"""Fresh CPU preflight and one owned, bounded original-torch direct comparison."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from task_contract import (OWN, ROOT, OBS, PRIOR, PREFIX, REVISION, ENABLED, APPROVED,
                           stamp, write, digest, implementation, enabled_plan, freeze)
from resource_guard import NVML, Idle
import ownership
from direct_counter import recover


def environment(*, cpu=False):
    e = {k:v for k,v in os.environ.items() if not k.startswith(('P0_','WAM_P0_','LINGBOT_','NSYS_'))}
    e.pop('CUDA_LAUNCH_BLOCKING',None)
    e.update(CUDA_VISIBLE_DEVICES='' if cpu else '0',PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='7',MKL_NUM_THREADS='7',OPENBLAS_NUM_THREADS='7',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    e[ENABLED] = e[APPROVED] = '1'
    return e


def create_job():
    old = json.loads((PRIOR / 'FINAL_RESULT.json').read_text())
    job = ROOT / '投机采样/results/runtime' / (PREFIX + stamp().replace('-','').replace(':','').replace('T','-')[:15])
    job.mkdir()
    p = dict(job=str(job),created_at=stamp(),execution_revision=REVISION,
        model_python='/rjs/cxz/miniconda3/envs/va/bin/python',
        authorization_text='可以的继续吧；你就不可以和原始torch实现对比吗',
        scope='direct original torch comparison and shared-address repair; fixed numerical gate',
        hard_target_forward_cap=0,direct_attention_micro_call_cap=24,GPU_hard_limit_seconds=300,
        GPU_attempt_cap=1,GPU_model_worker_cap=1,new_full_episode_cap=0,EGL_initialization_cap=0,
        complete_policy_chunk_cap=0,training=False,RL=False,no_retries=True,layers=[0,8,29],
        aggregate_prior_target_intents=old['aggregate_target_intents'],
        aggregate_prior_GPU_seconds=old['aggregate_GPU_seconds'],aggregate_target_cap=900,
        aggregate_GPU_seconds_cap=2700,implementation_sha256=implementation(),
        original_reference_and_bitwise_gate_unchanged=True,performance_measured=False,
        shared_prefix_rows=2,private_current_rows=14,original_SDPA_is_only_gold=True)
    freeze(p); write(job / 'plan.json',p,exclusive=True)
    print(json.dumps(dict(event='CREATED',job=str(job))),flush=True)
    return job,p


def gpu(job,p):
    p = enabled_plan(job,cpu_gate=True)
    nvml, child, pins, start = NVML(), None, {}, None
    def cleanup():
        if child is None: return
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:
                now,pin = ownership.read_process(child.pid),pins[child.pid]
                if (now.start_ticks,now.uid,now.pgid,now.argv) == (pin.start_ticks,pin.uid,pin.pgid,pin.argv):
                    os.killpg(child.pid,sig)
            except (FileNotFoundError,ProcessLookupError): pass
            try: child.wait(timeout=4)
            except subprocess.TimeoutExpired: pass
    identity = dict(job=str(job),plan_sha256=digest(job / 'plan.json'),kind=REVISION)
    try:
        idle,limit = Idle(),time.monotonic()+180
        while True:
            sample = nvml.sample()
            with (job / 'resource-samples.jsonl').open('a') as f:
                f.write(json.dumps(dict(timestamp=stamp(),sample=sample))+'\n')
            if idle.add(sample,time.monotonic()): break
            if time.monotonic() >= limit: raise RuntimeError('stable idle unavailable; zero GPU attempts')
            time.sleep(1)
        if sample['memory_total_bytes'] != 40*2**30 or shutil.disk_usage(job).free < 20*2**30:
            raise ValueError('A10040GiB/disk20GiB gate')
        write(job / 'RESOURCE-GATE.json',dict(status='PASS',sample=sample,stable_samples=15),exclusive=True)
        freeze(p); start = time.monotonic()
        write(job / 'GPU-ATTEMPT.json',dict(attempt=1,execution_revision=REVISION,
            monotonic_start=start,monotonic_deadline=start+300,timestamp=stamp(),
            includes_loading_IO_synchronize_cleanup=True),exclusive=True)
        cmd = [p['model_python'],str(OWN / 'micro_worker.py'),'--job',str(job)]
        with (job / 'micro-worker.log').open('xb') as log:
            child = subprocess.Popen(cmd,env=environment(),cwd=OWN,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ownership.pin_roots([dict(pid=child.pid,process_group=child.pid,command=cmd)],pins)
        write(job / 'owned-processes.json',[dict(pid=child.pid,start_ticks=pins[child.pid].start_ticks,
            process_group=child.pid,command=cmd)],exclusive=True)
        print(json.dumps(dict(event='GPU_START',pid=child.pid,job=str(job))),flush=True)
        while child.poll() is None:
            sample = nvml.sample()
            apps = '\n'.join(str(v['pid'])+',0' for v in sample['compute']+sample['graphics'])
            foreign = ownership.foreign_pids(apps,pins)
            with (job / 'ownership-samples.jsonl').open('a') as f:
                f.write(json.dumps(dict(timestamp=stamp(),sample=sample,foreign=foreign))+'\n')
            if foreign or sample['memory_free_bytes'] < 8*2**30 or time.monotonic() >= start+285:
                raise RuntimeError('ownership/resource/deadline guard: '+str(foreign))
            time.sleep(2)
        if child.returncode: raise RuntimeError('micro worker failed: '+str(child.returncode))
        cleanup()
        recovery = recover(job / 'direct-counter.jsonl',identity,24)
        if recovery['status'] != 'CLOSED' or recovery['inflight'] or recovery['counter']['attempted'] != 24:
            raise ValueError('24 closed direct intents required')
        row = json.loads((job / 'MICRO-RESULT.json').read_text())
        sample = nvml.sample()
        if any(v['pid'] in pins for v in sample['compute']+sample['graphics']): raise ValueError('owned GPU PID remains')
        freeze(p); elapsed = time.monotonic()-start
        if elapsed > 300: raise TimeoutError('inclusive time cap')
        row.update(GPU_elapsed_seconds=elapsed,recovery=recovery,
            aggregate_target_intents=p['aggregate_prior_target_intents'],
            aggregate_GPU_seconds=p['aggregate_prior_GPU_seconds']+elapsed,
            cleanup=dict(owned_worker_exited=True,other_processes_not_signalled=True,sample=sample))
        write(job / 'FINAL_RESULT.json',row,exclusive=True)
        print(json.dumps(dict(event=row['status'],job=str(job),GPU_elapsed_seconds=elapsed)),flush=True)
    except BaseException as e:
        if start is not None: (job / 'STOP_REQUESTED').write_text(repr(e)+'\n')
        cleanup()
        row = dict(status='FAIL',error=repr(e),traceback=traceback.format_exc(),
                   GPU_started=start is not None,GPU_elapsed_seconds=None if start is None else time.monotonic()-start)
        try: row['recovery'] = recover(job / 'direct-counter.jsonl',identity,24)
        except Exception as exc: row['recovery'] = dict(status='UNKNOWN',error=repr(exc))
        write(job / 'FAILED.json',row,exclusive=True); raise
    finally:
        cleanup(); nvml.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--new',action='store_true')
    a = ap.parse_args()
    if not a.new: raise ValueError('fresh job required')
    os.environ[ENABLED] = os.environ[APPROVED] = '1'
    job,p = create_job()
    with (job / 'cpu-tests.log').open('xb') as log:
        subprocess.run([p['model_python'],str(OWN / 'test_cpu.py'),'--job',str(job)],
                       env=environment(cpu=True),cwd=OWN,stdout=log,stderr=subprocess.STDOUT,
                       check=True,timeout=180)
    print(json.dumps(dict(event='CPU_PASS',job=str(job))),flush=True)
    gpu(job,p)


if __name__ == '__main__': main()

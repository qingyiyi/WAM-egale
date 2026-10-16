"""Fresh finite full-episode experiment. No old zero-episode plan is changed."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
RESEARCH = ROOT/'投机采样'
WORKTREE = RESEARCH/'wam-p0-worktree'
BASE = OWN.parent/'single_worker_chc_gpu_v4'
for p in (BASE,):
    if str(p) not in sys.path:
        sys.path.append(str(p))
ENABLED = 'P0_SPECULATIVE_E2E_V3_ENABLED'
APPROVED = 'P0_SPECULATIVE_E2E_V3_GPU_APPROVED'
PREFIX = 'p0-speculative-e2e-v3-'
REVISION = 'natural-full-episode-original-K2-versus-S0-finite-disk-v3'
VA = '/rjs/cxz/miniconda3/envs/va/bin/python'
CLIENT = '/rjs/cxz/miniconda3/envs/libero-va/bin/python'

def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path, value, *, exclusive=False, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    tmp = path if exclusive else path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('x') as f:
        f.write(body)
        f.flush()
        if durable:
            os.fsync(f.fileno())
    if not exclusive:
        os.replace(tmp, path)
    if durable:
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)

def implementation():
    return {str(p): digest(p) for p in OWN.iterdir()
            if p.is_file() and p.suffix in ('.py', '.md', '.mjs')}

def freeze(p):
    for name, h in {**p['frozen_sha256'], **p['implementation_sha256']}.items():
        if digest(name) != h:
            raise ValueError('immutable input changed: '+name)
    if (ROOT/'.pipeline/cycle.yaml').exists():
        raise ValueError('protected cycle absence changed')
    for item in p['target_weight_inventory']:
        st = Path(item['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError('frozen weight stat changed: '+item['path'])

def plan(job, *, gpu=False):
    job = Path(job).resolve()
    if os.environ.get(ENABLED) != '1':
        raise ValueError('E2E experiment default off')
    if job.parent != RESEARCH/'results/runtime' or not job.name.startswith(PREFIX):
        raise ValueError('fresh private E2E job required')
    p = json.loads((job/'plan.json').read_text())
    exact = dict(job=str(job), execution_revision=REVISION, arms=['D','S'],
                 inits=[3,4], full_episode_cap=4, EGL_initialization_cap=4,
                 parallel_EGL_workers=2, model_worker_cap=4, no_retries=True,
                 training=False, RL=False, trace_control_changed=False,
                 profile_OFF=True, full_trace_OFF=True, teacher_OFF=True,
                 prompt_shared=False, rowproof_reuse=False,
                 per_episode_forward_cap=4000, probe_forward_cap=0,
                 GPU_hard_limit_seconds=5400, aggregate_target_cap=35000,
                 aggregate_GPU_seconds_cap=18000,
                 aggregate_prior_target_intents=17551,
                 aggregate_prior_GPU_seconds=5705.285101825371,
                 disk_initial_min_bytes=16*2**30,disk_runtime_floor_bytes=12*2**30,
                 new_output_bytes_cap=2**30)
    for k,v in exact.items():
        if type(p.get(k)) is not type(v) or p[k] != v:
            raise ValueError('full E2E scope mismatch: '+k)
    if set(p['implementation_sha256']) != set(implementation()):
        raise ValueError('implementation set changed')
    if p['aggregate_prior_target_intents']+4*4000 > p['aggregate_target_cap']:
        raise ValueError('aggregate forward reservations exceed cap')
    if p['aggregate_prior_GPU_seconds']+5400 > p['aggregate_GPU_seconds_cap']:
        raise ValueError('cumulative five-hour cap exceeded')
    freeze(p)
    previous = json.loads(Path(p['reference_target_job'],'RECONCILED-CLEANUP.json').read_text())
    if (previous['status'] != 'PASS_FAILED_JOB_OWNED_ORPHAN_CLEANUP'
            or previous['aggregate_target_intents'] != p['aggregate_prior_target_intents']
            or previous['aggregate_GPU_seconds'] != p['aggregate_prior_GPU_seconds']):
        raise ValueError('failed-job consumption/cleanup reconciliation required')
    for n in ('online_shared.py','runtime.py','client_entry.py','server_entry.py',
              'prefix_adapter.py','prefix_attention.py','prefix_kernel.py'):
        if digest(OWN/n) != digest(OWN.parent/'speculative_e2e_v1'/n):
            raise ValueError('only orchestration may change; common model/client code changed: '+n)
    numerical = json.loads(Path(p['reference_target_job'],'probe/PROBE-RESULT.json').read_text())
    if numerical['status'] != 'PASS_FULL_CHUNK_BITWISE':
        raise ValueError('unchanged full-chunk original/S0 bitwise gate required')
    if gpu:
        if (os.environ.get(APPROVED) != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'):
            raise ValueError('separate GPU boundary')
        g = json.loads((job/'preflight/CPU-GATE.json').read_text())
        r = json.loads((job/'AUTHORIZATION.json').read_text())
        if g['status'] != 'PASS' or g['plan_sha256'] != digest(job/'plan.json'):
            raise ValueError('fresh CPU gate required')
        if (r['plan_sha256'] != digest(job/'plan.json') or not r['authorized']
                or r['GPU_seconds_cap'] != 18000 or not r['user_text']):
            raise ValueError('plan-bound persistent user authorization required')
    return p

def worker_entry():
    job = Path(os.environ['P0_SPEC_E2E_JOB']).resolve()
    p = plan(job, gpu=True)
    if any((job/f).exists() for f in ('FAILED.json','FINAL_RESULT.json','STOP_REQUESTED')):
        raise ValueError('terminal/stopped job cannot start worker')
    a = json.loads((job/'GPU-ATTEMPT.json').read_text())
    if time.monotonic() >= a['monotonic_deadline']-30:
        raise ValueError('bounded GPU deadline')
    out = Path(os.environ['P0_SPEC_E2E_OUT']).resolve()
    if not out.is_relative_to(job):
        raise ValueError('owned worker output')
    out.mkdir(parents=True, exist_ok=True)
    reservation = json.loads((out/'FORWARD-RESERVATION.json').read_text())
    if (reservation['out'] != str(out) or reservation['plan_sha256'] != digest(job/'plan.json')
            or reservation['cap'] != (400 if out == job/'probe' else 4000)
            or a['plan_sha256'] != digest(job/'plan.json')):
        raise ValueError('durable finite worker reservation required')
    import ownership
    pins = {r['pid']:ownership.Process(**{**r,'argv':tuple(r['argv'])})
            for r in json.loads((job/'OWNED-PROCESSES.json').read_text())}
    if not ownership.is_owned(ownership.read_process(os.getpid()),pins):
        raise ValueError('model worker must descend from exact pinned owned root')
    write(out/'GPU-ENTRY.json', dict(pid=os.getpid(), timestamp=stamp(),
                                    plan_sha256=digest(job/'plan.json')), exclusive=True,
          durable=True)
    return job, out, p

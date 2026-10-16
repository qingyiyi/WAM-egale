"""Default-off, pinned, one-shot numerical comparison; no model forwards."""
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
BASE = OWN.parent / 'single_worker_chc_gpu_v4'
OBS = ROOT / '投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534'
PRIOR = ROOT / '投机采样/results/runtime/p0-shared-prefix-torch-v1-20261012-193802'
sys.path.append(str(BASE))
PREFIX = 'p0-shared-prefix-torch-v2-'
REVISION = 'original-torch-flash-build-align-v2'
ENABLED = 'P0_SHARED_TORCH_V2_ENABLED'
APPROVED = 'P0_SHARED_TORCH_V2_GPU_APPROVED'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as f:
            f.write(body)
    else:
        tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
        with tmp.open('x') as f:
            f.write(body)
        os.replace(tmp, path)


def implementation():
    return {str(p): digest(p) for p in OWN.rglob('*') if p.is_file()
            and 'build' not in p.relative_to(OWN).parts and '__pycache__' not in p.parts
            and (p.suffix in ('.py', '.h', '.hpp', '.cuh', '.cu', '.so', '.diff')
                 or p.name in ('SOURCE-PROVENANCE.json', 'FROZEN-INPUTS.json'))}


def freeze(p):
    frozen = json.loads((OWN / 'FROZEN-INPUTS.json').read_text())['files']
    for path, h in {**frozen, **p['implementation_sha256']}.items():
        if digest(path) != h:
            raise ValueError('frozen source/evidence changed: ' + path)
    if (ROOT / '.pipeline/cycle.yaml').exists():
        raise ValueError('protected cycle absence changed')


def enabled_plan(job, *, cpu_gate=False):
    if os.environ.get(ENABLED) != '1' or os.environ.get(APPROVED) != '1':
        raise ValueError('original-torch numerical experiment is default off')
    job = Path(job).resolve()
    if job.parent != ROOT / '投机采样/results/runtime' or not job.name.startswith(PREFIX):
        raise ValueError('new private job required')
    p = json.loads((job / 'plan.json').read_text())
    exact = dict(job=str(job), execution_revision=REVISION, hard_target_forward_cap=0,
                 direct_attention_micro_call_cap=24, GPU_hard_limit_seconds=300,
                 GPU_attempt_cap=1, GPU_model_worker_cap=1, new_full_episode_cap=0,
                 EGL_initialization_cap=0, complete_policy_chunk_cap=0,
                 training=False, RL=False, no_retries=True, layers=[0,8,29])
    if any(type(p.get(k)) is not type(v) or p[k] != v for k,v in exact.items()):
        raise ValueError('numerical comparison scope mismatch')
    old = json.loads((PRIOR / 'FINAL_RESULT.json').read_text())
    if (p['aggregate_prior_target_intents'] != old['aggregate_target_intents']
            or p['aggregate_prior_GPU_seconds'] != old['aggregate_GPU_seconds']
            or p['aggregate_target_cap'] != 900 or p['aggregate_GPU_seconds_cap'] != 2700):
        raise ValueError('prior consumption must be carried forward')
    if set(p['implementation_sha256']) != set(implementation()):
        raise ValueError('implementation file set changed')
    if cpu_gate:
        gate = json.loads((job / 'CPU-GATE.json').read_text())
        if gate['status'] != 'PASS' or gate['plan_sha256'] != digest(job / 'plan.json'):
            raise ValueError('fresh plan-bound CPU gate required')
    freeze(p)
    return p


def gpu_entry_gate(job, *, claim=False):
    p = enabled_plan(job, cpu_gate=True)
    job = Path(job)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('one GPU required')
    if any((job / f).exists() for f in ('FAILED.json','FINAL_RESULT.json','STOP_REQUESTED')):
        raise ValueError('terminal job cannot be reused')
    a = json.loads((job / 'GPU-ATTEMPT.json').read_text())
    if a['execution_revision'] != REVISION or a['attempt'] != 1 or time.monotonic() >= a['monotonic_deadline'] - 15:
        raise ValueError('fresh finite attempt required')
    marker = job / 'GPU-ENTRY.json'
    if claim:
        write(marker, dict(pid=os.getpid(), job=str(job), timestamp=stamp()), exclusive=True)
    elif json.loads(marker.read_text())['pid'] != os.getpid():
        raise ValueError('owned worker required')
    return p


def counter_enabled():
    raise RuntimeError('target-model forwards forbidden in this micro comparison')

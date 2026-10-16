"""Bounded diagnostic contract; frozen implementation helpers remain read-only."""
import json
import os
from pathlib import Path
import sys
import time

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
PREFIX = OWN.parent / 'prefix_acceptance_v1'
sys.path.insert(0, str(PREFIX))
from paths import WORKTREE, EXP, S0, load
from experiment import write, stamp, digest

VA = '/rjs/cxz/miniconda3/envs/va/bin/python'


def slots():
    return [dict(slot=0, arm='RAW_CURRENT', lane='oracle'),
            dict(slot=1, arm='CURRENT', lane='warmup'),
            dict(slot=2, arm='PROFILE_CURRENT', lane='diagnostic')]


def plan(job, gpu=False):
    job = Path(job).resolve()
    p = json.loads((job / 'plan.json').read_text())
    assert p['job'] == str(job) and p['fixed_slots'] == slots()
    assert p['fixed_target_cap'] == 1500 and p['GPU_hard_limit_seconds'] == 900
    assert p['aggregate_prior_GPU_seconds'] + 900 <= 36000
    assert p['aggregate_prior_target_intents'] + 1500 <= p['aggregate_target_cap']
    assert p['complete_chunks'] == 9 and p['model_worker_cap'] == 1
    assert p['profiled_verifier_calls'] == 6 and p['formal_timing_samples'] == 0
    assert p['selector'] == 'prefix_then_cost' and p['EGL_initialization_cap'] == 0
    assert p['profile_OFF'] is False and p['training'] is False
    assert not (ROOT / '.pipeline/cycle.yaml').exists()
    prior = json.loads(Path(p['source_carry']).read_text())
    assert p['aggregate_prior_GPU_seconds'] == prior['aggregate_GPU_seconds']
    assert p['aggregate_prior_target_intents'] == prior['aggregate_target_intents']
    assert p['aggregate_target_cap'] == prior['aggregate_target_cap']
    for path, sha in p['source_sha256'].items():
        if digest(path) != sha:
            raise ValueError('frozen source changed: ' + path)
    for item in p['target_weight_inventory']:
        s = Path(item['path']).stat()
        assert (s.st_size, s.st_mtime_ns) == (item['bytes'], item['mtime_ns'])
    if gpu:
        assert os.environ.get('MV_PROFILE_GPU_APPROVED') == '1'
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        h = digest(job / 'plan.json')
        gate = json.loads((job / 'preflight/CPU-GATE.json').read_text())
        auth = json.loads((job / 'AUTHORIZATION.json').read_text())
        assert gate['status'] == 'PASS' and not gate['CUDA_initialized']
        assert gate['plan_sha256'] == auth['plan_sha256'] == h and auth['authorized']
    return p


def worker_entry():
    job = Path(os.environ['P0_SPEC_E2E_JOB']).resolve()
    out = Path(os.environ['P0_SPEC_E2E_OUT']).resolve()
    p = plan(job, gpu=True)
    assert out == job / 'fixed/worker0'
    for name in ('FAILED.json', 'GPU-COLLECTION.json', 'STOP_REQUESTED'):
        assert not (job / name).exists()
    attempt = json.loads((job / 'GPU-ATTEMPT.json').read_text())
    assert time.monotonic() < attempt['monotonic_deadline'] - 25
    reservation = json.loads((out / 'FORWARD-RESERVATION.json').read_text())
    assert reservation['cap'] == p['fixed_target_cap']
    assert reservation['plan_sha256'] == digest(job / 'plan.json')
    import ownership
    limit = time.monotonic() + 3
    while True:
        records = json.loads((job / 'OWNED-PROCESSES.json').read_text())
        pins = {x['pid']: ownership.Process(**{**x, 'argv': tuple(x['argv'])}) for x in records}
        if ownership.is_owned(ownership.read_process(os.getpid()), pins):
            break
        if time.monotonic() > limit:
            raise ValueError('worker must have pinned owned ancestry')
        time.sleep(.01)
    write(out / 'GPU-ENTRY.json', dict(pid=os.getpid(), at=stamp(), cap=p['fixed_target_cap']), exclusive=True)
    return job, out, p

"""Freeze a fresh diagnostic job and validate its launch without importing torch."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import types

from mv_profile_contract import OWN, ROOT, PREFIX, load, write, stamp, digest, slots, plan

PRIOR_JOB = Path('/var/tmp/cxz-wam-quality-20261014/p0-current-chunk-lengths-v1-20261014-183953')


def prepare(job):
    if os.sched_getaffinity(0) != {0}:
        raise RuntimeError('preparation is CPU0 only')
    job = Path(job).resolve()
    assert str(job).startswith('/var/tmp/cxz-wam-quality-20261014/p0-megaverifier-profile-v1-')
    job.mkdir(parents=True, exist_ok=False)
    (job / 'preflight').mkdir()
    p = json.loads((PRIOR_JOB / 'plan.json').read_text())
    carry = json.loads((PRIOR_JOB / 'GPU-COLLECTION.json').read_text())
    assert carry['owned_roots_exited'] and carry['status'] == 'PASS_CURRENT_CHUNK_COLLECTION'
    for path, expected in p['source_sha256'].items():
        assert digest(path) == expected, 'old validated source changed: ' + path
    p = copy.deepcopy(p)
    p.update(job=str(job), fixed_slots=slots(), fixed_target_cap=1500, GPU_hard_limit_seconds=900,
        aggregate_prior_GPU_seconds=carry['aggregate_GPU_seconds'],
        aggregate_prior_target_intents=carry['aggregate_target_intents'],
        source_carry=str(PRIOR_JOB / 'GPU-COLLECTION.json'), complete_chunks=9,
        profiled_verifier_calls=6, formal_timing_samples=0, profile_OFF=False,
        CPU_affinity=[1, 2, 3, 4, 5, 6, 7], CPU_threads=7, ports=[19320, 19321],
        at=stamp(), previous_attempt=None,
        workload='real short/middle/long same-algorithm oracle, capture and profile',
        trace_per_file_max_bytes=128*1024*1024,
        source_prior_benchmark=str(PRIOR_JOB),
        no_latency_claim=True, architecture_training_on_GPU=False)
    for name in ('fix', 'resource_recheck'):
        p.pop(name, None)
    required = {'mv_profile_contract.py', 'supervisor.py', 'profile_worker.py',
                'profile_capture.py', 'prepare_profile.py'}
    assert required <= {x.name for x in OWN.glob('*.py')}
    checks = []
    for path in sorted(OWN.glob('*.py')):
        ast.parse(path.read_text(), filename=str(path))
        p['source_sha256'][str(path)] = digest(path)
        checks.append('syntax:' + path.name)
    write(job / 'plan.json', p, exclusive=True, durable=True)
    assert plan(job) == p
    for port in p['ports']:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    checks.append('ports available at CPU preparation')
    module = load('mv_profile_preflight_supervisor', OWN / 'supervisor.py')
    assert Path(module.__file__).resolve() == OWN / 'supervisor.py'
    obj = module.Campaign.__new__(module.Campaign)
    obj.p, obj.job = p, job
    env = obj.environment('CURRENT', job / 'fixed/worker0')
    expected = {'PREFIX_ACCEPT_ENABLED': '1', 'PREFIX_ACCEPT_GPU_APPROVED': '1',
                'MV_PROFILE_GPU_APPROVED': '1', 'CUDA_VISIBLE_DEVICES': '0',
                'WAM_P0_ACTION_DRAFT_MODE': 'verifier_k2_b2_bonus',
                'WAM_P0_ACTION_DRAFT_SELECTOR': 'global', 'WAM_P0_PHASE_PROFILE': '0'}
    assert all(env.get(k) == value for k, value in expected.items())
    checks.append('actual supervisor child launch environment and namespace')
    kernel = load('mv_profile_preflight_prefix_kernel', PREFIX / 'prefix_kernel.py')
    original_environment = dict(os.environ)
    missing = object()
    native_before = sys.modules.get('_shared_torch_v1', missing)
    sentinel = object()
    try:
        sys.modules['_shared_torch_v1'] = types.SimpleNamespace(shared=lambda *args: sentinel)
        for enable, approve in [('0', '1'), ('1', '0')]:
            os.environ.update(PREFIX_ACCEPT_ENABLED=enable, PREFIX_ACCEPT_GPU_APPROVED=approve)
            try:
                kernel.run(*([None] * 7))
            except ValueError:
                checks.append('frozen kernel gate blocks ' + enable + '/' + approve)
            else:
                raise AssertionError('default-off frozen kernel gate failed')
        os.environ.update(expected)
        assert kernel.run(*([None] * 7)) is sentinel
        checks.append('both authorized gates reach mocked native function; no CUDA')
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
        if native_before is missing:
            del sys.modules['_shared_torch_v1']
        else:
            sys.modules['_shared_torch_v1'] = native_before
    assert 'torch' not in sys.modules
    checks.append('no torch import or GPU compute during CPU preparation')
    write(job / 'AUTHORIZATION.json', dict(authorized=True, at=stamp(),
        plan_sha256=digest(job / 'plan.json'),
        source='User approved continuing draft architecture experiment and current-verifier profile; exclusive local GPU and cumulative existing cap retained.',
        no_extra_GPU_budget=True), exclusive=True, durable=True)
    write(job / 'preflight/CPU-GATE.json', dict(status='PASS', at=stamp(),
        plan_sha256=digest(job / 'plan.json'), CUDA_initialized=False, torch_imported=False,
        checks=checks, source_files=len(p['source_sha256']),
        actual_child_environment={k: env[k] for k in expected},
        independent_review_required_before_launch=True), exclusive=True, durable=True)
    print(json.dumps(dict(status='PREPARED_NO_GPU_LAUNCH', job=str(job),
        plan_sha256=digest(job / 'plan.json'), checks=len(checks)), ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    prepare(ap.parse_args().job)

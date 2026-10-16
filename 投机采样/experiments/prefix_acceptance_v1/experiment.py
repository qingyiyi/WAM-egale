"""Finite bonus experiment; frozen target/draft and accepted O1/O2/O3 core."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from paths import ROOT, OWN, S0, WORKTREE

RESEARCH = ROOT / '投机采样'
VA = '/rjs/cxz/miniconda3/envs/va/bin/python'
CLIENT = '/rjs/cxz/miniconda3/envs/libero-va/bin/python'
ENABLED = 'PREFIX_ACCEPT_ENABLED'
APPROVED = 'PREFIX_ACCEPT_GPU_APPROVED'
PREFIX = 'p0-prefix-acceptance-v1-'
MODES = {'T': 'off', 'DEPTH': 'verifier_k2_b2_bonus', 'PREFIX': 'verifier_k2_b2_bonus'}
SELECTORS = {'T':'global','DEPTH':'depth_then_cost','PREFIX':'prefix_then_cost'}


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def write(path, value, *, exclusive=False, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path if exclusive else path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        if durable:
            os.fsync(f.fileno())
    if not exclusive:
        os.replace(tmp, path)


def assignments():
    return [
        [dict(arm='T', init=6, cap=4000), dict(arm='PREFIX', init=7, cap=4000)],
        [dict(arm='DEPTH', init=6, cap=4000), dict(arm='T', init=7, cap=4000)],
        [dict(arm='PREFIX', init=6, cap=4000), dict(arm='DEPTH', init=7, cap=4000)],
    ]


def fixed_slots():
    return [dict(slot=i, arm=a, selector=SELECTORS[a.split('_')[-1]], optimized=not a.startswith('RAW_'))
        for i,a in enumerate(['RAW_DEPTH', 'DEPTH', 'RAW_PREFIX', 'PREFIX'])]


def implementation():
    return {str(p): digest(p) for p in OWN.glob('*.py')}


def metadata(path):
    s = Path(path).stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                ino=s.st_ino, dev=s.st_dev)


def freeze(p, *, full=False):
    sets = [p['implementation_sha256'], p['critical_source_sha256']]
    if full:
        sets.append(p['frozen_sha256'])
    for files in sets:
        for path, h in files.items():
            if digest(path) != h:
                raise ValueError('immutable input changed: ' + path)
    for path, expected in p['frozen_file_stats'].items():
        if metadata(path) != expected:
            raise ValueError('immutable input metadata changed: ' + path)
    for item in p['target_weight_inventory']:
        s = Path(item['path']).stat()
        if (s.st_size, s.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError('target weights changed')
    if (ROOT / '.pipeline/cycle.yaml').exists():
        raise ValueError('protected Cycle absence changed')


def plan(job, *, gpu=False, full=False):
    job = Path(job).resolve()
    if job.parent != RESEARCH / 'results/runtime' or not job.name.startswith(PREFIX):
        raise ValueError('fresh private bonus job required')
    p = json.loads((job / 'plan.json').read_text())
    exact = dict(job=str(job), arms=['T', 'DEPTH', 'PREFIX'], inits=[6, 7],
        parallel_EGL_workers=2, full_episode_cap=6, EGL_initialization_cap=6,
        model_worker_cap=7, fixed_model_workers=1, fixed_target_cap=640,
        fixed_full_chunk_cap=8, target_cap=24640, GPU_hard_limit_seconds=2100,
        aggregate_prior_target_intents=46654, aggregate_prior_GPU_seconds=14118.697842976078,
        aggregate_target_cap=75000, aggregate_GPU_seconds_cap=18000,
        initial_resource_wait_seconds=900, profile_OFF=True, teacher_OFF=True,
        full_trace_OFF=True, training=False, no_retries=True, prompt_shared=False,
        rowproof_reuse=False, trace_control_changed=False,
        artifact_plumbing_original=True, speed_comparison_deferred=True, chunk_events_ON=True, disk_initial_min_bytes=15 * 2**30,
        disk_runtime_floor_bytes=12 * 2**30, new_output_bytes_cap=2 * 2**30)
    if any(type(p.get(k)) is not type(v) or p[k] != v for k, v in exact.items()):
        raise ValueError('bounded bonus scope mismatch')
    if p['assignments'] != assignments() or p['fixed_slots'] != fixed_slots():
        raise ValueError('original/previous/candidate balanced assignments changed')
    if set(p['implementation_sha256']) != set(implementation()):
        raise ValueError('source manifest changed')
    if p['aggregate_prior_target_intents'] + p['target_cap'] > p['aggregate_target_cap']:
        raise ValueError('bounded target reservation exceeded')
    if p['aggregate_prior_GPU_seconds'] + p['GPU_hard_limit_seconds'] > 18000:
        raise ValueError('user five-hour cap exceeded')
    freeze(p, full=full)
    if gpu:
        if (os.getenv(ENABLED) != '1' or os.getenv(APPROVED) != '1'
                or os.getenv('CUDA_VISIBLE_DEVICES') != '0'):
            raise ValueError('explicit default-off GPU boundary')
        g = json.loads((job / 'preflight/CPU-GATE.json').read_text())
        a = json.loads((job / 'AUTHORIZATION.json').read_text())
        h = digest(job / 'plan.json')
        if g['status'] != 'PASS' or g['plan_sha256'] != h or g['CUDA_initialized']:
            raise ValueError('fresh CPU gate required')
        if not a['authorized'] or a['plan_sha256'] != h:
            raise ValueError('persistent user authorization required')
    return p


def worker_entry(kind='episode'):
    job = Path(os.environ['P0_SPEC_E2E_JOB']).resolve()
    p = plan(job, gpu=True)
    out = Path(os.environ['P0_SPEC_E2E_OUT']).resolve()
    if kind == 'fixed':
        if out != job / 'fixed/worker0':
            raise ValueError('single deployment fixture worker required')
        cap = p['fixed_target_cap']
    elif kind == 'episode':
        arm, init = os.environ['P0_SPEC_E2E_ARM'], int(os.environ['BONUS_ACCEPT_INIT'])
        match = [x for wave in assignments() for x in wave if (x['arm'], x['init']) == (arm, init)]
        if len(match) != 1 or out != job / 'campaigns' / arm / 'workers' / f'worker{init-6}':
            raise ValueError('exact full-episode assignment required')
        cap = match[0]['cap']
    else:
        raise ValueError('unsupported worker kind')
    if any((job / f).exists() for f in ('FAILED.json', 'FINAL_RESULT.json', 'STOP_REQUESTED')):
        raise ValueError('terminal or stopped job')
    attempt = json.loads((job / 'GPU-ATTEMPT.json').read_text())
    if time.monotonic() >= attempt['monotonic_deadline'] - 25:
        raise ValueError('bounded GPU deadline')
    r = json.loads((out / 'FORWARD-RESERVATION.json').read_text())
    if r['cap'] != cap or r['plan_sha256'] != digest(job / 'plan.json'):
        raise ValueError('durable forward reservation required')
    import ownership
    limit = time.monotonic() + 3
    while True:
        records = json.loads((job / 'OWNED-PROCESSES.json').read_text())
        pins = {x['pid']: ownership.Process(**{**x, 'argv': tuple(x['argv'])}) for x in records}
        if ownership.is_owned(ownership.read_process(os.getpid()), pins):
            break
        if time.monotonic() > limit:
            raise ValueError('worker must be an owned descendant')
        time.sleep(.01)
    for k, v in [('MUJOCO_GL', 'egl'), ('PYOPENGL_PLATFORM', 'egl'), ('MUJOCO_EGL_DEVICE_ID', '0')]:
        if os.getenv(k) != v:
            raise ValueError('EGL provenance missing')
    write(out / 'GPU-ENTRY.json', dict(pid=os.getpid(), kind=kind, cap=cap, at=stamp()),
          exclusive=True)
    return job, out, p

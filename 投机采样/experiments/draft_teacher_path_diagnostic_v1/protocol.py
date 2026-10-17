"""Bounded quality diagnosis, fixed public base; no training admission."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = OWN.parents[2]
RUNTIME = ROOT / '投机采样/results/runtime'
COMMIT = 'b0dfac3a02f65de33a1217246b83003615166640'
FIRST = RUNTIME / 'p0-task4-direction-diagnostic-v1-20261016-r1'
SECOND = RUNTIME / 'p0-task4-direction-diagnostic-completion-v1-20261016-r2'
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
                  noninferiority_established=False, formal_performance_permission=False)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(job):
    job = Path(job).resolve()
    require(job.parent == RUNTIME and not job.exists(), 'fresh own runtime namespace required')
    require(subprocess.check_output(['git', '-C', str(ROOT / 'review_exports/WAM-egale'),
        'rev-parse', 'HEAD'], text=True).strip() == COMMIT, 'pinned base differs')
    p = read(SECOND / 'plan.json')
    for path, expected in p['source_sha256'].items():
        require(sha(path) == expected, 'previous frozen source changed: ' + path)
    arms = {}
    for arm, source_job in (('T', FIRST), ('TF_CONT', SECOND)):
        key = arm + '-task4-init26-repeat0'
        folder = source_job / 'episodes' / key
        files = dict(trace=str(folder / 'direction_trace/chunk-000.pt'),
                     counter=str(folder / 'COUNTER.json'),
                     observations=str(folder / 'client/initial_observations.npz'),
                     rng=str(source_job / 'arms' / arm / 'TARGET-INITIAL-RNG.pt'))
        for prefix in ('actions', 'latents'):
            found = list((folder / 'server_output').rglob(prefix + '_0.pt'))
            require(len(found) == 1, 'one original chunk-zero ' + prefix + ' required')
            files[prefix] = str(found[0])
        arms[arm] = files
    evidence = [FIRST / 'FAILED.json', SECOND / 'FAILED.json', SECOND / 'plan.json',
        RUNTIME / 'p0-draft-freeze-quality-v1-local-20261016-r1/RESULT.json']
    evidence += [Path(v) for row in arms.values() for v in row.values()]
    sources = {**p['source_sha256'], **{str(x): sha(x) for x in OWN.glob('*.py')}}
    extra = EXP / 'fixed_load_cost/snapshot.py'
    sources[str(extra)] = sha(extra)
    extra = EXP / 'wam_quality_diagnostic_v1/resource_probe.py'
    sources[str(extra)] = sha(extra)
    now = datetime.now(timezone.utc)
    plan = dict(schema='draft-teacher-path-diagnostic/v1', job=str(job), commit=COMMIT,
        goal='Prioritize draft usability: isolate first teacher discrepancy before any new training',
        created_at=now.isoformat(), reference_plan=p, arms=arms,
        source_sha256=sources, evidence_sha256={str(x): sha(x) for x in evidence},
        permissions=PERMISSIONS, candidate='fixed TF64 CONT3000 research fixture only',
        known_case=dict(task=4, init=26, repeat=0, chunk=0, scheduler_index=1),
        fixed_paths=['B2 ordinary', 'B2 readonly', 'B14 dense', 'B14 existing shared-prefix'],
        full_policy_chunk_replays=2, full_episode_count=0, new_environment_commands=0,
        gate='Both recorded full policy chunks/actions/video/directions must reproduce exactly before final same-query claims',
        target_forward_cap=256, max_run_wall_sec=1200, active_wall_sec=1020,
        cleanup_reserve_sec=180, absolute_cleanup_deadline=(now + timedelta(minutes=35)).isoformat(),
        GPU='local NVIDIA A100-SXM4-40GB', workers=1, shared_quality_GPU_allowed=True,
        CPU_affinity=[12, 13], CPU_threads=2, master_port=29658,
        storage_cap_bytes=2**30, disk_floor_bytes=5*2**30, initial_free_GiB=22,
        scope='Real query and complete recorded policy chunk correctness; NOT robot episode or formal timing',
        no_training=True, no_threshold_change=True, no_120=True, no_AutoDL=True,
        no_architecture_sweep=True, no_system_optimization=True, no_automatic_retry=True,
        retained_legacy_failures=True, performance_claim=False,
        cleanup='Independent CPU supervisor and OS timeout; exact spawned identities via existing ownership cleanup',
        terminal_status='REGISTERED_NOT_STARTED')
    job.mkdir()
    write(job / 'PREREGISTRATION.json', plan)
    write(job / 'plan.json', {**plan, 'preregistration_sha256': sha(job / 'PREREGISTRATION.json')})
    return plan


def load(job):
    job = Path(job).resolve()
    p = read(job / 'plan.json')
    require(p['job'] == str(job) and p['commit'] == COMMIT and p['permissions'] == PERMISSIONS,
            'identity/permissions differ')
    require((p['target_forward_cap'], p['max_run_wall_sec'], p['active_wall_sec'],
        p['cleanup_reserve_sec'], p['storage_cap_bytes']) == (256, 1200, 1020, 180, 2**30),
        'fixed numeric diagnostic budget differs')
    require(sha(job / 'PREREGISTRATION.json') == p['preregistration_sha256'], 'registration changed')
    for group in ('source_sha256', 'evidence_sha256'):
        for path, expected in p[group].items():
            require(sha(path) == expected, 'frozen input/source changed: ' + path)
    for path, info in p['reference_plan']['target_weight_stats'].items():
        s = Path(path).stat()
        require((s.st_size, s.st_mtime_ns) == (info['bytes'], info['mtime_ns']), 'target file stat changed')
    return p

"""Conditional, one-shot four-shard DEV dispatcher; never handles TEST."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

EXP = Path(__file__).resolve().parent.parent
ROOT = EXP.parent.parent
sys.path.insert(0, str(EXP))
from draft_frozen_validation_v1.protocol import (
    MANIFEST, SHARDS, _manifest, _registration, _today_gate,
    load, prepare, read, sha, write,
)

WINDOW_START = datetime.fromisoformat('2026-10-17T08:00:00+08:00')
WINDOW_HARD = datetime.fromisoformat('2026-10-17T20:00:00+08:00')
DISPATCH_WHOLE_LIFE_SECONDS = 32*3600
LAUNCH_SOURCE = Path(__file__).resolve().parent/'launch_one.py'


def cpu2():
    allowed = sorted(os.sched_getaffinity(0))[:2]
    if len(allowed) != 2:
        raise RuntimeError('two CPU cores unavailable for dispatcher')
    os.sched_setaffinity(0, allowed)
    return allowed


def birth_monotonic():
    fields = Path('/proc/self/stat').read_text().rpartition(') ')[2].split()
    birth = int(fields[19])/os.sysconf('SC_CLK_TCK')
    if not 0 <= time.monotonic()-birth < 86400:
        raise ValueError('dispatcher actual process birth unavailable')
    return birth, int(fields[19])


def exact_registration(path):
    _today_gate()
    doc = _manifest(MANIFEST)
    path, reg = _registration(path, doc)
    start = datetime.fromisoformat(reg['cohort_started_at'])
    hard = datetime.fromisoformat(reg['absolute_cleanup_deadline'])
    if start < WINDOW_START or hard > WINDOW_HARD:
        raise ValueError('tomorrow 08:00 not-before/20:00 absolute hard boundary differs')
    if reg['test_manifest_sha256'] != sha(EXP/'draft_frozen_validation_v1/TEST-MANIFEST.json'):
        raise ValueError('reserved TEST manifest changed')
    return path, reg


def remaining(deadline, current=None):
    now = time.monotonic()
    if now >= deadline:
        raise TimeoutError('dispatcher whole-life/absolute CPU wait deadline reached')
    if current is not None and (current/'STOP_REQUESTED').exists():
        raise RuntimeError('active shard STOP_REQUESTED; preserve original terminal')
    return deadline-now


def terminal(job):
    if not (job/'RESULT.json').is_file() or not (job/'CLOCK-RESULT.json').is_file():
        return None
    plan = load(job)
    result = read(job/'RESULT.json')
    collection = read(job/'COLLECTION.json')
    final = read(job/'FINAL-OWNED-CLEANUP.json')
    clock = read(job/'CLOCK-RESULT.json')
    if (result.get('status') != 'COMPLETE30_FROZEN_VALIDATION_SHARD_ANALYSIS'
            or collection.get('status') != 'PASS_COMPLETE_FROZEN_VALIDATION_SHARD_COLLECTION'
            or collection.get('completed') != [row['key'] for row in plan['cases']]
            or final.get('all_roots_reaped') is not True
            or clock.get('status') != 'CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS'
            or clock.get('all_observed_owned_live_processes_exited') is not True
            or clock.get('original_quality_RESULT_present') is not True
            or clock.get('original_quality_FAILED_present') is not False
            or clock.get('external_stop_reason') is not None
            or clock.get('error') is not None or clock.get('signals')
            or clock.get('no_foreign_signals') is not True
            or any(job.rglob('*FAILED*.json')) or any(job.rglob('*STOP*'))):
        raise ValueError('actual shard terminal/cleanup/failure-first admission rejected')
    if any(read(job/'arms'/arm/'OWNED-CLEANUP.json').get('all_roots_reaped') is not True
           for arm in plan['arms']):
        raise ValueError('one arm subtree cleanup is incomplete')
    return dict(plan=plan, result=result, clock=clock)


def wait_terminal(job, deadline):
    while True:
        # A stopped or failed supervisor still needs the independent clock's
        # owned-process cleanup receipt before this dispatcher signs failure.
        remaining(deadline)
        if any(job.rglob('*FAILED*.json')) or any(job.rglob('*STOP*')):
            if (job/'CLOCK-RESULT.json').is_file() or not (job/'RUN-STARTED.json').is_file():
                raise RuntimeError('shard failed/stopped; original evidence preserved')
        if (job/'CLOCK-RESULT.json').is_file() and not (job/'RESULT.json').is_file():
            raise RuntimeError('supervisor exited without complete RESULT; clock receipt preserved')
        found = terminal(job)
        if found is not None:
            return found
        time.sleep(min(5, remaining(deadline)))


def wait_boundary(metadata, deadline, next_shard):
    marker = metadata/'PAUSE_AFTER_SHARD'
    if not marker.exists():
        return
    write(metadata/f'DISPATCH-PAUSED-{next_shard}.json', dict(status='PAUSED_ONLY_AT_COMPLETE_SHARD_BOUNDARY',
        marker=str(marker), next_shard=next_shard, at=datetime.now(timezone.utc).isoformat()))
    while marker.exists():
        time.sleep(min(5, remaining(deadline)))
    write(metadata/f'DISPATCH-RESUMED-{next_shard}.json', dict(status='EXPLICIT_MARKER_REMOVED_AFTER_COMPLETE_SHARD',
        next_shard=next_shard, at=datetime.now(timezone.utc).isoformat()))


def observe_failed_launch_cleanup(job, deadline):
    if not (job/'ROOT-CLOCK.log').exists() and not (job/'CLOCK-GUARD.json').exists():
        return dict(status='NO_ATTACHED_CLOCK_START_OBSERVED',
                    owned_cleanup_observed=False, supervisor_GPU_admission_unverified=True)
    limit = min(deadline, time.monotonic()+300)
    while time.monotonic() < limit:
        receipt = job/'CLOCK-RESULT.json'
        if receipt.is_file():
            try:
                actual = read(receipt)
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            return dict(status='ORIGINAL_CLOCK_TERMINAL_OBSERVED',
                        clock_result_sha256=sha(receipt), clock_status=actual.get('status'),
                        owned_cleanup_observed=actual.get('all_observed_owned_live_processes_exited') is True,
                        no_foreign_signals=actual.get('no_foreign_signals'))
        time.sleep(min(2, max(0, limit-time.monotonic())))
    return dict(status='CLOCK_TERMINAL_NOT_OBSERVED_WITHIN_REMAINING_300_SECONDS',
                owned_cleanup_observed=False, original_clock_may_continue_independently=True)


def _merge_worker(metadata, shard_jobs):
    from draft_frozen_validation_v1.analyze import merge
    merged = merge(shard_jobs, MANIFEST)
    pending = metadata/'COHORT-ANALYSIS.pending'
    with pending.open('x') as stream:
        json.dump(merged, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    pending.replace(metadata/'COHORT-ANALYSIS.json')


def bounded_real_merge(metadata, reg, deadline):
    work_left = (datetime.fromisoformat(reg['absolute_work_stop'])-
                 datetime.now(timezone.utc)).total_seconds()
    limit = min(remaining(deadline)-20, work_left)
    if limit <= 0:
        raise TimeoutError('registered work time exhausted before full raw cohort merge')
    process = multiprocessing.get_context('spawn').Process(
        target=_merge_worker,
        args=(metadata, {shard: reg['shard_jobs'][shard] for shard in SHARDS}))
    process.start()
    process.join(limit)
    if process.is_alive():
        process.terminate()
        process.join(min(10, max(0, deadline-time.monotonic())))
        if process.is_alive():
            process.kill()
            process.join(min(10, max(0, deadline-time.monotonic())))
        raise TimeoutError('full raw cohort merge exceeded registered work/dispatcher deadline')
    if process.exitcode != 0 or not (metadata/'COHORT-ANALYSIS.json').is_file():
        raise RuntimeError('actual CPU full raw cohort merge failed; no complete receipt')
    return read(metadata/'COHORT-ANALYSIS.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    affinity = cpu2()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    born, ticks = birth_monotonic()
    deadline = min(born+DISPATCH_WHOLE_LIFE_SECONDS, time.monotonic()+max(0,
        (WINDOW_HARD-datetime.now(timezone.utc)).total_seconds()))
    path, reg = exact_registration(args.registration)
    registration_sha256 = sha(path)
    metadata = Path(reg['cohort_metadata_root'])
    dispatch_source_sha256 = sha(__file__)
    launcher_source_sha256 = sha(LAUNCH_SOURCE)
    if (metadata/'DISPATCH-STARTED.json').exists() or any(metadata.glob('DISPATCH-FAILED*.json')):
        raise ValueError('one-shot dispatcher; no retry or concurrent second dispatcher')
    write(metadata/'DISPATCH-STARTED.json', dict(
        status='CONDITIONAL_FOUR_SHARD_DISPATCH_STARTED',
        pid=os.getpid(), start_ticks=ticks, CPU_affinity=affinity,
        started_at=datetime.now(timezone.utc).isoformat(),
        registration=str(path), registration_sha256=registration_sha256,
        source_sha256=dispatch_source_sha256, launcher_source_sha256=launcher_source_sha256,
        absolute_deadline=WINDOW_HARD.isoformat(), max_cpu_lifetime_seconds=DISPATCH_WHOLE_LIFE_SECONDS,
        no_GPU_in_dispatcher=True, TEST_launch_forbidden=True))
    current = None
    try:
        while datetime.now(timezone.utc) < WINDOW_START:
            remaining(deadline)
            if sha(path) != registration_sha256:
                raise ValueError('frozen registration changed during conditional CPU wait')
            time.sleep(min(5, remaining(deadline)))
        for index, shard in enumerate(SHARDS):
            remaining(deadline)
            if sha(path) != registration_sha256:
                raise ValueError('frozen registration changed before a complete shard')
            exact_registration(path)
            if sha(__file__) != dispatch_source_sha256 or sha(LAUNCH_SOURCE) != launcher_source_sha256:
                raise ValueError('reviewed dispatch or detached-launch source changed during cohort')
            job = Path(reg['shard_jobs'][shard])
            wait_boundary(metadata, deadline, shard)
            if (sha(path) != registration_sha256 or sha(__file__) != dispatch_source_sha256
                    or sha(LAUNCH_SOURCE) != launcher_source_sha256):
                raise ValueError('registration/dispatcher/launcher changed across a shard-boundary pause')
            if index:
                previous = Path(reg['shard_jobs'][SHARDS[index-1]])
                prior = terminal(previous)
                if prior is None:
                    raise ValueError('previous complete shard and independent clock absent')
                write(previous/'NEXT-SHARD-RELEASE.json', dict(
                    status='ALLOW_NEXT_FROZEN_VALIDATION_SHARD', next_shard=shard,
                    prior_result_sha256=sha(previous/'RESULT.json'),
                    prior_clock_result_sha256=sha(previous/'CLOCK-RESULT.json'),
                    at=datetime.now(timezone.utc).isoformat(),
                    exact_complete_shard_boundary=True))
            plan = prepare(job, manifest=MANIFEST, shard=shard, registration=path)
            copy = job/'ROOT-DETACHED-LAUNCH.py'
            if copy.exists():
                raise ValueError('launcher copy already exists; no retry')
            shutil.copyfile(LAUNCH_SOURCE, copy)
            if sha(copy) != sha(LAUNCH_SOURCE):
                raise ValueError('copied actual pinned launcher source differs')
            current = job
            va = plan['va_python']
            launch = subprocess.run([va, '-B', str(copy)], cwd=ROOT, env=dict(os.environ),
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
            write(job/'DISPATCH-LAUNCH-EXIT.json', dict(returncode=launch.returncode,
                stdout_tail=launch.stdout[-3000:], stderr_tail=launch.stderr[-3000:],
                launcher_source_sha256=sha(copy)))
            if launch.returncode:
                raise RuntimeError('actual pinned detached launch failed; see original job receipts')
            wait_terminal(job, deadline)
            current = None
        merged = bounded_real_merge(metadata, reg, deadline)
        if (merged.get('status') != 'COMPLETE120_EXPANDED_DEV_DESCRIPTIVE_COHORT'
                or merged.get('complete') is not True or merged.get('full_episodes') != 120):
            raise ValueError('real all-120 CPU merge returned PARTIAL; no complete cohort')
        write(metadata/'DISPATCH-COMPLETE.json', dict(
            status='FOUR_SHARDS_FULLY_EXECUTED_AND_INDEPENDENTLY_CLEANED',
            shard_jobs=reg['shard_jobs'], original_RESULTS_only=True,
            cohort_analysis_sha256=sha(metadata/'COHORT-ANALYSIS.json'),
            at=datetime.now(timezone.utc).isoformat(),
            no_TEST_run=True, no_formal_performance_permission=True))
    except BaseException as exc:
        cleanup = None if current is None else observe_failed_launch_cleanup(current, deadline)
        write(metadata/'DISPATCH-FAILED.json', dict(
            status='FAILED_OR_PARTIAL_FROZEN_VALIDATION_DISPATCH',
            error=repr(exc), traceback=traceback.format_exc(),
            active_shard=None if current is None else str(current),
            active_shard_cleanup=cleanup,
            at=datetime.now(timezone.utc).isoformat(),
            no_retry=True, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()

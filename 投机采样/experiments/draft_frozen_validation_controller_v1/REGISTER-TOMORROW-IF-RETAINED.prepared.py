"""One-shot CPU registration/dispatcher launch. Prepared source: do not run early."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/rjs/cxz/论文探索')
EXP = ROOT/'投机采样/experiments'
RUNTIME = ROOT/'投机采样/results/runtime'
sys.path.insert(0, str(EXP))
from draft_frozen_validation_v1 import protocol

OWN = EXP/'draft_frozen_validation_v1'
DISPATCH = EXP/'draft_frozen_validation_dispatch_v1/dispatch.py'
LAUNCH = DISPATCH.with_name('launch_one.py')
CLOCK = EXP/'draft_frozen_validation_clock_v1/watch.py'
COHORT = RUNTIME/'p0-draft-frozen-validation-v1-cohort-local-20261017-r1'
REGISTRATION = COHORT/'REGISTRATION.json'
START = datetime.fromisoformat('2026-10-17T08:00:00+08:00')
WORK = datetime.fromisoformat('2026-10-17T19:55:00+08:00')
HARD = datetime.fromisoformat('2026-10-17T20:00:00+08:00')
SHARDS = protocol.SHARDS
CPUS = (0, 1)
EXPECTED_SOURCE_SHA256 = {
    str(DISPATCH): '6943399d890431300efadd687cc169cded485f28593b3deb9c4ae59b73b8035f',
    str(LAUNCH): 'f27374c45e54c0ea43f25eb9248e7fd6b976c259949a1e7491df0b8c6e84cb2e',
    str(CLOCK): 'd9d932ed8cb0b57719123008e96fc3e6389eafb50e316d6b1f1215a40ab6499f',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    tmp = path.with_name(path.name+'.pending')
    with tmp.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def live_pin(pid, argv):
    proc = Path('/proc')/str(pid)
    fields = (proc/'stat').read_text().rpartition(') ')[2].split()
    observed = [os.fsdecode(v) for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    if (fields[0] in ('Z', 'X') or observed != argv or int(fields[2]) != pid
            or proc.stat().st_uid != os.geteuid()
            or sorted(os.sched_getaffinity(pid)) != list(CPUS)):
        raise ValueError('actual detached CPU dispatcher pin/argv/affinity differs')
    return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]),
                start_ticks=int(fields[19]), uid=os.geteuid(), argv=observed,
                CPU_affinity=list(CPUS))


def set_dispatch_affinity():
    os.sched_setaffinity(0, CPUS)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    if datetime.now(timezone.utc) >= START:
        raise TimeoutError('CPU waiter registration must be launched before tomorrow 08:00')
    if not set(CPUS).issubset(os.sched_getaffinity(0)):
        raise ValueError('fixed dispatcher CPU cores 0,1 unavailable')
    os.sched_setaffinity(0, CPUS)
    if COHORT.exists() or any((RUNTIME/f'p0-draft-frozen-validation-v1-local-{s}-20261017-r1').exists() for s in SHARDS):
        raise ValueError('one-shot cohort/shard path already exists; no retry or overwrite')
    for path, expected in EXPECTED_SOURCE_SHA256.items():
        if sha(path) != expected:
            raise ValueError('independently reviewed dispatcher/clock source changed: '+path)
    evidence = protocol._today_gate()  # Original full RESULT, retention and all cleanup only.
    manifest = protocol._manifest(protocol.MANIFEST)
    frozen_sources = {str(p): sha(p) for p in protocol.source_files()}
    if len(frozen_sources) != 181:
        raise ValueError('future complete source inventory is not the independently reviewed 181 files')
    today_plan = protocol.read(protocol.TODAY_JOB/'plan.json')
    va = Path(today_plan['va_python'])
    resolved_va = va.resolve(strict=True)
    if not va.is_absolute() or not os.access(resolved_va, os.X_OK):
        raise ValueError('actual today50 Python executable unavailable')
    jobs = {s: str(RUNTIME/f'p0-draft-frozen-validation-v1-local-{s}-20261017-r1') for s in SHARDS}
    registration = dict(
        schema='draft-frozen-validation-execution-registration/v1',
        status='FROZEN_BEFORE_EXPANDED_DEV', site='local', profile='expanded_DEV_init36_39',
        cohort_metadata_root=str(COHORT), cohort_started_at=START.isoformat(),
        absolute_work_stop=WORK.isoformat(), absolute_cleanup_deadline=HARD.isoformat(),
        manifest_path=str(protocol.MANIFEST), manifest_sha256=sha(protocol.MANIFEST),
        test_manifest_path=str(protocol.TEST_MANIFEST), test_manifest_sha256=sha(protocol.TEST_MANIFEST),
        shard_order=list(SHARDS), shard_jobs=jobs,
        clock_wrapper_path=str(CLOCK), clock_wrapper_sha256=sha(CLOCK),
        clock_provenance_name='DRAFT-FROZEN-VALIDATION-CLOCK-PROVENANCE.json',
        per_shard_max_seconds=14400, cleanup_reserve_seconds=300,
        cohort_gpu_job_cap_seconds=28800, cohort_wall_cap_seconds=43200,
        per_shard_output_budget_GiB=3, cohort_output_budget_GiB=5, disk_floor_GiB=5,
        today_terminal_receipts=evidence,
        fixed_candidate_sha256={arm: today_plan['selected'][arm]['checkpoint_sha256']
                                for arm in ('TF_CONT', 'MLP_CONT')},
        frozen_source_sha256=frozen_sources,
        dispatch_source_sha256=sha(DISPATCH), launcher_source_sha256=sha(LAUNCH),
        original_va_python_argv=str(va), resolved_va_python=str(resolved_va),
        registered_by_source_sha256=sha(__file__),
        no_TEST_launch=True, formal_performance_permission=False)
    COHORT.mkdir()  # Exclusive creation: an interrupted registration is not retried.
    atomic_new(REGISTRATION, registration)
    protocol._registration(REGISTRATION, manifest)
    if protocol._today_gate() != evidence:
        raise ValueError('today50 original terminal bytes changed during registration')
    command = [str(va), '-B', str(DISPATCH), '--registration', str(REGISTRATION)]
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    hz = os.sysconf('SC_CLK_TCK')
    child = None
    try:
        with (COHORT/'DISPATCH-ROOT.log').open('xb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True, preexec_fn=set_dispatch_affinity)
        pin = live_pin(child.pid, command)
        atomic_new(COHORT/'DISPATCH-DETACHED-LAUNCH.json', dict(
            status='PINNED_DETACHED_CPU_WAITING_DISPATCHER', command=command,
            start_new_session=True, pin=pin, boot_id=boot, SC_CLK_TCK=hz,
            registered_at=datetime.now(timezone.utc).isoformat(),
            registration=str(REGISTRATION), registration_sha256=sha(REGISTRATION),
            controller_source_sha256=sha(__file__),
            dispatcher_source_sha256=sha(DISPATCH), launcher_source_sha256=sha(LAUNCH),
            CPU_waiter_whole_life_cap_seconds=115200,
            absolute_dispatcher_hard_deadline=HARD.isoformat(),
            GPU_job_started=False, future_shards_prepared=False))
        limit = time.monotonic()+30
        receipt = COHORT/'DISPATCH-STARTED.json'
        while not receipt.is_file():
            if child.poll() is not None or time.monotonic() >= limit:
                raise RuntimeError('detached dispatcher did not produce actual CPU start receipt')
            time.sleep(.2)
        started = protocol.read(receipt)
        if (started.get('pid') != pin['pid'] or started.get('start_ticks') != pin['start_ticks']
                or started.get('CPU_affinity') != list(CPUS)
                or started.get('registration_sha256') != sha(REGISTRATION)
                or started.get('source_sha256') != sha(DISPATCH)):
            raise ValueError('actual dispatcher start receipt differs from detached process')
        print(json.dumps(dict(status='ACTUAL_CPU_WAITER_LAUNCHED_NOT_GPU_JOB',
            registration=str(REGISTRATION), dispatcher_pid=pin['pid'],
            dispatcher_start_ticks=pin['start_ticks'], dispatcher_receipt=str(receipt)),
            ensure_ascii=False), flush=True)
    except BaseException as exc:
        stopped = False
        if child is not None and child.poll() is None:
            try:
                live_pin(child.pid, command)
                child.terminate()
                child.wait(timeout=10)
                stopped = True
            except (OSError, ValueError, subprocess.TimeoutExpired):
                # A PID that cannot be pinned must never receive a signal.
                pass
        atomic_new(COHORT/'REGISTRATION-LAUNCH-FAILED.json', dict(
            status='FAILED_OR_UNVERIFIED_CPU_WAITER_LAUNCH', error=repr(exc),
            traceback=traceback.format_exc(), dispatcher_pid=None if child is None else child.pid,
            actual_cpu_dispatcher_may_still_be_live=child is not None and not stopped,
            exact_pinned_dispatcher_stopped_on_failure=stopped,
            no_GPU_job_claim=True, no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()

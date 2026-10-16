"""Copy into one registered shard job as ROOT-DETACHED-LAUNCH.py; run once."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/rjs/cxz/论文探索')
EXP = ROOT/'投机采样/experiments'
sys.path.insert(0, str(EXP))
from draft_frozen_validation_v1.protocol import load, sha, write

JOB = Path(__file__).resolve().parent
SOURCE = EXP/'draft_frozen_validation_dispatch_v1/launch_one.py'
RUN = EXP/'draft_frozen_validation_v1/run.py'
WORKER_CPUS = (12, 13, 14, 15)


def set_worker_affinity():
    # The launcher and attached clock stay on two cores. The detached
    # supervisor gets the four-core allocation actually used by today50.
    os.sched_setaffinity(0, WORKER_CPUS)


def live_pin(pid, command):
    proc = Path('/proc')/str(pid)
    fields = (proc/'stat').read_text().rpartition(') ')[2].split()
    actual = [os.fsdecode(x) for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if (fields[0] in ('Z', 'X') or actual != command
            or proc.stat().st_uid != os.geteuid() or int(fields[2]) != pid):
        raise ValueError('actual detached direct child argv/UID/PGID differs')
    return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]),
                start_ticks=int(fields[19]), uid=os.geteuid(), argv=actual)


def stop_owned(child, command):
    if child is None or child.poll() is not None:
        return
    live_pin(child.pid, command)
    child.terminate()
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        # The independent pinned clock owns later descendants; never signal
        # a process group or an uncertain/reused PID here.
        pass


def wait_file(path, child, seconds, label):
    deadline = time.monotonic()+seconds
    while not path.is_file():
        if child.poll() is not None:
            raise RuntimeError(label+' exited before receipt; inspect its log')
        if time.monotonic() >= deadline:
            raise TimeoutError(label+' receipt absent within fixed admission window')
        time.sleep(.1)


def main():
    if Path(__file__).resolve() != JOB/'ROOT-DETACHED-LAUNCH.py' or sha(__file__) != sha(SOURCE):
        raise ValueError('one exact reviewed launcher copy inside the shard job required')
    plan = load(JOB)
    watch = Path(plan['clock_wrapper_path']).resolve(strict=True)
    if (sha(watch) != plan['clock_wrapper_sha256']
            or datetime.now(timezone.utc) >= datetime.fromisoformat(plan['absolute_work_stop'])):
        raise ValueError('registered clock/source/work window expired')
    if any((JOB/name).exists() for name in
           ('RUN-STARTED.json', 'ROOT-DETACHED-LAUNCH.json', 'CLOCK-GUARD.json',
            'COLLECTION.json', 'RESULT.json', 'FAILED.json', 'ROOT-LAUNCH-ERROR.json')):
        raise ValueError('no retry or preexisting run/clock/terminal receipt permitted')
    affinity = sorted(os.sched_getaffinity(0))[:2]
    if len(affinity) != 2:
        raise RuntimeError('CPU two-core launcher allocation unavailable')
    os.sched_setaffinity(0, affinity)
    va = plan['va_python']
    command = [va, '-B', str(RUN), '--job', str(JOB)]
    watch_command = [va, '-B', str(watch), '--attach', '--job', str(JOB)]
    worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    clock_env = dict(worker_env, CUDA_VISIBLE_DEVICES='')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    hz = os.sysconf('SC_CLK_TCK')
    worker = guard = None
    try:
        with (JOB/'ROOT-RUN.log').open('xb') as stream:
            worker = subprocess.Popen(command, cwd=ROOT, env=worker_env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True, preexec_fn=set_worker_affinity)
        wait_file(JOB/'RUN-STARTED.json', worker, 60, 'supervisor')
        worker_pin = live_pin(worker.pid, command)
        worker_affinity = sorted(os.sched_getaffinity(worker.pid))
        if worker_affinity != list(WORKER_CPUS):
            raise ValueError('actual detached supervisor did not receive fixed four-core CPU affinity')
        started = json.loads((JOB/'RUN-STARTED.json').read_text())
        if (started.get('pid') != worker_pin['pid']
                or started.get('start_ticks') != worker_pin['start_ticks']
                or started.get('shard') != plan['shard']):
            raise ValueError('actual supervisor RUN birth differs from detached child')
        source_sha = sha(__file__)
        write(JOB/'ROOT-DETACHED-LAUNCH.json', dict(
            status='PINNED_DETACHED_EXPLICIT_FROZEN_VALIDATION_SHARD_LAUNCH',
            at=datetime.now(timezone.utc).isoformat(), command=command,
            start_new_session=True, pin=worker_pin, boot_id=boot, SC_CLK_TCK=hz,
            source_sha256=source_sha, CPU_affinity=affinity,
            launcher_CPU_affinity=affinity,
            supervisor_CPU_affinity=worker_affinity,
            registered_plan_sha256=sha(JOB/'plan.json'), no_GPU_in_launcher=True))
        checked = subprocess.run([va, '-B', str(watch), '--inspect', '--job', str(JOB)],
            cwd=ROOT, env=clock_env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=20, check=False)
        if checked.returncode:
            raise RuntimeError('real clock inspect rejected: '+checked.stdout[-1000:]+checked.stderr[-1000:])
        inspected = json.loads(checked.stdout)
        if (inspected.get('status') != 'PASS_ACTUAL_REGISTERED_DEV_SHARD_CPU_INSPECT_ONLY_NOT_ATTACHED'
                or inspected.get('supervisor_pin', {}).get('pid') != worker.pid
                or inspected.get('supervisor_pin', {}).get('start_ticks') != worker_pin['start_ticks']):
            raise ValueError('clock inspection did not bind actual supervisor birth')
        write(JOB/'ROOT-CLOCK-INSPECT.json', inspected)
        with (JOB/'ROOT-CLOCK.log').open('xb') as stream:
            guard = subprocess.Popen(watch_command, cwd=ROOT, env=clock_env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        wait_file(JOB/'CLOCK-GUARD.json', guard, 20, 'independent attached clock')
        if worker.poll() is not None:
            raise RuntimeError('supervisor exited before clock guard admission')
        guard_pin = live_pin(guard.pid, watch_command)
        write(JOB/'ROOT-CLOCK-DETACHED-LAUNCH.json', dict(
            status='ACTUAL_DETACHED_CPU_CLOCK_ATTACHED', command=watch_command,
            start_new_session=True, pin=guard_pin, boot_id=boot, SC_CLK_TCK=hz,
            launcher_source_sha256=source_sha))
        print(json.dumps(dict(status='LAUNCHED_GUARDED_FROZEN_VALIDATION_SHARD',
            job=str(JOB), shard=plan['shard'], supervisor_pid=worker.pid,
            guard_pid=guard.pid, full_episodes=30), ensure_ascii=False), flush=True)
    except BaseException as exc:
        write(JOB/'ROOT-LAUNCH-ERROR.json', dict(status='FAILED_GUARDED_SHARD_LAUNCH',
            error=repr(exc), at=datetime.now(timezone.utc).isoformat()))
        try:
            stop_owned(worker, command)
        except (OSError, ValueError, ProcessLookupError):
            pass
        # A started clock must remain alive to reap the supervisor's owned
        # process tree. Before its guard receipt exists, the supervisor's
        # startup barrier still prevents any model or GPU work.
        raise


if __name__ == '__main__':
    main()

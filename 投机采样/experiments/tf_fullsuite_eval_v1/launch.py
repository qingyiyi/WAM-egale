"""Start this registered full-episode campaign once; no test, probe or digest."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

OWN = Path(__file__).resolve().parent
sys.path.insert(0, str(OWN.parent))
from tf_fullsuite_eval_v1.protocol import load, read, write, require


def birth(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()[19])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    a = parser.parse_args()
    job = a.job.resolve(strict=True)
    p = load(job)
    require(p['source_code_commit'] not in ('PENDING_NOT_EXECUTABLE', ''), 'source must be frozen')
    require(not (job/'LAUNCH-STARTED.json').exists(), 'one attempt; no implicit retry')
    require((job/'DEPLOYMENT-COMPLETE.json').is_file(), 'actual source/asset staging receipt required')
    write(job/'LAUNCH-STARTED.json', dict(status='STARTING_REGISTERED_COMPLETE_EPISODES',
        at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        source_commit=p['source_code_commit'], independent_tests=False, remote_SHA=False))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    children = []
    try:
        with (job/'guard.log').open('xb') as f:
            guard = subprocess.Popen([p['va_python'], '-B', str(OWN/'guard.py'), '--job', str(job)],
                stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env,
                close_fds=True, start_new_session=True)
        children.append(guard)
        guard_birth = birth(guard.pid)
        write(job/'GUARD-PROCESS.json', dict(pid=guard.pid, birth_ticks=guard_birth,
            pgid=os.getpgid(guard.pid), owner_job=str(job), role='independent_absolute_guard'))
        until = time.monotonic()+15
        while not (job/'GUARD-STARTED.json').exists():
            require(guard.poll() is None and time.monotonic() < until, 'actual guard did not start')
            time.sleep(.1)
        require(read(job/'GUARD-STARTED.json')['pid'] == guard.pid, 'guard receipt identity differs')
        with (job/'runner.log').open('xb') as f:
            runner = subprocess.Popen([p['va_python'], '-B', str(OWN/'runner.py'), '--job', str(job)],
                stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env,
                close_fds=True, start_new_session=True)
        children.append(runner)
        pin = dict(pid=runner.pid, birth_ticks=birth(runner.pid), pgid=os.getpgid(runner.pid),
            uid=os.getuid(), owner_job=str(job), role='fullsuite_runner',
            registered_at=datetime.now(timezone.utc).isoformat())
        write(job/f'OWNED-{runner.pid}.json', pin)
        write(job/'LAUNCH-RESULT.json', dict(status='DETACHED_RUNNER_AND_GUARD_STARTED',
            guard_pid=guard.pid, runner=pin, no_episode_result_claim=True,
            at=datetime.now(timezone.utc).isoformat()))
        print(str(job), runner.pid, guard.pid, flush=True)
        return 0
    except BaseException as exc:
        write(job/'LAUNCH-FAILED.json', dict(status='FAILED_LAUNCH', error=repr(exc),
            children=[dict(pid=x.pid, returncode=x.poll()) for x in children]))
        # The independently running guard observes this explicit stop and
        # signals only registered birth-pinned groups, never other jobs.
        if not (job/'STOP_REQUESTED').exists():
            (job/'STOP_REQUESTED').write_text('launch failed; preserve attempt\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())

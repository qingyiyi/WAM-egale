"""Start a registered suite phase once, with its independent owned-process guard."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from tf_suite_pipeline_v1.protocol import load, read, write, require, PERMISSIONS
from tf_suite_pipeline_v1.runner import process, matches, forensic_groups


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    p = load(job)
    require(p['source_code_commit'] not in ('PENDING_NOT_EXECUTABLE', ''), 'suite source must be frozen')
    require((job / 'DEPLOYMENT-COMPLETE.json').is_file(), 'actual suite asset/source staging receipt required')
    work = datetime.fromisoformat(p['absolute_work_stop']).timestamp()
    hard = datetime.fromisoformat(p['absolute_cleanup_deadline']).timestamp()
    require(time.time() < work, 'no phase launch after registered work stop')
    control = p['CPU_affinity']['control']
    require(isinstance(control, list) and len(control) == 1 and type(control[0]) is int,
        'one registered suite control CPU required')
    write(job / 'LAUNCH-STARTED.json', dict(schema='tf-suite-phase-launch/v1',
        status='STARTING_REGISTERED_COMPLETE_EPISODES', at=stamp(), pid=os.getpid(),
        source_code_commit=p['source_code_commit'], inherited_CPU_affinity=sorted(os.sched_getaffinity(0)),
        control_CPU_affinity=control, no_independent_tests=True, remote_SHA=False, permissions=PERMISSIONS))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    children = []

    def spawn(role, filename):
        argv = [p['va_python'], '-B', str(OWN / filename), '--job', str(job)]
        previous = os.sched_getaffinity(0)
        with (job / (role + '.log')).open('xb') as stream:
            try:
                os.sched_setaffinity(0, set(control))
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    env=env, close_fds=True, start_new_session=True, cwd=OWN)
            finally:
                os.sched_setaffinity(0, previous)
        record = dict(proc=child, pin=None, role=role)
        children.append(record)
        actual = process(child.pid)
        if actual is None:
            child.wait(timeout=5)
            raise RuntimeError('fresh suite ' + role + ' exited before lifetime registration')
        pin = dict(pid=child.pid, birth_ticks=actual['birth_ticks'], pgid=actual['pgid'], uid=actual['uid'],
            owner_job=str(job), role='suite_runner' if role == 'runner' else 'independent_suite_guard',
            argv=argv, CPU_affinity=control, registered_at=stamp())
        record['pin'] = pin
        require(pin['pgid'] == pin['pid'] and pin['uid'] == os.getuid(), 'fresh suite session identity differs')
        write(job / (f'OWNED-{child.pid}.json' if role == 'runner' else 'GUARD-PROCESS.json'), pin)
        return record

    def signal_owned(record, sig, signals):
        if record['proc'].poll() is not None:
            return
        pin = record['pin']
        if pin is None or not matches(pin):
            signals.append(dict(pid=record['proc'].pid, refused=True, reason='birth/UID/PGID no longer proven'))
            return
        try:
            os.killpg(pin['pgid'], sig)
            signals.append(dict(pid=pin['pid'], birth_ticks=pin['birth_ticks'], pgid=pin['pgid'],
                uid=pin['uid'], signal=sig.name, at=stamp()))
        except ProcessLookupError:
            pass

    try:
        guard = spawn('guard', 'guard.py')
        until = min(time.time() + 15, work)
        while not (job / 'GUARD-STARTED.json').is_file():
            require(guard['proc'].poll() is None and time.time() < until, 'actual independent suite guard did not start')
            time.sleep(.1)
        observed = read(job / 'GUARD-STARTED.json')
        require(observed['schema'] == 'tf-suite-independent-guard/v1'
            and observed['pid'] == guard['pin']['pid'] and observed['birth_ticks'] == guard['pin']['birth_ticks']
            and observed['uid'] == os.getuid() and observed['owner_job'] == str(job)
            and observed['work_stop'] == work and observed['hard_deadline'] == hard
            and observed['permissions'] == PERMISSIONS, 'actual suite guard identity/deadlines differ')
        runner = spawn('runner', 'runner.py')
        write(job / 'LAUNCH-RESULT.json', dict(schema='tf-suite-phase-launch-result/v1',
            status='DETACHED_SUITE_RUNNER_AND_GUARD_STARTED', at=stamp(), guard=guard['pin'], runner=runner['pin'],
            no_episode_result_claim=True, permissions=PERMISSIONS))
        print(str(job), runner['pin']['pid'], guard['pin']['pid'], flush=True)
        return 0
    except BaseException as exc:
        # Keep the launch attempt and stop only its exact fresh child lifetimes.
        # The independent guard remains alive while the runner closes its children.
        if not (job / 'STOP_REQUESTED').exists():
            try:
                (job / 'STOP_REQUESTED').write_text('suite launch failed; no automatic retry\n')
            except OSError:
                pass
        signals, waits = [], []
        runners = [r for r in children if r['role'] == 'runner']
        guards = [r for r in children if r['role'] == 'guard']
        for r in runners:
            signal_owned(r, signal.SIGTERM, signals)
            try:
                r['proc'].wait(timeout=max(0, min(25, hard - time.time() - 5)))
            except subprocess.TimeoutExpired:
                signal_owned(r, signal.SIGKILL, signals)
                try:
                    r['proc'].wait(timeout=max(0, min(5, hard - time.time())))
                except subprocess.TimeoutExpired:
                    pass
            waits.append(dict(pid=r['proc'].pid, role='runner', returncode=r['proc'].poll()))
        # Let the guard clean registered model/client children before asking
        # the guard itself to exit. A still-live owner remains supervised.
        for r in guards:
            try:
                r['proc'].wait(timeout=max(0, min(65, hard - time.time())))
            except subprocess.TimeoutExpired:
                owned = []
                try:
                    for file in job.glob('OWNED-*.json'):
                        candidate = read(file)
                        if (candidate.get('owner_job') == str(job) and candidate.get('uid') == os.getuid()
                                and matches(candidate)):
                            owned.append(candidate)
                except (OSError, ValueError, KeyError):
                    owned = ['UNKNOWN']
                if not owned:
                    signal_owned(r, signal.SIGTERM, signals)
                    try:
                        r['proc'].wait(timeout=max(0, min(5, hard - time.time())))
                    except subprocess.TimeoutExpired:
                        signal_owned(r, signal.SIGKILL, signals)
                        try:
                            r['proc'].wait(timeout=max(0, min(5, hard - time.time())))
                        except subprocess.TimeoutExpired:
                            pass
                else:
                    waits.append(dict(pid=r['proc'].pid, role='guard',
                        retained_until_registered_absolute_deadline=True, unresolved_owned=owned))
            waits.append(dict(pid=r['proc'].pid, role='guard', returncode=r['proc'].poll()))
        proof = forensic_groups([r['pin'] for r in children if r['pin'] is not None])
        write(job / 'LAUNCH-FAILED.json', dict(schema='tf-suite-phase-launch-failed/v1', status='FAILED_LAUNCH',
            error=repr(exc), traceback=traceback.format_exc(), at=stamp(), signals=signals, waits=waits,
            FORENSIC_group_members=proof, all_launch_child_groups_empty=proof['scan_complete']
                and not any(g['members'] for g in proof['groups']),
            no_implicit_retry=True, permissions=PERMISSIONS))
        raise


if __name__ == '__main__':
    raise SystemExit(main())

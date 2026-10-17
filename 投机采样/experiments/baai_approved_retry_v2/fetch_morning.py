"""Return the bounded 09:15 remote CPU report to the original project at 09:20."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import io
import json
import os
import shlex
import signal
import subprocess
import tarfile
import time
import traceback


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    p = json.loads((job / 'REGISTRATION.json').read_text())
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    target = datetime.fromisoformat(p['fetch_at']).timestamp()
    stop = datetime.fromisoformat(p['cleanup_at']).timestamp()
    interrupted = False

    def on_signal(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    write(job / 'STARTED.json', dict(pid=os.getpid(), at=time.time(), CPU=1, GPU=0))
    child = None
    try:
        while time.time() < target:
            assert not interrupted and not (job / 'STOP_REQUESTED').exists()
            time.sleep(min(30, target - time.time()))
        ssh = ['ssh', '-T', '-x', '-a', '-i', p['identity_file'],
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + p['known_hosts'],
            '-p', str(p['port']), p['host']]
        code = r'''
from pathlib import Path
import io,sys,tarfile
base=(Path.home()/'cxz').resolve(strict=True)
assert Path.cwd().resolve()==base
job=base/'WAM-egale/投机采样/results/runtime/p0-long-first-round-morning-cpu1-v2-20261018-r2'
if not (job/'TERMINAL.json').exists():raise SystemExit(75)
paths=[p for p in job.rglob('*') if p.is_file()]
assert sum(p.stat().st_size for p in paths)<=16*2**20
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as out:
 for p in paths:out.add(p,arcname=str(p.relative_to(job)),recursive=False)
'''
        attempts = 0
        while True:
            assert not interrupted and time.time() + 30 < stop
            attempts += 1
            command = 'cd "$HOME/cxz" && PYTHONDONTWRITEBYTECODE=1 python3 -B -c ' + shlex.quote(code)
            completed = subprocess.run(ssh + [command], capture_output=True,
                                       timeout=min(60, stop-time.time()-10), start_new_session=True)
            write(job / f'READ-ATTEMPT-{attempts:02d}.json', dict(
                returncode=completed.returncode, at=time.time(), bytes=len(completed.stdout),
                stderr=completed.stderr.decode(errors='replace')[-2000:],
                read_only_no_episode_retry=True))
            if completed.returncode == 75:
                time.sleep(20)
                continue
            assert completed.returncode == 0, 'read-only report return failed'
            assert len(completed.stdout) <= 17 * 2**20
            destination = job / 'remote-originals'
            destination.mkdir(exist_ok=False)
            with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode='r:') as tar:
                for item in tar:
                    rel = Path(item.name)
                    assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
                    dest = destination / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(item) as source, dest.open('xb') as out:
                        out.write(source.read())
            terminal = json.loads((destination / 'TERMINAL.json').read_text())
            assert terminal['status'] == 'CPU_REPORT_COMPLETE'
            result = json.loads((destination / 'report/RESULT.json').read_text())
            assert not any(result['permissions'].values())
            hw = Path(p['HW_result_file'])
            hw.parent.mkdir(parents=True, exist_ok=True)
            with hw.open('x') as stream:
                stream.write((destination / 'report/LONG-RESULT.md').read_text())
                stream.write('\n本机已回传原件：' + str(destination / 'report') + '\n')
                stream.write('观测时间：' + result['observed_at'] + '\n')
            write(job / 'TERMINAL.json', dict(status='MORNING_REPORT_RETURNED_TO_LOCAL_HW',
                complete_cases=result['complete_common_cases'], report_status=result['status'],
                HW_file=str(hw), source=str(destination / 'report/RESULT.json'),
                at=datetime.now(timezone.utc).isoformat(), no_SHA=True, no_GPU_jobs=True))
            break
    except BaseException as exc:
        write(job / 'TERMINAL.json', dict(status='REPORT_RETURN_PARTIAL_OR_FAILED',
            error=repr(exc), traceback=traceback.format_exc(), at=time.time()))
        raise
    finally:
        write(job / 'CLEANUP.json', dict(status='CPU_REPORT_RETURNER_EXITS', at=time.time(),
            no_GPU_processes_started=True, no_foreign_processes_signalled=True))


if __name__ == '__main__':
    main()

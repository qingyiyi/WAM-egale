"""单个有上限作业的owned监督；只向PID+出生时间均匹配的子进程发信号。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    path = Path(f'/proc/{pid}')
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, birth_ticks=int(fields[19]), uid=path.stat().st_uid)
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--name', required=True)
    p.add_argument('--cap', required=True, type=int)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    directory = args.registration.parent / args.name
    directory.mkdir(exist_ok=False)
    prior = list(args.registration.parent.glob('*/OWNED-RESULT.json'))
    consumed = sum(json.loads(f.read_text()).get('lifecycle_seconds', 0) for f in prior)
    available = min(reg['GPU_lifecycle_seconds_cap']-consumed,
        datetime.fromisoformat(reg['absolute_cleanup_deadline']).timestamp()-time.time()-30,
        args.cap)
    if available < 30:
        raise RuntimeError('没有完整作业及清理预算')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    command = [sys.executable if x == '{python}' else x for x in command]
    if Path(command[0]).name == 'ncu':
        if '--clock-control' not in command or command[command.index('--clock-control')+1] != 'none':
            raise PermissionError('项目禁止锁频；NCU必须显式使用 --clock-control none')
        if '--cache-control' not in command:
            raise ValueError('NCU必须显式登记cache-control口径')
    record = dict(command=command, started_at=datetime.now().astimezone().isoformat(),
        consumed_prior_lifecycle_seconds=consumed, cap_seconds=available, status='RUNNING')
    def save():
        (directory / 'OWNED-RESULT.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    start = time.monotonic()
    with (directory / 'worker.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        owned = identity(child.pid)
        record['owned'] = owned
        save()
        try:
            child.wait(timeout=available-15)
            record.update(status='NATURAL_EXIT', exit_code=child.returncode)
        except subprocess.TimeoutExpired:
            record['status'] = 'TIMEOUT'
            if identity(child.pid) == owned:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if identity(child.pid) == owned:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            record['exit_code'] = child.returncode
        finally:
            record.update(lifecycle_seconds=time.monotonic()-start,
                finished_at=datetime.now().astimezone().isoformat(), owned_exited=child.poll() is not None)
            save()
    print(json.dumps(record, ensure_ascii=False))
    if child.returncode:
        raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()

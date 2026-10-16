"""CPU-validated private guard prototype; default-disabled, no foreign signals.

Ownership follows live ancestry to pinned launcher identities, NOT PGID alone.
This does not authorize a GPU retry or change the failed job/11 model files.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    pgid: int
    start_ticks: int
    uid: int
    argv: tuple


def read_process(pid):
    base = Path('/proc') / str(pid)
    raw = (base / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    argv = tuple(x.decode('utf-8', errors='surrogateescape')
                 for x in (base / 'cmdline').read_bytes().split(b'\0') if x)
    uid = int(next(x for x in (base / 'status').read_text().splitlines()
                   if x.startswith('Uid:')).split()[2])
    info = Process(pid, int(fields[1]), int(fields[2]), int(fields[19]), uid, argv)
    again = (base / 'stat').read_text()
    check = again[again.rfind(')') + 2:].split()
    if (int(check[1]), int(check[2]), int(check[19])) != (info.ppid, info.pgid, info.start_ticks):
        raise RuntimeError('process changed during ownership inspection')
    return info


def pin_roots(records, pins, lookup=read_process, uid=None):
    uid = os.geteuid() if uid is None else uid
    for record in records:
        pid = int(record['pid'])
        if pid in pins:
            continue
        try:
            info = lookup(pid)
        except FileNotFoundError:
            continue  # Completed registered CPU processes are not allowlisted.
        if (info.uid != uid or info.pgid != record['process_group']
                or info.argv != tuple(record['command'])):
            raise RuntimeError('registered root identity does not match live process')
        pins[pid] = info


def is_owned(info, pins, lookup=read_process):
    seen = set()
    for _ in range(256):
        if info.pid in seen:
            return False
        seen.add(info.pid)
        if info.pid in pins:
            root = pins[info.pid]
            return (info.start_ticks, info.uid, info.pgid, info.argv) == (
                root.start_ticks, root.uid, root.pgid, root.argv)
        if info.ppid <= 1:
            return False
        try:
            parent = lookup(info.ppid)
        except FileNotFoundError:
            return False  # Live orphan is not silently accepted.
        if parent.uid != info.uid or parent.start_ticks > info.start_ticks:
            return False
        info = parent
    return False


def foreign_pids(apps, pins, lookup=read_process):
    foreign = []
    for row in apps.splitlines():
        if not row.strip():
            continue
        pid = int(row.split(',', 1)[0].strip())
        if pid <= 0:
            raise ValueError('invalid GPU PID')
        try:
            info = lookup(pid)
        except FileNotFoundError:
            continue  # NVML/proc exit race; no live PID to classify.
        if not is_owned(info, pins, lookup):
            foreign.append(pid)
    return sorted(set(foreign))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--watch-approved-job', action='store_true')
    args = parser.parse_args()
    if not args.watch_approved_job or os.environ.get('P0_FIXED_FOREIGN_GPU_GUARD') != '1':
        raise SystemExit('default-disabled foreign GPU guard')
    sys.path.insert(0, '/rjs/cxz/论文探索/投机采样/experiments/fixed_load_cost')
    from contract import require_execution, write
    require_execution(args.job)
    job = Path(args.job).resolve()
    manifest = json.loads((job / 'launch-guards.json').read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if (manifest['status'] != 'PASS_CPU_GUARD_ONLY'
            or manifest.get('ownership_version') != 2
            or manifest['plan_sha256'] != sha(job / 'plan.json')
            or manifest['guard_sha256'] != sha(__file__)
            or manifest['self_test_sha256'] != sha(job / 'preflight/foreign-GPU-guard-tests.json')):
        raise SystemExit('private guard manifest changed/not CPU-validated v2')
    samples, pins = 0, {}
    while True:
        try:
            state = json.loads((job / 'status.json').read_text())
            if state.get('state') in ['completed', 'failed']:
                write(job / 'foreign-GPU-guard-result.json', dict(
                    status='EXITED_WITH_TASK', samples=samples, ownership_version=2,
                    foreign_process_signalling=False, task_state=state['state']))
                return
            if state.get('GPU_started'):
                records = json.loads((job / 'owned-processes.json').read_text())
                pin_roots(records, pins)
                apps = subprocess.check_output([
                    'nvidia-smi', '-i', '0', '--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader'], text=True, timeout=10)
                foreign = foreign_pids(apps, pins)
                row = dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                           pinned_roots={pid: vars(info) for pid, info in pins.items()},
                           compute_apps=apps.strip(), foreign_pids=foreign, ownership_version=2)
                with (job / 'foreign-GPU-samples.jsonl').open('a') as file:
                    file.write(json.dumps(row) + '\n')
                samples += 1
                if foreign:
                    raise RuntimeError('non-owned GPU compute processes: ' + str(foreign))
        except Exception as error:
            write(job / 'foreign-GPU-guard-result.json', dict(
                status='FAIL_CONTENTION_OR_INSPECTION', error=repr(error), samples=samples,
                ownership_version=2, foreign_process_signalling=False))
            (job / 'STOP_REQUESTED').write_text('GPU contention/inspection guard: ' + repr(error) + '\n')
            return
        time.sleep(15)


if __name__ == '__main__':
    main()

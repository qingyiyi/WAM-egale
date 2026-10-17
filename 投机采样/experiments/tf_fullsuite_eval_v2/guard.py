"""Independent absolute clock; signal only this campaign's birth-pinned groups."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.guard-tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def birth(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def owned_live(job):
    rows = []
    for path in sorted(Path(job).glob('OWNED-*.json')):
        try:
            row = read(path)
            if (row.get('owner_job') == str(Path(job).resolve())
                    and birth(row['pid']) == row['birth_ticks']
                    and os.getpgid(row['pid']) == row['pgid']
                    and row['pgid'] == row['pid']):
                rows.append(row)
        except (OSError, KeyError, ValueError):
            pass
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    plan = read(job / 'plan.json')
    hard = datetime.fromisoformat(plan['absolute_cleanup_deadline']).timestamp()
    work = datetime.fromisoformat(plan['absolute_work_stop']).timestamp()
    write(job / 'GUARD-STARTED.json', dict(pid=os.getpid(), birth_ticks=birth(os.getpid()),
          started_at=datetime.now(timezone.utc).isoformat(), hard_deadline=hard,
          work_stop=work, unowned_signals=False))
    signals = []
    reason = None
    stop_started = None
    while True:
        if (job / 'ALL-OWNED-CLEANUP.json').exists() and not owned_live(job):
            break
        now = time.time()
        if now >= work:
            reason = 'absolute_work_stop'
        setup = job / 'PREPARATION-STARTED.json'
        if setup.exists() and not (job / 'DEPLOYMENT-COMPLETE.json').exists():
            if now >= read(setup)['prepare_deadline_unix']:
                reason = 'preparation_two_hour_cap'
        if reason:
            if stop_started is None:
                stop_started = now
            if not (job / 'STOP_REQUESTED').exists():
                (job / 'STOP_REQUESTED').write_text(reason + '\n')
            for row in owned_live(job):
                if not any(x['pid'] == row['pid'] and x['signal'] == 'TERM' for x in signals):
                    try:
                        os.killpg(row['pgid'], signal.SIGTERM)
                        signals.append(dict(pid=row['pid'], birth_ticks=row['birth_ticks'],
                                            signal='TERM', at=time.time()))
                    except ProcessLookupError:
                        pass
            if now >= min(hard - 30, stop_started + 60):
                for row in owned_live(job):
                    try:
                        os.killpg(row['pgid'], signal.SIGKILL)
                        signals.append(dict(pid=row['pid'], birth_ticks=row['birth_ticks'],
                                            signal='KILL', at=time.time()))
                    except ProcessLookupError:
                        pass
            if not owned_live(job) or now >= hard:
                break
        time.sleep(1)
    remaining = owned_live(job)
    write(job / 'GUARD-FINAL.json', dict(status='CLEAN' if not remaining else 'OWNED_REMAIN',
          at=datetime.now(timezone.utc).isoformat(), reason=reason, signals=signals,
          remaining=remaining, instance_shutdown='NOT_REQUESTED_NOT_EXECUTED'))


if __name__ == '__main__':
    main()

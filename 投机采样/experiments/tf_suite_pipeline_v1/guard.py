"""Independent suite deadlines/budgets and birth/UID/PGID owned cleanup only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import sys
import time

OWN = Path(__file__).resolve().parent
sys.path.insert(0, str(OWN.parent))
from tf_suite_pipeline_v1.protocol import PERMISSIONS, SCHEMA, read, write, require
from tf_suite_pipeline_v1.runner import process, matches, forensic_groups


def pins(job):
    result = []
    for path in job.glob('OWNED-*.json'):
        row = read(path)
        require(row['owner_job'] == str(job) and row['pgid'] == row['pid'] and row['uid'] == os.getuid(),
            'owned pin identity/UID differs')
        result.append(row)
    return result


def live(rows):
    result = []
    for r in rows:
        actual = process(r['pid'])
        if (actual is not None and actual['birth_ticks'] == r['birth_ticks']
                and actual['pgid'] == r['pgid'] == r['pid'] and actual['uid'] == r['uid']
                and actual['state'] not in ('Z', 'X')):
            result.append(r)
    return result


def tree_bytes(root):
    total = 0
    for path in root.rglob('*') if root.exists() else ():
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    p = read(job / 'plan.json')
    require(p['schema'] == SCHEMA and p['frozen_permissions'] == PERMISSIONS, 'new suite clock identity required')
    work = datetime.fromisoformat(p['absolute_work_stop']).timestamp()
    hard = datetime.fromisoformat(p['absolute_cleanup_deadline']).timestamp()
    self_state = process(os.getpid())
    started = time.time()
    write(job / 'GUARD-STARTED.json', dict(schema='tf-suite-independent-guard/v1', pid=os.getpid(),
        birth_ticks=self_state['birth_ticks'], pgid=self_state['pgid'], uid=self_state['uid'], owner_job=str(job),
        work_stop=work, hard_deadline=hard, started_at=datetime.now(timezone.utc).isoformat(), permissions=PERMISSIONS))
    reason, stop_at, completed_at, checks_at = None, None, None, 0
    signals, refused, error = [], [], None
    final_rows = []
    try:
        while True:
            now = time.time()
            rows = pins(job)
            final_rows = rows
            current_live = live(rows)
            if now >= work:
                reason = reason or 'absolute_work_stop'
            if (job / 'STOP_REQUESTED').exists():
                reason = reason or 'explicit_hard_STOP_REQUESTED'
            runner_rows = [r for r in rows if r['role'] == 'suite_runner']
            if runner_rows and not live(runner_rows) and not (job / 'ALL-OWNED-CLEANUP.json').is_file():
                reason = reason or 'owned_runner_disappeared_without_terminal_cleanup'
            if not runner_rows and now >= started + 60:
                reason = reason or 'launcher_failed_to_register_runner60s'
            if now - checks_at >= 10:
                checks_at = now
                if tree_bytes(job) > p['new_evidence_cap_bytes']:
                    reason = reason or 'suite_evidence_storage_cap'
                if p['phase'] != 'FINAL_EVAL' and tree_bytes(Path(p['suite_storage_root'])) > 2**30:
                    reason = reason or 'shared_suite_compact_one_GiB_cap'
                if shutil.disk_usage(job).free < p['remote_data_disk_floor_bytes']:
                    reason = reason or 'registered_disk_free_floor'
                pulse = job / 'server-worker0/LIVE-COUNTER.json'
                if pulse.is_file():
                    count = read(pulse)['target_forward_attempts']
                    if type(count) is not int or not 0 <= count <= p['job_forward_attempt_cap']:
                        reason = reason or 'actual_forward_attempt_cap'
                for r in rows:
                    if r['role'] == 'server-0' and matches(r) and not (job / 'server-worker0/READY.json').is_file():
                        if time.monotonic() >= r['birth_ticks'] / os.sysconf('SC_CLK_TCK') + 900:
                            reason = reason or 'actual_model_setup900s_cap'
                    if r['role'].startswith('client-row') and matches(r):
                        if time.monotonic() >= r['birth_ticks'] / os.sysconf('SC_CLK_TCK') + 1200:
                            reason = reason or 'actual_complete_episode1200s_cap'
            if reason:
                stop_at = now if stop_at is None else stop_at
                if not (job / 'STOP_REQUESTED').exists():
                    (job / 'STOP_REQUESTED').write_text(reason + '\n')
                for r in current_live:
                    sig = signal.SIGKILL if now >= min(hard - 15, stop_at + 45) else signal.SIGTERM
                    if any(s['pid'] == r['pid'] and s['birth_ticks'] == r['birth_ticks'] and s['signal'] == sig.name for s in signals):
                        continue
                    if not matches(r):
                        refused.append(dict(pid=r['pid'], reason='birth/UID/PGID proof changed'))
                        continue
                    try:
                        os.killpg(r['pgid'], sig)
                        signals.append(dict(pid=r['pid'], birth_ticks=r['birth_ticks'], pgid=r['pgid'],
                            uid=r['uid'], signal=sig.name, at_unix=time.time()))
                    except ProcessLookupError:
                        pass
            if ((job / 'ALL-OWNED-CLEANUP.json').is_file() or reason) and not live(rows):
                completed_at = now if completed_at is None else completed_at
                forensic = forensic_groups(rows)
                if forensic['scan_complete'] and not any(g['members'] for g in forensic['groups']):
                    break
                if now >= min(hard, completed_at + 60):
                    break
            if now >= hard:
                break
            time.sleep(1)
    except BaseException as exc:
        error = repr(exc)
        reason = reason or 'independent_guard_exception'
        # The error itself remains FAILED/UNKNOWN. Cleanup still uses only
        # lifetime-proven roots, with bounded escalation and no foreign kill.
        if not (job / 'STOP_REQUESTED').exists():
            try:
                (job / 'STOP_REQUESTED').write_text(reason + '\n')
            except OSError:
                pass
        failed_at = time.time()
        until = min(hard, failed_at + 60)
        while time.time() < until:
            try:
                current = pins(job)
                known = {(r['pid'], r['birth_ticks']): r for r in final_rows}
                known.update({(r['pid'], r['birth_ticks']): r for r in current})
                final_rows = list(known.values())
            except BaseException:
                pass
            current_live = live(final_rows)
            if not current_live:
                break
            for r in current_live:
                sig = signal.SIGKILL if time.time() >= min(hard - 5, failed_at + 15) else signal.SIGTERM
                if any(s['pid'] == r['pid'] and s['birth_ticks'] == r['birth_ticks'] and s['signal'] == sig.name for s in signals):
                    continue
                if not matches(r):
                    continue
                try:
                    os.killpg(r['pgid'], sig)
                    signals.append(dict(pid=r['pid'], birth_ticks=r['birth_ticks'], pgid=r['pgid'],
                        uid=r['uid'], signal=sig.name, reason='guard_exception', at_unix=time.time()))
                except ProcessLookupError:
                    pass
            time.sleep(.2)
    finally:
        final = forensic_groups(final_rows)
        remaining = live(final_rows)
        empty = final['scan_complete'] and not any(g['members'] for g in final['groups'])
        write(job / 'GUARD-FINAL.json', dict(schema='tf-suite-independent-guard-final/v1',
            status='CLEAN' if not remaining and empty and error is None else 'OWNED_REMAIN_OR_UNKNOWN',
            reason=reason, error=error, signals=signals, refused=refused, remaining=remaining,
            all_owned_groups_empty=empty, FORENSIC_group_members=final,
            elapsed_wall_sec=time.time() - started, permissions=PERMISSIONS,
            instance_shutdown='NOT_REQUESTED_NOT_EXECUTED'))


if __name__ == '__main__':
    main()

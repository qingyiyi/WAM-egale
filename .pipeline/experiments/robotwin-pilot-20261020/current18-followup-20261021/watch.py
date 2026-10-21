"""Continue an already authorized fixed chain after all 18 complete episodes.

CPU waiting holds no GPU lease. Collection accounts its own GPU lifecycle.
No robot retry, checkpoint selection, threshold change, or new experiment arm.
"""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import os
import signal
import subprocess
import time
import traceback

B = Path(__file__).resolve().parent
R = B.parent
M = R / 'tf-migration-20261021'
PY = '/root/cxz/envs/va/bin/python'
os.sched_setaffinity(0, {18, 19})
plan = json.loads((B / 'PLAN.json').read_text())
deadline = datetime.fromisoformat(plan['deadline']).timestamp()
assert not (B / 'STATUS.json').exists(), 'No automatic restart'
state = dict(status='WAITING_COMPLETE_CURRENT18', start=time.time(), jobs=[],
             GPU_lifecycle_seconds=0., downstream_DEV_started=False)


def save():
    p = B / 'STATUS.json'
    p.with_suffix('.tmp').write_text(json.dumps(state, indent=2) + '\n')
    p.with_suffix('.tmp').replace(p)


def identity(pid):
    return dict(pid=pid, pgid=os.getpgid(pid), birth_ticks=int(
        Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]))


def alive(row):
    try:
        return identity(row['pid']) == row
    except (FileNotFoundError, ProcessLookupError):
        return False


def run(name, script, input_plan, expected_status, cap, gpu):
    assert time.time() + cap + 60 < deadline, 'Insufficient bounded stage reserve'
    locks = []
    if gpu:
        for file in ('GPU1-LEASE.lock', 'GPU1.lease'):
            lock = (R / file).open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        ledger = json.loads((R / 'GPU-LEDGER.json').read_text())
        assert ledger['consumed_seconds'] + cap + 60 <= ledger['cap_seconds']
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader'], text=True)
        assert plan['GPU_uuid'] not in apps, 'GPU1 occupied'
    output = B / name
    assert not output.exists()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['GPU_uuid'] if gpu else '',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(M), TMPDIR=str(B / 'tmp'), XDG_CACHE_HOME=str(B / 'cache'),
               CUDA_CACHE_PATH=str(B / 'cache/cuda'))
    for d in ('tmp', 'cache/cuda'):
        (B / d).mkdir(parents=True, exist_ok=True)
    process = None
    row = dict(name=name, start=time.time(), gpu=gpu, cap_seconds=cap)
    state['jobs'].append(row)
    state['status'] = 'RUNNING_' + name
    save()
    try:
        with (B / (name + '.log')).open('x') as log:
            process = subprocess.Popen([PY, str(script), str(input_plan), str(output)],
                cwd=B, env=env, stdout=log, stderr=log, start_new_session=True)
        row['owned'] = identity(process.pid)
        save()
        try:
            row['returncode'] = process.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            row['timeout'] = True
    finally:
        if process is not None:
            if process.poll() is None and alive(row['owned']):
                os.kill(row['owned']['pid'], signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if alive(row['owned']):
                    os.kill(row['owned']['pid'], signal.SIGKILL)
                process.wait(timeout=15)
            row.update(returncode=process.returncode, root_reaped=True,
                       identity_alive=alive(row['owned']))
        row.update(end=time.time(), lifecycle_seconds=time.time() - row['start'])
        if gpu:
            with (R / 'GPU-LEDGER.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ledger = json.loads((R / 'GPU-LEDGER.json').read_text())
                ledger['consumed_seconds'] += row['lifecycle_seconds']
                ledger['jobs'].append(dict(namespace=str(B), **row,
                    GPU_lifecycle_seconds=row['lifecycle_seconds']))
                p = R / 'GPU-LEDGER.followup.tmp'
                p.write_text(json.dumps(ledger, indent=2) + '\n')
                p.replace(R / 'GPU-LEDGER.json')
            state['GPU_lifecycle_seconds'] += row['lifecycle_seconds']
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        save()
    assert row['returncode'] == 0 and not row.get('timeout'), name + ' failed; no retry'
    result = json.loads((output / 'RESULT.json').read_text())
    assert result['status'] == expected_status, (name, result['status'])
    assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file()) < plan['storage_cap_bytes']


try:
    save()
    while True:
        assert time.time() + 2400 < deadline, 'No time for full downstream stage'
        dep = json.loads(Path(plan['collection_status']).read_text())
        # Wait for finalized cleanup/ledger, not just completion written by loop.
        if 'end' in dep:
            assert dep['status'] == 'COMPLETE_CURRENT18_VISITED_COLLECTION', dep['status']
            assert len(dep['episodes']) == 18 and not dep['remaining_owned']
            break
        state['complete_collection_episodes'] = len(dep['episodes'])
        state['CPU_wait_seconds'] = time.time() - state['start']
        save()
        time.sleep(20)
    run('AUDIT', M / 'audit_observed.py', B / 'AUDIT-PLAN.json',
        'PASS_COMPLETE_FIXED18_REACHED_SUPERVISION', 1800, False)
    run('ON3000', M / 'train_branch.py', B / 'ON3000-PLAN.json',
        'COMPLETE_FIXED_BRANCH_NOT_DEPLOYMENT_CAL_OR_ROBOT_CERTIFICATE', 1800, True)
    run('CAL', B / 'cal_current.py', B / 'CAL-PLAN.json',
        'PASS_FINITE_GPU_ADAPTER_DECODE_NOT_CLOSED_LOOP', 300, True)
    state.update(status='COMPLETE_CURRENT_AUDIT_ON_AND_INTERFACE_CAL',
                 next='Fixed DEV90 requires its reviewed wrapper/dispatcher; not started by this script')
except BaseException as error:
    state.update(status='FAILED_OR_BLOCKED_NO_AUTOMATIC_RETRY', error=repr(error),
                 traceback=traceback.format_exc())
finally:
    state.update(end=time.time(), wall_seconds=time.time() - state['start'])
    save()

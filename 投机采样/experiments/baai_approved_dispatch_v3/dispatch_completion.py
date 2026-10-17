"""Wait on the existing lifecycle, then launch only the eight authorized cases.

Waiting, failure and replacement attempts retain the original absolute budget.
This CPU-only dispatcher never signals existing or foreign model processes.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent.parent / 'baai_approved_retry_v2'
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from tf_fullsuite_eval_v2.protocol import read, write, require, PERMISSIONS
from tf_fullsuite_eval_v2.runner import process, matches

APPROVED = [219, 220, 221, 222, 223, 225, 226, 227]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    queue = args.job.resolve(strict=True)
    registration = read(queue / 'REGISTRATION.json')
    workspace = (Path.home() / 'cxz').resolve(strict=True)
    require(queue.is_relative_to(workspace), 'queue must stay inside cxz')
    old = Path(registration['wait_for_job']).resolve(strict=True)
    p0 = read(old / 'plan.json')
    stop = datetime.fromisoformat(p0['absolute_work_stop']).timestamp()
    hard = datetime.fromisoformat(p0['absolute_cleanup_deadline']).timestamp()
    os.sched_setaffinity(0, {43})
    write(queue / 'STARTED.json', dict(**process(os.getpid()), at=time.time(), CPU=[43], GPU=0,
                                     original_work_stop=stop, original_cleanup_deadline=hard))
    child = None
    child_pin = None
    terminal = None
    interrupted = False

    def interrupt(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        require(registration['authorized_retry_case_ids'] == APPROVED
                and registration['explicit_user_authorization'] is True, 'new explicit permission required')
        last = 0
        while True:
            require(not interrupted and not (queue / 'STOP_REQUESTED').exists(), 'dispatcher stopped')
            # Worst complete-case reserve for four fixed workers, plus setup and cleanup.
            counts = [sum(p0['cases'][cid]['execution_worker_index'] == i for cid in APPROVED)
                      for i in range(4)]
            reserve = 900 + max(counts) * 3 * 1320
            require(time.time() + reserve < stop and time.time() + reserve + 900 < hard,
                    'original budget lacks full replacement-panel reserve')
            if (old / 'SUPERVISOR-RESULT.json').exists() and (old / 'GUARD-FINAL.json').exists():
                require(read(old / 'FINAL-OWNED-CLEANUP.json')['all_owned_groups_empty']
                        and read(old / 'GUARD-FINAL.json')['status'] == 'CLEAN',
                        'previous model and guard cleanup must be real')
                pins = [path for path in list(old.glob('OWNED-*.json')) + [old / 'SUPERVISOR-PROCESS.json'] if path.exists()]
                # Cleanup receipts can precede the supervising process's actual
                # final exit by a few milliseconds. Wait; never race a new model
                # launch against that old lifetime or treat the race as a retry.
                still_live = [read(path) for path in pins if matches(read(path))
                              and process(read(path)['pid'])['state'] not in ('Z', 'X')]
                if not still_live:
                    break
            if time.time() - last >= 60:
                last = time.time()
                temp = queue / 'status.new'
                temp.write_text(json.dumps(dict(status='WAITING_EXISTING_NORMAL_PANEL_NO_GPU',
                    at=time.time(), wait_for=str(old), authorized_cases=APPROVED,
                    original_deadline=hard, permissions=PERMISSIONS), indent=2))
                temp.replace(queue / 'status.json')
            time.sleep(10)

        previous = read(old / 'SUPERVISOR-RESULT.json')
        complete = sorted(previous['complete_cases'])
        require(complete == sorted(p0['unstarted_case_ids']),
                'normal panel did not fully close; do not retry or silently widen its scope')
        require(len(complete) + len(p0['prior_completed_case_ids']) == 492,
                'expected exactly 492 normal cases before eight replacements')
        plan = copy.deepcopy(p0)
        target = Path(registration['target_job']).resolve()
        require(target.is_relative_to(workspace) and not target.exists(), 'one new cxz replacement namespace')
        target.mkdir()
        roots = plan['historical_source_roots'] + [dict(logical_source_job=str(old), physical_source_job=str(old))]
        debit = sum(read(Path(r['physical_source_job']) / f'server-worker{i}/LIVE-COUNTER.json')['target_forward_attempts']
                    for r in roots for i in range(read(Path(r['physical_source_job']) / 'plan.json')['max_GPU_workers']))
        sources = plan['prior_complete_source_cases'] + [dict(case_id=cid, source_job=str(old)) for cid in complete]
        sources.sort(key=lambda value: value['case_id'])
        plan.update(job=str(target), prior_job=str(old), source_code_commit=registration['source_commit'],
            code_namespace=str(OWN), migration_schema='baai-explicit-approved-infra-retry/v1',
            historical_source_roots=roots, prior_target_forward_debit=debit,
            prior_complete_source_cases=sources, prior_completed_case_ids=[r['case_id'] for r in sources],
            blocked_prior_partial_case_ids=[], unstarted_case_ids=[], authorized_retry_case_ids=APPROVED,
            explicit_user_authorization=True, migration_shards=[APPROVED], evidence_path=str(target),
            ports={str(i): 19781+i for i in range(4)},
            master_ports={str(i): 29781+i for i in range(4)},
            worker_forward_caps={str(i): sum(plan['cases'][cid]['execution_worker_index'] == i
                                            for cid in APPROVED) * 12000 + 250 for i in range(4)})
        require(sum(plan['worker_forward_caps'].values()) + debit <= plan['job_forward_attempt_cap'],
                'all previous actual debits plus replacement reservations must fit original budget')
        write(target / 'plan.json', plan)
        write(target / 'AUTHORIZATION.json', dict(source='User approved the eight listed infrastructure replacement cases',
            at=time.time(), approved=APPROVED, original_partial_overwritten=False,
            original_budget_reset=False, new_independent_initial_states=False, permissions=PERMISSIONS))
        write(target / 'SOURCE-FREEZE.json', dict(source_commit=registration['source_commit'],
            numerical_server_client='unchanged tf_fullsuite_eval_v2', protocol='baai_approved_retry_v2', no_SHA=True))
        write(target / 'DEPLOYMENT-COMPLETE.json', dict(status='EXISTING_ASSETS_ORIGINAL_MATH_AND_EXPLICIT_RETRY_PROTOCOL',
            no_model_tests=True, no_SHA=True, permissions=PERMISSIONS))
        admission = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=15)
        write(target / 'GPU-ADMISSION.json', dict(returncode=admission.returncode, stdout=admission.stdout,
              stderr=admission.stderr, formal_performance=False, no_foreign_processes_signalled=True))
        require(admission.returncode == 0 and len(admission.stdout.strip().splitlines()) == 2
                and all(int(line.rsplit(',', 1)[1]) >= 32000 for line in admission.stdout.strip().splitlines()),
                'four-worker real resource admission; do not disturb foreign work')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                   TMPDIR=str(workspace / 'tmp'), XDG_CACHE_HOME=str(workspace / 'cache'),
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (target / 'supervisor.log').open('xb') as log:
            child = subprocess.Popen([plan['va_python'], '-B', str(OWN / 'launch.py'), '--job', str(target)],
                cwd=Path(plan['root']), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=env, close_fds=True, start_new_session=True)
        child_pin = dict(**process(child.pid), owner_job=str(target), role='approved_retry_supervisor')
        write(target / 'SUPERVISOR-PROCESS.json', child_pin)
        write(queue / 'LAUNCHED.json', dict(at=time.time(), target_job=str(target), supervisor=child_pin,
                                          permissions=PERMISSIONS))
        while child.poll() is None:
            require(time.time() < hard + 20 and not interrupted, 'original absolute cleanup or dispatcher stop')
            time.sleep(5)
        terminal = dict(status='APPROVED_REPLACEMENT_LIFECYCLE_EXITED', child_returncode=child.returncode,
                        target_job=str(target), permissions=PERMISSIONS)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            target = Path(registration['target_job'])
            if not (target / 'STOP_REQUESTED').exists():
                (target / 'STOP_REQUESTED').write_text('dispatcher exiting; retain original guard and close owned work\n')
        terminal = dict(status='BLOCKED_OR_PARTIAL_NO_AUTOMATIC_RETRY', error=repr(exc),
                        traceback=traceback.format_exc(), child_returncode=child.poll() if child else None,
                        original_budget_reset=False, permissions=PERMISSIONS)
    finally:
        write(queue / 'TERMINAL.json', dict(**terminal, finished_at=time.time()))
        write(queue / 'CLEANUP.json', dict(status='CPU_DISPATCHER_EXITS', no_foreign_signals=True,
            child_returncode=child.poll() if child else None,
            research_original_guard_retained_if_child_still_running=child is not None and child.poll() is None))
    return 0 if terminal['status'] == 'APPROVED_REPLACEMENT_LIFECYCLE_EXITED' else 2


if __name__ == '__main__':
    raise SystemExit(main())

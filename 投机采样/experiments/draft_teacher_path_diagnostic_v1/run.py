"""CPU supervisor; exact owned cleanup, external timeout, no auto reruns."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent
sys.path.insert(0, str(OWN.parent))
from draft_teacher_path_diagnostic_v1.protocol import EXP, ROOT, RUNTIME, load, prepare, read, write, require


def execute(job, p):
    from tf_usable_quality_v1.legacy import launcher
    Base, cleanup = launcher()
    class Driver:
        def __init__(self):
            self.roots, self.pins = [], {}
    driver = Driver()
    begin = time.monotonic()
    end = min(begin + p['max_run_wall_sec'], begin +
        (datetime.fromisoformat(p['absolute_cleanup_deadline']) - datetime.now(timezone.utc)).total_seconds())
    work = end - p['cleanup_reserve_sec']
    require(work > begin, 'registered deadline already expired')
    write(job/'RUN-STARTED.json', dict(status='EXPLICIT_NUMERIC_DIAGNOSIS', pid=os.getpid(),
        monotonic=begin, active_deadline_monotonic=work, deadline_monotonic=end, performance_claim=False))
    q = p['reference_plan']
    env = {k:v for k,v in os.environ.items() if not k.startswith(('WAM_P0_', 'P0_', 'LINGBOT_',
        'PREFIX_ACCEPT_', 'RESET_TEXT_BANK_', 'FORMAL_SPEED_', 'TARGET_TRACE_', 'DIRECTION_DIAG_'))}
    env.update(CUDA_VISIBLE_DEVICES='0', LINGBOT_SEED=str(q['seed']), PYTHONHASHSEED='0',
        PYTHONDONTWRITEBYTECODE='1', LINGBOT_DISABLE_FSDP='1', LINGBOT_CACHE_MODE='original',
        LINGBOT_ATTN_MODE='torch', LINGBOT_NSYS_PROFILE='off', LINGBOT_VA_LIBERO_SUITE='libero_10',
        LINGBOT_VA_LIBERO_CHECKPOINT=q['target_checkpoint_path'], LINGBOT_VA_LIBERO_CHECKPOINT_ID=q['target_checkpoint_id'],
        LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=q['registry'], WAM_VAE_DEVICE='cuda', WAM_P0_PHASE_PROFILE='0',
        WAM_P0_ACTION_DRAFT_MODE='off', WAM_P0_ACTION_DRAFT_SELECTOR='global',
        WAM_P0_ACTION_DRAFT_THRESHOLD=str(q['threshold']), WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=q['selected']['TF_CONT']['checkpoint'],
        RESET_TEXT_BANK_ENABLED='1', PREFIX_ACCEPT_ENABLED='1', PREFIX_ACCEPT_GPU_APPROVED='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(p['master_port']))
    seq = 0
    def spawn(cmd, out, e):
        proc = Base.spawn(driver, cmd, out, e)
        write(job/f'OWNED-{proc.pid}.json', asdict(driver.pins[proc.pid]))
        return proc
    def probe():
        nonlocal seq
        seq += 1
        out = job/f'GPU-{seq:03d}.json'
        child = spawn([q['va_python'], '-B', str(EXP/'wam_quality_diagnostic_v1/resource_probe.py'),
            '--out', str(out)], job/f'probe-{seq:03d}.log', dict(env, CUDA_VISIBLE_DEVICES=''))
        stop = min(time.monotonic()+12, end-20)
        while child.poll() is None and time.monotonic() < stop:
            time.sleep(.05)
        if child.poll() is None:
            cleanup([child], driver.pins, deadline=min(time.monotonic()+15, end), wait_seconds=0.)
            raise TimeoutError('bounded metadata probe timed out')
        require(child.returncode == 0 and out.exists(), 'actual NVML probe failed')
        return read(out)
    failure = final = closed = None
    try:
        first = probe()
        require(first['memory_free_bytes'] >= p['initial_free_GiB']*2**30, '22GiB available memory required; no wait')
        require(shutil.disk_usage(job).free >= p['disk_floor_bytes'], '5GiB free disk required')
        worker = spawn([q['va_python'], '-B', str(OWN/'worker.py'), '--job', str(job), '--execute-reviewed'],
            job/'worker.log', env)
        next_probe = time.monotonic()+30
        while worker.poll() is None:
            require(time.monotonic() < work, '1020s active deadline; cleanup reserved')
            require(not (job/'STOP_REQUESTED').exists(), 'user/agent STOP_REQUESTED')
            require(shutil.disk_usage(job).free >= p['disk_floor_bytes'], 'disk floor reached')
            require(sum(x.stat().st_size for x in job.rglob('*') if x.is_file()) <= p['storage_cap_bytes'], '1GiB output cap')
            if time.monotonic() >= next_probe:
                require(probe()['memory_free_bytes'] >= .5*2**30, 'runtime memory floor reached')
                next_probe = time.monotonic()+30
            time.sleep(.2)
        require(worker.returncode == 0 and (job/'WORKER-RESULT.json').exists(), 'worker diagnostic failed')
        require(not (job/'WORKER-FAILED.json').exists(), 'worker failure takes precedence')
    except BaseException as exc:
        failure = dict(status='FAILED_OR_PARTIAL_NUMERIC_DIAGNOSTIC', error=repr(exc), traceback=traceback.format_exc())
    finally:
        closed = cleanup(driver.roots, driver.pins, deadline=end, wait_seconds=.5)
        write(job/'FINAL-OWNED-CLEANUP.json', closed)
        final = probe()
        require(all(x['pid'] not in driver.pins for x in final['compute']+final['graphics']), 'own GPU PID remains')
    if failure:
        write(job/'FAILED.json', {**failure, 'permissions':p['permissions'], 'elapsed_sec':time.monotonic()-begin})
    write(job/'SUPERVISOR-RESULT.json', dict(status='FAILED_OR_PARTIAL' if failure else 'COMPLETE_DIAGNOSTIC',
        failure=failure, cleanup=closed, observed_pins=[asdict(x) for x in driver.pins.values()],
        elapsed_sec=time.monotonic()-begin, output_bytes=sum(x.stat().st_size for x in job.rglob('*') if x.is_file()),
        performance_claim=False, permissions=p['permissions'], own_GPU_processes_empty=True))
    if failure:
        raise RuntimeError(failure['error'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute-reviewed', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        require(not args.execute_reviewed, 'prepare is CPU-only')
        p = prepare(args.job)
        print(dict(status='REGISTERED_CPU_ONLY', job=p['job'], GPU_started=False))
        return
    require(args.execute_reviewed, 'explicit execution flag required')
    p = load(args.job)
    require(not (args.job/'RUN-STARTED.json').exists(), 'one Attempt only')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', p['master_port']))
    locks = []
    try:
        for name in ('.speculative_minimal_gpu.lock', '.shared-prefix-GPU.lock'):
            f = (RUNTIME/name).open('a+')
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
            locks.append(f)
        def stop(*a):
            raise SystemExit('stop numeric diagnosis and clean own worker')
        signal.signal(signal.SIGTERM, stop)
        execute(args.job, p)
    finally:
        for f in reversed(locks):
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()


if __name__ == '__main__':
    main()

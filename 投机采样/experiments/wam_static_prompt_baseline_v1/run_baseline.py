"""One independent S04-dependent ordinary strong-baseline experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'wam_shape_audit_v1'))
from run_graph import bounded, NVML, Idle, ownership, pin, lifetime, ROOT, OWN as SHAPE
from baseline_contract import prepare, load, read, require, write, ARMS, DEFAULT_S04
OWN = Path(__file__).resolve().parent
RUNTIME = ROOT / '投机采样/results/runtime'

def execute(job, plan):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    require(not any((job / n).exists() for n in
        ('RUN-STARTED.json', 'SUPERVISOR-RESULT.json', 'BASELINE-RESULT.json', 'FAILED.json', 'STOP_REQUESTED')),
        'one reviewed attempt only')
    require(not list(job.rglob('*FAILED*.json')), 'failure markers block execution')
    locks = []
    child = None
    pins = {}
    samples = []
    cleanups = []
    error = None
    nv = None
    started = None

    def refresh():
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            try:
                info = ownership.read_process(int(path.name))
                if ownership.is_owned(info, pins): pins[info.pid] = info
            except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError): continue

    def live():
        result = []
        for pid, known in list(pins.items()):
            try:
                info = ownership.read_process(pid)
                raw = Path('/proc', str(pid), 'stat').read_text()
                if lifetime(info) == lifetime(known) and raw[raw.rfind(')') + 2:].split()[0] != 'Z':
                    result.append(pid)
            except (FileNotFoundError, ProcessLookupError): pass
            except (RuntimeError, PermissionError): result.append(pid)
        return result

    def stop(sig):
        refresh()
        for pid in live():
            try:
                info = ownership.read_process(pid)
                require(lifetime(info) == lifetime(pins[pid]), 'refuse reused PID')
                os.kill(pid, sig)
                cleanups.append(dict(pid=pid, signal=int(sig)))
            except (FileNotFoundError, ProcessLookupError): pass

    def stop_signal(*_): raise SystemExit('signal requested owned cleanup')

    signal.signal(signal.SIGTERM, stop_signal)
    try:
        for name in ('.speculative_minimal_gpu.lock', '.shared-prefix-GPU.lock'):
            handle = (RUNTIME / name).open('a+')
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException: handle.close(); raise RuntimeError('project GPU lock busy: ' + name)
            locks.append(handle)
        nv = bounded(NVML)
        idle = Idle()
        for _ in range(60):
            sample = bounded(nv.sample)
            samples.append(sample)
            if idle.add(sample, time.monotonic()): break
            time.sleep(1)
        require(len(idle.samples) >= 15, 'exclusive GPU window unavailable')
        require(samples[-1]['memory_free_bytes'] >= plan['initial_free_bytes'], '22GiB headroom required')
        require(shutil.disk_usage(job).free >= plan['disk_floor_bytes'], '5GiB disk floor required')
        with socket.socket() as sock: sock.bind(('127.0.0.1', plan['master_port']))
        prior = read(job / 'SOURCE-PLAN.json')
        ref = prior['frozen_reference']
        env = {k: v for k, v in os.environ.items() if not k.startswith(('WAM_P0_', 'P0_', 'LINGBOT_'))}
        env.update(CUDA_VISIBLE_DEVICES='0', WAM_SHAPE_OUT=str(job),
            OMP_NUM_THREADS='7', MKL_NUM_THREADS='7', OPENBLAS_NUM_THREADS='7', NUMEXPR_NUM_THREADS='7',
            PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            PYTHONHASHSEED='0', LINGBOT_SEED='20260922', LINGBOT_DISABLE_FSDP='1',
            LINGBOT_CACHE_MODE='original', LINGBOT_ATTN_MODE='torch', LINGBOT_NSYS_PROFILE='off',
            LINGBOT_VA_LIBERO_SUITE='libero_10', LINGBOT_VA_LIBERO_CHECKPOINT=ref['target_checkpoint_path'],
            LINGBOT_VA_LIBERO_CHECKPOINT_ID=ref['target_checkpoint_id'],
            LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=ref['registry'], WAM_VAE_DEVICE='cuda',
            WAM_P0_PHASE_PROFILE='0', WAM_P0_ACTION_DRAFT_MODE='off',
            WAM_P0_ACTION_DRAFT_SELECTOR='prefix_then_cost',
            WAM_P0_ACTION_DRAFT_THRESHOLD=str(ref['threshold']),
            WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=prior['frozen_A_off'],
            PREFIX_ACCEPT_ENABLED='1', PREFIX_ACCEPT_GPU_APPROVED='1',
            MASTER_ADDR='127.0.0.1', MASTER_PORT=str(plan['master_port']), RANK='0', LOCAL_RANK='0', WORLD_SIZE='1',
            PYTHONPATH=':'.join(map(str, [OWN, SHAPE, OWN.parent / 'prefix_acceptance_v1',
                                      ROOT / '投机采样/wam-p0-worktree', ROOT / '投机采样/scripts'])))
        cmd = ['/rjs/cxz/miniconda3/envs/va/bin/python', '-B', str(OWN / 'baseline_worker.py'),
               '--config-name', 'libero', '--port', '19390', '--save_root', str(job / 'server')]
        with (job / 'worker.log').open('x') as log:
            started = time.monotonic()
            write(job / 'RUN-STARTED.json', dict(started_monotonic=started,
                active_deadline_monotonic=started + 540, final_deadline_monotonic=started + 600,
                no_automatic_retry=True, command=cmd, CUDA_visible_devices='0'))
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=ROOT / '投机采样/wam-p0-worktree')
            identity = pin(child, cmd, on_boot=lambda info: pins.update({info.pid: info}))
            pins[identity.pid] = identity
            write(job / 'OWNED-PROCESS.json', vars(identity))
            while child.poll() is None:
                refresh()
                sample = bounded(nv.sample)
                samples.append(sample)
                foreign = [x for kind in ('compute', 'graphics') for x in sample[kind] if x['pid'] not in pins]
                require(not foreign, 'foreign GPU process invalidates incremental timing: ' + repr(foreign))
                require(time.monotonic() - started < 540, '540s work cap, sixty seconds cleanup reserve')
                require(sample['memory_free_bytes'] >= plan['runtime_free_bytes'], 'runtime half-GiB headroom')
                require(shutil.disk_usage(job).free >= plan['disk_floor_bytes'], 'disk floor reached')
                require(sum(x.stat().st_size for x in job.rglob('*') if x.is_file()) <= plan['output_cap_bytes'],
                        '128MiB output cap')
                require(not (job / 'STOP_REQUESTED').exists(), 'STOP_REQUESTED')
                time.sleep(.5)
            require(child.returncode == 0, 'worker failed: ' + str(child.returncode))
        result = read(job / 'BASELINE-RESULT.json')
        require(result['status'] == 'COMPLETE_STATIC_SAME_SHAPE_PROMPT_BASELINE_PENDING_CLEANUP'
                and len(result['rows']) == 6 and len(result['admission']) == 36
                and result['all_36_eager_admissions_passed']
                and result['all_six_queries_all_six_arms_complete']
                and result['all_valid_KV_RNG_inputs_slots_versions_unchanged']
                and result['target_forward_intents'] == 410
                and result['captured_forward_intents'] == 36
                and result['actual_device_forward_equivalent'] == 374,
                'complete actual six-arm same-shape Graph result required')
        require(all(z['output']['bitwise'] and z['output']['finite'] for z in result['admission']),
                'all real eager admission outputs must pass')
        require(len(result['prefills']) == 2 and len(result['oracles']) == 12
                and {p['batch'] for p in result['prefills']} == {2, 14}
                and all(p['saved_projections'] == 61 and p['original_M'] == p['batch'] * 512
                    and p['own_original_same_shape_output']['bitwise']
                    and p['own_original_same_shape_output']['finite'] for p in result['prefills']),
                'both real original full-shape first fills and all twelve raw oracles required')
        require(all(row['valid_KV_RNG_inputs_slots_versions_and_cached_outputs_unchanged']
                    and all(len(row['timing'][arm]['samples']) == 6
                            and all(z['same_shape_original_bitwise_and_finite'] for z in row['timing'][arm]['samples'])
                            for arm in ARMS) for row in result['rows']),
                'all six replay samples per query per arm must pass')
        require(not list(job.rglob('*FAILED*.json')), 'failure markers take precedence')
        require(sum(x.stat().st_size for x in job.rglob('*') if x.is_file()) <= plan['output_cap_bytes'],
                'final output cap')
    except BaseException as exc:
        error = dict(error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            if child is not None and child.poll() is None:
                if not (job / 'STOP_REQUESTED').exists():
                    (job / 'STOP_REQUESTED').write_text('supervisor owned cleanup\n')
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stop(signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: stop(signal.SIGKILL); child.wait(timeout=5)
            refresh()
            if live():
                stop(signal.SIGTERM)
                limit = time.monotonic() + 5
                while live() and time.monotonic() < limit: time.sleep(.1)
                if live(): stop(signal.SIGKILL)
                limit = time.monotonic() + 5
                while live() and time.monotonic() < limit: time.sleep(.1)
            require(not live(), 'owned process remains live: ' + repr(live()))
        except BaseException as exc:
            error = error or dict(error='cleanup: ' + repr(exc))
        final = None
        if nv is not None:
            try:
                final = bounded(nv.sample)
                bounded(nv.close)
                require(not any(x['pid'] in pins for k in ('compute', 'graphics') for x in final[k]),
                        'own GPU process remains in NVML')
            except BaseException as exc: error = error or dict(error='final NVML: ' + repr(exc))
        for handle in reversed(locks):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        if list(job.rglob('*FAILED*.json')): error = error or dict(error='failure marker has priority')
        if (job / 'STOP_REQUESTED').exists(): error = error or dict(error='STOP_REQUESTED has priority')
        terminal = dict(status='COMPLETE_STATIC_SAME_SHAPE_PROMPT_BASELINE' if error is None else 'FAILED_OR_BLOCKED',
            production_integration_permission=False, formal_robot_performance=False,
            failure=error, own_process_reaped=child is None or child.poll() is not None,
            all_owned_live_exited=not live(), owned_tree=[vars(v) for v in pins.values()],
            cleanup_events=cleanups, final_resources=final, samples=samples,
            GPU_wall_sec=0 if started is None else time.monotonic() - started)
        if error is not None and not (job / 'FAILED.json').exists(): write(job / 'FAILED.json', error)
        write(job / 'SUPERVISOR-RESULT.json', terminal)
    if error is not None: raise RuntimeError(error['error'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--s04-job', type=Path, default=DEFAULT_S04)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--execute-reviewed', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        p = prepare(args.job, args.s04_job)
        print(json.dumps(dict(status='PREPARED_CPU_ONLY_REAL_S04_PASS', job=p['job'], GPU_used=False)))
    else:
        p = load(args.job)
        if args.inspect:
            print(json.dumps(dict(status='PASS_FROZEN_REAL_SOURCE_PLAN_CPU_ONLY', job=p['job'], GPU_used=False)))
        else:
            execute(args.job.resolve(), p)

if __name__ == '__main__': main()

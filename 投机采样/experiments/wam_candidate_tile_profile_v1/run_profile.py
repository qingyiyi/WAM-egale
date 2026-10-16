"""One reviewed real attention profile, exclusive GPU, bounded owned cleanup."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
RUNTIME = ROOT / '投机采样/results/runtime'
sys.path.insert(0, str(OWN.parent / 'wam_shape_audit_v1'))
from run_matrix import bounded, NVML, Idle, ownership, pin
NCU = '/opt/nvidia/nsight-compute/2026.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu'


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gates():
    spec = importlib.util.spec_from_file_location('candidate_profile_pilot_gate',
        OWN.parent / 'wam_candidate_tile_graph_pilot_v1/run_pilot.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    receipts = {}
    source_plan = None
    for run in (1, 2):
        folder = RUNTIME / f'p0-wam-candidate-tile-graph-pilot-v1-20261016-r{run}'
        source_plan = gate.load(folder)
        terminal, result = read(folder / 'SUPERVISOR-RESULT.json'), read(folder / 'PILOT-RESULT.json')
        assert terminal['status'] == 'COMPLETE_THREE_ARM_STATIC_GRAPH_PILOT'
        assert terminal['failure'] is None and terminal['own_process_reaped'] and terminal['all_owned_live_exited']
        assert result['status'] == 'COMPLETE_THREE_ARM_STATIC_GRAPH_PILOT_PENDING_CLEANUP'
        assert result['all_18_actual_eager_admissions_passed'] and len(result['rows']) == 6
        assert result['all_valid_KV_RNG_inputs_slots_versions_unchanged']
        assert result['target_forward_intents'] == 198 and result['actual_device_forward_equivalent'] == 180
        assert not list(folder.rglob('*FAILED*.json')) and not (folder / 'STOP_REQUESTED').exists()
        for name in ('PLAN.json', 'SUPERVISOR-RESULT.json', 'PILOT-RESULT.json'):
            receipts[str(folder / name)] = sha(folder / name)
    return source_plan, receipts


def prepare(out):
    assert out.parent == RUNTIME and not out.exists()
    source, receipts = gates()
    files = [OWN / 'run_profile.py', OWN / 'profile_worker.py', OWN / 'PLAN.txt',
             OWN.parent / 'wam_shape_audit_v1/run_matrix.py']
    sources = {**source['sources'], **{str(p): sha(p) for p in files}}
    obs = RUNTIME / 'p0-shared-prefix-observe-v2-20261012-182534/observation'
    inputs = {}
    for suffix in ('tensors.pt', 'metadata.json'):
        p = obs / ('layer00-' + suffix)
        stat = p.stat()
        inputs[str(p)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    out.mkdir()
    write(out / 'PLAN.json', dict(schema='candidate-attention-real-profile/v1', job=str(out),
        GPU_wall_cap=240, active_seconds=180, cleanup_seconds=60, CPU=2,
        storage_cap=64 * 2**20, ncu_launches=3, Python_attention_calls=9,
        profiler_internal_replays_additional=True, exclusive=True, no_retry=True,
        clock_control='none', cache_control='none', formal_timing=False,
        sources=sources, inputs=inputs, prerequisites=receipts,
        kernels_in_order=['native_shared_BM128', 'research_G1_BM64', 'research_G2_BM64']))


def load(out):
    _, receipts = gates()
    p = read(out / 'PLAN.json')
    assert p['schema'] == 'candidate-attention-real-profile/v1' and p['job'] == str(out)
    assert (p['GPU_wall_cap'], p['active_seconds'], p['cleanup_seconds'], p['CPU'], p['ncu_launches']) == (240, 180, 60, 2, 3)
    assert p['prerequisites'] == receipts and p['exclusive'] and p['no_retry'] and not p['formal_timing']
    for path, expected in {**p['sources'], **p['prerequisites']}.items():
        assert sha(path) == expected, 'source/receipt changed: ' + path
    for path, expected in p['inputs'].items():
        stat = Path(path).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (expected['bytes'], expected['mtime_ns'])
    return p


def parse_csv(contents):
    lines = contents.splitlines()
    start = next(i for i, row in enumerate(lines) if row.startswith('"ID",'))
    rows = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
    units = [r for r in rows if r.get('ID', '').strip() == '']
    data = [r for r in rows if r.get('ID', '').strip() != '']
    assert len(units) == 1 and len(data) == 3 and sorted(r['ID'] for r in data) == ['0', '1', '2']
    assert units[0]['gpu__time_duration.sum'] in ('nsecond', 'usecond', 'msecond', 'second', 'ns', 'us', 'ms', 's')
    by_id = {r['ID']: r for r in data}
    assert 'shared_kernel' in by_id['0']['Kernel Name']
    assert all(re.search(r'grouped_kernel<\s*' + str(i) + r'\s*>', by_id[str(i)]['Kernel Name']) for i in (1, 2))
    for key in ('dram__bytes.sum', 'lts__t_bytes.sum'):
        assert units[0].get(key) in ('byte', 'Kbyte', 'Mbyte', 'Gbyte'), 'actual bytes counter unit required: ' + key
        for row in data:
            number = float(row[key].replace(',', ''))
            assert math.isfinite(number) and number >= 0, 'actual finite nonnegative bytes required'
    return data, units[0]


def execute(out, plan):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not any((out / n).exists() for n in ('RUN-STARTED.json', 'RESULT.json', 'FAILED.json', 'STOP_REQUESTED'))
    os.sched_setaffinity(0, {12, 13})
    cmd = [NCU, '--target-processes', 'all', '--clock-control', 'none', '--cache-control', 'none',
        '--nvtx', '--nvtx-include', 'CANDIDATE_PROFILE/', '--kernel-name', 'regex:.*(shared_kernel|grouped_kernel).*',
        '--launch-count', '3', '--section', 'SpeedOfLight', '--section', 'MemoryWorkloadAnalysis',
        '--section', 'LaunchStats', '--section', 'Occupancy', '--section', 'SchedulerStats',
        '--metrics', 'dram__bytes.sum,lts__t_bytes.sum', '--export', str(out / 'real-attention'),
        sys.executable, '-B', str(OWN / 'profile_worker.py'), str(out)]
    pins, samples, locks, cleanup, children = {}, [], [], [], []
    nv = child = None
    failure = None
    start = time.monotonic()
    launch_at = None
    write(out / 'RUN-STARTED.json', dict(monotonic=start, command=cmd,
        active_deadline=start + 180, cleanup_deadline=start + 240))

    def lifetime(info):
        return info.pid, info.pgid, info.start_ticks, info.uid

    def refresh():
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                info = ownership.read_process(int(path.name))
                if ownership.is_owned(info, pins):
                    pins[info.pid] = info
            except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError):
                pass

    def live():
        answer = []
        for pid, known in pins.items():
            try:
                info = ownership.read_process(pid)
                if lifetime(info) == lifetime(known):
                    raw = Path('/proc', str(pid), 'stat').read_text()
                    if raw[raw.rfind(')') + 2:].split()[0] != 'Z':
                        answer.append(pid)
            except (FileNotFoundError, ProcessLookupError):
                pass
            except RuntimeError:
                answer.append(pid)
        return answer

    def stop(sig):
        refresh()
        for pid, known in list(pins.items()):
            try:
                if lifetime(ownership.read_process(pid)) == lifetime(known):
                    os.kill(pid, sig)
                    cleanup.append(dict(pid=pid, signal=int(sig)))
            except (FileNotFoundError, ProcessLookupError):
                pass

    def interrupt(*_):
        raise RuntimeError('owned profile interrupted')

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for name in ('.speculative_minimal_gpu.lock', '.shared-prefix-GPU.lock'):
            handle = (RUNTIME / name).open('a+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        nv, idle = bounded(NVML), Idle()
        for _ in range(30):
            row = bounded(nv.sample)
            samples.append(row)
            if idle.add(row, time.monotonic()):
                break
            time.sleep(1)
        assert len(idle.samples) >= 15 and samples[-1]['memory_free_bytes'] >= 2 * 2**30, 'exclusive idle unavailable'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        with (out / 'ncu.log').open('x') as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            launch_at = time.monotonic()
            info = pin(child, cmd, on_boot=lambda value: pins.update({value.pid: value}))
            pins[child.pid] = info
            while child.poll() is None:
                refresh()
                row = bounded(nv.sample)
                samples.append(row)
                foreign = [x['pid'] for key in ('compute', 'graphics') for x in row[key] if x['pid'] not in pins]
                assert not foreign, 'foreign GPU processes: ' + repr(foreign)
                assert time.monotonic() < start + 180 and not (out / 'STOP_REQUESTED').exists()
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < plan['storage_cap']
                time.sleep(.5)
        assert child.returncode == 0, 'profile worker failed'
        inputs = read(out / 'INPUTS.json')
        assert inputs['status'] == 'COMPLETE_REAL_ATTENTION_PROFILE_INPUTS' and len(inputs['rows']) == 3
        assert inputs['valid_KV_Q_current_KV_slots_CFG_RNG_versions_unchanged']
        assert all(r['frozen_bitwise'] and r['finite'] for r in inputs['rows'])
        assert time.monotonic() + 20 < start + 180, 'no CPU export time remains'
        export_cmd = [NCU, '--import', str(out / 'real-attention.ncu-rep'), '--csv', '--page', 'raw']
        with (out / 'metrics.csv').open('x') as csv_file, (out / 'export.stderr').open('x') as err_file:
            export = subprocess.Popen(export_cmd, stdout=csv_file, stderr=err_file,
                env=dict(env, CUDA_VISIBLE_DEVICES=''), start_new_session=True)
            children.append(export)
            info = pin(export, export_cmd, on_boot=lambda value: pins.update({value.pid: value}))
            pins[export.pid] = info
            deadline = min(start + 180, time.monotonic() + 20)
            while export.poll() is None:
                refresh()
                assert time.monotonic() < deadline and not (out / 'STOP_REQUESTED').exists(), 'offline export deadline'
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < plan['storage_cap']
                time.sleep(.1)
        assert export.returncode == 0, (out / 'export.stderr').read_text()
        rows, units = parse_csv((out / 'metrics.csv').read_text())
        write(out / 'COUNTERS.json', dict(rows=rows, units=units, order=plan['kernels_in_order'], profile_only=True))
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) <= plan['storage_cap']
    except BaseException as exc:
        failure = dict(error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            refresh()
            if live():
                stop(signal.SIGTERM)
            for process in children:
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        stop(signal.SIGKILL)
                        process.wait(timeout=5)
            limit = time.monotonic() + 5
            while live() and time.monotonic() < limit:
                time.sleep(.1)
            if live():
                stop(signal.SIGKILL)
            limit = time.monotonic() + 5
            while live() and time.monotonic() < limit:
                time.sleep(.1)
            assert not live(), 'owned process remains'
        except BaseException as exc:
            failure = failure or dict(error='cleanup: ' + repr(exc))
        final = None
        if nv is not None:
            try:
                final = bounded(nv.sample)
                bounded(nv.close)
                assert not any(x['pid'] in pins for key in ('compute', 'graphics') for x in final[key])
            except BaseException as exc:
                failure = failure or dict(error='final NVML: ' + repr(exc))
        for handle in reversed(locks):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        if (out / 'STOP_REQUESTED').exists():
            failure = failure or dict(error='STOP_REQUESTED has priority')
        if failure:
            write(out / 'FAILED.json', failure)
        write(out / 'RESULT.json', dict(status='COMPLETE_REAL_ATTENTION_PROFILE' if failure is None else 'FAILED_OR_BLOCKED',
            failure=failure, GPU_wall_sec=0 if launch_at is None else time.monotonic() - launch_at,
            total_budget_wall_sec=time.monotonic() - start, all_owned_live_exited=not live(),
            own_root_reaped=all(p.poll() is not None for p in children), resources=final,
            owned=[vars(v) for v in pins.values()], samples=samples, cleanup=cleanup, formal_timing=False))
    if failure:
        raise RuntimeError(failure['error'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--execute-reviewed', action='store_true')
    args = parser.parse_args()
    out = args.job.resolve()
    if args.prepare:
        prepare(out)
        print('PREPARED_CPU_ONLY')
    elif args.inspect:
        load(out)
        print('PASS_CPU_PLAN_LOAD')
    else:
        execute(out, load(out))


if __name__ == '__main__':
    main()

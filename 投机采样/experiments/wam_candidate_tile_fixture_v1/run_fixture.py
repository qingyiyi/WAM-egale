"""Independent, bounded real-fixture numerical comparison; no timing."""
import argparse
import fcntl
import hashlib
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
from run_graph import bounded, NVML, Idle, ownership, pin, lifetime, ROOT, OWN as SHAPE, PRIOR
OWN = Path(__file__).resolve().parent
KERNEL = OWN.parent / 'wam_candidate_tile_v1'

RUNTIME = ROOT / '投机采样/results/runtime'
FIXTURE = RUNTIME / 'p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value, text):
    if not value: raise RuntimeError(text)
def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(job, fixture):
    job = Path(job).resolve()
    fixture = Path(fixture).resolve()
    require(job.parent == RUNTIME and not job.exists(), 'fresh isolated runtime directory required')
    require(fixture == FIXTURE.resolve(), 'only actual complete r4 fixtures accepted')
    producer = read(fixture / 'MATRIX-RESULT.json')
    supervisor = read(fixture / 'SUPERVISOR-RESULT.json')
    require(producer['status'] == 'COMPLETE_THREE_LENGTH_FULL_VERIFIER_MECHANISM'
        and len(producer['rows']) == 18 and len(producer['chunks']) == 3,
        'actual complete 18-row, three-chunk producer required')
    require(supervisor['status'] == 'COMPLETE_MECHANISM_MATRIX'
        and supervisor['reason'] is None and supervisor['all_owned_live_exited']
        and supervisor['own_process_reaped'], 'producer complete cleanup required')
    require(not list(fixture.rglob('*FAILED*.json')) and not (fixture / 'STOP_REQUESTED').exists(),
            'producer failure/stop takes precedence')
    prior = read(PRIOR)
    provenance, administrative = {}, []
    for path, expected in prior['source_sha256'].items():
        actual = sha(path)
        if path in (str(ROOT / '.pipeline/config.yaml'), str(ROOT / 'AGENTS.md')):
            administrative.append(dict(path=path, original=expected, admitted=actual,
                                       reason='authorized HW administrative change'))
        else: require(actual == expected, 'frozen implementation drift: ' + path)
        provenance[path] = actual
    for path in [SHAPE / 'full_matrix.py', SHAPE / 'run_graph.py', OWN / 'run_fixture.py',
                 OWN / 'fixture_worker.py', OWN / 'PLAN.txt', KERNEL / 'GENERATION.json',
                 KERNEL / 'generate.py', KERNEL / 'setup.py', KERNEL / 'build.py',
                 *KERNEL.glob('csrc/*'), *KERNEL.glob('_wam_candidate_tile*.so')]:
        provenance[str(path)] = sha(path)
    require(len(list(KERNEL.glob('_wam_candidate_tile*.so'))) == 1, 'one existing research binary required')
    numeric_root = RUNTIME / 'p0-wam-candidate-tile-numeric-v1-20261016-r1'
    numeric = read(numeric_root / 'WORKER-RESULT.json')
    terminal = read(numeric_root / 'RESULT.json')
    require(terminal['status'] == 'COMPLETE_KERNEL_NUMERIC_RESEARCH'
            and terminal['error'] is None and terminal['cleanup']['all_roots_reaped'],
            'actual previous numerical cleanup required')
    require(numeric['status'] == 'COMPLETE_FINITE_KERNEL_NUMERIC_RESEARCH'
            and numeric['all_real_inputs_and_RNG_unchanged']
            and numeric['group_exact_on_three_long_fixtures']['1']
            and numeric['group_exact_on_three_long_fixtures']['2'],
            'actual G1/G2 long numerical admission required')
    require(not list(numeric_root.rglob('*FAILED*.json'))
            and not (numeric_root / 'STOP_REQUESTED').exists(), 'previous failure-first')
    previous_plan = read(numeric_root / 'PLAN.json')
    for path, expected in previous_plan['sources'].items():
        if Path(path).suffix in ('.so', '.h', '.cu'):
            require(sha(path) == expected, 'numerically qualified kernel changed: ' + path)
    build_result = RUNTIME / 'p0-wam-candidate-tile-build-v1-20261016-r2/RESULT.json'
    build = read(build_result)
    require(build['status'] == 'COMPLETE_CPU_BUILD' and build['owned_root_reaped']
            and build['error'] is None and not build['GPU_used'], 'actual CPU build required')
    weights = prior['target_weight_inventory']
    for item in weights:
        stat = Path(item['path']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']),
                'target weight stat differs')
    fixtures = {}
    for label in ('short', 'middle', 'long'):
        path = fixture / label / 'real-verifier-fixture.pt'
        stat = path.stat()
        fixtures[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    prerequisites = {str(fixture / name): sha(fixture / name) for name in
                     ('MATRIX-RESULT.json', 'SUPERVISOR-RESULT.json', 'SOURCE-PROVENANCE.json')}
    for path in [numeric_root / 'PLAN.json', numeric_root / 'WORKER-RESULT.json',
                 numeric_root / 'RESULT.json', build_result]:
        prerequisites[str(path)] = sha(path)
    plan = dict(schema='candidate-tile-real-fixture-numeric/v1', job=str(job), fixture_root=str(fixture),
        expected_queries=6, expected_comparisons=198, G2_expected_comparisons=180, G1_expected_comparisons=18,
        candidate_comparison_cap=256, attention_call_cap=512, expected_attention_calls=378,
        target_forward_cap=12, expected_target_forwards=6, full_policy_chunks=0, episodes=0,
        GPU_wall_cap_seconds=600, active_seconds=540, cleanup_seconds=60,
        output_cap_bytes=128 * 2**20, disk_floor_bytes=5 * 2**30,
        initial_free_bytes=22 * 2**30, runtime_free_bytes=2**29, CPU_worker=7,
        groups=[1, 2], G1_layers=[0, 8, 29], G2_layers=list(range(30)),
        original_native_always_returned=True, early_mismatch_finish_current_native_query=True,
        shared_GPU_allowed=True, performance_claim=False, integration_permission=False,
        no_sampling=True, no_EGL=True, no_retry=True, master_port=29619,
        sources=provenance, administrative_exceptions=administrative,
        weight_stats=weights, fixture_stats=fixtures, prerequisites=prerequisites,
        original_source_plan=str(PRIOR), original_source_plan_sha256=sha(PRIOR))
    job.mkdir()
    write(job / 'PLAN.json', plan)
    write(job / 'SOURCE-PLAN.json', prior)
    return plan


def load(job):
    job = Path(job).resolve()
    p = read(job / 'PLAN.json')
    require(p['schema'] == 'candidate-tile-real-fixture-numeric/v1' and p['job'] == str(job),
            'job identity changed')
    require((p['expected_queries'], p['expected_comparisons'], p['expected_target_forwards'],
        p['GPU_wall_cap_seconds'], p['active_seconds'], p['cleanup_seconds'], p['CPU_worker'],
        p['output_cap_bytes'], p['candidate_comparison_cap'], p['attention_call_cap'],
        p['target_forward_cap'], p['full_policy_chunks'], p['episodes']) ==
        (6, 198, 6, 600, 540, 60, 7, 128 * 2**20, 256, 512, 12, 0, 0),
        'fixed numerical experiment bounds changed')
    require(p['groups'] == [1, 2] and p['G1_layers'] == [0, 8, 29]
            and p['G2_layers'] == list(range(30)) and p['original_native_always_returned']
            and p['early_mismatch_finish_current_native_query'] and p['shared_GPU_allowed']
            and not p['performance_claim'] and not p['integration_permission'],
            'fixed numeric scope changed')
    require(p['fixture_root'] == str(FIXTURE.resolve()), 'real fixture identity changed')
    for path, expected in {**p['sources'], **p['prerequisites']}.items():
        require(sha(path) == expected, 'frozen source/prerequisite changed: ' + path)
    require(sha(PRIOR) == p['original_source_plan_sha256'], 'original source plan changed')
    require(read(job / 'SOURCE-PLAN.json') == read(PRIOR), 'saved original source plan differs')
    for path, expected in p['fixture_stats'].items():
        stat = Path(path).stat()
        require((stat.st_size, stat.st_mtime_ns) == (expected['bytes'], expected['mtime_ns']),
                'real fixture bytes/mtime changed')
    for item in p['weight_stats']:
        stat = Path(item['path']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']),
                'target weight bytes/mtime changed')
    return p


def execute(job, plan):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    require(not any((job / n).exists() for n in
        ('RUN-STARTED.json', 'SUPERVISOR-RESULT.json', 'FIXTURE-RESULT.json', 'FAILED.json', 'STOP_REQUESTED')),
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
        sample = bounded(nv.sample)
        samples.append(sample)
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
        cmd = ['/rjs/cxz/miniconda3/envs/va/bin/python', '-B', str(OWN / 'fixture_worker.py'),
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
                # Shared-GPU numerical experiment; never signals foreign processes.
                require(time.monotonic() - started < 540, '540s work cap, sixty seconds cleanup reserve')
                require(sample['memory_free_bytes'] >= plan['runtime_free_bytes'], 'runtime half-GiB headroom')
                require(shutil.disk_usage(job).free >= plan['disk_floor_bytes'], 'disk floor reached')
                require(sum(x.stat().st_size for x in job.rglob('*') if x.is_file()) <= plan['output_cap_bytes'],
                        '128MiB output cap')
                require(not (job / 'STOP_REQUESTED').exists(), 'STOP_REQUESTED')
                time.sleep(.5)
            require(child.returncode == 0, 'worker failed: ' + str(child.returncode))
        result = read(job / 'FIXTURE-RESULT.json')
        allowed = ('COMPLETE_SIX_FIXTURE_FINITE_NUMERIC_GATE_PENDING_CLEANUP',
                   'STOPPED_NUMERICAL_MISMATCH_NATIVE_QUERY_VERIFIED_PENDING_CLEANUP')
        require(result['status'] in allowed and result['all_completed_queries_frozen_exact_and_valid_state_unchanged'],
                'complete original native query and valid-state verification required')
        require(result['target_forwards'] <= plan['target_forward_cap']
                and result['candidate_comparisons'] <= plan['candidate_comparison_cap']
                and result['attention_calls'] <= plan['attention_call_cap'], 'actual finite call caps')
        require(not result['integration_permission'] and not result['performance_claim'], 'numerical scope only')
        if result['finite_fixture_gate_passed']:
            require(result['status'] == allowed[0] and len(result['queries']) == 6
                    and len(result['rows']) == 198 and result['candidate_comparisons'] == 198
                    and result['target_forwards'] == 6 and result['native_oracle_calls'] == 180
                    and result['attention_calls'] == 378 and result['mismatch'] is None,
                    'all six complete queries and all 198 exact candidate comparisons required')
        else:
            require(result['status'] == allowed[1] and result['mismatch'] is not None
                    and len(result['queries']) >= 1
                    and result['queries'][-1]['completed_original_native_query_after_mismatch'],
                    'honest negative must finish original current query before terminal')
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
        verdict = None if error is not None else read(job / 'FIXTURE-RESULT.json')['finite_fixture_gate_passed']
        terminal = dict(status=('COMPLETE_SIX_FIXTURE_FINITE_NUMERIC_GATE' if verdict else
            'STOPPED_NUMERICAL_MISMATCH_NATIVE_QUERY_VERIFIED') if error is None else 'FAILED_OR_BLOCKED',
            finite_fixture_gate_passed=verdict, integration_permission=False, performance_claim=False,
            failure=error, own_process_reaped=child is None or child.poll() is not None,
            all_owned_live_exited=not live(), owned_tree=[vars(v) for v in pins.values()],
            cleanup_events=cleanups, final_resources=final, samples=samples,
            GPU_wall_sec=0 if started is None else time.monotonic() - started,
            formal_robot_performance=False)
        if error is not None and not (job / 'FAILED.json').exists(): write(job / 'FAILED.json', error)
        write(job / 'SUPERVISOR-RESULT.json', terminal)
    if error is not None: raise RuntimeError(error['error'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--fixture-root', type=Path, default=FIXTURE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--execute-reviewed', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        p = prepare(args.job, args.fixture_root)
        print(json.dumps(dict(status='PREPARED_CPU_ONLY', job=p['job'], GPU_used=False)))
    else:
        p = load(args.job)
        if args.inspect:
            print(json.dumps(dict(status='PASS_ACTUAL_FROZEN_PLAN_LOAD_CPU_ONLY', job=p['job'], GPU_used=False)))
        else: execute(args.job.resolve(), p)


if __name__ == '__main__': main()

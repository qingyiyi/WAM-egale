"""Prepared bounded RoboTwin phase runner, one model and two isolated clients.

Deployment requires an explicit job, reviewed canonical result and data panel.
No panel, GPU reservation or authorization is inferred by this module.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def birth(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def result_tree_bytes(root):
    """Count a live output tree without following asset links.

    An atomic STATUS replacement may remove a temporary file between walk and
    lstat. Only that missing-entry race is ignored; access/IO errors still fail.
    """
    def onerror(error):
        if not isinstance(error, FileNotFoundError):
            raise error

    size = os.lstat(root).st_size
    for base, directories, files in os.walk(root, followlinks=False, onerror=onerror):
        for name in directories + files:
            try:
                size += os.lstat(Path(base) / name).st_size
            except FileNotFoundError:
                continue
    return size


def resolve_case(template, target=None, previous_seed=None):
    """Materialize a registered group without changing its split or arm."""
    case = dict(template)
    if case['arm'] == 'T':
        assert target is None
        if previous_seed is not None:
            assert case['scene_sequence'] == 'after_previous_valid'
            case['seed_start'] = previous_seed + 1
    else:
        assert target['complete'] and target['identity']['arm'] == 'T'
        for key in ('group', 'task', 'task_config', 'official_episode_index',
                    'instruction_generation_count', 'model_seed'):
            assert case[key] == target['identity'][key], key
        case['seed_start'] = target['scene_seed']
        case['max_candidate_seeds'] = 1
        case['paired_target_episode'] = target['identity']['episode']
    assert case['seed_start'] >= case['seed_range'][0]
    assert case['seed_start'] + case['max_candidate_seeds'] <= case['seed_range'][1]
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    job = read(args.job)
    reg = read(root / 'REGISTRATION.json')
    assert job['phase'] in ('collect', 'train', 'dev', 'full')
    assert digest(root / 'PANEL.json') == job['panel_sha256']
    assert digest(root / 'CANONICAL-REVIEW.json') == reg['canonical_review_sha256']
    assert read(root / 'CANONICAL-REVIEW.json')['passed']
    for name, sha in job['source_sha256'].items():
        assert digest(root / 'code' / name) == sha, name
    assert job['card_gpu_cap_seconds'] == 86400
    assert job['prior_gpu_seconds'] >= 0
    remaining = job['card_gpu_cap_seconds'] - job['prior_gpu_seconds']
    assert 0 < job['gpu_lifecycle_cap_seconds'] <= remaining
    assert len(job['CPU_cores']) >= 6
    assert set(job['CPU_cores']).issubset(os.sched_getaffinity(0))
    if job['phase'] != 'train':
        assert len(job['lanes']) == 2
    scene_stream_lanes = {}
    planned_episodes = set()
    for lane_id, lane in enumerate(job.get('lanes', [])):
        for group in lane:
            arms = [case['arm'] for case in group['cases']]
            assert arms == (['T'] if job['phase'] == 'collect' else ['T', 'S', 'P'])
            for case in group['cases']:
                assert case['episode'] not in planned_episodes
                assert Path(case['episode']).name == case['episode']
                planned_episodes.add(case['episode'])
                if case.get('scene_sequence') == 'after_previous_valid':
                    key = (case['task'], case['task_config'], case['split'])
                    assert scene_stream_lanes.setdefault(key, lane_id) == lane_id
    assert Path(job['id']).name == job['id']
    if job['phase'] != 'train':
        assert Path(job['server_id']).name == job['server_id']
    out = root / 'results' / job['id']
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    deadline = min(reg['deadline_epoch'] - 300, job['deadline_epoch'],
                   start + job['gpu_lifecycle_cap_seconds'])
    state = dict(status='PREFLIGHT', start=start, deadline=deadline, pid=os.getpid(),
                 job=job, episodes=[], prior_gpu_seconds=job['prior_gpu_seconds'])
    lock = threading.RLock()
    cancel = threading.Event()
    owned = []
    accepted = set()
    gpu_start = None
    server = None
    last_size_check = 0.

    def stop_signal(*_):
        cancel.set()
        raise InterruptedError('owned phase supervisor was stopped')

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    def save():
        with lock:
            state['updated'] = time.time()
            path = out / 'STATUS.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(path)

    def guard():
        nonlocal last_size_check
        assert not cancel.is_set(), 'another owned worker failed'
        assert time.time() < deadline, 'registered phase deadline'
        assert shutil.disk_usage(root).free >= job['free_floor_bytes'], 'free disk floor'
        if time.time() - last_size_check > 60:
            with lock:
                if time.time() - last_size_check > 60:
                    size = result_tree_bytes(root / 'results')
                    assert size <= job['results_cap_bytes'], ('result storage cap', size)
                    last_size_check = time.time()
                    state['results_bytes'] = size

    def environment(slot, client):
        env = os.environ.copy()
        cache = root / 'runtime' / job['id'] / slot
        for key, folder in dict(TMPDIR='tmp', TORCHINDUCTOR_CACHE_DIR='inductor',
                TRITON_CACHE_DIR='triton', CUDA_CACHE_PATH='cuda',
                TORCH_EXTENSIONS_DIR='torch-extensions', WARP_CACHE_PATH='warp',
                HF_HOME='hf', PYTHONPYCACHEPREFIX='pycache', XDG_CACHE_HOME='xdg',
                MPLCONFIGDIR='matplotlib').items():
            path = cache / folder
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']), OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MAX_JOBS='2',
            TORCHINDUCTOR_COMPILE_THREADS='2', TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        # Each host provides its checked native-library paths. Never copy a
        # remote graphics driver or put Python3.12 packages on the client path.
        config = job['client_environment' if client else 'model_environment']
        for key, value in config.items():
            assert key in ('LD_LIBRARY_PATH', 'VK_ICD_FILENAMES', 'SAPIEN_RENDER_DEVICE')
            env[key] = value
        return env

    def launch(name, command, slot, lane=None):
        client = lane is not None
        cores = job['CPU_cores'][2 + 2*lane:4 + 2*lane] if client else job['CPU_cores'][:2]
        with (out / (name + '.log')).open('xb') as log:
            process = subprocess.Popen(['taskset', '-c', ','.join(map(str, cores)), *map(str, command)],
                cwd=root, env=environment(slot, client), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        row = dict(name=name, pid=process.pid, birth_ticks=birth(process.pid), start=time.time())
        with lock:
            owned.append((process, row))
        return process, row

    def wait(process):
        while process.poll() is None:
            guard()
            if server is not None and process is not server:
                assert server.poll() is None, 'model server exited'
                server_state = read(root / 'results' / job['server_id'] / 'STATUS.json')
                assert server_state['status'] != 'FAILED_LATCHED', server_state
            time.sleep(2)
        assert process.returncode == 0, ('owned process failed', process.pid, process.returncode)

    def lane_worker(index):
        previous = {}
        try:
            for group in job['lanes'][index]:
                target = None
                for template in group['cases']:
                    guard()
                    stream = (template['task'], template['task_config'], template['split'])
                    earlier = previous.get(stream) if template.get('scene_sequence') == 'after_previous_valid' else None
                    case = resolve_case(template, target, earlier if template['arm'] == 'T' else None)
                    case_path = out / (case['episode'] + '.case.json')
                    with case_path.open('x') as handle:
                        handle.write(json.dumps(case, indent=2) + '\n')
                    process, row = launch(case['episode'], [root / 'client-env/bin/python', '-u',
                        root / 'code/client.py', '--case', case_path, '--port', job['port']],
                        'client-slot' + str(index), index)
                    wait(process)
                    result = read(root / 'results/episodes' / case['episode'] / 'EPISODE.json')
                    assert result['complete'] and result['identity'] == case
                    if case['arm'] == 'T':
                        target = result
                        key = (case['task'], case['task_config'], result['scene_seed'])
                        with lock:
                            assert key not in accepted, ('duplicate accepted scene', key)
                            accepted.add(key)
                        previous[stream] = result['scene_seed']
                    with lock:
                        row.update(end=time.time(), returncode=process.returncode)
                        state['episodes'].append(dict(episode=case['episode'], group=case['group'],
                            arm=case['arm'], split=case['split'], task=case['task'],
                            scene_seed=result['scene_seed'], success=result['success'],
                            commands=result['commands'], chunks=result['chunks']))
                    save()
        except BaseException:
            cancel.set()
            raise

    try:
        save()
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(job['gpu']),
            '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        while True:
            guard()
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader'], text=True)
            if not any(line.startswith(uuid) for line in processes.splitlines()):
                break
            state['status'] = 'WAITING_FREE_REGISTERED_GPU'
            save()
            time.sleep(5)
        gpu_start = time.time()
        if job['phase'] == 'train':
            process, row = launch('train', [root / 'env/bin/python', '-u', root / 'code/train.py'], 'train')
            state['status'] = 'TRAINING'
            save()
            wait(process)
            assert read(root / 'training/STATUS.json')['status'] == 'COMPLETE'
        else:
            server, row = launch('server', [root / 'env/bin/python', '-u', root / 'code/server.py',
                '--port', job['port'], '--worker', job['server_id'], '--assets', job['assets']], 'server')
            state.update(status='LOADING', server=row)
            save()
            server_path = root / 'results' / job['server_id'] / 'STATUS.json'
            while not server_path.exists() or read(server_path)['status'] not in ('READY', 'SERVING'):
                guard()
                assert server.poll() is None, 'model load failed'
                if server_path.exists():
                    assert read(server_path)['status'] not in ('FAILED_LATCHED', 'STOPPED_OR_FAILED')
                time.sleep(2)
            state['status'] = 'RUNNING_TWO_CLIENTS'
            save()
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in as_completed([pool.submit(lane_worker, i) for i in range(2)]):
                    future.result()
            expected = sum(len(group['cases']) for lane in job['lanes'] for group in lane)
            assert len(state['episodes']) == expected
        state['status'] = 'COMPLETE_REQUIRES_PHASE_REVIEW'
    except BaseException as exc:
        state.update(status='FAILED_OR_PARTIAL', error=repr(exc), traceback=traceback.format_exc())
    finally:
        cancel.set()
        for process, row in reversed(owned):
            if process.poll() is None:
                assert birth(process.pid) == row['birth_ticks'], 'owned PID identity changed'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=20)
            row.update(returncode=process.returncode, end=time.time())
        state.update(owned=[row for _, row in owned], end=time.time(),
            gpu_window_seconds=0 if gpu_start is None else time.time() - gpu_start,
            owned_alive=any(process.poll() is None for process, _ in owned))
        save()


if __name__ == '__main__':
    main()

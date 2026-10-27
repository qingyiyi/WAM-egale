"""One bounded A100 server with two isolated EGL clients for an explicit job panel."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('--job', required=True, type=Path)
args = P.parse_args()
ROOT = Path(__file__).resolve().parent.parent
job = json.loads(args.job.read_text())
OUT = ROOT/'results'/job['id']
OUT.mkdir(exist_ok=False)
reg = json.loads((ROOT/'REGISTRATION.json').read_text())
gpu, port = str(job['gpu']), int(job['port'])
LIBERO, ASSETS = Path(job['libero']), Path(job['assets'])
owned, cancel, save_lock = [], threading.Event(), threading.Lock()
cores = sorted(os.sched_getaffinity(0))[job['cpu_offset']:job['cpu_offset']+6]
assert len(cores) == 6
start = time.time()
deadline = min(reg['deadline_epoch']-300, start+job['wall_cap_seconds'])
status = dict(status='PREFLIGHT', job=job, pid=os.getpid(), start=start, deadline=deadline, episodes=[])
gpu_start = None
server = None


def save():
    with save_lock:
        status['updated'] = time.time()
        p = OUT/'STATUS.json'
        q = p.with_suffix('.tmp')
        q.write_text(json.dumps(status, indent=2)+'\n')
        q.replace(p)


def guard():
    assert time.time() < deadline, 'registered phase deadline'
    assert shutil.disk_usage(ROOT).free >= 25*2**30, 'minimum free storage guard'
    if gpu_start:
        assert time.time()-gpu_start < job['gpu_seconds_cap'], 'registered GPU lifecycle cap'


def environment(who, client=False):
    e = os.environ.copy()
    w = ROOT/'runtime'/who
    for folder in ('tmp', 'cache', 'config/libero', 'unused-datasets'):
        (w/folder).mkdir(parents=True, exist_ok=True)
    conf = dict(benchmark_root=LIBERO/'libero/libero', bddl_files=LIBERO/'libero/libero/bddl_files',
                init_states=LIBERO/'libero/libero/init_files', assets=LIBERO/'libero/libero/assets', datasets=w/'unused-datasets')
    (w/'config/libero/config.yaml').write_text('\n'.join(k+': '+str(v) for k, v in conf.items())+'\n')
    e.update(CUDA_VISIBLE_DEVICES=gpu, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        MUJOCO_EGL_DEVICE_ID=gpu, EGL_DEVICE_ID=gpu, LIBERO_CONFIG_PATH=str(w/'config/libero'),
        FASTWAM_SUITE=job['suite'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false', MAX_JOBS='2', TORCHINDUCTOR_COMPILE_THREADS='2',
        PYTHONPATH=str(LIBERO) if client else str(ROOT/'source/FastWAM/src'))
    for key, folder in dict(TMPDIR='tmp', TMP='tmp', TEMP='tmp', HF_HOME='cache/hf',
        TORCH_EXTENSIONS_DIR='cache/extensions', TORCHINDUCTOR_CACHE_DIR='cache/inductor',
        TRITON_CACHE_DIR='cache/triton', CUDA_CACHE_PATH='cache/cuda', XDG_CACHE_HOME='cache/xdg',
        PYTHONPYCACHEPREFIX='cache/pycache', MPLCONFIGDIR='cache/matplotlib',
        __GL_SHADER_DISK_CACHE_PATH='cache/graphics').items():
        e[key] = str(w/folder)
    parent = Path(job['client_parent'] if client else job['model_parent'])
    sp = parent/'lib'/('python3.10' if client else 'python3.12')/'site-packages'
    paths = [str(sp/'torch/lib'), str(parent/'lib')]+[str(p) for p in (sp/'nvidia').glob('*/lib')]
    if job.get('glvnd'):
        paths += sorted({str(p.parent) for p in Path(job['glvnd']).rglob('libGL.so*')})
    paths += [p for p in e.get('LD_LIBRARY_PATH', '').split(':') if p and '/torch/lib' not in p]
    e['LD_LIBRARY_PATH'] = ':'.join(paths)
    return e


def launch(name, command, env, cpus):
    with (OUT/(name+'.log')).open('xb') as log:
        p = subprocess.Popen(['taskset', '-c', ','.join(map(str, cpus)), *map(str, command)], env=env,
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    row = dict(name=name, pid=p.pid, start=time.time(),
        birth_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    owned.append((p, row))
    return p, row


def stop(p, row):
    if p.poll() is None:
        stat = Path(f'/proc/{p.pid}/stat')
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[19] == row['birth_ticks']:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=20)
    row.update(end=time.time(), returncode=p.poll())


def worker(index):
    try:
        for case in job['cases'][index::2]:
            for arm in job['arms']:
                if cancel.is_set():
                    return
                guard()
                name = f'{case["split"]}-task{case["task"]:02d}-init{case["init"]:02d}-{arm}'
                output = ROOT/'results/episodes'/name
                assert not output.exists(), ('existing attempt is immutable', name)
                p, row = launch(name, [ROOT/'client-env/bin/python', '-u', ROOT/'code/client.py',
                    '--port', port, '--task', case['task'], '--init', case['init'], '--split', case['split'], '--arm', arm],
                    environment(job['id']+'-client'+str(index), True), cores[2+2*index:4+2*index])
                try:
                    while p.poll() is None:
                        guard()
                        assert server.poll() is None, 'model server exited'
                        assert time.time()-row['start'] < 1850, 'episode cleanup budget'
                        time.sleep(2)
                    result = json.loads((output/'EPISODE.json').read_text())
                    row['result'] = result
                    status['episodes'].append(row)
                    save()
                    assert p.returncode == 0 and result.get('complete'), result
                finally:
                    stop(p, row)
    except BaseException:
        cancel.set()
        raise


signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('owned supervisor stopped')))
try:
    save()
    assert job['suite'] == json.loads((ROOT/'PANEL.json').read_text())['suite']
    assert job['authorization_record'] and job['cases']
    status['status'] = 'WAITING_FREE_GPU'
    save()
    uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    while True:
        guard()
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if not any(line.startswith(uuid) for line in apps.splitlines()) and free >= 35000:
            break
        time.sleep(15)
    gpu_start = time.time()
    server_name = job['id']+'-server'
    server, server_row = launch('server', [ROOT/'env/bin/python', '-u', ROOT/'code/server.py',
        '--port', port, '--worker', server_name, '--assets', ASSETS], environment('server-gpu'+gpu), cores[:2])
    status.update(status='LOADING_MODEL', server=server_row)
    save()
    server_status = ROOT/'results'/server_name/'STATUS.json'
    while not server_status.exists() or json.loads(server_status.read_text()).get('phase') not in ('READY', 'SERVING'):
        guard()
        assert server.poll() is None, 'model loading failed'
        assert time.time()-gpu_start < 1800, 'model loading cap'
        time.sleep(2)
    status['status'] = 'RUNNING'
    save()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(worker, i) for i in (0, 1)]):
            future.result()
    assert len(status['episodes']) == len(job['cases'])*len(job['arms'])
    status['status'] = 'COMPLETE'
except BaseException as exc:
    status.update(status='FAILED_OR_PARTIAL', error=repr(exc), traceback=traceback.format_exc())
finally:
    for p, row in reversed(owned):
        stop(p, row)
    status.update(end=time.time(), owned=[r for p, r in owned],
                  gpu_window_seconds=0 if gpu_start is None else time.time()-gpu_start)
    save()

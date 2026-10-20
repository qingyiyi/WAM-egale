"""Bound an authorized real-scene repair attempt and retain owned-process evidence."""
from pathlib import Path
from datetime import datetime
import json, os, signal, subprocess, sys, time, traceback

R = Path('/root/cxz/robotwin')
mode = sys.argv[1]
if mode not in ('base', 'inherited'):
    raise ValueError(mode)
J = R / 'runs' / ('driver-interop-' + mode + '-1')
J.mkdir(exist_ok=False)
os.sched_setaffinity(0, {18})
start = time.time()
state = dict(status='PREPARING', start=start, mode=mode, policy_episodes=0, teacher_forwards=0)
proc = None
owned = None
seen = {}

def save():
    (J / 'GUARD.json').write_text(json.dumps(state, indent=2) + '\n')

def identity(pid):
    raw = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, birth_ticks=int(raw[19]), pgid=os.getpgid(pid))

def same(rec):
    try:
        return identity(rec['pid']) == rec
    except (FileNotFoundError, ProcessLookupError):
        return False

try:
    auth = json.loads((R / 'runs/AUTHORIZATION.json').read_text())
    ledger = json.loads((R / 'runs/GPU-LEDGER.json').read_text())
    deadline = datetime.fromisoformat(auth['CPU_preparation_deadline']).timestamp()
    if start + 660 > deadline or ledger['consumed_seconds'] + 660 > auth['GPU_lifecycle_cap_seconds']:
        raise RuntimeError('original deadline or total budget would be exceeded')
    previous = [json.loads(f.read_text()) for f in (R / 'runs').glob('driver-interop-*/GUARD.json')
                if f.parent != J]
    if len(previous) >= 3 or sum(x.get('lifecycle_seconds', 0) for x in previous) + 660 > 1800:
        raise RuntimeError('repair sub-budget exhausted')
    uuid = auth['GPU_uuid']
    contexts = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader'], text=True)
    if any(line.startswith(uuid) for line in contexts.splitlines()):
        raise RuntimeError('GPU1 occupied; no foreign-process intervention')
    env = {**os.environ, **json.loads((R / 'runs/CLIENT-ENVIRONMENT.json').read_text()),
           'CUDA_VISIBLE_DEVICES': uuid, 'PYTHONPATH': str(R / 'runs'),
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'HF_HUB_OFFLINE': '1', 'CUDA_CACHE_PATH': str(R / 'cache/cuda'),
           'WARP_CACHE_PATH': str(R / 'cache/warp'), 'OIDN_VERBOSE': '2'}
    if mode == 'base':
        lib = R / 'deps/system-driver535-links'
        lib.mkdir(exist_ok=True)
        for name in ['libcuda.so.1', 'libcuda.so', 'libnvidia-ptxjitcompiler.so.1', 'libnvidia-nvvm.so.4']:
            source = (Path('/usr/lib/x86_64-linux-gnu') / name).resolve(strict=True)
            if '535.161.08' not in source.name:
                raise RuntimeError('unexpected installed driver library ' + str(source))
            target = lib / name
            if not target.exists():
                target.symlink_to(source)
        paths = [p for p in env.get('LD_LIBRARY_PATH', '').split(':') if p and '/compat/' not in p]
        env['LD_LIBRARY_PATH'] = ':'.join([str(lib)] + paths)
    state.update(GPU=uuid, worker_cap_seconds=600, cleanup_reserve_seconds=60,
                 repair_total_cap_seconds=1800, storage_cap_bytes=256 * 2**20,
                 fixed_case=dict(task='place_object_basket', seed=10000),
                 library_path=env['LD_LIBRARY_PATH'], goal='original complete expert and observation; no policy')
    (J / 'REGISTRATION.json').write_text(json.dumps(state, indent=2) + '\n')
    with (J / 'worker.log').open('x') as log:
        proc = subprocess.Popen([str(R / 'envs/client/bin/python'), '-B',
                                 str(R / 'runs/render_driver_scene.py'), str(J / 'scene')],
                                cwd=R, env=env, stdout=log, stderr=log, start_new_session=True)
        owned = identity(proc.pid)
        seen[proc.pid] = owned
        state.update(status='RUNNING', owned=owned)
        save()
        while proc.poll() is None:
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():
                    continue
                try:
                    rec = identity(int(path.name))
                    if rec['pgid'] == owned['pgid']:
                        seen[rec['pid']] = rec
                except (FileNotFoundError, ProcessLookupError):
                    pass
            try:
                maps = Path(f'/proc/{proc.pid}/maps').read_text()
                selected = sorted({s.split()[-1] for s in maps.splitlines()
                                   if any(x in s for x in ('libcuda.', 'libOpenImageDenoise',
                                                           'libvulkan', 'libGLX_nvidia', 'libnvidia-'))})
                (J / 'LOADED-LIBRARIES.json').write_text(json.dumps(selected, indent=2) + '\n')
            except FileNotFoundError:
                pass
            tail = (J / 'worker.log').read_text(errors='replace')[-24000:]
            if 'OIDN Error:' in tail and 'illegal memory access' in tail:
                state['stop_reason'] = 'OIDN_ILLEGAL_MEMORY_ACCESS'
                break
            if time.time() - start >= 600:
                state['stop_reason'] = 'WALL_CAP'
                break
            if sum(f.stat().st_size for f in J.rglob('*') if f.is_file()) > 256 * 2**20:
                state['stop_reason'] = 'STORAGE_CAP'
                break
            time.sleep(2)
except BaseException as exc:
    state.update(error=repr(exc), traceback=traceback.format_exc())
finally:
    if proc is not None:
        if proc.poll() is None and owned is not None and same(owned):
            os.killpg(owned['pgid'], signal.SIGTERM)
            state['cleanup_signal'] = 'SIGTERM'
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if owned is not None and same(owned):
                os.killpg(owned['pgid'], signal.SIGKILL)
                state['cleanup_signal'] = 'SIGKILL'
            proc.wait(timeout=15)
        for rec in seen.values():
            if rec['pid'] != proc.pid and same(rec):
                os.kill(rec['pid'], signal.SIGTERM)
        state.update(returncode=proc.returncode, root_reaped=True,
                     owned_processes=list(seen.values()), remaining_owned=[r for r in seen.values() if same(r)])
    scene = J / 'scene/STATUS.json'
    state['scene'] = json.loads(scene.read_text()) if scene.exists() else None
    good = state['scene'] is not None and state['scene']['status'] == 'COMPLETE_ORIGINAL_EXPERT_AND_OBSERVATION'
    state.update(status='COMPLETE' if good and state.get('returncode') == 0 else 'FAILED_OR_PARTIAL',
                 end=time.time(), lifecycle_seconds=time.time() - start)
    save()
    if proc is not None:
        ledger = json.loads((R / 'runs/GPU-LEDGER.json').read_text())
        ledger['consumed_seconds'] += state['lifecycle_seconds']
        ledger['jobs'].append(dict(namespace=str(J), **state))
        (R / 'runs/GPU-LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n')

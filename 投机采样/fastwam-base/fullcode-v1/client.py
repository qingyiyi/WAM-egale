"""Complete official-protocol Long episode; isolated simulator and seed per process."""
import argparse
import hashlib
import json
import os
import pickle
import random
import signal
import socket
import struct
import time
import traceback
from pathlib import Path
import numpy as np
import torch

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, required=True)
P.add_argument('--task', type=int, required=True)
P.add_argument('--init', type=int, required=True)
P.add_argument('--split', required=True)
P.add_argument('--arm', choices=['T', 'S', 'P'], required=True)
args = P.parse_args()
ROOT = Path(__file__).resolve().parent.parent
suite_name = json.loads((ROOT/'PANEL.json').read_text())['suite']
horizon = {'libero_10':700, 'libero_spatial':400, 'libero_object':400, 'libero_goal':400}[suite_name]
identity = dict(suite=suite_name, task=args.task, init=args.init, split=args.split,
                noise_seed=42, env_seed=42, arm=args.arm)
name = f'{args.split}-task{args.task:02d}-init{args.init:02d}-{args.arm}'
OUT = ROOT / 'results/episodes' / name
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT/'EPISODE.json').exists(), 'terminal episode is immutable'
start = time.time()
record = dict(identity=identity, start=start, status='INITIALIZING', episode=name,
              action_steps=0, wait_steps=0, chunks=0, policy_seconds=0., success=False,
              renderer='official LIBERO original MSAA unchanged')


def write(path, value):
    p = Path(path)
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    q.replace(p)


def receive(sock, count):
    out = bytearray()
    while len(out) < count:
        data = sock.recv(count-len(out))
        if not data: raise EOFError()
        out.extend(data)
    return bytes(out)


def rpc(req):
    data = pickle.dumps(req, protocol=4)
    with socket.create_connection(('127.0.0.1', args.port), timeout=1200) as sock:
        sock.sendall(struct.pack('!I', len(data))+data)
        n = struct.unpack('!I', receive(sock, 4))[0]
        response = pickle.loads(receive(sock, n))
    if not response['ok']: raise RuntimeError(response)
    return response['result']


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def obs_subset(obs):
    return {k: np.array(obs[k], copy=True) for k in ['agentview_image', 'robot0_eye_in_hand_image',
            'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']}


env = None
actions = []
try:
    reg = json.loads((ROOT/'REGISTRATION.json').read_text())
    deadline = min(reg['deadline_epoch']-300, start+1800)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('complete episode budget exceeded')))
    signal.alarm(max(1, int(deadline-time.time())))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(args.task)
    # Pinned official LIBERO init files contain NumPy arrays. Explicitly retain
    # the official pre-Torch2.6 pickle-loading semantics for these trusted files.
    init_path = Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
    states = torch.load(init_path, weights_only=False)
    assert len(states) > args.init
    init = states[args.init]
    record.update(task_name=task.name, task_description=task.language, init_count=len(states),
                  init_vector_sha256=digest(init))
    env = OffScreenRenderEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,
                            camera_heights=256, camera_widths=256)
    env.seed(42)
    env.reset()
    obs = env.set_init_state(init)
    record['reset_physics_sha256'] = digest(env.sim.get_state().flatten())
    for _ in range(30):
        obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        record['wait_steps'] += 1
    record['ready_physics_sha256'] = digest(env.sim.get_state().flatten())
    record['first_image_sha256'] = {k: digest(obs[k]) for k in ['agentview_image', 'robot0_eye_in_hand_image']}
    np.savez_compressed(OUT/'initial-state.npz', physics=env.sim.get_state().flatten(), **obs_subset(obs))
    record['status'] = 'RUNNING'
    write(OUT/'STATUS.json', record)
    pending = []
    counts = {}
    while record['action_steps'] < horizon:
        if not pending:
            ident = dict(identity, episode=name, chunk=record['chunks'])
            reply = rpc(dict(op='predict', arm=args.arm, identity=ident, task=task.language, obs=obs_subset(obs)))
            record.setdefault('server_identity', reply['server_identity'])
            assert record['server_identity'] == reply['server_identity'], 'server changed within episode'
            action = reply['action']
            assert action.shape == (32, 7) and np.isfinite(action).all()
            pending = action[:10].tolist()
            record['chunks'] += 1
            record['policy_seconds'] += reply['seconds']
            for k, v in reply['counts'].items(): counts[k] = counts.get(k, 0) + v
            record['counts'] = counts
            write(OUT/'STATUS.json', record)
        action = pending.pop(0)
        obs, _, done, _ = env.step(action)
        actions.append(action)
        record['action_steps'] += 1
        if done:
            record['success'] = True
            break
    assert record['counts']['committed_updates'] == 10*record['chunks']
    record.update(status='COMPLETE_SUCCESS' if record['success'] else 'COMPLETE_TIMEOUT_FAILURE',
                  complete=True, horizon=horizon, physical_steps=30+record['action_steps'],
                  commands=record['action_steps'], actual_final_check_success=bool(env.check_success()))
    assert record['success'] == record['actual_final_check_success']
except BaseException as exc:
    record.update(status='PARTIAL_OR_INVALID', complete=False, error=repr(exc), traceback=traceback.format_exc())
finally:
    record.update(end=time.time(), wall_seconds=time.time()-start)
    np.save(OUT/'executed-actions.npy', np.array(actions, dtype=np.float32))
    write(OUT/'EPISODE.json', record)
    write(OUT/'STATUS.json', record)
    if env is not None: env.close()
    signal.alarm(0)
print(json.dumps(record, ensure_ascii=False), flush=True)
if not record.get('complete'): raise SystemExit(1)

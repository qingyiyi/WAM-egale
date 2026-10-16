"""Lean, deterministic full-horizon EGL client; original loop retained.

No per-step hashes, disk writes or full native observer. Commands are copied
to RAM. Initial/final simulator/controller/RNG and commands are saved after
the original task timer has stopped. Env construction is a single attempt.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

SEED = 20260922
ROOT = Path('/rjs/cxz/论文探索')

def seed():
    random.seed(SEED)
    np.random.seed(SEED)

def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def rng():
    n = np.random.get_state()
    return dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                numpy=hashlib.sha256(n[0].encode()+n[1].tobytes()+repr(n[2:]).encode()).hexdigest())

def state(env):
    e = env.env
    sim = e.sim
    data = sim.data._data
    s = {'flattened':np.array(sim.get_state().flatten(),copy=True)}
    for name in ('qacc_warmstart','ctrl','act','qacc','qfrc_applied','xfrc_applied',
                 'mocap_pos','mocap_quat','qpos','qvel'):
        s[name] = np.array(getattr(data,name),copy=True)
    for name in ('body_xpos','body_xquat','site_xpos','site_xmat'):
        s[name] = np.array(getattr(sim.data,name),copy=True)
    ctrl = e.robots[0].controller
    for name in ('goal_pos','goal_ori','joint_pos','joint_vel','initial_joint'):
        if getattr(ctrl,name,None) is not None:
            s['controller_'+name] = np.array(getattr(ctrl,name),copy=True)
    return s

def install(client, out):
    original_init, original_episode = client._init_env, client._run_episode
    memory = {}

    def construct(cls, args):
        seed()
        return cls(**args)  # one actual EGL initialization, no hidden retry

    def init(env, init_state):
        seed()
        env.seed(SEED)
        obs, timestep = original_init(env,init_state)
        memory['initial'] = state(env)
        memory['initial_obs'] = {k:np.array(v,copy=True) for k,v in obs.items()}
        memory['initial_rng'] = rng()
        memory['init_state'] = np.array(init_state,copy=True)
        memory['xml'] = env.env.sim.model.get_xml()
        memory['commands'] = []
        original_step, original_close = env.step, env.close
        def step(command):
            result = original_step(command)
            memory['commands'].append(np.array(command,dtype=np.float32,copy=True))
            return result
        def close():
            # Base client creates its elapsed_sec record BEFORE this finally.
            memory['final'] = state(env)
            memory['final_rng'] = rng()
            return original_close()
        env.step, env.close = step, close
        return obs, timestep

    def episode(**kwargs):
        memory.clear()
        # Retain the original per-chunk quality events and full-horizon loop.
        if kwargs.get('chunk_events_path') is None:
            raise ValueError('diagnostic chunk events required')
        start = time.monotonic()
        record = original_episode(**kwargs)
        record['outer_episode_wall_sec'] = time.monotonic()-start
        record['outer_episode_includes_construction_and_close'] = True
        record['instrumentation'] = 'RAM commands + initial/final snapshots + original per-chunk quality events; not a speed benchmark'
        out.mkdir(parents=True,exist_ok=True)
        np.savez(out/'initial_state.npz',**memory['initial'])
        np.savez(out/'final_state.npz',**memory['final'])
        np.save(out/'commands.npy',np.asarray(memory['commands'],dtype=np.float32))
        meta = dict(seed=SEED, actual_EGL_initialization_attempts=1,
                    model_xml_sha256=hashlib.sha256(memory['xml'].encode()).hexdigest(),
                    init_state_sha256=array_hash(memory['init_state']),
                    initial_state_hashes={k:array_hash(v) for k,v in memory['initial'].items()},
                    final_state_hashes={k:array_hash(v) for k,v in memory['final'].items()},
                    initial_obs_hashes={k:array_hash(v) for k,v in memory['initial_obs'].items()},
                    initial_rng=memory['initial_rng'], final_rng=memory['final_rng'],
                    command_count=len(memory['commands']),
                    no_policy_chunk_truncation=True, per_step_file_IO=False,
                    initial_final_artifact_writes_after_task_timer=True,
                    chunk_event_logging=True, heavy_native_observer=False)
        if meta['command_count'] != record['env_steps']:
            raise ValueError('actual command/env_steps coverage mismatch')
        (out/'repro_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        return record

    client._construct_env, client._init_env, client._run_episode = construct, init, episode

def main():
    if (os.getenv('MUJOCO_GL') != 'egl' or os.getenv('PYOPENGL_PLATFORM') != 'egl'
            or os.getenv('MUJOCO_EGL_DEVICE_ID') != '0' or os.getenv('CUDA_VISIBLE_DEVICES') != '0'
            or os.getenv('P0_NATIVE_DIAGNOSTIC_DIR') or os.getenv('WAM_P0_STATE_TEACHER_DIR')):
        raise SystemExit('formal EGL/diagnostic boundary mismatch')
    sys.path.insert(0,str(ROOT/'投机采样/scripts'))
    import p0_libero_multichunk_client as client
    args = client.parse_args()
    if (args.episodes_per_task != 1 or args.episode_start not in (3,4)
            or args.trace_mode != 'off' or args.latency_benchmark
            or args.benchmarks != ['libero_10'] or args.task_indices != [0]):
        raise SystemExit('fresh full-episode assignment mismatch')
    seed()
    install(client,args.out_dir)
    return client.main()

if __name__ == '__main__':
    raise SystemExit(main())

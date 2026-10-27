"""Prepared one-complete-episode client with explicit RoboTwin replay identity.

Uses the pinned expert validity test and complete task horizon. Target records
the accepted scene and Python RNG state; paired arms replay those identities.
No GPU run is authorized by this file's existence.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import socket
import struct
import sys
import time
import traceback

import numpy as np


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, list) else value


def instrument_eval(text):
    replacements = {
        '    st_seed = 100000 * (1 + seed)\n': '    st_seed = audit.seed_start\n',
        '    now_id = 0\n': '    now_id = audit.official_episode_index\n',
        '    while succ_seed < test_num:\n': '    while succ_seed < test_num:\n        audit.begin(now_seed)\n',
        '        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)\n':
            '        results = generate_episode_descriptions(args["task_name"], episode_info_list, audit.instruction_generation_count)\n',
        '        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction\n':
            '        audit.set_instruction(TASK_ENV, instruction, episode_info_list)\n',
        '        if succ:\n            TASK_ENV.suc += 1':
            '        audit.finish(now_seed, succ, TASK_ENV.take_action_cnt, TASK_ENV.step_lim)\n        if succ:\n            TASK_ENV.suc += 1',
        '            now_seed += 1\n            args["render_freq"] = render_freq\n            continue':
            '            audit.invalid(now_seed, "expert_check_failed")\n            now_seed += 1\n            args["render_freq"] = render_freq\n            continue',
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, ('pinned eval changed', old, text.count(old))
        text = text.replace(old, new)
    assert text.count('                now_seed += 1\n') == 2
    text = text.replace('                now_seed += 1\n',
                        '                audit.invalid(now_seed, "expert_setup_exception_logged")\n                now_seed += 1\n')
    assert text.count('            succ_seed -= 1\n') == 2
    text = text.replace('            succ_seed -= 1\n',
                        '            audit.invalid(now_seed, "rollout_setup_exception_logged")\n            succ_seed -= 1\n')
    return text


def physics_snapshot(env):
    actors = []
    for i, actor in enumerate(env.scene.get_all_actors()):
        pose = actor.get_pose()
        actors.append(dict(index=i, name=actor.get_name(), p=pose.p.tolist(), q=pose.q.tolist()))
    articulations = []
    for i, actor in enumerate(env.scene.get_all_articulations()):
        pose = actor.get_pose()
        articulations.append(dict(index=i, name=actor.get_name(), p=pose.p.tolist(), q=pose.q.tolist(),
                                  qpos=actor.get_qpos().tolist(), qvel=actor.get_qvel().tolist()))
    return dict(actors=actors, articulations=articulations)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    vendor = root / 'source/RoboTwin'
    reg = json.loads((root / 'REGISTRATION.json').read_text())
    case = json.loads(a.case.read_text())
    review = root / 'CANONICAL-REVIEW.json'
    assert hashlib.sha256(review.read_bytes()).hexdigest() == reg['canonical_review_sha256']
    assert json.loads(review.read_text())['passed']
    assert case['arm'] in ('T', 'S', 'P') and case['split'] in ('FIT', 'CAL', 'DEV', 'FULL')
    assert case['arm'] == 'T' or case['split'] in ('DEV', 'FULL')
    assert case['task_config'] in ('demo_clean', 'demo_randomized') and case['model_seed'] == 42
    out = root / 'results/episodes' / case['episode']
    assert out.parent == root / 'results/episodes' and out.name == case['episode']
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'EPISODE.json').exists()
    paired = None
    if case.get('paired_target_episode'):
        reference = root / 'results/episodes' / case['paired_target_episode']
        assert reference.parent == root / 'results/episodes'
        target = json.loads((reference / 'EPISODE.json').read_text())
        paired = json.loads((reference / 'REPLAY-IDENTITY.json').read_text())
        assert target['complete'] and target['identity']['arm'] == 'T'
        assert target['identity']['group'] == case['group']
        assert target['identity']['task_config'] == case['task_config']
        assert target['identity']['task'] == case['task']
        assert paired['scene_seed'] == case['seed_start']
        assert paired['official_episode_index'] == case['official_episode_index']
        assert paired['instruction_generation_count'] == case['instruction_generation_count']
    assert case['arm'] == 'T' or paired is not None, 'candidate needs its paired target identity'
    start = time.time()
    cap = min(reg['episode_cap_seconds'], reg['deadline_epoch'] - 300 - start)
    assert cap > 0
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('registered complete-episode cap')))
    signal.alarm(max(1, int(cap)))
    status = dict(identity=case, episode=case['episode'], status='INITIALIZING', start=start,
                  complete=False, success=False, commands=0, chunks=0, invalid_attempts=[])

    def write(name, value):
        path = out / name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)

    def receive(sock, count):
        data = bytearray()
        while len(data) < count:
            block = sock.recv(count - len(data))
            if not block:
                raise EOFError()
            data.extend(block)
        return bytes(data)

    class Policy:
        def __init__(self):
            self.pending = deque()
            self.commands = []
            self.chunks = 0
            self.policy_seconds = 0.
            self.server_identity = None

        def reset(self):
            assert self.chunks == 0, 'one complete episode per isolated client'
            self.pending.clear()

        def should_request_observation(self):
            return not self.pending

        def step(self, env, obs):
            if not self.pending:
                assert obs is not None
                small = dict(observation={c: dict(rgb=np.array(obs['observation'][c]['rgb'], copy=True))
                    for c in ('head_camera', 'left_camera', 'right_camera')},
                    joint_action=dict(vector=np.array(obs['joint_action']['vector'], copy=True)))
                if self.chunks == 0:
                    snapshot = physics_snapshot(env)
                    write('INITIAL-PHYSICS.json', snapshot)
                    status.update(first_rgb_sha256={c: hashlib.sha256(v['rgb'].tobytes()).hexdigest()
                        for c, v in small['observation'].items()},
                        initial_joint=small['joint_action']['vector'].tolist(), step_limit=int(env.step_lim),
                        initial_physics_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest())
                ident = dict(case, chunk=self.chunks, scene_seed=audit.current_seed)
                request = dict(op='predict', arm=case['arm'], identity=ident, obs=small, instruction=env.get_instruction())
                payload = pickle.dumps(request, protocol=4)
                with socket.create_connection(('127.0.0.1', a.port), timeout=1200) as sock:
                    sock.sendall(struct.pack('!I', len(payload)) + payload)
                    size = struct.unpack('!I', receive(sock, 4))[0]
                    assert 0 < size < 16*1024*1024
                    reply = pickle.loads(receive(sock, size))
                assert reply['ok'], reply
                result = reply['result']
                action = result['action']
                assert action.shape == (32, 14) and np.isfinite(action).all()
                assert result['counts']['committed_updates'] == 10
                if self.server_identity is None:
                    self.server_identity = result['server_identity']
                assert self.server_identity == result['server_identity']
                self.pending.extend(action[:24])
                self.chunks += 1
                self.policy_seconds += result['seconds']
            action = self.pending.popleft()
            before = env.take_action_cnt
            env.take_action(action, action_type='qpos')
            assert env.take_action_cnt == before + 1, 'submitted command was not executed'
            self.commands.append(action.copy())
            status.update(status='RUNNING', commands=len(self.commands), chunks=self.chunks,
                          policy_seconds=self.policy_seconds, server_identity=self.server_identity)
            write('STATUS.json', status)

    policy = Policy()

    class Audit:
        seed_start = int(case['seed_start'])
        official_episode_index = int(case['official_episode_index'])
        instruction_generation_count = int(case['instruction_generation_count'])

        def begin(self, seed):
            self.current_seed = int(seed)
            assert self.seed_start <= seed < self.seed_start + case['max_candidate_seeds']
            if paired:
                assert seed == paired['scene_seed'], 'paired invalid scene cannot be replaced'
                random.setstate(tuple_tree(paired['python_random_state_before_expert']))
            self.python_state = random.getstate()
            write(f'python-rng-before-seed-{seed}.json', dict(seed=seed, state=self.python_state,
                replayed_from_target=paired is not None))

        def invalid(self, seed, reason):
            status['invalid_attempts'].append(dict(seed=int(seed), reason=reason))
            write('STATUS.json', status)
            if paired:
                raise RuntimeError('paired expert/setup validity changed; retain invalid identity')

        def set_instruction(self, env, instruction, episode_info):
            instruction = str(instruction)
            if paired:
                assert instruction == paired['instruction'], 'paired generated instruction differs'
            env.set_instruction(instruction=instruction)
            status.update(scene_seed=self.current_seed, instruction=instruction)
            write('REPLAY-IDENTITY.json', dict(scene_seed=self.current_seed, instruction=instruction,
                official_episode_index=self.official_episode_index,
                instruction_generation_count=self.instruction_generation_count,
                python_random_state_before_expert=self.python_state, episode_info=episode_info,
                note='Target observes official unseeded Python random state; paired arms restore that exact state. NumPy/Torch use the official setup seeds.'))

        def finish(self, seed, success, commands, limit):
            assert len(policy.commands) == commands and policy.chunks == (commands + 23)//24
            assert seed == self.current_seed
            status.update(status='COMPLETE_SUCCESS' if success else 'COMPLETE_TIMEOUT_FAILURE',
                success=bool(success), complete=True, scene_seed=int(seed), commands=int(commands),
                action_steps=int(commands), step_limit=int(limit), horizon=int(limit), chunks=policy.chunks,
                policy_seconds=policy.policy_seconds, server_identity=policy.server_identity,
                end=time.time(), wall_seconds=time.time()-start)
            np.save(out / 'executed-actions.npy', np.asarray(policy.commands, dtype=np.float32))
            write('EPISODE.json', status)

    audit = Audit()
    try:
        os.chdir(vendor)
        sys.path[:0] = [str(vendor), str(vendor/'policy'), str(vendor/'description/utils')]
        source = vendor/'script/eval_policy.py'
        text = instrument_eval(source.read_text())
        (out/'INSTRUMENTED-EVAL.py').write_text(text)
        namespace = dict(__name__='fixed_robotwin_eval', __file__=str(source), audit=audit)
        exec(compile(text, str(source), 'exec'), namespace)

        def decorator(_name, method):
            if method == 'get_model':
                return lambda _: policy
            if method == 'reset_model':
                return lambda model: model.reset()
            if method == 'eval':
                return lambda env, model, obs: model.step(env, obs)
            raise ValueError(method)

        namespace['eval_function_decorator'] = decorator
        namespace['main'](dict(task_name=case['task'], task_config=case['task_config'],
            ckpt_setting='fastwam-base-8eaceeb', policy_name='fastwam_rpc', instruction_type='unseen',
            eval_num_episodes=1, seed=42, eval_output_dir=str(out/'official'), skip_get_obs_within_replan=True))
        assert status['complete'], 'no complete valid episode'
    except BaseException as exc:
        if not status['complete']:
            status.update(status='PARTIAL_OR_INVALID', error=repr(exc), traceback=traceback.format_exc(),
                          end=time.time(), wall_seconds=time.time()-start)
            np.save(out/'executed-actions.npy', np.asarray(policy.commands, dtype=np.float32))
            write('EPISODE.json', status)
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()

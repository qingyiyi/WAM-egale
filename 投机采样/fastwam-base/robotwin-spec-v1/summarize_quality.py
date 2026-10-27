"""Audit complete RoboTwin triads without dropping failures or invalid cases.

PANEL[DEV/FULL] rows contain group/task/task_config and an episodes{T,S,P}
mapping. This reports engineering evidence; it never sets scientific permission
or admits a new full panel by itself.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from policy_schedule import policy_role


def read(path):
    return json.loads(Path(path).read_text())


def describe(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return dict(N=len(array), mean=float(array.mean()), p50=float(np.median(array)),
                p90=float(np.quantile(array, .9)), max=float(array.max()))


def wilson(success, total):
    if total == 0:
        return None
    z = 1.95996398454
    p = success/total
    den = 1+z*z/total
    center = (p+z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half, center+half]


def summarize(root, split):
    panel = read(root/'PANEL.json')
    rows = panel[split]
    assert len({row['group'] for row in rows}) == len(rows)
    planned_names = [row['episodes'][arm] for row in rows for arm in ('T', 'S', 'P')]
    assert len(set(planned_names)) == len(planned_names)
    assert all(Path(name).name == name for name in planned_names)
    result = dict(schema='fastwam-robotwin-triarm-quality/v1', created=time.time(), split=split,
        source=str(root), panel_sha256=hashlib.sha256((root/'PANEL.json').read_bytes()).hexdigest(),
        quality={}, per_task=[], paired={}, acceptance={}, audits=[], errors=[],
        exposure=panel.get('exposure', 'Not independent TEST; actual FIT/CAL/DEV exposure must remain explicit.'),
        timing='Concurrent quality wall/policy seconds only; not an exclusive benchmark.',
        four_permissions={key: False for key in ('engineering_quality_screen', 'independent_test_complete',
                                                'noninferiority_established', 'formal_performance_permission')})
    episodes = {}
    counts = {arm: Counter() for arm in ('T', 'S', 'P')}
    for row in rows:
        for arm in ('T', 'S', 'P'):
            name = row['episodes'][arm]
            folder = root/'results/episodes'/name
            if not (folder/'EPISODE.json').exists():
                continue
            episode = read(folder/'EPISODE.json')
            episodes[(row['group'], arm)] = episode
            if not episode.get('complete'):
                continue
            try:
                identity = episode['identity']
                assert identity['episode'] == name and identity['arm'] == arm
                assert identity['group'] == row['group'] and identity['split'] == split
                assert identity['task'] == row['task'] and identity['task_config'] == row['task_config']
                assert episode['scene_seed'] >= identity['seed_start']
                assert episode['scene_seed'] < identity['seed_start']+identity['max_candidate_seeds']
                assert 0 < episode['commands'] <= episode['step_limit']
                physical = read(folder/'INITIAL-PHYSICS.json')
                assert hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest() == episode['initial_physics_sha256']
                chunks = sorted(folder.glob('chunk-*.json'))
                assert len(chunks) == episode['chunks'] == (episode['commands']+23)//24
                executed = np.load(folder/'executed-actions.npy')
                assert executed.shape == (episode['commands'], 14) and np.isfinite(executed).all()
                expected = []
                local = Counter()
                for ordinal, path in enumerate(chunks):
                    chunk = read(path)
                    role = policy_role(arm, ordinal)
                    assert chunk['arm'] == arm and chunk['policy_role'] == role
                    assert chunk['identity']['chunk'] == ordinal and chunk['identity']['episode'] == name
                    action = np.asarray(chunk['action'], dtype=np.float32)
                    normalized = np.asarray(chunk['normalized_action'])
                    assert action.shape == normalized.shape == (32, 14)
                    assert np.isfinite(action).all() and np.isfinite(normalized).all()
                    c = chunk['counts']
                    assert c['committed_updates'] == 10
                    assert c.get('PF_refresh_chunks', 0) == int(arm == 'P' and role == 'T')
                    if role == 'S':
                        cursor = 0
                        for rnd in chunk['rounds']:
                            assert rnd['step'] == cursor
                            assert rnd['g'] == (rnd['depth']+1 if rnd['kind'] == 'tree' else 1)
                            if rnd['kind'] == 'tree':
                                assert rnd['step'] <= 7 and rnd['depth'] in (0, 1, 2)
                            cursor += rnd['g']
                            assert cursor <= 10
                        assert cursor == 10
                        assert c['candidate_expansions'] == 6*c['tree_count']
                        assert c['target_B7_calls'] == c['tree_count']
                        assert c['target_state_rows'] == c['target_B1_calls']+7*c['target_B7_calls']
                        assert sum(c.get(key, 0) for key in ('bootstrap_updates', 'draft_updates', 'bonus_updates',
                                                            'fallback_updates', 'boundary_target_updates')) == 10
                    else:
                        assert c['target_calls'] == c['target_state_rows'] == 10
                    local.update(c)
                    expected.extend(action[:24])
                assert np.array_equal(executed, np.asarray(expected[:len(executed)], dtype=np.float32))
                assert local['committed_updates'] == 10*episode['chunks']
                assert local['PF_refresh_chunks'] == (episode['chunks']//3 if arm == 'P' else 0)
                counts[arm].update(local)
                result['audits'].append(dict(episode=name, passed=True, chunks=len(chunks), commands=len(executed)))
            except BaseException as exc:
                result['errors'].append(dict(episode=name, error=repr(exc)))
    for arm in ('T', 'S', 'P'):
        planned = [episodes.get((row['group'], arm)) for row in rows]
        complete = [episode for episode in planned if episode and episode.get('complete')]
        success = sum(episode['success'] for episode in complete)
        result['quality'][arm] = dict(N_planned=len(rows), N_valid=len(complete), success=success,
            failure=len(complete)-success, missing=sum(episode is None for episode in planned),
            partial_or_invalid=sum(episode is not None and not episode.get('complete') for episode in planned),
            SR=success/len(complete) if complete else None, Wilson95=wilson(success, len(complete)),
            actual_commands=describe([episode['commands'] for episode in complete]),
            policy_chunks=describe([episode['chunks'] for episode in complete]),
            concurrent_wall_seconds=describe([episode['wall_seconds'] for episode in complete]))
        c = counts[arm]
        trees = c['tree_count']
        attempted = trees+c['depth_1']+c['depth_2']
        result['acceptance'][arm] = dict(counts=dict(c),
            conditional_path_step_pass=c['draft_updates']/attempted if attempted else None,
            depth_fraction={str(d): c[f'depth_{d}']/trees if trees else None for d in range(3)})
    for task, config in sorted({(row['task'], row['task_config']) for row in rows}):
        item = dict(task=task, task_config=config)
        for arm in ('T', 'S', 'P'):
            selected = [episodes.get((row['group'], arm)) for row in rows
                        if row['task'] == task and row['task_config'] == config]
            valid = [episode for episode in selected if episode and episode.get('complete')]
            item[arm] = dict(N_planned=len(selected), N_valid=len(valid), success=sum(e['success'] for e in valid))
        result['per_task'].append(item)
    for arm in ('S', 'P'):
        pairs = []
        for row in rows:
            target, candidate = [episodes.get((row['group'], a)) for a in ('T', arm)]
            if not target or not candidate or not target.get('complete') or not candidate.get('complete'):
                continue
            identity = all(target['identity'][key] == candidate['identity'][key]
                for key in ('group', 'task', 'task_config', 'model_seed', 'official_episode_index', 'instruction_generation_count'))
            identity &= target['scene_seed'] == candidate['scene_seed'] and target['instruction'] == candidate['instruction']
            hardware = all(target['server_identity'][key] == candidate['server_identity'][key]
                for key in ('host', 'gpu', 'visible_devices', 'torch', 'cuda', 'compile_action_infer'))
            paths = [root/'results/episodes'/row['episodes'][a] for a in ('T', arm)]
            replay = [read(path/'REPLAY-IDENTITY.json') for path in paths]
            rng = replay[0]['python_random_state_before_expert'] == replay[1]['python_random_state_before_expert']
            physical = target['initial_physics_sha256'] == candidate['initial_physics_sha256']
            pixels = target['first_rgb_sha256'] == candidate['first_rgb_sha256']
            pairs.append(dict(group=row['group'], task=row['task'], task_config=row['task_config'],
                scene_seed=target['scene_seed'], T_success=target['success'], candidate_success=candidate['success'],
                identity_exact=identity, hardware_exact=hardware, instruction_RNG_exact=rng,
                initial_physics_exact=physical, initial_pixels_exact=pixels,
                T_commands=target['commands'], candidate_commands=candidate['commands'],
                command_ratio=candidate['commands']/target['commands'],
                T_chunks=target['chunks'], candidate_chunks=candidate['chunks']))
        common = [pair for pair in pairs if pair['T_success'] and pair['candidate_success']]
        result['paired'][arm] = dict(N=len(pairs), both_success=len(common),
            target_only=sum(pair['T_success'] and not pair['candidate_success'] for pair in pairs),
            candidate_only=sum(not pair['T_success'] and pair['candidate_success'] for pair in pairs),
            both_failure=sum(not pair['T_success'] and not pair['candidate_success'] for pair in pairs),
            common_command_ratio=describe([pair['command_ratio'] for pair in common]), pairs=pairs)
    conditions = dict(all_planned_valid=all(q['N_valid'] == len(rows) for q in result['quality'].values()),
                      actual_commands_and_scheduler=not result['errors'] and len(result['audits']) == 3*len(rows))
    for arm in ('S', 'P'):
        pairs = result['paired'][arm]['pairs']
        conditions[arm+'_paired_identity'] = len(pairs) == len(rows) and all(
            all(pair[key] for key in ('identity_exact', 'hardware_exact', 'instruction_RNG_exact', 'initial_physics_exact'))
            for pair in pairs)
    result['data_audit_conditions'] = conditions
    result['data_audit_complete'] = all(conditions.values())
    result['expansion_permission'] = 'Not decided by this descriptive script; requires preregistered screen, actual B7 audits and scene-split audit.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--split', choices=('DEV', 'FULL'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.split)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(quality=result['quality'], data_audit_conditions=result['data_audit_conditions'])))

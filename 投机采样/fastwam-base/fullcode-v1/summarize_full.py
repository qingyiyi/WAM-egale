"""Three-arm engineering screen with raw commit audit; no permission promotion."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from policy_schedule import policy_role

P = argparse.ArgumentParser()
P.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
args = P.parse_args()
ROOT = args.root
panel = json.loads((ROOT/'PANEL.json').read_text())
registration = json.loads((ROOT/'OVERNIGHT-EXTENSION-v1.json').read_text())


def describe(xs):
    a = np.asarray(xs, dtype=float)
    return None if not len(a) else dict(N=len(a), mean=float(a.mean()),
        p50=float(np.median(a)), p90=float(np.quantile(a, .9)), max=float(a.max()))


def wilson(k, n):
    if not n:
        return None
    z = 1.95996398454
    p, den = k/n, 1+z*z/n
    center = (p+z*z/(2*n))/den
    width = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-width, center+width]


def acceptance(counts):
    trees = counts.get('tree_count', 0)
    drafts = counts.get('draft_updates', 0)
    attempted_path_steps = trees+counts.get('depth_1', 0)+counts.get('depth_2', 0)
    return dict(counts=dict(counts),
        depth_fraction={str(d): counts.get(f'depth_{d}', 0)/trees if trees else None for d in range(3)},
        mean_accepted_depth=drafts/trees if trees else None,
        accepted_draft_fraction_of_two_slots=drafts/(2*trees) if trees else None,
        conditional_path_step_pass=drafts/attempted_path_steps if attempted_path_steps else None,
        denominator_note='Tree-depth distribution is primary. Six candidate expansions are not six submitted updates; PF target refreshes are outside the tree denominator.')


episodes, aggregate, raw_audits, errors = {}, {a: Counter() for a in ('T', 'S', 'P')}, [], []
for case in panel['FULL']:
    for arm in ('T', 'S', 'P'):
        name = f'FULL-task{case["task"]:02d}-init{case["init"]:02d}-{arm}'
        folder = ROOT/'results/episodes'/name
        path = folder/'EPISODE.json'
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        episodes[(case['task'], case['init'], arm)] = row
        if not row.get('complete'):
            continue
        try:
            chunks = [json.loads(p.read_text()) for p in sorted(folder.glob('chunk-*.json'))]
            assert len(chunks) == row['chunks'] == (row['action_steps']+9)//10
            command = np.load(folder/'executed-actions.npy')
            assert command.shape == (row['action_steps'], 7) and np.isfinite(command).all()
            expected, counts = [], Counter()
            for ordinal, chunk in enumerate(chunks):
                role = policy_role(arm, ordinal)
                assert chunk['arm'] == arm and chunk['identity']['chunk'] == ordinal
                assert chunk.get('policy_role', chunk['arm']) == role
                assert chunk['counts']['committed_updates'] == 10
                if role == 'S':
                    rounds = chunk['rounds']
                    assert sum(r['g'] for r in rounds) == 10
                    cursor = 0
                    for r in rounds:
                        assert r['step'] == cursor and cursor+r['g'] <= 10
                        if r['kind'] == 'tree':
                            assert r['g'] == r['depth']+1 and r['step'] <= 7
                        else:
                            assert r['g'] == 1
                        cursor += r['g']
                else:
                    assert chunk['counts']['target_calls'] == chunk['counts']['target_state_rows'] == 10
                expected.extend(chunk['action'][:10])
                counts.update(chunk['counts'])
            assert np.array_equal(command, np.asarray(expected[:len(command)], dtype=np.float32))
            assert counts['committed_updates'] == 10*row['chunks']
            if arm == 'P':
                assert counts['PF_refresh_chunks'] == row['chunks']//3
            aggregate[arm].update(counts)
            raw_audits.append(dict(episode=name, chunks=row['chunks'], commands=len(command), passed=True))
        except Exception as exc:
            errors.append(dict(episode=name, error=repr(exc)))

quality, per_task, paired = {}, [], {}
for arm in ('T', 'S', 'P'):
    planned = [episodes.get((c['task'], c['init'], arm)) for c in panel['FULL']]
    complete = [r for r in planned if r and r.get('complete')]
    successes = sum(r['success'] for r in complete)
    quality[arm] = dict(N_planned=len(planned), N_valid=len(complete), success=successes,
        failure=sum(not r['success'] for r in complete), missing=sum(r is None for r in planned),
        partial_or_invalid=sum(r is not None and not r.get('complete') for r in planned),
        SR=successes/len(complete) if complete else None, Wilson95=wilson(successes, len(complete)),
        actual_commands=describe([r['action_steps'] for r in complete]),
        policy_chunks=describe([r['chunks'] for r in complete]),
        horizon_capped_commands=describe([r['action_steps'] if r['success'] else r['horizon'] for r in complete]),
        concurrent_wall_seconds=describe([r['wall_seconds'] for r in complete]))
for task in range(10):
    row = dict(task=task)
    for arm in ('T', 'S', 'P'):
        valid = [r for (t, i, a), r in episodes.items() if t == task and a == arm and r.get('complete')]
        row[arm] = dict(N=len(valid), success=sum(r['success'] for r in valid))
    per_task.append(row)

for arm in ('S', 'P'):
    pairs = []
    for c in panel['FULL']:
        t, s = [episodes.get((c['task'], c['init'], a)) for a in ('T', arm)]
        if not (t and s and t.get('complete') and s.get('complete')):
            continue
        physics = all(t[k] == s[k] for k in ('init_vector_sha256', 'reset_physics_sha256', 'ready_physics_sha256'))
        hardware = all(t['server_identity'][k] == s['server_identity'][k] for k in ('host', 'gpu', 'visible_devices', 'torch', 'cuda'))
        identity = all(t['identity'][k] == s['identity'][k] for k in ('suite', 'task', 'init', 'noise_seed', 'env_seed'))
        pixel = {}
        tdir, sdir = [ROOT/'results/episodes'/r['episode'] for r in (t, s)]
        with np.load(tdir/'initial-state.npz') as tz, np.load(sdir/'initial-state.npz') as sz:
            for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
                diff = np.abs(tz[camera].astype(np.int16)-sz[camera].astype(np.int16))
                pixel[camera] = dict(channels_different=int(np.count_nonzero(diff)),
                    pixels_different=int(np.count_nonzero(diff.any(-1))), max_abs=int(diff.max()), mean_abs=float(diff.mean()))
        pairs.append(dict(task=c['task'], init=c['init'], T_success=t['success'], candidate_success=s['success'],
            initial_physics_exact=physics, hardware_exact=hardware, identity_exact=identity,
            initial_images_exact=t['first_image_sha256'] == s['first_image_sha256'], initial_image_differences=pixel,
            T_commands=t['action_steps'], candidate_commands=s['action_steps'],
            command_ratio=s['action_steps']/t['action_steps'], chunk_ratio=s['chunks']/t['chunks'],
            concurrent_wall_ratio=s['wall_seconds']/t['wall_seconds']))
    common = [p for p in pairs if p['T_success'] and p['candidate_success']]
    paired[arm] = dict(N=len(pairs), both_success=len(common),
        T_only=sum(p['T_success'] and not p['candidate_success'] for p in pairs),
        candidate_only=sum(p['candidate_success'] and not p['T_success'] for p in pairs),
        both_failure=sum(not p['T_success'] and not p['candidate_success'] for p in pairs),
        initial_physics_exact=sum(p['initial_physics_exact'] for p in pairs),
        initial_images_exact=sum(p['initial_images_exact'] for p in pairs),
        common_success_command_ratio=describe([p['command_ratio'] for p in common]),
        common_success_chunk_ratio=describe([p['chunk_ratio'] for p in common]),
        common_success_concurrent_wall_ratio=describe([p['concurrent_wall_ratio'] for p in common]), pairs=pairs)

conditions = dict(all_arms_complete=all(q['N_valid'] == len(panel['FULL']) for q in quality.values()),
    raw_commits_and_commands=not errors and len(raw_audits) == 3*len(panel['FULL']))
for arm in ('S','P'):
    pp=paired[arm]['pairs']
    conditions[arm+'_paired_identity'] = len(pp)==len(panel['FULL']) and all(x['initial_physics_exact'] and x['hardware_exact'] and x['identity_exact'] for x in pp)
# Every actual speculative server must supply its independent cold numerical audit.
audit_paths = list((ROOT/'results').glob('*/B7-AUDIT.json'))
b7 = [json.loads(p.read_text()) for p in audit_paths]
conditions['trained_B7_audits'] = len(b7) >= 3 and all(rows and all(row['pass_all'] for row in rows) for rows in b7)
result = dict(schema='fastwam-full-triarm-quality/v1', quality=quality, per_task=per_task,
    paired=paired, acceptance={a: acceptance(c) for a, c in aggregate.items()}, raw_audits=raw_audits,
    audit_errors=errors, data_audit_conditions=conditions, full_data_audit_complete=all(conditions.values()),
    exposure='Full official50-init benchmark includes FIT0..9, CAL10, DEV47/48; remaining indices not exposed to THIS recipe selection, historical exposure remains. Not independent TEST.',
    timing='Concurrent A100 wall/policy times are ancillary, not formal speedups.',
    initial_pixels='Nonexact pixels are reported; only physical-init statistical pairing is claimed in those cases.',
    four_permissions=registration['permissions'])
(ROOT/'results/QUALITY-FULL-TRIARM.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ('quality', 'data_audit_conditions', 'full_data_audit_complete')}, indent=2))

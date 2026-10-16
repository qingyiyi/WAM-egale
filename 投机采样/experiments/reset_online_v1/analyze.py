"""Complete original/bank pairs must agree, including valid model state."""
import json
from pathlib import Path

import numpy as np

from protocol import load, read, write


def only_record(path):
    manifest = path / 'episodes.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 1 or rows[0].get('status') != 'completed':
        raise ValueError('one completed full-horizon record required')
    return rows[0]


def main(job):
    job = Path(job).resolve()
    p = load(job)
    rows = []
    for arm in p['arms']:
        for task in sorted({r['task'] for r in p['cases']}):
            pair = [r for r in p['cases'] if r['arm'] == arm and r['task'] == task]
            if len(pair) != 2 or {r['text_bank'] for r in pair} != {False, True}:
                raise ValueError('one original/bank pair required')
            a, b = sorted(pair, key=lambda r: r['text_bank'])
            da, db = (job / 'episodes' / r['key'] for r in (a, b))
            ca, cb = da / 'client', db / 'client'
            ma, mb = read(ca / 'repro_metadata.json'), read(cb / 'repro_metadata.json')
            ra, rb = only_record(ca), only_record(cb)
            reset_a, reset_b = read(da / 'RESET.json'), read(db / 'RESET.json')
            counter_a, counter_b = read(da / 'COUNTER.json'), read(db / 'COUNTER.json')
            checks = dict(
                commands=np.array_equal(np.load(ca / 'commands.npy'), np.load(cb / 'commands.npy')),
                reset_state=reset_a['state'] == reset_b['state'],
                final_server_state=read(da / 'FINAL-SERVER-STATE.json') == read(db / 'FINAL-SERVER-STATE.json'),
                target_forwards=counter_a['target_forwards'] == counter_b['target_forwards'],
                success=ra['success'] == rb['success'], env_steps=ra['env_steps'] == rb['env_steps'],
                infer_chunks=ra['infer_chunks'] == rb['infer_chunks'])
            for field in ('model_xml_sha256', 'init_state_sha256', 'initial_state_hashes',
                          'final_state_hashes', 'initial_obs_hashes', 'initial_rng', 'final_rng'):
                checks[field] = ma[field] == mb[field]
            aa = [{k: v for k, v in r.items() if k != 'context'} for r in counter_a['acceptance']]
            ab = [{k: v for k, v in r.items() if k != 'context'} for r in counter_b['acceptance']]
            checks['all_acceptance_trees'] = aa == ab
            rows.append(dict(arm=arm, task=task, init=a['init'], checks=checks,
                passed=all(checks.values()), success=ra['success'], env_steps=ra['env_steps'],
                infer_chunks=ra['infer_chunks'], target_forwards=counter_a['target_forwards'],
                original_reset_sec=reset_a['reset_wall_sec'], bank_reset_sec=reset_b['reset_wall_sec'],
                original_episode_sec=ra['elapsed_sec'], bank_episode_sec=rb['elapsed_sec']))
    result = dict(status='PASS_COMPLETE_RESET_BANK_EQUIVALENCE' if all(r['passed'] for r in rows)
                  else 'FAIL_COMPLETE_RESET_BANK_EQUIVALENCE', pairs=len(rows),
                  full_episodes=len(p['cases']), rows=rows,
                  performance_claim=False, shared_GPU=True,
                  limitation='Finite two-task state/trajectory equivalence; shared-GPU diagnostic timings are not formal paper speed or general quality proof')
    write(job / 'RESULT.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'].startswith('FAIL'):
        raise ValueError('complete reset equivalence failed; keep bank default-off')


if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]))

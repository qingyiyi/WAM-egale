"""CPU-only verification of the portable FastWAM evidence snapshot."""
from pathlib import Path
import csv
import hashlib
import json
import math

B = Path(__file__).resolve().parent
ROOT = B.parent.parent
M = json.loads((B / 'MANIFEST.json').read_text())
for record in M['sources']:
    p = ROOT / record['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == record['public_sha256'], str(p)

def rows(name):
    with (B / 'data' / name).open(newline='') as f:
        return list(csv.DictReader(f))

episodes = rows('episodes-6000.csv')
assert len(episodes) == 6000
assert len({(x['suite'], x['task'], x['init'], x['arm']) for x in episodes}) == 6000
expected = {'T': 1950, 'S': 1945, 'P': 1944}
for arm, count in expected.items():
    selected = [x for x in episodes if x['arm'] == arm]
    assert len(selected) == 2000 and sum(int(x['success']) for x in selected) == count
    assert all(int(x['complete']) == int(x['audit_passed']) == 1 for x in selected)
for t in rows('tasks-120.csv'):
    selected = [x for x in episodes if all(x[k] == t[k] for k in ['suite', 'task', 'arm'])]
    assert len(selected) == int(t['N']) == 50
    assert sum(int(x['success']) for x in selected) == int(t['success'])
    assert math.isclose(sum(int(x['commands']) for x in selected)/50, float(t['commands_mean']), abs_tol=1e-10)
assert len(rows('paired-quality-4000.csv')) == 4000
speed = rows('cached-speed-pairs-1125.csv')
assert len(speed) == 1125 and all(int(x['count_committed_updates']) == 10 for x in speed)
F = ROOT / '.pipeline/experiments/fastwam-full-triarm-20261027'
result = json.loads((F / 'TEXT-CACHE-v1/REVIEW-4suites-v1/RESULT.json').read_text())
for suite, v in result['suites'].items():
    selected = [x for x in speed if x['suite'] == suite]
    for field, measurement in [('chunk', 'full_chunk'), ('action', 'action_denoising')]:
        ratio = sum(float(x[f'T_{field}_seconds']) for x in selected)/sum(float(x[f'S_{field}_seconds']) for x in selected)
        assert math.isclose(ratio, v['measurements'][measurement]['speedup'], abs_tol=1e-12)
    for line in (F / 'TEXT-CACHE-v1' / suite / 'results/speed-triarm-attempt01/FULL-CHUNKS.jsonl').read_text().splitlines():
        x = json.loads(line)
        cursor = 0
        for r in x['rounds']:
            assert r['step'] == cursor
            assert r['g'] == (r['depth']+1 if r['kind'] == 'tree' else 1)
            cursor += r['g']
        assert cursor == x['counts']['committed_updates'] == 10
scores = rows('calibration-scores.csv')
assert len(scores) == 4960
for model in json.loads((B / 'data/draft-selection.json').read_text()):
    values = sorted(float(x['score']) for x in scores if x['suite'] == model['suite'])
    index = (len(values)-1)*.8
    lower = int(index)
    q = values[lower] + (index-lower)*(values[min(lower+1, len(values)-1)]-values[lower])
    assert abs(q-model['tau']) < 3e-9  # FP32 torch.quantile vs scalar double interpolation.
print(json.dumps(dict(status='PASS', source_files=len(M['sources']), episodes=6000,
                      speed_pairs=1125, calibration_scores=4960, success=expected,
                      scope='portable scalar audit; no GPU, no raw tensor or closed-loop replay')))

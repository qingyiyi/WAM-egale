"""CPU-only counterfactual over actual archived costs, never rescore old tensors."""
import argparse
from collections import Counter
from pathlib import Path
import json
import os
from types import SimpleNamespace

from paths import ROOT, load
from experiment import digest, write
from selection import choose, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only analysis required')
    import torch
    api = load('archived_costs_api', ROOT / '投机采样/scripts/p0_batched_verify.py')
    directory = ROOT / '投机采样/traces/p0-libero-libero_10-verifier-k2b2-global-1init-20260930-r2'
    # Discover the exact existing trace path, since historical collectors vary.
    if not directory.is_dir():
        matches = [p for p in (ROOT / '投机采样/results').rglob(directory.name) if p.is_dir()]
        if len(matches) != 1:
            raise ValueError('unique preserved historical global trace required')
        directory = matches[0]
    counts, by_stage, inventory = Counter(), {}, []
    for path in sorted(directory.rglob('*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        trees = [s for s in payload.get('steps', []) if s['kind'] == 'k2_b2_verifier_tree']
        for step in trees:
            score = SimpleNamespace(first_costs=step['first_costs'], second_costs=step['second_costs'])
            t = float(step['threshold'])
            g = api.select_k2_b2_route(score, selector='global')
            d = choose(api, score, t)
            gd, dd = depth(g.first_cost, g.second_cost, t), depth(d.first_cost, d.second_cost, t)
            assert gd == step['accepted_depth'] and dd >= gd
            key = 'step_' + str(int(step['step_index']) // 10 * 10)
            c = by_stage.setdefault(key, Counter())
            for counter in (c, counts):
                counter['trees'] += 1
                counter['global_depth_'+str(gd)] += 1
                counter['depth_depth_'+str(dd)] += 1
                counter['recoverable_trees'] += dd > gd
                counter['recoverable_updates'] += dd-gd
        inventory.append(dict(path=str(path), sha256=digest(path), trees=len(trees)))
    assert not torch.cuda.is_initialized()
    write(a.out, dict(status='PASS_ARCHIVED_SAME_TREE_CPU_COUNTERFACTUAL', counts=dict(counts),
        by_denoising_stage={k:dict(v) for k,v in by_stage.items()}, files=inventory,
        CUDA_initialized=False, exact_stored_costs=True,
        limitation='Historical pre-A_off, bonus-OFF trace distribution; not a current closed-loop acceptance or quality result.'), exclusive=True)
    print(json.dumps(dict(counts)))


if __name__ == '__main__':
    main()

"""Critical commit/boundary checks with the real scheduler and a synthetic target.

This does not validate the loaded RoboTwin backbone or any learned acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from draft import DirectionPredictor, load
from real_speculative import CanonicalCollector, RealSpeculative, choose

p = argparse.ArgumentParser()
p.add_argument('--scheduler', type=Path, required=True)
p.add_argument('--old-checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(2)
torch.manual_seed(42)
spec = importlib.util.spec_from_file_location('official_scheduler', a.scheduler)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
scheduler = module.WanContinuousFlowMatchScheduler(shift=5.)
ts, ds = scheduler.build_inference_schedule(10, device='cpu', dtype=torch.bfloat16, shift_override=5.)
model = SimpleNamespace(infer_action_scheduler=scheduler, device='cpu')
x = torch.randn(1, 32, 14, dtype=torch.bfloat16)
ctx, mask = torch.randn(1, 129, 8), torch.ones(1, 129, dtype=torch.bool)
ks, vs = [torch.randn(1, 20, 16)], [torch.randn(1, 20, 16)]
original_conditions = [t.clone() for t in [ctx, mask, *ks, *vs]]


def target(**kw):
    states = kw['latents_action']
    b = len(states)
    assert states.shape == (b, 32, 14) and b in (1, 7)
    assert kw['context'].shape == (b, 129, 8)
    assert kw['context_mask'].shape == (b, 129)
    assert len(kw['timestep_action']) == b
    return torch.full_like(states, .25)


class ScriptedDraft:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, state, previous, timestep, delta):
        step = int((ts == timestep).nonzero().item())
        out = torch.full((2, 32, 14), 1.25, dtype=state.dtype)
        # The second head is correct, which exercises nonzero parent/leaf indices.
        if self.mode == 'accept_two' or (self.mode == 'accept_one' and step % 2 == 1):
            out[1].fill_(.25)
        return out


canonical = x.clone()
for d in ds:
    canonical = scheduler.step(torch.full_like(x, .25), d, canonical)
records = []
expected = dict(reject_all=dict(tree_count=7, depth_0=7, fallback_updates=7, boundary_target_updates=2),
                accept_one=dict(tree_count=4, depth_1=4, draft_updates=4, fallback_updates=4, boundary_target_updates=1),
                accept_two=dict(tree_count=3, depth_2=3, draft_updates=6, bonus_updates=3, boundary_target_updates=0))
for mode, counts in expected.items():
    method = RealSpeculative(model, ScriptedDraft(mode), tau=0.)
    method.validate = mode == 'accept_two'
    actual = method.loop(x.clone(), ts, ds, ctx, mask, ks, vs, None, target, False)
    assert torch.equal(actual, canonical), mode
    assert method.counts['bootstrap_updates'] == 1 and method.counts['committed_updates'] == 10
    assert sum(r['g'] for r in method.rounds) == 10
    assert all(method.counts[k] == v for k, v in counts.items()), (mode, method.counts)
    assert all(r['step']+r['g'] <= 10 for r in method.rounds)
    assert method.counts['candidate_expansions'] == 6*counts['tree_count']
    assert method.counts['target_state_rows'] == method.counts['target_B1_calls']+7*method.counts['target_B7_calls']
    assert all(torch.equal(s, t) for s, t in zip(original_conditions, [ctx, mask, *ks, *vs]))
    if mode == 'accept_two':
        assert all(r['parent'] == r['child'] == 1 for r in method.rounds if r['kind'] == 'tree')
        assert all(r['pass_all'] for r in method.audit_records)
        assert method.acceptance_audit['accepted_edge_flips'] == 0
        assert not method.acceptance_audit['chosen_path_changed']
    records.append(dict(mode=mode, counts=dict(method.counts), canonical_exact=True,
                        commits=[dict(step=r['step'], g=r['g'], kind=r['kind']) for r in method.rounds]))

assert choose([1., 1., 1., 1., 1., 1.], 1.) == (2, 0, 0), 'threshold includes equality'
for scores in [[float('nan')]+[0.]*5, [float('inf')]+[0.]*5, [-1.]+[0.]*5]:
    try:
        choose(scores, 1.)
    except AssertionError:
        pass
    else:
        raise AssertionError('invalid score accepted')

collector = CanonicalCollector(model)
collected = collector.loop(x.clone(), ts, ds, ctx, mask, ks, vs, None, target, False)
assert torch.equal(collected, canonical)
assert collector.columns['target'].shape == (10, 32, 14)
assert torch.equal(collector.columns['target'], torch.full((10, 32, 14), .25))
assert torch.equal(collector.columns['has_previous'], torch.arange(10) > 0)

draft = DirectionPredictor().eval().requires_grad_(False)
with torch.no_grad():
    proposals = draft(x, torch.ones_like(x), ts[1], ds[1])
assert proposals.shape == (2, 32, 14) and proposals.dtype == torch.bfloat16
assert torch.isfinite(proposals).all()
assert sum(v.numel() for v in draft.parameters()) == 55900
try:
    load(a.old_checkpoint, device='cpu')
except AssertionError:
    old_rejected = True
else:
    raise AssertionError('LIBERO checkpoint accepted as RoboTwin')

result = dict(status='PASS_CPU_ONLY', scheduler_sha256=hashlib.sha256(a.scheduler.read_bytes()).hexdigest(),
              timestep=ts.float().tolist(), delta=ds.float().tolist(), modes=records,
              predictor_parameters=55900, predictor_untrained=True, old_LIBERO_checkpoint_rejected=old_rejected,
              teacher_direction_label_shape=[10, 32, 14], candidate_rows=7, horizon=32,
              synthetic_B7_audit_and_acceptance_sensitivity_checked=True,
              GPU_calls=0, scope='Synthetic target with actual BF16 scheduler; loaded-model parity, B7 numerics and closed-loop quality remain unverified.')
with a.output.open('x') as f:
    f.write(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))

"""K2/B2 seven-state verification, actual threshold acceptance, legal ten-step commit.

No historical depth forcing. Selector is original depth_then_cost (not PF2).
EmpiricalReplay is imported solely for its B1/B7 adapter and numerical audit.
"""
import math
import time
from collections import Counter
import torch
from simulation_reference import EmpiricalReplay


def choose(scores, tau):
    assert math.isfinite(tau) and tau >= 0 and len(scores) == 6
    routes = []
    for p in range(2):
        for c in range(2):
            a, b = scores[p], scores[2 + p * 2 + c]
            assert math.isfinite(a + b) and min(a, b) >= 0
            depth = 0 if a > tau else 1 if b > tau else 2
            routes.append((depth, a + b, p, c))
    d, _, p, c = min(routes, key=lambda r: (-r[0], r[1], r[2], r[3]))
    return d, p, c


class RealSpeculative(EmpiricalReplay):
    def __init__(self, model, draft, tau):
        super().__init__(model, draft)
        self.tau = float(tau)
        self.profile = False

    def loop(self, state, timesteps, deltas, ctx, mask, cache_k, cache_v, attn, fn, compiled):
        assert len(timesteps) == len(deltas) == 10 and state.shape == (1, 32, 7)
        self.counts, self.rounds = Counter(), []
        step = self.model.infer_action_scheduler.step
        i, previous = 0, None
        begin = time.perf_counter()
        while i < 10:
            if self.profile: torch.cuda.synchronize()
            start = time.perf_counter()
            if previous is None or 10 - i < 3:
                v = self.target(state, timesteps[i:i+1], ctx, mask, cache_k, cache_v, attn, fn, compiled)
                state = step(v, deltas[i], state)
                previous = v
                self.counts['bootstrap_updates' if i == 0 else 'boundary_target_updates'] += 1
                self.counts['committed_updates'] += 1
                if self.profile: torch.cuda.synchronize()
                self.rounds.append(dict(step=i, kind='bootstrap' if i == 0 else 'tail', g=1,
                                        seconds=time.perf_counter()-start, synchronized=self.profile))
                i += 1
                continue
            dirs1 = self.draft(state, previous, timesteps[i], deltas[i])
            parents = step(dirs1, deltas[i], state.expand(2, -1, -1))
            dirs2 = torch.stack([self.draft(parents[j:j+1], dirs1[j:j+1], timesteps[i+1], deltas[i+1])
                                 for j in range(2)])
            leaves = step(dirs2.reshape(4, 32, 7), deltas[i+1],
                          parents[:, None].expand(-1, 2, -1, -1).reshape(4, 32, 7))
            self.counts['draft_calls'] += 3
            if self.profile: torch.cuda.synchronize()
            after_draft = time.perf_counter()
            packed = torch.cat((state, parents, leaves), 0)
            times = torch.cat((timesteps[i:i+1], timesteps[i+1:i+2].expand(2), timesteps[i+2:i+3].expand(4)))
            if self.profile: torch.cuda.synchronize()
            after_pack = time.perf_counter()
            values = self.target(packed, times, ctx, mask, cache_k, cache_v, attn, fn, compiled)
            if self.profile: torch.cuda.synchronize()
            after_verify = time.perf_counter()
            first = (dirs1.float()-values[0:1].float()).flatten(1).norm(dim=1)*deltas[i].float().abs()
            second = (dirs2.float()-values[1:3, None].float()).flatten(2).norm(dim=2)*deltas[i+1].float().abs()
            scores = torch.cat((first, second.flatten())).tolist()
            depth, parent, child = choose(scores, self.tau)
            if self.validate and not self.audit_records:
                self.audit_inputs = dict(states=packed.detach().cpu(), timesteps=times.detach().cpu(),
                    deltas=torch.cat((deltas[i:i+1], deltas[i+1:i+2].expand(2),
                                      deltas[i+2:i+3].expand(4))).detach().cpu(),
                    batched_target=values.detach().cpu(), condition_source='raw observation saved by caller',
                    step=i, state_source='actual trained-draft candidate states')
                self.audit(packed, times, ctx, mask, cache_k, cache_v, attn, fn, compiled, values,
                           torch.cat((deltas[i:i+1], deltas[i+1:i+2].expand(2), deltas[i+2:i+3].expand(4))))
            if depth == 0:
                state = step(values[0:1], deltas[i], state)
                previous = values[0:1]
                self.counts['rejected_trees'] += 1
                self.counts['fallback_updates'] += 1
            elif depth == 1:
                state = step(values[1+parent:2+parent], deltas[i+1], parents[parent:parent+1])
                previous = values[1+parent:2+parent]
                self.counts['draft_updates'] += 1
                self.counts['fallback_updates'] += 1
            else:
                leaf = 3 + 2*parent + child
                state = step(values[leaf:leaf+1], deltas[i+2], leaves[leaf-3:leaf-2])
                previous = values[leaf:leaf+1]
                self.counts['draft_updates'] += 2
                self.counts['bonus_updates'] += 1
            self.counts['tree_count'] += 1
            self.counts[f'depth_{depth}'] += 1
            self.counts['candidate_expansions'] += 6
            self.counts['committed_updates'] += depth + 1
            if self.profile: torch.cuda.synchronize()
            end = time.perf_counter()
            self.rounds.append(dict(step=i, kind='tree', depth=depth, parent=parent, child=child, g=depth+1,
                scores=scores, tau=self.tau, seconds=end-start, draft_seconds=after_draft-start,
                pack_seconds=after_pack-after_draft, verify_seconds=after_verify-after_pack,
                accept_commit_seconds=end-after_verify, synchronized=self.profile))
            i += depth + 1
        assert i == self.counts['committed_updates'] == 10
        self.host_envelope_seconds = time.perf_counter() - begin
        return state


class CanonicalCollector:
    def __init__(self, model):
        self.model = model

    def loop(self, state, timesteps, deltas, ctx, mask, cache_k, cache_v, attn, fn, compiled):
        rows = []
        previous = torch.zeros_like(state)
        for i, (t, d) in enumerate(zip(timesteps, deltas)):
            if compiled: torch.compiler.cudagraph_mark_step_begin()
            v = fn(latents_action=state, timestep_action=t.unsqueeze(0).to(dtype=state.dtype, device=self.model.device),
                   context=ctx, context_mask=mask, video_cache_k=cache_k, video_cache_v=cache_v,
                   action_attention_mask=attn).clone()
            rows.append((state.clone(), previous.clone(), v.clone()))
            state = self.model.infer_action_scheduler.step(v, d, state)
            previous = v
        self.columns = dict(state=torch.cat([r[0] for r in rows]).float().cpu(),
                            previous=torch.cat([r[1] for r in rows]).float().cpu(),
                            target=torch.cat([r[2] for r in rows]).float().cpu(),
                            timestep=timesteps.float().cpu(), delta=deltas.float().cpu(),
                            has_previous=torch.arange(10) > 0)
        return state

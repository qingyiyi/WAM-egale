"""B1/B7 adapter and diagnostics only; no forced acceptance path."""
from collections import Counter
import torch
class BatchVerifierAdapter:

    def __init__(self, model, draft):
        self.model = model
        self.draft = draft
        self.trace = []
        self.validate = False
        self.audit_records = []

    def target(self, states, timesteps, ctx, mask, cache_k, cache_v, attn, fn, compiled):
        b = states.shape[0]
        assert b in (1, 7) and states.shape[1:] == (32, 14)
        if compiled:
            torch.compiler.cudagraph_mark_step_begin()
        self.counts['target_calls'] += 1
        self.counts[f'target_B{b}_calls'] += 1
        self.counts['target_state_rows'] += b
        return fn(latents_action=states, timestep_action=timesteps, context=ctx.expand(b, -1, -1), context_mask=mask.expand(b, -1), video_cache_k=[x.expand(b, -1, -1) for x in cache_k], video_cache_v=[x.expand(b, -1, -1) for x in cache_v], action_attention_mask=attn).clone()

    def audit(self, states, times, ctx, mask, ks, vs, attn, fn, compiled, batch, deltas):
        old_counts = self.counts.copy()
        snapshots = [x.clone() for x in ks + vs] + [ctx.clone(), mask.clone()]
        serial = torch.cat([self.target(states[j:j + 1], times[j:j + 1], ctx, mask, ks, vs, attn, fn, compiled) for j in range(7)], 0)
        diff = batch.float() - serial.float()
        rel = float(diff.norm() / serial.float().norm().clamp_min(1e-06))
        maxdiff = float(diff.abs().max())
        limit = 0.05 + 0.02 * float(serial.float().abs().max())
        row = dict(check='B7_vs_serial_B1', relative_L2=rel, max_abs=maxdiff, relative_L2_limit=0.01, max_abs_limit=limit, pass_all=rel <= 0.01 and maxdiff <= limit)
        self.audit_records.append(row)
        assert row['pass_all'], row
        for j in range(7):
            ref = serial[j].float()
            delta = batch[j].float() - ref
            r = float(delta.norm() / ref.norm().clamp_min(1e-06))
            a = float(delta.abs().max())
            limit = 0.05 + 0.02 * float(ref.abs().max())
            batched_update = self.model.infer_action_scheduler.step(batch[j], deltas[j], states[j])
            serial_update = self.model.infer_action_scheduler.step(serial[j], deltas[j], states[j])
            update_diff = float((batched_update.float() - serial_update.float()).abs().max())
            update_limit = float(deltas[j].float().abs()) * a + 2 * torch.finfo(torch.bfloat16).eps * max(1.0, float(serial_update.float().abs().max()))
            per_row = dict(check='candidate_direction_and_scheduler_update', candidate=j, relative_L2=r, max_abs=a, relative_L2_limit=0.01, max_abs_limit=limit, update_max_abs=update_diff, update_limit=update_limit, pass_all=r <= 0.01 and a <= limit and (update_diff <= update_limit))
            self.audit_records.append(per_row)
            assert per_row['pass_all'], per_row
        mutated = states.clone()
        mutated[6] += torch.tensor(0.125, device=states.device, dtype=states.dtype)
        other = self.target(mutated, times, ctx, mask, ks, vs, attn, fn, compiled)
        unchanged = torch.equal(batch[:6], other[:6])
        condition_unchanged = all((torch.equal(a, b) for a, b in zip(snapshots, ks + vs + [ctx, mask])))
        row = dict(check='candidate_isolation_and_fixed_condition', other_six_exact=unchanged, condition_unchanged=condition_unchanged, pass_all=unchanged and condition_unchanged)
        self.audit_records.append(row)
        assert row['pass_all'], row
        self.counts = old_counts
        return serial

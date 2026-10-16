"""Return original native attention while checking fixed G1/G2 on real fixtures.

This is a numerical diagnostic, with no timing or production integration.
"""
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = OWN.parents[2]
SHAPE = EXP / 'wam_shape_audit_v1'
sys.path.insert(0, str(SHAPE))
import full_matrix as entry
from full_matrix import write, load


def run(server, source, out, prior):
    import torch
    import adapter
    import state_check

    sys.path.insert(0, str(EXP / 'shared_prefix_torch_v2'))
    import _shared_torch_v1 as native
    sys.path.insert(0, str(EXP / 'wam_candidate_tile_v1'))
    import _wam_candidate_tile as candidate
    snap = load('candidate_tile_fixture_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    plan = json.loads((out / 'PLAN.json').read_text())
    started = json.loads((out / 'RUN-STARTED.json').read_text())
    fixture_root = Path(plan['fixture_root'])
    result = dict(status='RUNNING', rows=[], queries=[], target_forwards=0,
        candidate_comparisons=0, native_oracle_calls=0, attention_calls=0,
        full_policy_chunks=0, episodes=0, production_integrated=False,
        integration_permission=False, performance_claim=False,
        scope='six actual frozen B14 queries; original native outputs always returned',
        modules=dict(native=str(Path(native.__file__).resolve()),
                     candidate=str(Path(candidate.__file__).resolve())))
    mismatch = None

    def check():
        if time.monotonic() >= started['active_deadline_monotonic']:
            raise TimeoutError('540s activity cap with sixty-second cleanup reserve')
        if (out / 'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED')
        if result['target_forwards'] > plan['target_forward_cap']:
            raise RuntimeError('target forward cap')
        if result['candidate_comparisons'] > plan['candidate_comparison_cap']:
            raise RuntimeError('candidate comparison cap')
        if result['attention_calls'] > plan['attention_call_cap']:
            raise RuntimeError('attention call cap')

    def call_attention(fn, *, oracle):
        check()
        if result['attention_calls'] + 1 > plan['attention_call_cap']:
            raise RuntimeError('attention call reservation refused')
        result['attention_calls'] += 1
        if oracle:
            result['native_oracle_calls'] += 1
        else:
            if result['candidate_comparisons'] + 1 > plan['candidate_comparison_cap']:
                raise RuntimeError('candidate comparison reservation refused')
            result['candidate_comparisons'] += 1
        return fn()

    def dev(value):
        if isinstance(value, torch.Tensor): return value.to(server.device)
        if isinstance(value, dict): return {k: dev(v) for k, v in value.items()}
        if isinstance(value, tuple): return tuple(dev(v) for v in value)
        if isinstance(value, list): return [dev(v) for v in value]
        return value

    def tensors(value):
        if isinstance(value, torch.Tensor): return [value]
        if isinstance(value, dict): return [t for v in value.values() for t in tensors(v)]
        if isinstance(value, (tuple, list)): return [t for v in value for t in tensors(v)]
        return []

    def compare(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError('output shape/dtype differs')
        aa = a.detach().cpu().contiguous()
        bb = b.detach().cpu().contiguous()
        delta = aa.float() - bb.float()
        return dict(bitwise=torch.equal(aa.view(torch.uint8), bb.view(torch.uint8)),
            finite=bool(torch.isfinite(aa).all()) and bool(torch.isfinite(bb).all()),
            different=int(torch.count_nonzero(delta)), elements=aa.numel(),
            max_abs=float(delta.abs().max()),
            relative_RMS=float(delta.square().mean().sqrt() /
                               (bb.float().square().mean().sqrt() + 1e-30)))

    check()
    torch.set_num_threads(7)
    # This diagnostic uses direct model forwards; no draft lazy loader is used.
    server.infer(dict(reset=True, prompt=None))
    model = server.transformer
    assert len(model.blocks) == 30
    result['CPU_threads'] = torch.get_num_threads()
    result['CPU_affinity'] = sorted(os.sched_getaffinity(0))
    try:
        for label in ('short', 'middle', 'long'):
            check()
            data = torch.load(fixture_root / label / 'real-verifier-fixture.pt',
                              map_location='cpu', weights_only=False)
            assert len(data['queries']) == 2 and len(data['history']) == 30
            for block, saved in zip(model.blocks, data['history']):
                snap.restore_cache(block.attn1.attn_caches[server.cache_name], saved)
            refs = state_check.references(model, server.cache_name)
            initial_rng = snap.fingerprint(snap.rng(True))
            slots = []
            for block, saved in zip(model.blocks, data['history']):
                valid = saved['slots']
                assert valid.dtype == torch.int64 and valid.ndim == 1
                assert valid.numel() > 0 and valid.numel() + 16 == data['total_KV']
                assert bool((valid >= 0).all())
                assert bool((valid < saved['pool_shape'][1]).all())
                assert bool((valid[1:] > valid[:-1]).all())
                assert saved['pool_shape'][0] == 2
                assert all(bool(torch.isfinite(saved[k]).all()) for k in ('k_valid', 'v_valid'))
                slots.append(valid.to(server.device))
            originals = [block.attn1.forward for block in model.blocks]
            try:
                for query in data['queries']:
                    check()
                    inputs = dev(query['input'])
                    kwargs = dev(query['kwargs'])
                    kwargs.pop('input_dict', None)
                    rows = kwargs['candidate_cache_rows']
                    assert inputs['noisy_latents'].shape[0] == 14 and kwargs['update_cache'] == 0
                    assert torch.equal(rows, torch.tensor([0] * 7 + [1] * 7, device=server.device))
                    query_rows_start = len(result['rows'])
                    native_start = result['native_oracle_calls']
                    # Raw C++ writes bypass tensor version counters; valid KV is
                    # compared by state_check and ephemeral Q/K/V by exact clone.
                    tracked = list(model.parameters()) + tensors((inputs, kwargs, slots))
                    tracked += [v for cache in state_check.pools(model, server.cache_name)
                                for v in cache.values() if isinstance(v, torch.Tensor)]
                    versions = [(t, t._version) for t in tracked]
                    seen_layers = []

                    def make_history(layer, cache, valid):
                        def history(attention, q, k, v, c, name, r, update):
                            nonlocal mismatch
                            assert c is cache and name == server.cache_name and r is rows and update == 0
                            assert layer not in seen_layers
                            seen_layers.append(layer)
                            assert q.shape == (14, 16, 24, 128) and k.shape == q.shape and v.shape == q.shape
                            gold = call_attention(lambda: native.shared(q, cache['k'], cache['v'],
                                k, v, valid, rows), oracle=True)
                            if mismatch is not None:
                                return gold
                            saved_qkv = [t.clone() for t in (q, k, v)]
                            assert all(bool(torch.isfinite(t).all()) for t in (q, k, v, gold))
                            for group in (2, 1) if layer in (0, 8, 29) else (2,):
                                actual = call_attention(lambda: candidate.grouped(q, cache['k'], cache['v'],
                                    k, v, valid, group), oracle=False)
                                parity = compare(actual, gold)
                                row = dict(length=label, total_KV=data['total_KV'], query=query['index'],
                                    position='first' if query['index'] == 0 else 'last', layer=layer,
                                    n_v=7, group=group, output=parity, original_native_returned=True)
                                result['rows'].append(row)
                                if not (parity['bitwise'] and parity['finite']):
                                    mismatch = row
                                    break
                            assert all(torch.equal(actual.contiguous().view(torch.uint8),
                                                   before.contiguous().view(torch.uint8))
                                       for actual, before in zip((q, k, v), saved_qkv)), 'kernel changed actual Q/K/V bytes'
                            return gold
                        return history

                    try:
                        for i, block in enumerate(model.blocks):
                            callback = make_history(i, block.attn1.attn_caches[server.cache_name], slots[i])
                            block.attn1.forward = types.MethodType(adapter._private_forward(originals[i], callback), block.attn1)
                        check()
                        assert result['target_forwards'] + 1 <= plan['target_forward_cap']
                        result['target_forwards'] += 1
                        actual_output = model(inputs, **kwargs)
                        full_parity = compare(actual_output, query['output'].to(server.device))
                        assert full_parity['bitwise'] and full_parity['finite'], 'native full query no longer reproduces actual frozen output'
                        assert seen_layers == list(range(30))
                        assert result['native_oracle_calls'] - native_start == 30
                        assert state_check.equal(model, refs, server.cache_name), 'all-valid KV or cache metadata mutated'
                        assert all(t._version == version for t, version in versions), 'parameter/cache/input version mutated'
                        assert snap.fingerprint(snap.rng(True)) == initial_rng, 'RNG changed'
                        result['queries'].append(dict(length=label, total_KV=data['total_KV'],
                            query=query['index'], complete_native_layers=seen_layers,
                            full_frozen_output=full_parity,
                            candidate_comparisons=len(result['rows']) - query_rows_start,
                            all_valid_KV_RNG_and_tensor_versions_unchanged=True,
                            completed_original_native_query_after_mismatch=mismatch is not None))
                        print(json.dumps(dict(length=label, query=query['index'],
                            comparisons=len(result['rows']) - query_rows_start,
                            exact=full_parity['bitwise'], mismatch=mismatch)), flush=True)
                    finally:
                        for block, original in zip(model.blocks, originals):
                            block.attn1.forward = original
                    del actual_output, inputs, kwargs, tracked, versions
                    gc.collect()
                    torch.cuda.empty_cache()
                    if mismatch is not None:
                        break
            finally:
                for block, original in zip(model.blocks, originals):
                    block.attn1.forward = original
            del refs, data
            gc.collect()
            torch.cuda.empty_cache()
            if mismatch is not None:
                break
        if mismatch is None:
            assert len(result['queries']) == 6 and result['target_forwards'] == 6
            assert result['candidate_comparisons'] == 198 and len(result['rows']) == 198
            assert result['native_oracle_calls'] == 180 and result['attention_calls'] == 378
            assert sum(row['group'] == 2 for row in result['rows']) == 180
            assert sum(row['group'] == 1 for row in result['rows']) == 18
        result.update(status='COMPLETE_SIX_FIXTURE_FINITE_NUMERIC_GATE_PENDING_CLEANUP' if mismatch is None
                      else 'STOPPED_NUMERICAL_MISMATCH_NATIVE_QUERY_VERIFIED_PENDING_CLEANUP',
            mismatch=mismatch, finite_fixture_gate_passed=mismatch is None,
            all_completed_queries_frozen_exact_and_valid_state_unchanged=True,
            numerical_negative_is_not_integration_permission=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        write(out / 'FIXTURE-RESULT.json', result)


if __name__ == '__main__':
    entry.run = run
    entry.main()

"""Two-pass numerical gate for one fixed, geometry-guarded research kernel."""
import gc
import hashlib
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


def selected_group(prefix):
    if type(prefix) is not int or prefix <= 0:
        raise ValueError('actual positive validated prefix count required')
    return 2 if (prefix + 32 + 63) // 64 == (prefix + 16 + 63) // 64 else 1


def run(server, source, out, prior):
    import torch
    import adapter
    import state_check

    sys.path.insert(0, str(EXP / 'shared_prefix_torch_v2'))
    import _shared_torch_v1 as native
    sys.path.insert(0, str(EXP / 'wam_candidate_tile_v1'))
    import _wam_candidate_tile as candidate
    snap = load('guarded_v2_fixture_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    plan = json.loads((out / 'PLAN.json').read_text())
    started = json.loads((out / 'RUN-STARTED.json').read_text())
    for module, key in ((native, 'native_binary'), (candidate, 'candidate_binary')):
        actual = Path(module.__file__).resolve()
        assert actual == Path(plan[key]).resolve()
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == plan['sources'][str(actual)]
    fixture_root = Path(plan['fixture_root'])
    result = dict(status='RUNNING', probe_rows=[], probe_queries=[], actual_rows=[], actual_queries=[],
        target_forwards=0, candidate_comparisons=0, native_oracle_calls=0,
        actual_guarded_calls=0, attention_calls=0, probe_fully_passed=False,
        full_policy_chunks=0, episodes=0, integration_permission=False,
        production_integrated=False, performance_claim=False,
        scope='one post-result geometry rule; six real frozen B14 queries',
        native_binary_binding_is_new_not_retrospective=True)
    mismatch = None

    def check():
        if time.monotonic() >= started['active_deadline_monotonic']:
            raise TimeoutError('540s activity cap; sixty-second cleanup reserve')
        if (out / 'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED')
        assert result['target_forwards'] <= plan['target_forward_cap']
        assert result['candidate_comparisons'] <= plan['candidate_comparison_cap']
        assert result['attention_calls'] <= plan['attention_call_cap']

    def call_attention(fn, kind):
        check()
        assert result['attention_calls'] + 1 <= plan['attention_call_cap']
        result['attention_calls'] += 1
        key = dict(oracle='native_oracle_calls', probe='candidate_comparisons',
                   actual='actual_guarded_calls')[kind]
        if kind == 'probe':
            assert result[key] + 1 <= plan['candidate_comparison_cap']
        result[key] += 1
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

    def byte_equal(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    def compare(a, b):
        assert a.shape == b.shape and a.dtype == b.dtype
        aa, bb = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        delta = aa.float() - bb.float()
        return dict(bitwise=byte_equal(aa, bb),
            finite=bool(torch.isfinite(aa).all()) and bool(torch.isfinite(bb).all()),
            different=int(torch.count_nonzero(delta)), elements=aa.numel(),
            max_abs=float(delta.abs().max()),
            relative_RMS=float(delta.square().mean().sqrt() /
                               (bb.float().square().mean().sqrt() + 1e-30)))

    check()
    torch.set_num_threads(7)
    server.infer(dict(reset=True, prompt=None))
    model = server.transformer
    assert len(model.blocks) == 30
    result.update(CPU_threads=torch.get_num_threads(), CPU_affinity=sorted(os.sched_getaffinity(0)))
    try:
        for phase in ('probe', 'actual'):
            if phase == 'actual':
                assert result['probe_fully_passed'] and mismatch is None
                assert len(result['probe_queries']) == 6 and len(result['probe_rows']) == 180
                assert result['candidate_comparisons'] == result['native_oracle_calls'] == 180
                assert result['target_forwards'] == 6 and result['attention_calls'] == 360
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
                for saved in data['history']:
                    valid = saved['slots']
                    assert valid.dtype == torch.int64 and valid.ndim == 1
                    assert valid.numel() > 0 and valid.numel() + 16 == data['total_KV']
                    assert bool((valid >= 0).all()) and bool((valid < saved['pool_shape'][1]).all())
                    assert bool((valid[1:] > valid[:-1]).all()) and saved['pool_shape'][0] == 2
                    assert all(bool(torch.isfinite(saved[k]).all()) for k in ('k_valid', 'v_valid'))
                    slots.append(valid.to(server.device))
                originals = [block.attn1.forward for block in model.blocks]
                try:
                    for query in data['queries']:
                        check()
                        inputs, kwargs = dev(query['input']), dev(query['kwargs'])
                        kwargs.pop('input_dict', None)
                        rows = kwargs['candidate_cache_rows']
                        assert inputs['noisy_latents'].shape[0] == 14 and kwargs['update_cache'] == 0
                        assert torch.equal(rows, torch.tensor([0] * 7 + [1] * 7, device=server.device))
                        tracked = list(model.parameters()) + tensors((inputs, kwargs, slots))
                        tracked += [v for cache in state_check.pools(model, server.cache_name)
                                    for v in cache.values() if isinstance(v, torch.Tensor)]
                        versions = [(t, t._version) for t in tracked]
                        seen_layers = []
                        row_start = len(result[phase + '_rows'])

                        def make_history(layer, cache, valid):
                            prefix = valid.numel()
                            group = selected_group(prefix)
                            def history(attention, q, k, v, c, name, r, update):
                                nonlocal mismatch
                                assert c is cache and name == server.cache_name and r is rows and update == 0
                                assert layer not in seen_layers
                                seen_layers.append(layer)
                                assert q.shape == k.shape == v.shape == (14, 16, 24, 128)
                                if phase == 'probe':
                                    gold = call_attention(lambda: native.shared(q, cache['k'], cache['v'],
                                        k, v, valid, rows), 'oracle')
                                    if mismatch is not None:
                                        return gold
                                saved_qkv = [t.clone() for t in (q, k, v)]
                                assert all(bool(torch.isfinite(t).all()) for t in (q, k, v))
                                actual = call_attention(lambda: candidate.grouped(q, cache['k'], cache['v'],
                                    k, v, valid, group), 'probe' if phase == 'probe' else 'actual')
                                assert all(byte_equal(t, before) for t, before in zip((q, k, v), saved_qkv)), 'actual Q/K/V bytes mutated'
                                row = dict(phase=phase, length=label, total_KV=data['total_KV'],
                                    query=query['index'], layer=layer, prefix=prefix, prefix_mod64=prefix % 64,
                                    reference_tiles=(prefix + 16 + 63) // 64,
                                    G2_virtual_tiles=(prefix + 32 + 63) // 64, selected_group=group,
                                    original_native_returned=phase == 'probe')
                                if phase == 'probe':
                                    row['output'] = compare(actual, gold)
                                    result['probe_rows'].append(row)
                                    if not (row['output']['bitwise'] and row['output']['finite']):
                                        mismatch = row
                                    return gold
                                assert bool(torch.isfinite(actual).all()), 'actual guarded output nonfinite'
                                result['actual_rows'].append(row)
                                return actual
                            return history

                        try:
                            for i, block in enumerate(model.blocks):
                                callback = make_history(i, block.attn1.attn_caches[server.cache_name], slots[i])
                                block.attn1.forward = types.MethodType(adapter._private_forward(originals[i], callback), block.attn1)
                            check()
                            assert result['target_forwards'] + 1 <= plan['target_forward_cap']
                            result['target_forwards'] += 1
                            actual_output = model(inputs, **kwargs)
                            parity = compare(actual_output, query['output'].to(server.device))
                            result[phase + '_queries'].append(dict(length=label, query=query['index'],
                                full_frozen_output=parity, complete_layers=seen_layers,
                                custom_rows=len(result[phase + '_rows']) - row_start,
                                actual_guarded_returned_at_every_layer=phase == 'actual',
                                native_current_query_completed_after_mismatch=mismatch is not None))
                            assert parity['bitwise'] and parity['finite'], 'full query differs from original frozen output'
                            assert seen_layers == list(range(30))
                            assert state_check.equal(model, refs, server.cache_name), 'valid KV or metadata mutated'
                            assert all(t._version == version for t, version in versions), 'parameter/cache/input version mutated'
                            assert all(byte_equal(v.cpu(), saved['slots']) for v, saved in zip(slots, data['history'])), 'actual slots bytes mutated'
                            assert snap.fingerprint(snap.rng(True)) == initial_rng, 'RNG changed'
                            result[phase + '_queries'][-1]['all_valid_KV_slots_RNG_and_versions_unchanged'] = True
                            print(json.dumps(dict(phase=phase, length=label, query=query['index'],
                                rows=len(result[phase + '_rows']) - row_start, mismatch=mismatch)), flush=True)
                        finally:
                            for block, original in zip(model.blocks, originals): block.attn1.forward = original
                        del actual_output, inputs, kwargs, tracked, versions
                        gc.collect()
                        torch.cuda.empty_cache()
                        if mismatch is not None: break
                finally:
                    for block, original in zip(model.blocks, originals): block.attn1.forward = original
                del data, refs
                gc.collect()
                torch.cuda.empty_cache()
                if mismatch is not None: break
            if mismatch is not None: break
            if phase == 'probe':
                assert len(result['probe_queries']) == 6 and len(result['probe_rows']) == 180
                assert all(row['output']['bitwise'] and row['output']['finite'] for row in result['probe_rows'])
                result['probe_fully_passed'] = True
        if mismatch is None:
            assert len(result['actual_queries']) == 6 and len(result['actual_rows']) == 180
            assert result['target_forwards'] == 12 and result['candidate_comparisons'] == 180
            assert result['native_oracle_calls'] == result['actual_guarded_calls'] == 180
            assert result['attention_calls'] == 540
        result.update(status='COMPLETE_GUARDED_V2_PROBE_AND_ACTUAL_FIXTURES_PENDING_CLEANUP'
                      if mismatch is None else 'STOPPED_GUARDED_V2_PROBE_MISMATCH_NATIVE_QUERY_VERIFIED_PENDING_CLEANUP',
            mismatch=mismatch, finite_fixture_gate_passed=mismatch is None,
            all_completed_queries_frozen_exact_and_valid_state_unchanged=True,
            all_input_geometry_bitwise_proof=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        write(out / 'GUARDED-RESULT.json', result)


if __name__ == '__main__':
    entry.run = run
    entry.main()

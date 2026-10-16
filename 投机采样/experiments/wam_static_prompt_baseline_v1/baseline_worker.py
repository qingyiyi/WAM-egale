"""One S04-dependent six-arm ordinary baseline diagnostic; no deployment path."""
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import types

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
sys.path.insert(0, str(EXP / 'wam_shape_audit_v1'))
import full_matrix as entry
from full_matrix import load, write
from baseline_contract import ARMS, ORDERS
from static_projection import OriginalShapePack


def run(server, source, out, prior):
    import torch
    import adapter
    import state_check
    sys.path.insert(0, str(EXP / 'shared_prefix_torch_v2'))
    import _shared_torch_v1 as native
    sys.path.insert(0, str(EXP / 'wam_candidate_tile_v1'))
    import _wam_candidate_tile as candidate
    snap = load('static_prompt_baseline_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    plan = json.loads((out / 'PLAN.json').read_text())
    started = json.loads((out / 'RUN-STARTED.json').read_text())
    for module, name in ((native, 'native_binary'), (candidate, 'candidate_binary')):
        path = Path(module.__file__).resolve()
        assert path == Path(plan[name]).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == plan['sources'][str(path)]
    result = dict(status='RUNNING', rows=[], admission=[], prefills=[], oracles=[],
        target_forward_intents=0, captured_forward_intents=0, actual_device_forward_equivalent=0,
        all_36_eager_admissions_passed=False, formal_robot_performance=False,
        production_integration_permission=False, full_policy_chunks=0, episodes=0,
        scope='ordinary strong baseline: same-shape full B2/B14 prompt projections and static Graph',
        guard_hoisting='static admitted fixture only; Python guards do not run on Graph replay',
        input_updates_Graph_build_and_prompt_lifecycle_excluded=True)

    def check():
        if time.monotonic() >= started['active_deadline_monotonic']:
            raise TimeoutError('540s active work cap with sixty-second cleanup reserve')
        if (out / 'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED')
        assert result['target_forward_intents'] <= 500

    def reserve(*, capture=False):
        check()
        assert result['target_forward_intents'] < 500
        result['target_forward_intents'] += 1
        result['captured_forward_intents' if capture else 'actual_device_forward_equivalent'] += 1

    def dev(value):
        if isinstance(value, torch.Tensor):
            return value.to(server.device)
        if isinstance(value, dict):
            return {k: dev(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(dev(v) for v in value)
        return value

    def subset(value, indices):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 14:
            return value[indices].contiguous()
        if isinstance(value, dict):
            return {k: subset(v, indices) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(subset(v, indices) for v in value)
        return value

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [t for v in value.values() for t in tensors(v)]
        if isinstance(value, (tuple, list)):
            return [t for v in value for t in tensors(v)]
        return []

    def byte_equal(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    def parity(a, b):
        a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        assert a.shape == b.shape and a.dtype == b.dtype
        return dict(bitwise=byte_equal(a, b), finite=bool(torch.isfinite(a).all())
                    and bool(torch.isfinite(b).all()),
                    different_bytes=int(torch.count_nonzero(a.view(torch.uint8) != b.view(torch.uint8))))

    def group(prefix):
        return 2 if (prefix + 32 + 63) // 64 == (prefix + 16 + 63) // 64 else 1

    torch.set_num_threads(7)
    server.infer(dict(reset=True, prompt=None))
    model = server.transformer
    assert len(model.blocks) == 30 and all(not m.training for m in model.modules())
    originals = [block.attn1.forward for block in model.blocks]
    rope_original = {k: getattr(model.rope, k) for k in
                     ('f_freqs_base', 'h_freqs_base', 'w_freqs_base')}
    rope_gpu = {k: v.to(server.device) for k, v in rope_original.items()}
    for name, value in rope_gpu.items():
        setattr(model.rope, name, value)
    packs = {1: OriginalShapePack(model, 2), 7: OriginalShapePack(model, 14)}
    oracle_cpu = {}
    prefill_output_cpu = {}
    prompt_hashes = {}
    pack_fingerprints = {}

    def restore_attention():
        for block, original in zip(model.blocks, originals):
            block.attn1.forward = original

    def install(arm, slots, rows):
        restore_attention()
        for i, block in enumerate(model.blocks):
            pool, valid = block.attn1.attn_caches[server.cache_name], slots[i]
            def factory(c0, v0, engine):
                def history(attention, q, k, v, cache, name, mapping, update):
                    assert cache is c0 and name == server.cache_name and mapping is rows and update == 0
                    if engine.startswith('n1_dense'):
                        hk, hv = c0['k'][:, v0], c0['v'][:, v0]
                        return attention.attn_op(q,
                            torch.cat((hk.index_select(0, rows), k), dim=1),
                            torch.cat((hv.index_select(0, rows), v), dim=1))
                    if engine.startswith('n7_native'):
                        return native.shared(q, c0['k'], c0['v'], k, v, v0, rows)
                    return candidate.grouped(q, c0['k'], c0['v'], k, v, v0, group(v0.numel()))
                return history
            block.attn1.forward = types.MethodType(adapter._private_forward(
                originals[i], factory(pool, valid, arm)), block.attn1)

    try:
        for phase in ('admission', 'Graph'):
            if phase == 'Graph':
                assert result['all_36_eager_admissions_passed'] and result['target_forward_intents'] == 50
            for label in ('short', 'middle', 'long'):
                check()
                data = torch.load(Path(plan['fixture_root']) / label / 'real-verifier-fixture.pt',
                                  map_location='cpu', weights_only=False, mmap=True)
                for block, saved in zip(model.blocks, data['history']):
                    snap.restore_cache(block.attn1.attn_caches[server.cache_name], saved)
                refs = state_check.references(model, server.cache_name)
                initial_rng = snap.fingerprint(snap.rng(True))
                slots = [saved['slots'].to(server.device) for saved in data['history']]
                assert all(v.numel() + 16 == data['total_KV'] for v in slots)
                for query in data['queries']:
                    inp, kwargs = dev(query['input']), dev(query['kwargs'])
                    kwargs.pop('input_dict', None)
                    assert inp['noisy_latents'].shape[0] == 14 and kwargs['update_cache'] == 0
                    index = torch.tensor([0, 7], device=server.device)
                    inputs = {7: (inp, kwargs), 1: (subset(inp, index), subset(kwargs, index))}
                    input_hash = snap.fingerprint(snap.cpu(inputs))
                    for n, (si, sk) in inputs.items():
                        assert torch.equal(sk['candidate_cache_rows'], torch.tensor(
                            [0] * n + [1] * n, device=server.device))
                        ph = snap.fingerprint(si['text_emb'])
                        if n not in prompt_hashes:
                            prompt_hashes[n] = ph
                        assert prompt_hashes[n] == ph, 'actual prompt bytes vary across queries'
                        if len(packs[n].entries) == 0:
                            assert phase == 'admission' and label == 'short' and query['index'] == 0
                            restore_attention()
                            torch.cuda.synchronize()
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            wall = time.perf_counter()
                            begin.record()
                            with packs[n].activate(build=True):
                                reserve()
                                filled = model(si, **sk)
                            end.record()
                            end.synchronize()
                            fill_wall_ms = 1000 * (time.perf_counter() - wall)
                            p = parity(filled, query['output']) if n == 7 else dict(
                                finite=bool(torch.isfinite(filled).all()),
                                same_shape_original_oracle_check_deferred=True)
                            assert p['finite'] and (n == 1 or p['bitwise'])
                            prefill_output_cpu[n] = filled.detach().cpu().clone()
                            result['prefills'].append(dict(n_v=n, first_fill_wall_ms=fill_wall_ms,
                                first_fill_cuda_ms=begin.elapsed_time(end), output=p, **packs[n].summary()))
                            del filled
                        packs[n].admit(si['text_emb'])
                        if n not in pack_fingerprints:
                            pack_fingerprints[n] = snap.fingerprint(packs[n].tensors())
                    tracked = list(model.parameters()) + tensors((inputs, slots, rope_gpu))
                    tracked += [v for c in state_check.pools(model, server.cache_name)
                                for v in c.values() if isinstance(v, torch.Tensor)]
                    tracked += [t for pack in packs.values() for t in pack.tensors()]
                    versions = [(t, t._version) for t in tracked]
                    graphs, outputs, first = {}, {}, {}
                    graph = output = None
                    try:
                        if phase == 'admission':
                            restore_attention()
                            for n, (si, sk) in inputs.items():
                                reserve()
                                gold = model(si, **sk)
                                assert bool(torch.isfinite(gold).all())
                                if n == 7:
                                    p = parity(gold, query['output'])
                                    assert p['bitwise'] and p['finite']
                                oracle_cpu[(label, query['index'], n)] = gold.detach().cpu().clone()
                                if label == 'short' and query['index'] == 0:
                                    p = parity(prefill_output_cpu[n], gold)
                                    assert p['bitwise'] and p['finite'], 'first-fill full output differs from original same shape'
                                    next(v for v in result['prefills'] if v['n_v'] == n)['own_original_same_shape_output'] = p
                                result['oracles'].append(dict(length=label, query=query['index'], n_v=n,
                                    original_same_shape=True, output_hash=snap.fingerprint(gold)))
                                del gold
                            for arm in ARMS:
                                n = 1 if arm.startswith('n1_') else 7
                                si, sk = inputs[n]
                                install(arm, slots, sk['candidate_cache_rows'])
                                context = packs[n].activate() if arm.endswith('_cached') and not arm.endswith('_uncached') else nullcontext()
                                with context:
                                    reserve()
                                    actual = model(si, **sk)
                                p = parity(actual, oracle_cpu[(label, query['index'], n)])
                                assert p['bitwise'] and p['finite'], 'one baseline eager arm differs from same-shape oracle'
                                result['admission'].append(dict(length=label, query=query['index'], arm=arm, output=p))
                                del actual
                        else:
                            for arm in ARMS:
                                n = 1 if arm.startswith('n1_') else 7
                                si, sk = inputs[n]
                                install(arm, slots, sk['candidate_cache_rows'])
                                context = packs[n].activate() if arm.endswith('_cached') and not arm.endswith('_uncached') else nullcontext()
                                with context:
                                    stream = torch.cuda.Stream()
                                    stream.wait_stream(torch.cuda.current_stream())
                                    with torch.cuda.stream(stream):
                                        for _ in range(2):
                                            reserve()
                                            model(si, **sk)
                                    torch.cuda.current_stream().wait_stream(stream)
                                    torch.cuda.synchronize()
                                    graph = torch.cuda.CUDAGraph()
                                    with torch.cuda.graph(graph):
                                        reserve(capture=True)
                                        output = model(si, **sk)
                                graphs[arm], outputs[arm] = graph, output
                                reserve()
                                graph.replay()
                                torch.cuda.synchronize()
                                first[arm] = parity(output, oracle_cpu[(label, query['index'], n)])
                                assert first[arm]['bitwise'] and first[arm]['finite']
                            samples = {arm: [] for arm in ARMS}
                            orders = []
                            for order in ORDERS:
                                orders.append([ARMS[i] for i in order])
                                for i in order:
                                    arm = ARMS[i]
                                    n = 1 if arm.startswith('n1_') else 7
                                    reserve()
                                    torch.cuda.synchronize()
                                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    wall = time.perf_counter()
                                    begin.record()
                                    graphs[arm].replay()
                                    end.record()
                                    end.synchronize()
                                    sample = dict(wall_ms=1000*(time.perf_counter()-wall), cuda_ms=begin.elapsed_time(end))
                                    p = parity(outputs[arm], oracle_cpu[(label, query['index'], n)])
                                    assert p['bitwise'] and p['finite']
                                    sample['same_shape_original_bitwise_and_finite'] = True
                                    samples[arm].append(sample)
                            result['rows'].append(dict(length=label, total_KV=data['total_KV'], query=query['index'],
                                timing={arm: dict(samples=v, median_wall_ms=statistics.median(x['wall_ms'] for x in v),
                                    median_cuda_ms=statistics.median(x['cuda_ms'] for x in v)) for arm,v in samples.items()},
                                orders=orders, first_actual_replay=first, inputs_resident=True,
                                original_same_shape_oracle=True, independent_Graphs_and_pools=True,
                                python_guards_not_executed_on_Graph_replay=True))
                            torch.save(dict(original_n1=oracle_cpu[(label, query['index'], 1)],
                                original_n7=oracle_cpu[(label, query['index'], 7)],
                                **{arm: value.detach().cpu() for arm,value in outputs.items()}),
                                out / f'{label}-q{query["index"]}-outputs.pt')
                        assert state_check.equal(model, refs, server.cache_name)
                        assert all(t._version == v for t,v in versions)
                        assert snap.fingerprint(snap.cpu(inputs)) == input_hash
                        assert all(byte_equal(v.cpu(), h['slots']) for v,h in zip(slots,data['history']))
                        assert snap.fingerprint(snap.rng(True)) == initial_rng
                        for n, (si, sk) in inputs.items():
                            packs[n].admit(si['text_emb'])
                            assert snap.fingerprint(packs[n].tensors()) == pack_fingerprints[n], 'owned projection/prompt bytes changed'
                        if phase == 'Graph':
                            result['rows'][-1]['valid_KV_RNG_inputs_slots_versions_and_cached_outputs_unchanged'] = True
                        print(json.dumps(dict(phase=phase, length=label, query=query['index'],
                            intents=result['target_forward_intents'])), flush=True)
                    finally:
                        restore_attention()
                        graphs.clear()
                        outputs.clear()
                        graph = output = None
                        gc.collect()
                        torch.cuda.empty_cache()
                    del inputs, inp, kwargs, tracked, versions
                del refs, data
                gc.collect()
                torch.cuda.empty_cache()
            if phase == 'admission':
                assert len(result['admission']) == 36 and len(result['oracles']) == 12
                result['all_36_eager_admissions_passed'] = True
        assert len(result['prefills']) == 2 and len(result['rows']) == 6
        assert result['target_forward_intents'] == 410 and result['captured_forward_intents'] == 36
        assert result['actual_device_forward_equivalent'] == 374
        result.update(status='COMPLETE_STATIC_SAME_SHAPE_PROMPT_BASELINE_PENDING_CLEANUP',
            all_six_queries_all_six_arms_complete=True,
            all_valid_KV_RNG_inputs_slots_versions_unchanged=True,
            cached_projection_packs={str(n): pack.summary() for n,pack in packs.items()},
            CPU_threads=torch.get_num_threads(), CPU_affinity=sorted(os.sched_getaffinity(0)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        restore_attention()
        for name,value in rope_original.items():
            setattr(model.rope, name, value)
        for pack in packs.values():
            pack.close()
        write(out / 'BASELINE-RESULT.json', result)


if __name__ == '__main__':
    entry.run = run
    entry.main()

"""Conditional three-arm B14 Graph pilot; all actual numerical checks precede timing."""
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
ROOT = OWN.parents[2]
sys.path.insert(0, str(EXP / 'wam_shape_audit_v1'))
import full_matrix as entry
from full_matrix import write, load

ARMS = ('native_shared_BM128', 'research_G1_BM64', 'guarded_v2')
ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2))


def group_for(prefix):
    if type(prefix) is not int or prefix <= 0:
        raise ValueError('actual validated positive prefix required')
    return 2 if (prefix + 32 + 63) // 64 == (prefix + 16 + 63) // 64 else 1


def run(server, source, out, prior):
    import torch
    import adapter
    import state_check

    sys.path.insert(0, str(EXP / 'shared_prefix_torch_v2'))
    import _shared_torch_v1 as native
    sys.path.insert(0, str(EXP / 'wam_candidate_tile_v1'))
    import _wam_candidate_tile as candidate
    snap = load('candidate_graph_pilot_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    plan = json.loads((out / 'PLAN.json').read_text())
    started = json.loads((out / 'RUN-STARTED.json').read_text())
    for module, key in ((native, 'native_binary'), (candidate, 'candidate_binary')):
        path = Path(module.__file__).resolve()
        assert path == Path(plan[key]).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == plan['sources'][str(path)]
    fixture = Path(plan['fixture_root'])
    result = dict(status='RUNNING', admission=[], rows=[], target_forward_intents=0,
        captured_forward_intents=0, actual_device_forward_equivalent=0,
        all_18_actual_eager_admissions_passed=False, full_policy_chunks=0, episodes=0,
        production_engine=False, formal_robot_performance=False,
        scope='six frozen B14 same-shape complete30 static Graph queries',
        input_update_and_capture_excluded=True, numerical_checks_excluded_from_timing=True)

    def check():
        if time.monotonic() >= started['active_deadline_monotonic']:
            raise TimeoutError('540s active budget with sixty-second cleanup reserve')
        if (out / 'STOP_REQUESTED').exists(): raise RuntimeError('STOP_REQUESTED')
        assert result['target_forward_intents'] <= plan['target_forward_cap']

    def reserve(*, capture=False):
        check()
        assert result['target_forward_intents'] + 1 <= plan['target_forward_cap']
        result['target_forward_intents'] += 1
        result['captured_forward_intents' if capture else 'actual_device_forward_equivalent'] += 1

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

    def parity(a, b):
        assert a.shape == b.shape and a.dtype == b.dtype
        a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        diff = a.float() - b.float()
        return dict(bitwise=byte_equal(a, b), finite=bool(torch.isfinite(a).all())
            and bool(torch.isfinite(b).all()), different=int(torch.count_nonzero(diff)),
            max_abs=float(diff.abs().max()))

    check()
    torch.set_num_threads(7)
    server.infer(dict(reset=True, prompt=None))
    model = server.transformer
    assert len(model.blocks) == 30
    originals = [b.attn1.forward for b in model.blocks]
    rope_original = {k: getattr(model.rope, k) for k in
                     ('f_freqs_base', 'h_freqs_base', 'w_freqs_base')}
    # Keep these exact GPU bases alive until every Graph has been destroyed.
    rope_gpu = {k: v.to(server.device) for k, v in rope_original.items()}
    for name, value in rope_gpu.items(): setattr(model.rope, name, value)
    result.update(CPU_threads=torch.get_num_threads(), CPU_affinity=sorted(os.sched_getaffinity(0)))

    def restore_attention():
        for block, original in zip(model.blocks, originals): block.attn1.forward = original

    def install(engine, slots, rows):
        restore_attention()
        for i, block in enumerate(model.blocks):
            cache, valid = block.attn1.attn_caches[server.cache_name], slots[i]
            chosen = 1 if engine == 'research_G1_BM64' else group_for(valid.numel())
            def factory(c0, valid0, chosen0, engine0):
                def history(attention, q, k, v, c, name, r, update):
                    assert c is c0 and name == server.cache_name and r is rows and update == 0
                    if engine0 == 'native_shared_BM128':
                        return native.shared(q, c0['k'], c0['v'], k, v, valid0, rows)
                    return candidate.grouped(q, c0['k'], c0['v'], k, v, valid0, chosen0)
                return history
            block.attn1.forward = types.MethodType(adapter._private_forward(
                originals[i], factory(cache, valid, chosen, engine)), block.attn1)

    try:
        # All six queries and all three actual engines qualify before any Graph.
        for phase in ('admission', 'Graph'):
            if phase == 'Graph':
                assert result['all_18_actual_eager_admissions_passed']
                assert result['target_forward_intents'] == 18
            for label in ('short', 'middle', 'long'):
                check()
                data = torch.load(fixture / label / 'real-verifier-fixture.pt',
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
                    slots.append(valid.to(server.device))
                for query in data['queries']:
                    check()
                    inputs, kwargs = dev(query['input']), dev(query['kwargs'])
                    kwargs.pop('input_dict', None)
                    rows = kwargs['candidate_cache_rows']
                    assert inputs['noisy_latents'].shape[0] == 14 and kwargs['update_cache'] == 0
                    assert torch.equal(rows, torch.tensor([0] * 7 + [1] * 7, device=server.device))
                    input_hash = snap.fingerprint(snap.cpu((inputs, kwargs)))
                    tracked = list(model.parameters()) + tensors((inputs, kwargs, slots, rope_gpu))
                    tracked += [v for c in state_check.pools(model, server.cache_name)
                                for v in c.values() if isinstance(v, torch.Tensor)]
                    versions = [(t, t._version) for t in tracked]
                    graphs, outputs = {}, {}
                    graph = output = None
                    first_replay = {}
                    try:
                        if phase == 'admission':
                            for engine in ARMS:
                                install(engine, slots, rows)
                                reserve()
                                actual = model(inputs, **kwargs)
                                p = parity(actual, query['output'])
                                result['admission'].append(dict(length=label, query=query['index'],
                                    arm=engine, output=p, actual_eager_forward=True))
                                assert p['bitwise'] and p['finite'], 'one actual engine admission differs from frozen output'
                            del actual
                        else:
                            for engine in ARMS:
                                install(engine, slots, rows)
                                stream = torch.cuda.Stream()
                                stream.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(stream):
                                    for _ in range(2):
                                        reserve()
                                        model(inputs, **kwargs)
                                torch.cuda.current_stream().wait_stream(stream)
                                torch.cuda.synchronize()
                                graph = torch.cuda.CUDAGraph()
                                reserve(capture=True)
                                with torch.cuda.graph(graph): output = model(inputs, **kwargs)
                                # Capture records work; first real replay supplies valid output.
                                reserve()
                                graph.replay()
                                torch.cuda.synchronize()
                                p = parity(output, query['output'])
                                assert p['bitwise'] and p['finite'], 'first actual Graph replay differs'
                                graphs[engine], outputs[engine], first_replay[engine] = graph, output, p
                            restore_attention()
                            samples = {engine: [] for engine in ARMS}
                            orders = []
                            for order in ORDERS:
                                orders.append([ARMS[i] for i in order])
                                for i in order:
                                    engine = ARMS[i]
                                    reserve()
                                    torch.cuda.synchronize()
                                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    wall = time.perf_counter()
                                    begin.record()
                                    graphs[engine].replay()
                                    end.record()
                                    end.synchronize()
                                    sample = dict(wall_ms=1000 * (time.perf_counter() - wall),
                                                  cuda_ms=begin.elapsed_time(end))
                                    p = parity(outputs[engine], query['output'])
                                    assert p['bitwise'] and p['finite'], 'measured Graph replay differs'
                                    sample['frozen_output_bitwise_and_finite'] = True
                                    samples[engine].append(sample)
                            result['rows'].append(dict(length=label, total_KV=data['total_KV'],
                                query=query['index'], position='first' if query['index'] == 0 else 'last',
                                timing={engine: dict(samples=values,
                                    median_wall_ms=statistics.median(v['wall_ms'] for v in values),
                                    median_cuda_ms=statistics.median(v['cuda_ms'] for v in values))
                                    for engine, values in samples.items()},
                                orders=orders, first_actual_replay=first_replay,
                                guarded_groups_per_layer=[group_for(v.numel()) for v in slots],
                                independent_Graphs_and_pools=True, same_GPU_resident_inputs=True,
                                private_fixed_metadata_for_all_arms=True,
                                static_full30_only=True, capture_and_input_updates_not_timed=True))
                            torch.save(dict(frozen_original=query['output'],
                                **{engine: value.detach().cpu() for engine, value in outputs.items()}),
                                out / f'{label}-q{query["index"]}-outputs.pt')
                        assert state_check.equal(model, refs, server.cache_name), 'all-valid cache changed'
                        assert all(t._version == v for t, v in versions), 'parameter/cache/input version changed'
                        assert snap.fingerprint(snap.cpu((inputs, kwargs))) == input_hash, 'actual input bytes changed'
                        assert all(byte_equal(v.cpu(), h['slots']) for v, h in zip(slots, data['history'])), 'slots bytes changed'
                        assert snap.fingerprint(snap.rng(True)) == initial_rng, 'RNG changed'
                        if phase == 'Graph': result['rows'][-1]['valid_KV_RNG_inputs_slots_versions_unchanged'] = True
                        print(json.dumps(dict(phase=phase, length=label, query=query['index'],
                            intents=result['target_forward_intents'])), flush=True)
                    finally:
                        restore_attention()
                        graphs.clear()
                        outputs.clear()
                        # Last loop references also own a Graph/output; drop before GC.
                        graph = output = None
                        gc.collect()
                        torch.cuda.empty_cache()
                    del inputs, kwargs, tracked, versions
                del data, refs
                gc.collect()
                torch.cuda.empty_cache()
            if phase == 'admission':
                assert len(result['admission']) == 18
                result['all_18_actual_eager_admissions_passed'] = True
        assert len(result['rows']) == 6 and result['target_forward_intents'] == 198
        assert result['captured_forward_intents'] == 18 and result['actual_device_forward_equivalent'] == 180
        result.update(status='COMPLETE_THREE_ARM_STATIC_GRAPH_PILOT_PENDING_CLEANUP',
            all_six_queries_all_three_arms_complete=True,
            all_valid_KV_RNG_inputs_slots_versions_unchanged=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        restore_attention()
        for name, value in rope_original.items(): setattr(model.rope, name, value)
        write(out / 'PILOT-RESULT.json', result)


if __name__ == '__main__':
    entry.run = run
    entry.main()

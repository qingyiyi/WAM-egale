"""Six real same-seven-state sequential B2 Graph baselines; no policy rollout.

Each graph contains seven complete 30-layer original-shape B2 forwards and
their CFG-ordered concatenation. Static metadata admission is shared by private
eager and Graph; both must reproduce seven unmodified B2 oracle forwards.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import types

import full_matrix as entry
from full_matrix import EXP, OWN, write, load


def run(server, source, out, prior):
    import torch
    import adapter
    import state_check

    snap = load('same_seven_graph_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    plan = json.loads((out / 'PLAN.json').read_text())
    started = json.loads((out / 'RUN-STARTED.json').read_text())
    fixture_root = Path(plan['fixture_root'])
    result = dict(status='RUNNING', rows=[], target_forwards=0, full_policy_chunks=0,
                  episodes=0, candidate_states_per_query=7, production_engine=False,
                  formal_robot_performance=False, guard_hoisting='same fixed metadata for private eager and Graph',
                  comparison='same seven distinct real states; seven original B2 forwards plus CFG cat')

    def check():
        if time.monotonic() >= started['active_deadline_monotonic']:
            raise TimeoutError('540 second active budget, sixty second cleanup reserve')
        if (out / 'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED')
        if result['target_forwards'] > plan['target_forward_cap']:
            raise RuntimeError('finite complete target-forward cap exceeded')

    def reserve(count):
        check()
        if result['target_forwards'] + count > plan['target_forward_cap']:
            raise RuntimeError('cannot reserve complete target forward group')
        result['target_forwards'] += count

    def dev(value):
        if isinstance(value, torch.Tensor): return value.to(server.device)
        if isinstance(value, dict): return {k: dev(v) for k, v in value.items()}
        if isinstance(value, tuple): return tuple(dev(v) for v in value)
        if isinstance(value, list): return [dev(v) for v in value]
        return value

    def subset(value, indices):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 14:
            return value[indices].contiguous()
        if isinstance(value, dict): return {k: subset(v, indices) for k, v in value.items()}
        if isinstance(value, tuple): return tuple(subset(v, indices) for v in value)
        if isinstance(value, list): return [subset(v, indices) for v in value]
        return value

    def tensors(value):
        if isinstance(value, torch.Tensor): return [value]
        if isinstance(value, dict): return [x for v in value.values() for x in tensors(v)]
        if isinstance(value, (tuple, list)): return [x for v in value for x in tensors(v)]
        return []

    def parity(actual, expected):
        a = actual.detach().contiguous()
        b = expected.detach().to(a.device).contiguous()
        diff = (a.float() - b.float()).abs()
        return dict(bitwise=torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
                    max_abs=float(diff.max()), different=int(torch.count_nonzero(diff)),
                    finite=bool(torch.isfinite(a).all()))

    def cfg_cat(values):
        return torch.cat([v[:1] for v in values] + [v[1:] for v in values])

    def forward(inputs, kwargs):
        reserve(1)
        return model(inputs, **kwargs)

    def measure_once(fn):
        check()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        a.record()
        value = fn()
        b.record()
        b.synchronize()
        return dict(wall_ms=1000 * (time.perf_counter() - begin), cuda_ms=a.elapsed_time(b)), value

    check()
    torch.set_num_threads(7)
    server.infer(dict(reset=True, prompt=None))
    model = server.transformer
    assert len(model.blocks) == 30
    result.update(actual_CPU_threads=torch.get_num_threads(), CPU_affinity=sorted(os.sched_getaffinity(0)))
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
            slots = [c['slots'].to(server.device) for c in data['history']]
            originals = [block.attn1.forward for block in model.blocks]
            try:
                for query in data['queries']:
                    check()
                    inputs = dev(query['input'])
                    kwargs = dev(query['kwargs'])
                    kwargs.pop('input_dict', None)
                    assert inputs['noisy_latents'].shape[0] == 14
                    parts = []
                    for j in range(7):
                        indices = torch.tensor([j, j + 7], device=server.device)
                        ci, ck = subset(inputs, indices), subset(kwargs, indices)
                        assert ci['noisy_latents'].shape[0] == 2 and ck['update_cache'] == 0
                        assert torch.equal(ck['candidate_cache_rows'],
                                           torch.tensor([0, 1], device=server.device))
                        parts.append((ci, ck))
                    before = result['target_forwards']
                    state_hashes = [hashlib.sha256(ci['noisy_latents'].detach().cpu().contiguous()
                        .view(torch.uint8).numpy().tobytes()).hexdigest() for ci, _ in parts]
                    # These seven unmodified forwards are the actual same-state oracle.
                    raw_values = [forward(ci, ck) for ci, ck in parts]
                    oracle = cfg_cat(raw_values)
                    assert state_check.equal(model, refs, server.cache_name)
                    assert snap.fingerprint(snap.rng(True)) == initial_rng
                    rows = [ck['candidate_cache_rows'] for _, ck in parts]
                    rope_original = {k: getattr(model.rope, k) for k in
                                     ('f_freqs_base', 'h_freqs_base', 'w_freqs_base')}
                    graph = None
                    try:
                        for name, value in rope_original.items():
                            setattr(model.rope, name, value.to(server.device))
                        for i, block in enumerate(model.blocks):
                            cache = block.attn1.attn_caches[server.cache_name]
                            valid = slots[i]
                            assert valid.numel() + 16 == data['total_KV']

                            def history(attention, q, k, v, c, name, r, update,
                                        _cache=cache, _slots=valid):
                                assert c is _cache and name == server.cache_name and update == 0
                                assert any(r is allowed for allowed in rows)
                                hk = _cache['k'][:, _slots]
                                hv = _cache['v'][:, _slots]
                                return attention.attn_op(q, torch.cat((hk.index_select(0, r), k), 1),
                                                        torch.cat((hv.index_select(0, r), v), 1))

                            block.attn1.forward = types.MethodType(
                                adapter._private_forward(originals[i], history), block.attn1)

                        def eager():
                            return cfg_cat([forward(ci, ck) for ci, ck in parts])

                        private_output = eager()
                        admitted = parity(private_output, oracle)
                        assert admitted['bitwise'] and admitted['finite']
                        version_values = list(model.parameters())
                        version_values += [v for c in state_check.pools(model, server.cache_name)
                                           for v in c.values() if isinstance(v, torch.Tensor)]
                        version_values += tensors(parts) + slots
                        version_values += [getattr(model.rope, name) for name in rope_original]
                        versions = [(x, x._version) for x in version_values]
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(2): eager()
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        capture_begin = time.perf_counter()
                        with torch.cuda.graph(graph): graph_output = eager()
                        torch.cuda.synchronize()
                        capture_ms = 1000 * (time.perf_counter() - capture_begin)

                        def replay():
                            reserve(7)
                            graph.replay()
                            return graph_output

                        samples = dict(same7_private_eager=[], same7_sequential_Graph=[])
                        orders = []
                        last_values = {}
                        for repetition in range(6):
                            order = [('same7_private_eager', eager), ('same7_sequential_Graph', replay)]
                            if (repetition + (query['index'] != 0)) % 2: order.reverse()
                            orders.append([name for name, _ in order])
                            for name, fn in order:
                                sample, actual = measure_once(fn)
                                sample_check = parity(actual, oracle)
                                assert sample_check['bitwise'] and sample_check['finite']
                                sample['same7_raw_oracle_bitwise'] = True
                                samples[name].append(sample)
                                last_values[name] = actual
                        comparisons = {name: parity(actual, oracle) for name, actual in last_values.items()}
                        assert all(v['bitwise'] and v['finite'] for v in comparisons.values())
                        assert all(x._version == version for x, version in versions)
                        assert state_check.equal(model, refs, server.cache_name)
                        assert snap.fingerprint(snap.rng(True)) == initial_rng
                        timing = {name: dict(samples=values,
                            median_wall_ms=statistics.median(x['wall_ms'] for x in values),
                            median_cuda_ms=statistics.median(x['cuda_ms'] for x in values))
                            for name, values in samples.items()}
                        row = dict(length=label, total_KV=data['total_KV'], query=query['index'],
                            position='first' if query['index'] == 0 else 'last',
                            candidate_indices=list(range(7)), CFG_pairs=[[j, j + 7] for j in range(7)],
                            seven_actual_B2_state_input_sha256=state_hashes,
                            complete_original_B2_oracle_forwards=7, original_math_unchanged=True,
                            graph_contains_complete_B2_forwards=7, graph_contains_CFG_cat=True,
                            timing=timing, execution_orders=orders, private_vs_raw=admitted,
                            graph_replay_vs_raw=comparisons['same7_sequential_Graph'],
                            final_outputs_vs_raw=comparisons,
                            complete_target_forwards=result['target_forwards'] - before,
                            all_valid_KV_and_RNG_unchanged=True, all_tensor_versions_unchanged=True,
                            input_update_and_capture_not_timed=True, capture_ms_excluded=capture_ms,
                            GPU_inputs_resident=True, production_admission_not_measured=True,
                            no_B14_or_cross_shape_parity_claim=True)
                        assert row['complete_target_forwards'] == 119
                        torch.save(dict(raw_same7_oracle=oracle.detach().cpu(),
                            private_same7=private_output.detach().cpu(),
                            final_same7_Graph=graph_output.detach().cpu()),
                            out / f'{label}-q{query["index"]}-outputs.pt')
                        result['rows'].append(row)
                        print(json.dumps(dict(length=label, query=query['index'],
                            timing={name: x['median_wall_ms'] for name, x in timing.items()})), flush=True)
                    finally:
                        for block, original in zip(model.blocks, originals):
                            block.attn1.forward = original
                        for name, value in rope_original.items(): setattr(model.rope, name, value)
                        if graph is not None: del graph
                        gc.collect()
                        torch.cuda.empty_cache()
                    del oracle, raw_values, private_output, graph_output, parts, inputs, kwargs
            finally:
                for block, original in zip(model.blocks, originals): block.attn1.forward = original
            del refs, data
            gc.collect()
            torch.cuda.empty_cache()
        assert len(result['rows']) == 6 and result['target_forwards'] == 714
        result.update(status='COMPLETE_SAME_SEVEN_SEQUENTIAL_GRAPH_WORKER_PENDING_CLEANUP',
                      all_six_real_same7_queries_complete=True,
                      all_valid_KV_and_RNG_unchanged=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        write(out / 'SEQ-RESULT.json', result)


if __name__ == '__main__':
    entry.run = run
    entry.main()

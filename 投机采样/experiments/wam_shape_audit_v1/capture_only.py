"""Three real A_off chunks and qualified immutable verifier inputs; no timing.

This is a shared-GPU correctness fallback.  It neither runs an episode nor
replaces the original full_matrix/Graph/Q02 entry or any failed result.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import full_matrix as entry
from full_matrix import ROOT, EXP, OWN, write, load


def run(s, source, out, prior):
    import torch
    from runtime import Runtime
    from selection import install
    from measured_online import control
    import buffers
    import state_check
    from run_capture import load_plan

    plan = load_plan(out)
    started = json.loads((out / 'RUN-STARTED.json').read_text())
    assert started['schema'] == 'wam-real-fixture-capture-start/v1'
    origin = float(started['GPU_origin_monotonic'])
    snap = load('capture_only_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    old = load('capture_only_prior_profile', EXP / 'megaverifier_profile_v2/profile_worker.py')
    result = dict(status='RUNNING', complete_chunks=0, episodes=0,
        scope='three frozen real A_off policy chunks and immutable verifier fixtures',
        shared_GPU_allowed=True, formal_robot_performance=False, profiler_used=False,
        timing_samples=0, new_engine=False, current_TF_onpolicy=False, chunks=[], rows=[])
    rt = None
    all_handles = []

    def check():
        if (out / 'STOP_REQUESTED').exists():
            raise RuntimeError('capture supervisor requested stop')
        if time.monotonic() - origin > 1100:
            raise TimeoutError('capture active budget; cleanup remains reserved')
        if rt is not None and rt.attempted > plan['target_forward_cap']:
            raise RuntimeError('capture target-forward cap')

    def to_device(x):
        if isinstance(x, torch.Tensor): return x.to(s.device)
        if isinstance(x, dict): return {k: to_device(v) for k, v in x.items()}
        if isinstance(x, tuple): return tuple(to_device(v) for v in x)
        if isinstance(x, list): return [to_device(v) for v in x]
        return x

    def subset(x, idx):
        if isinstance(x, torch.Tensor) and x.ndim and x.shape[0] == 14:
            return x[idx].contiguous()
        if isinstance(x, dict): return {k: subset(v, idx) for k, v in x.items()}
        if isinstance(x, tuple): return tuple(subset(v, idx) for v in x)
        if isinstance(x, list): return [subset(v, idx) for v in x]
        return x

    def compare(a, b):
        a = a.detach().cpu().contiguous(); b = b.detach().cpu().contiguous()
        assert a.shape == b.shape and a.dtype == b.dtype
        d = (a.float() - b.float()).abs()
        return dict(bitwise=torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
            max_abs=float(d.max()), mean_abs=float(d.mean()),
            different=int(torch.count_nonzero(d)),
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))

    def linear_row(name, module, args):
        x = args[0]
        return dict(name=name, input=list(x.shape), stride=list(x.stride()),
            dtype=str(x.dtype), device=str(x.device), weight=list(module.weight.shape),
            weight_stride=list(module.weight.stride()), weight_dtype=str(module.weight.dtype),
            MNK=[x.numel() // x.shape[-1], module.weight.shape[0], x.shape[-1]])

    try:
        torch.set_num_threads(7)
        check()
        s.infer(dict(reset=True, prompt=None))
        s._p0_action_draft_mode = 'verifier_k2_b2_bonus'
        s._load_p0_action_draft_b2_predictor().model.to(s.device).eval()
        proxy = install(s, 'prefix_then_cost')
        buffers.prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
        all_handles.append(s.transformer.register_forward_pre_hook(lambda *a: check(), with_kwargs=True))
        rt = Runtime(s, out, 1800, shared=False, allow_cuda=True)
        short = torch.load(prior['short']['path'], map_location='cpu', weights_only=False)
        for key in ('state', 'request'):
            assert snap.fingerprint(short[key]) == prior['short'][key + '_hash']
        mid = old.middle(s, rt, snap, short, prior, out)
        assert rt.attempted == 18, 'middle must use the original18 cache forwards only'

        for label, history, frame, total in (
                ('short', 144, 4, 288), ('middle', 1008, 28, 1152), ('long', 2016, 56, 2160)):
            check(); rt.shared.close()
            payload = short if label == 'short' else mid if label == 'middle' else torch.load(
                prior['long']['path'], map_location='cpu', weights_only=False)
            ref = mid if label == 'middle' else prior[label]
            for key in ('state', 'request'):
                assert snap.fingerprint(payload[key]) == ref[key + '_hash']
            assert payload['state']['fields']['frame_st_id'] == frame
            assert len(payload['state']['caches']) == len(s.transformer.blocks) == 30
            assert all(c['slots'].numel() == history and c['pool_shape'] == [2, 2160, 24, 128]
                       for c in payload['state']['caches'])
            snap.restore(s, payload['state'])
            assert snap.fingerprint(state_check.capture_light(s)) == snap.fingerprint(state_check.light(payload['state']))
            s._p0_action_draft_mode = 'verifier_k2_b2_bonus'
            s._p0_action_draft_selector = proxy.effective = 'prefix_then_cost'
            proxy.records.clear(); proxy.pending = None
            rt.shared = control(s, 'O3', allow_cuda=True)
            cap = dict(active=False, count=0, first=None, last=None, history=None)

            def pre(model, args, kwargs):
                cap['active'] = kwargs.get('candidate_cache_rows') is not None
                if not cap['active']: return
                inp = args[0] if args else kwargs['input_dict']
                assert inp['noisy_latents'].shape[0] == 14 and kwargs['update_cache'] == 0
                if cap['history'] is None:
                    cap['history'] = [snap.capture_cache(b.attn1.attn_caches[s.cache_name]) for b in model.blocks]
                cap['current'] = dict(input=snap.cpu(inp), kwargs=snap.cpu(kwargs),
                    index=cap['count'], linear=[])

            def post(model, args, kwargs, output):
                if not cap['active']: return
                item = cap.pop('current'); item['output'] = snap.cpu(output)
                if cap['first'] is None: cap['first'] = item
                cap['last'] = item; cap['count'] += 1; cap['active'] = False

            def policy_linear(name):
                def hook(module, args):
                    if cap['active']: cap['current']['linear'].append(linear_row(name, module, args))
                return hook

            hooks = [s.transformer.register_forward_pre_hook(pre, with_kwargs=True),
                     s.transformer.register_forward_hook(post, with_kwargs=True)]
            for name, module in s.transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    hooks.append(module.register_forward_pre_hook(policy_linear(name)))
            folder = out / label; folder.mkdir()
            s.exp_save_root = str(folder / 'policy'); Path(s.exp_save_root).mkdir()
            before = dict(rt.phase); beforecount = rt.attempted; beforeevents = len(buffers.EVENTS)
            try:
                s.infer(copy.deepcopy(payload['request']))
            finally:
                for h in hooks: h.remove()
                shared = rt.shared.summary(); rt.shared.close()
            phase = {k: v - before.get(k, 0) for k, v in rt.phase.items() if v - before.get(k, 0)}
            events = copy.deepcopy(buffers.EVENTS[beforeevents:])
            assert phase['committed_scheduler_update_count'] == 50
            assert len(proxy.records) == phase['tree_count'] and proxy.pending is None
            assert cap['count'] == phase['verifier_forward_count'] and cap['count'] >= 2
            assert shared['last_request_clean'] and shared['candidates'] > 0
            assert len(events) == 1 and all(events[0].get(k) is True for k in ('enabled', 'ok', 'closed'))
            assert len(cap['history']) == 30 and all(c['slots'].numel() == total - 16 for c in cap['history'])
            assert torch.equal(cap['first']['input']['text_emb'], cap['last']['input']['text_emb'])
            cap['last']['input']['text_emb'] = cap['first']['input']['text_emb']
            for b, c in zip(s.transformer.blocks, cap['history']):
                snap.restore_cache(b.attn1.attn_caches[s.cache_name], c)
            refs = state_check.references(s.transformer, s.cache_name)
            rng = snap.fingerprint(snap.rng(True))
            versions = [(v, v._version) for v in s.transformer.parameters()]
            diagnostics = []
            for query in (cap['first'], cap['last']):
                inp = to_device(query['input']); kwargs = to_device(query['kwargs']); kwargs.pop('input_dict', None)
                ledger = []; handles = []
                def replay_linear(name):
                    def hook(module, args): ledger.append(linear_row(name, module, args))
                    return hook
                for name, module in s.transformer.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        handles.append(module.register_forward_pre_hook(replay_linear(name)))
                try:
                    raw = s.transformer(inp, **kwargs)
                    parity = compare(raw, query['output'])
                    assert parity['bitwise'] and parity['finite'], 'RAW_B14 must reproduce captured O3 exactly'
                    write(folder / f'LINEAR-query{query["index"]}-n7.json', dict(
                        scope='one actual original RAW_B14 forward; metadata hook only', linear=ledger))
                    diagnostics.append(dict(query_index=query['index'], n_v=7, CFG=2,
                        raw_vs_captured_O3=parity, repeated_timing=False))
                    for n in (1, 3):
                        check(); ledger.clear()
                        idx = torch.tensor(list(range(n)) + list(range(7, 7 + n)), device=s.device)
                        si = subset(inp, idx); sk = subset(kwargs, idx)
                        assert torch.equal(sk['candidate_cache_rows'], torch.tensor([0]*n+[1]*n, device=s.device))
                        y = s.transformer(si, **sk)
                        numeric = compare(y, raw[idx]); assert numeric['finite']
                        diagnostics.append(dict(query_index=query['index'], n_v=n, CFG=2,
                            subset_vs_full=numeric, bitwise_not_required_across_shapes=True, repeated_timing=False))
                        write(folder / f'LINEAR-query{query["index"]}-n{n}.json', dict(
                            scope='one actual original shape-subset forward; metadata hook only', linear=ledger))
                    assert state_check.equal(s.transformer, refs, s.cache_name), 'valid KV or metadata changed'
                    assert snap.fingerprint(snap.rng(True)) == rng, 'RNG changed'
                    assert all(v._version == ver for v, ver in versions), 'model parameter mutated'
                finally:
                    for h in handles: h.remove()
                del raw, inp, kwargs
            check()
            fixture = folder / 'real-verifier-fixture.pt'
            torch.save(dict(history=cap['history'], queries=[cap['first'], cap['last']],
                source=prior[label] if label != 'middle' else dict(source='real original18 cache-forward replay',
                    state_hash=mid['state_hash'], request_hash=mid['request_hash']),
                total_KV=total, capture_role='A_off on frozen T history, not current TF on-policy',
                no_profiler=True, formal_performance=False), fixture)
            chunk = dict(length=label, request_history=history, total_KV=total, frame=frame,
                candidates=cap['count'], phase=phase, forwards=rt.attempted-beforecount,
                complete_scheduler_50=True, closed_loop_episode=False, shared=shared, O3_events=events,
                records=copy.deepcopy(proxy.records), actual_candidate_linear_metadata=True,
                raw_first_last_B14_bitwise=True, all_valid_KV_RNG_parameters_unchanged=True,
                fixture=dict(path=str(fixture), bytes=fixture.stat().st_size))
            write(folder / 'CHUNK.json', chunk); write(folder / 'NUMERIC-COMPARISON.json', diagnostics)
            result['chunks'].append(chunk); result['rows'].extend(diagnostics); result['complete_chunks'] += 1
            print(json.dumps(dict(captured=label, total_KV=total, candidates=cap['count'],
                fixture_bytes=fixture.stat().st_size, correctness_only=True)), flush=True)
            del refs, cap, payload, versions; torch.cuda.empty_cache()
        assert result['complete_chunks'] == 3 and rt.attempted <= 1800
        source.save_async.__globals__['executor'].submit(lambda: None).result(timeout=45)
        source.save_async.__globals__['executor'].shutdown(wait=True)
        rt.close()
        result.update(status='COMPLETE_THREE_REAL_CHUNK_FIXTURES_PENDING_OWNED_CLEANUP',
            target_intents=rt.attempted, host_returned=rt.returned, prefix_cache_forwards=18,
            all_valid_KV_RNG_parameters_unchanged=True, baseline_model=prior['frozen_A_off'],
            elapsed_resource_budget_sec=time.monotonic()-origin, elapsed_is_not_a_speed_measurement=True)
    except BaseException as exc:
        result.update(status='FAILED_REAL_FIXTURE_CAPTURE', error=repr(exc),
            traceback=traceback.format_exc(), target_intents=rt.attempted if rt else 0)
        raise
    finally:
        if rt is not None:
            rt.shared.close()
        for h in all_handles: h.remove()
        write(out / 'CAPTURE-RESULT.json', result)


if __name__ == '__main__':
    if os.getenv('WAM_CAPTURE_EXECUTE_REVIEWED') != '1':
        raise SystemExit('capture worker requires the explicit reviewed supervisor')
    entry.run = run
    entry.main()

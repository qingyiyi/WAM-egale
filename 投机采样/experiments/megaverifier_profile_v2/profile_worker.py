"""Nine complete real chunks; candidate-only profile is never a speed result."""
import copy
import sys
import time
import traceback
from pathlib import Path

from mv_profile_contract import worker_entry, write, stamp, digest, load, EXP, S0, WORKTREE, OWN


def middle(s, rt, snap, short, p, out):
    rt.shared.close()
    snap.restore(s, short['state'])
    assert snap.fingerprint(snap.capture(s)) == p['short']['state_hash']
    s._p0_action_draft_mode = 'off'
    artifacts = {(x['frame'], x['kind']): x for x in p['middle_prefix_artifacts']}
    import torch
    start = rt.attempted
    s.exp_save_root = str(out / 'prefix_replay')
    Path(s.exp_save_root).mkdir()
    for frame in (4, 8, 12, 16, 20, 24):
        assert s.frame_st_id == frame
        tensors = {}
        for kind in ('latents', 'actions', 'obs_data'):
            item = artifacts[frame, kind]
            assert digest(item['path']) == item['sha256']
            tensors[kind] = torch.load(item['path'], map_location='cpu', weights_only=False)
        before = rt.attempted
        s.infer(dict(p0_counterfactual_prime_video_cache=True, frame_st_id=frame,
                     video_terminal_state=tensors['latents']))
        physical = s.postprocess_action(tensors['actions'])
        window = tensors['obs_data']
        s.infer(dict(compute_kv_cache=True, obs=window, state=physical))
        assert rt.attempted - before == 3 and s.frame_st_id == frame + 4
    assert rt.attempted - start == 18 and s.frame_st_id == 28
    snap.seed(20260922, True)
    state = snap.capture(s)
    assert len(state['caches']) == 30 and all(c['slots'].numel() == 1008 for c in state['caches'])
    context = copy.deepcopy(short['request']['p0_trace_context'])
    context['fixed_fixture'] = 'T-init3-chunk7'
    request = dict(obs=window[-1], p0_trace_context=context)
    result = dict(state=state, request=request, state_hash=snap.fingerprint(state),
                  request_hash=snap.fingerprint(request))
    write(out / 'MIDDLE-PREFIX.json', dict(status='PASS', forward_intents=18,
        frame=28, history=1008, state_hash=result['state_hash'], request_hash=result['request_hash'],
        source='original short state + recorded frames4,8,12,16,20,24', synthetic_KV=False), exclusive=True)
    return result


def run(s, source, job, out, p):
    import torch
    from runtime import Runtime
    from measured_online import control
    from selection import install
    import buffers
    import state_check
    capture = load('mv_candidate_profile_capture', OWN / 'profile_capture.py')
    snap = load('mv_profile_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    validate = load('mv_profile_validator', S0 / 'server_entry.py')
    validate.validate_server(s)
    s.infer(dict(reset=True, prompt=None))
    s._load_p0_action_draft_b2_predictor().model.to(s.device).eval()
    proxy = install(s, p['selector'])
    buffers.prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
    rt = Runtime(s, out, p['fixed_target_cap'], shared=False, allow_cuda=True)
    short = torch.load(p['short']['path'], map_location='cpu', weights_only=False)
    for key in ('state', 'request'):
        assert snap.fingerprint(short[key]) == p['short'][key + '_hash']
    mid = middle(s, rt, snap, short, p, out)
    save_original, saved = source.save_async, {}
    def save(value, path):
        name = Path(path).name
        key = 'video' if name.startswith('latents_') else 'normalized_action' if name.startswith('actions_') else None
        if key is None:
            raise ValueError('unexpected original policy artifact')
        saved[key] = value
        return save_original(value, path)
    source.save_async = save
    methods = [b.attn1.forward for b in s.transformer.blocks]
    transformer_forward = s.transformer.forward
    details, fixtures = [], []
    observer = None
    try:
        for label, history, frame, total in [('short', 144, 4, 288), ('middle', 1008, 28, 1152), ('long', 2016, 56, 2160)]:
            payload = short if label == 'short' else mid if label == 'middle' else torch.load(p['long']['path'], map_location='cpu', weights_only=False)
            ref = p[label] if label != 'middle' else mid
            for key in ('state', 'request'):
                assert snap.fingerprint(payload[key]) == ref[key + '_hash']
            assert payload['state']['fields']['frame_st_id'] == frame
            assert len(payload['state']['caches']) == 30
            assert all(c['slots'].numel() == history and c['pool_shape'][1] == 2160 for c in payload['state']['caches'])
            expected, end_state, initial, candidate_count = None, None, None, None
            for item in p['fixed_slots']:
                if (job / 'STOP_REQUESTED').exists():
                    raise RuntimeError('supervisor requested stop')
                arm, slot = item['arm'], item['slot']
                optimized = arm != 'RAW_CURRENT'
                rt.shared.close()
                s._p0_action_draft_mode = 'verifier_k2_b2_bonus'
                proxy.effective = s._p0_action_draft_selector = p['selector']
                proxy.records.clear()
                proxy.pending = None
                snap.restore(s, payload['state'])
                if initial is None:
                    assert snap.fingerprint(snap.capture(s)) == ref['state_hash']
                    initial = state_check.references(s.transformer, s.cache_name)
                assert state_check.equal(s.transformer, initial, s.cache_name)
                assert snap.fingerprint(state_check.capture_light(s)) == snap.fingerprint(state_check.light(payload['state']))
                if optimized:
                    rt.shared = control(s, 'O3', allow_cuda=True)
                mode = {'RAW_CURRENT': 'count', 'CURRENT': 'capture', 'PROFILE_CURRENT': 'profile'}[arm]
                indices = () if candidate_count is None else (0, candidate_count - 1)
                observer = capture.CandidateObserver(s, out, label, indices, mode)
                folder = out / 'calls' / label / f'{slot:02d}-{arm}'
                folder.mkdir(parents=True)
                s.exp_save_root = str(folder)
                saved.clear()
                before = dict(rt.phase)
                o3_before = len(buffers.EVENTS)
                request = copy.deepcopy(payload['request'])
                torch.cuda.synchronize()
                start = time.perf_counter()
                try:
                    returned = s.infer(request)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                finally:
                    # Per-candidate forward wrappers have already released;
                    # restore the observer packer before closing online O3.
                    observer.close()
                observation = observer.summary()
                observer = None
                shared = rt.shared.summary() if optimized else dict(enabled=False, reason='raw_same_algorithm_oracle')
                o3_events = copy.deepcopy(buffers.EVENTS[o3_before:])
                rt.shared.close()
                assert s.transformer.forward == transformer_forward
                assert [b.attn1.forward for b in s.transformer.blocks] == methods
                assert set(saved) == {'video', 'normalized_action'} and saved['normalized_action'].shape[1] == 30
                data = dict(saved=snap.cpu(saved), returned=snap.cpu(returned), final_state_light=state_check.capture_light(s))
                phase = {k: v - before.get(k, 0) for k, v in rt.phase.items() if v - before.get(k, 0)}
                assert phase.get('committed_scheduler_update_count') == 50
                assert len(proxy.records) == phase['tree_count'] and proxy.pending is None
                assert observation['candidate_forward_count'] == phase['verifier_forward_count']
                if optimized:
                    assert shared['last_request_clean'] and shared['candidates'] > 0
                    assert len(o3_events) == 1 and all(o3_events[0].get(k) is True for k in ('enabled', 'ok', 'closed'))
                else:
                    assert not o3_events
                identity = dict(data_hash=snap.fingerprint(data), phase=phase,
                    records=copy.deepcopy(proxy.records), forwards=rt.requests[-1]['forward_intents'])
                if expected is None:
                    expected = identity
                    end_state = state_check.references(s.transformer, s.cache_name)
                    candidate_count = observation['candidate_forward_count']
                    assert candidate_count >= 2
                    assert observation['observed_indices'] == []
                else:
                    assert identity == expected, 'same-algorithm output/routes/RNG mismatch'
                    assert observation['candidate_forward_count'] == candidate_count
                    assert observation['observed_indices'] == [0, candidate_count - 1]
                assert state_check.equal(s.transformer, end_state, s.cache_name), 'all30 valid KV mismatch'
                assert observation['closed']
                torch.save(data, out / f'{label}-{arm}-output.pt')
                row = dict(length=label, history=history, verifier_total_KV=total, **item,
                    diagnostic_wall_seconds=elapsed, diagnostic_wall_is_not_speed_measurement=True,
                    request_wall_seconds=rt.requests[-1]['wall_seconds'], **identity,
                    all30_GPU_valid_KV_bytes_equal=True, shared=shared,
                    actual_shared_enabled=optimized, O3_events=o3_events,
                    observation=observation, at=stamp())
                details.append(row)
                write(out / f'{label}-SLOT-{slot:02d}.json', row, exclusive=True, durable=True)
                print(f'PROFILE_CHUNK {label} slot={slot} arm={arm} candidates={candidate_count} mode={mode}', flush=True)
            fixtures.append(dict(length=label, request_history=history, verifier_total_KV=total,
                candidate_forward_count=candidate_count, selected_indices=[0, candidate_count - 1],
                raw_optimized_profiled_equal=True, all30_valid_KV_and_routes_RNG_equal=True))
            del payload, expected, end_state, initial, data
        save_original.__globals__['executor'].shutdown(wait=True)
        torch.cuda.synchronize()
        rt.close()
        profiled = sum(len(d['observation']['artifacts']) for d in details if d['arm'] == 'PROFILE_CURRENT')
        captured = sum(len(d['observation']['artifacts']) for d in details if d['arm'] == 'CURRENT')
        assert len(details) == 9 and profiled == captured == 6
        write(out / 'WORKER-RESULT.json', dict(status='PASS_MEGAVERIFIER_PROFILE', at=stamp(),
            chunks=len(details), profiled_verifier_calls=profiled, captured_block0_calls=captured,
            fixtures=fixtures, details=details, target_intents=rt.attempted,
            raw_current_vs_optimized_profiled_exact=True, all30_valid_KV_and_routes_RNG_equal=True,
            profiling_is_not_a_speed_measurement=True, capture_is_separate_from_profile=True,
            no_new_kernel_no_episode_quality_claim=True), exclusive=True, durable=True)
    finally:
        if observer is not None:
            observer.close()
        rt.shared.close()
        source.save_async = save_original


def main():
    job, out, p = worker_entry()
    sys.path.insert(0, str(WORKTREE))
    from wan_va import wan_va_server as source
    def execute(s, rank, host, port):
        try:
            assert rank == 0
            run(s, source, job, out, p)
        except BaseException as exc:
            write(out / 'WORKER-FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc()), exclusive=True, durable=True)
            raise
    source.run_async_server_mode = execute
    source.init_logger()
    source.main()


if __name__ == '__main__':
    main()

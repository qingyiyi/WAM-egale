"""S04 v2 six full replays; original ordinary scratch rollback is observed."""
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = OWN.parents[2]
sys.path.insert(0, str(EXP / 'wam_shape_audit_v1'))
import full_matrix as entry
from full_matrix import write, load
from observer import Observer, Difference
from private_adapter import control


def run(server, source, out, prior):
    import numpy as np
    import torch
    from runtime import Runtime
    from selection import install, depth, choose
    import buffers
    import state_check
    sys.path.insert(0, str(EXP / 'wam_candidate_tile_v1'))
    import _wam_candidate_tile as candidate
    snap = load('s04_full_chunk_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    recipe = load('s04_full_chunk_middle', EXP / 'megaverifier_profile_v2/profile_worker.py')
    plan = json.loads((out / 'PLAN.json').read_text())
    started = json.loads((out / 'RUN-STARTED.json').read_text())
    result = dict(status='RUNNING', complete_policy_cache_chunks=0, episodes=0, arms=[], pairs=[],
        actual_guarded_path_used=False, performance_claim=False, new_environment_closed_loop=False,
        TF_quality_certificate=False, production_integration_permission=False,
        scope='A_off original snapshots, own physical action and same recorded observations; numerical replay only',
        stop_at_first_difference=True, all_original_failures_preserved=True)
    result['original_FAILED_v1_preserved'] = plan['original_failed_v1']
    result['ordinary_scratch_observer_contract_only_revision'] = True
    rt = None
    observer = None
    closed = False
    try:
        torch.set_num_threads(7)
        for key in ('native_binary', 'candidate_binary'):
            p = Path(plan[key])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == plan['sources'][str(p)]
        assert Path(candidate.__file__).resolve() == Path(plan['candidate_binary']).resolve()
        server.infer(dict(reset=True, prompt=None))
        server._p0_action_draft_mode = 'verifier_k2_b2_bonus'
        predictor = server._load_p0_action_draft_b2_predictor()
        predictor.model.to(server.device).eval()
        assert server._p0_state_teacher is None
        assert float(server._p0_action_draft_threshold) == prior['frozen_reference']['threshold']
        proxy = install(server, 'prefix_then_cost')
        buffers.prepare(server._infer_action_verifier_k2_b2, server._load_p0_shadow_api())
        rt = Runtime(server, out, 580, shared=False, allow_cuda=True)

        def check():
            if time.monotonic() >= started['active_deadline_monotonic']:
                raise TimeoutError('780s work cap, sixty-second cleanup reserve')
            if (out / 'STOP_REQUESTED').exists(): raise RuntimeError('STOP_REQUESTED')
            assert rt.attempted <= 580
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 116 * 2**20:
                raise RuntimeError('116MiB output cap')
            old_bytes = sum(p.stat().st_size for p in Path(plan['original_failed_v1']).rglob('*') if p.is_file())
            new_bytes = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            if old_bytes + new_bytes > 128 * 2**20:
                raise RuntimeError('cumulative v1+v2 128MiB output cap')

        def drain():
            # Original save_async uses one FIFO executor; no global monkeypatch.
            source.save_async.__globals__['executor'].submit(lambda: None).result(timeout=30)

        def full_state():
            return snap.fingerprint(snap.capture(server))

        short = torch.load(prior['short']['path'], map_location='cpu', weights_only=False)
        for k in ('state', 'request'): assert snap.fingerprint(short[k]) == prior['short'][k + '_hash']
        before = rt.attempted
        middle = recipe.middle(server, rt, snap, short, prior, out)
        assert rt.attempted - before == 18
        drain()
        qualified_middle = json.loads((Path(plan['qualified_fixture_root']) / 'MIDDLE-PREFIX.json').read_text())
        assert middle['state_hash'] == qualified_middle['state_hash'] and middle['request_hash'] == qualified_middle['request_hash']
        result['middle_preparation_target_forwards'] = 18
        for label, frame, history in (('short', 4, 144), ('middle', 28, 1008), ('long', 56, 2016)):
            check()
            payload = short if label == 'short' else middle if label == 'middle' else torch.load(prior['long']['path'], map_location='cpu', weights_only=False)
            ref = middle if label == 'middle' else prior[label]
            for k in ('state', 'request'): assert snap.fingerprint(payload[k]) == ref[k + '_hash']
            obs_item = next(x for x in plan['post_policy_observations'] if x['frame'] == frame)
            obs = torch.load(obs_item['path'], map_location='cpu', weights_only=False)
            assert len(obs) == 16
            native_events = None
            native_summary = None
            for arm in ('native', 'guarded'):
                check()
                rt.shared.close()
                snap.restore(server, payload['state'])
                assert full_state() == ref['state_hash']
                assert server.frame_st_id == frame
                assert all(c['slots'].numel() == history for c in payload['state']['caches'])
                server._p0_action_draft_mode = 'verifier_k2_b2_bonus'
                server._p0_action_draft_selector = proxy.effective = 'prefix_then_cost'
                proxy.records.clear()
                proxy.pending = None
                folder = out / label / arm
                folder.mkdir(parents=True)
                server.exp_save_root = str(folder / 'policy')
                Path(server.exp_save_root).mkdir()
                count, phases = rt.attempted, dict(rt.phase)
                event_start = len(buffers.EVENTS)
                observer = Observer(server, snap, state_check, check, reference=native_events)
                rt.shared = control(server, arm, observer, candidate)
                complete = False
                try:
                    returned = server.infer(copy.deepcopy(payload['request']))
                    drain()
                    assert set(returned) == {'action'}
                    raw_actions = torch.load(Path(server.exp_save_root) / f'actions_{frame}.pt', map_location='cpu', weights_only=False)
                    video = torch.load(Path(server.exp_save_root) / f'latents_{frame}.pt', map_location='cpu', weights_only=False)
                    assert bool(np.isfinite(returned['action']).all()), 'actual physical action nonfinite'
                    assert bool(torch.isfinite(raw_actions).all()) and bool(torch.isfinite(video).all()), 'actual saved action/video nonfinite'
                    assert snap.fingerprint(server.postprocess_action(raw_actions)) == snap.fingerprint(returned['action'])
                    # Observer owns a temporary non-numerical bound step method,
                    # excluded explicitly from its otherwise complete light state.
                    observer.event('policy_final_state', dict(light=snap.fingerprint(observer.light()),
                        physical_action=snap.fingerprint(returned['action']),
                        normalized_action=snap.fingerprint(raw_actions), final_video=snap.fingerprint(video)))
                    policy_queries = observer.query_count
                    records = copy.deepcopy(proxy.records)
                    assert proxy.pending is None and records
                    for record in records:
                        assert all(math.isfinite(float(v)) and float(v) >= 0 for v in
                            [*record['first_costs'], *record['second_costs'][0], *record['second_costs'][1]]), 'actual verifier cost nonfinite or negative'
                        assert record['threshold'] == prior['frozen_reference']['threshold']
                        assert record['actual_depth'] == depth(record['first_costs'][record['actual_parent']],
                            record['second_costs'][record['actual_parent']][record['actual_child']], record['threshold'])
                        assert record['actual_depth'] == record['maximum_admissible_depth']
                    observer.event('actual_FP32_costs_routes_depth_bonus_contract', records)
                    observer.change_stage()
                    cache_input = dict(compute_kv_cache=True, obs=copy.deepcopy(obs), state=returned['action'])
                    observer.event('actual_own_action_recorded_observation_cache_request', dict(hash=snap.fingerprint(cache_input),
                        recorded_observation_hash=snap.fingerprint(obs)))
                    server.infer(cache_input)
                    drain()
                    assert server.frame_st_id == frame + 4
                    assert snap.fingerprint(torch.load(Path(server.exp_save_root) / f'obs_data_{frame}.pt', map_location='cpu', weights_only=False)) == snap.fingerprint(obs)
                    observer.event('cache_final_state', dict(light=snap.fingerprint(observer.light()),
                        all30_valid_KV=snap.fingerprint([snap.capture_cache(b.attn1.attn_caches[server.cache_name]) for b in server.transformer.blocks])))
                    assert len(observer.commits) == 50
                    complete = True
                finally:
                    rt.shared.close()
                    observer.close(complete=complete)
                after_state = full_state()
                phases = {k: v - phases.get(k, 0) for k, v in rt.phase.items() if v - phases.get(k, 0)}
                assert phases['committed_scheduler_update_count'] == 50 and phases['tree_count'] == len(records)
                assert observer.candidate_queries == len(records)
                workspace = buffers.EVENTS[event_start:]
                assert len(workspace) == 1 and workspace[0]['enabled'] and workspace[0]['ok'] and workspace[0]['closed']
                assert rt.shared.control is None and rt.shared.last_clean
                summary = dict(length=label, arm=arm, frame=frame, final_frame=server.frame_st_id,
                    policy_queries=policy_queries, cache_queries=2, target_forwards=rt.attempted - count,
                    phase=phases, actual_commits=len(observer.commits), actual_scheduler_transitions=len(observer.transitions),
                    candidate_B14_queries=observer.candidate_queries, actual_shared_layer_calls=observer.attention_calls,
                    all_valid_full_state_hash=after_state, event_count=len(observer.events),
                    events_hash=snap.fingerprint(observer.events), routes_hash=snap.fingerprint(records),
                    workspace=workspace, online_lifecycle=rt.shared.summary())
                write(folder / 'EVENTS.json', observer.events)
                write(folder / 'COMMITTED-50.json', observer.commits)
                write(folder / 'SCHEDULER-TRANSITIONS.json', observer.transitions)
                write(folder / 'ACCEPTANCE-RECORDS.json', records)
                write(folder / 'CACHE-FIELD-DIFFERENCES.json', observer.field_differences)
                write(folder / 'CHUNK.json', {**summary, 'actual_guarded_groups': dict(observer.groups),
                    'shared_attention_returned': 'original native' if arm == 'native' else 'actual guarded G1/G2 at every B14 layer'})
                if arm == 'native':
                    fixture = torch.load(Path(plan['qualified_fixture_root']) / label / 'real-verifier-fixture.pt',
                        map_location='cpu', weights_only=False, mmap=True)
                    candidate_inputs = [e['value'] for e in observer.events
                        if e['kind'] == 'target_input' and e['value']['candidate']]
                    target_outputs = {e['value']['index']: e['value'] for e in observer.events
                        if e['kind'] == 'target_output'}
                    assert len(candidate_inputs) == len(records)
                    for query in fixture['queries']:
                        captured = candidate_inputs[query['index']]
                        assert captured['input_hash'] == snap.fingerprint(((query['input'],), query['kwargs']))
                        assert target_outputs[captured['index']]['output_hash'] == snap.fingerprint(query['output'])
                    del fixture
                    native_events, native_summary = observer.events, summary
                else:
                    expected = {**native_summary, 'arm': 'guarded'}
                    assert summary == expected, 'full policy/cache numerical/counter/state summary mismatch'
                    assert observer.groups and sum(observer.groups.values()) == observer.attention_calls
                    result['pairs'].append(dict(length=label, initial_KV=history, verifier_KV=history + 144,
                        baseline_all_queries_preserved=True, actual_guarded_every_B14=True,
                        complete_policy_cache_exact=True, actual_commits=50, final_state=after_state))
                    result['actual_guarded_path_used'] = True
                result['arms'].append(summary)
                result['complete_policy_cache_chunks'] += 1
                observer = None
                gc.collect()
                torch.cuda.empty_cache()
                print(json.dumps(dict(length=label, arm=arm, status='COMPLETE_FULL_POLICY_CACHE_NUMERICAL_REPLAY',
                    commits=50, target_forwards=summary['target_forwards'])), flush=True)
            del payload, native_events
        assert len(result['pairs']) == 3 and result['complete_policy_cache_chunks'] == 6
        assert rt.attempted <= 580
        rt.close()
        closed = True
        drain()
        source.save_async.__globals__['executor'].shutdown(wait=True)
        result.update(status='COMPLETE_THREE_LENGTH_FULL_POLICY_CACHE_EXACT_PENDING_CLEANUP',
            total_target_forwards=rt.attempted, complete_50_update_sequences=6,
            all_target_inputs_outputs_exact=True, all_costs_routes_bonus_actual_commits_exact=True,
            all_valid_KV_RNG_scheduler_VAE_server_state_exact=True,
            original_history_not_polluted=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        result.update(status='STOPPED_AT_FIRST_DIFFERENCE' if isinstance(exc, Difference) else 'FAILED',
            error=repr(exc), traceback=traceback.format_exc(),
            first_difference=None if observer is None else observer.diff,
            total_target_forwards=0 if rt is None else rt.attempted)
        raise
    finally:
        if observer is not None:
            try:
                write(out / 'PARTIAL-OBSERVED-EVENTS.json', observer.events)
                write(out / 'PARTIAL-ACTUAL-COMMITS.json', observer.commits)
                write(out / 'PARTIAL-CACHE-FIELD-DIFFERENCES.json', observer.field_differences)
            except BaseException: pass
            try: observer.close(complete=False)
            except BaseException: pass
        if rt is not None and not closed:
            try:
                rt.shared.close()
                rt.close()
            except BaseException: pass
        write(out / 'CHUNK-RESULT.json', result)


if __name__ == '__main__':
    entry.run = run
    entry.main()

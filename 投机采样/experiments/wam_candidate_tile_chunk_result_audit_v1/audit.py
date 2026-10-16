"""Independent CPU audit of an actually terminal S04 v2; never launches work."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / '投机采样/experiments'
RUNTIME = ROOT / '投机采样/results/runtime'
JOB = RUNTIME / 'p0-wam-candidate-tile-chunk-correctness-v2-20261016-r1'


def read(p):
    return json.loads(Path(p).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def depth(first, second, tau): return 0 if first > tau else 1 if second > tau else 2


def select(row):
    pairs = []
    tau = row['threshold']
    for p in range(2):
        for c in range(2):
            a, b = row['first_costs'][p], row['second_costs'][p][c]
            assert math.isfinite(a) and math.isfinite(b) and min(a, b) >= 0
            assert struct.unpack('<f', struct.pack('<f', a))[0] == a
            assert struct.unpack('<f', struct.pack('<f', b))[0] == b
            d = depth(a, b, tau)
            pairs.append((p, c, a, b, d))
    if max(x[4] for x in pairs) == 0:
        return min(pairs, key=lambda x: (x[2] + x[3], x[0], x[1]))
    return min(pairs, key=lambda x: (-x[4], x[2] + x[3] if x[4] == 2 else x[2] if x[4] == 1 else 0., x[0], x[1]))


def audit(job, out):
    start = time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert Path(job).resolve() == JOB
    terminal = read(job / 'SUPERVISOR-RESULT.json')
    worker = read(job / 'CHUNK-RESULT.json')
    assert terminal['own_process_reaped'] and terminal['all_owned_live_exited']
    assert terminal['status'] == 'COMPLETE_THREE_LENGTH_FULL_POLICY_CACHE_EXACT' and terminal['failure'] is None
    assert worker['status'] == 'COMPLETE_THREE_LENGTH_FULL_POLICY_CACHE_EXACT_PENDING_CLEANUP'
    assert not list(job.rglob('*FAILED*.json')) and not (job / 'STOP_REQUESTED').exists()
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(EXP / 'wam_candidate_tile_chunk_correctness_v2'))
    import run_chunk
    plan = run_chunk.load(job)
    sys.path.insert(0, str(EXP / 'fixed_load_cost'))
    import snapshot as snap
    assert not torch.cuda.is_initialized()
    assert worker['complete_policy_cache_chunks'] == 6 and len(worker['arms']) == 6 and len(worker['pairs']) == 3
    assert worker['complete_50_update_sequences'] == 6 and worker['actual_guarded_path_used']
    assert worker['middle_preparation_target_forwards'] == 18
    assert worker['all_target_inputs_outputs_exact'] and worker['all_costs_routes_bonus_actual_commits_exact']
    assert worker['all_valid_KV_RNG_scheduler_VAE_server_state_exact'] and worker['original_history_not_polluted']
    assert not worker['performance_claim'] and not worker['production_integration_permission']
    assert not worker['new_environment_closed_loop'] and not worker['TF_quality_certificate']
    summaries = []
    receipts = {}
    scratch_versions = Counter()
    total_queries = total_candidates = total_attention = total_commits = total_transitions = 0
    old = Path(plan['original_failed_v1'])
    old_terminal = read(old / 'SUPERVISOR-RESULT.json')
    assert read(old / 'CHUNK-RESULT.json')['status'] == 'FAILED'
    assert old_terminal['status'] == 'FAILED_OR_STOPPED' and old_terminal['all_owned_live_exited']
    for label, frame in (('short', 4), ('middle', 28), ('long', 56)):
        saved = {}
        for arm in ('native', 'guarded'):
            folder = job / label / arm
            names = ['CHUNK.json', 'EVENTS.json', 'COMMITTED-50.json', 'SCHEDULER-TRANSITIONS.json',
                     'ACCEPTANCE-RECORDS.json', 'CACHE-FIELD-DIFFERENCES.json']
            chunk, events, commits, transitions, records, scratch = [read(folder / n) for n in names]
            for n in names: receipts[str(folder / n)] = sha(folder / n)
            assert chunk['length'] == label and chunk['arm'] == arm and chunk['frame'] == frame and chunk['final_frame'] == frame + 4
            assert chunk['actual_commits'] == len(commits) == 50
            assert [r['step'] for r in commits] == list(range(50))
            assert [r['index'] for r in transitions] == list(range(len(transitions)))
            assert all(commits[i]['projected_compact_next_hash'] == commits[i + 1]['state_hash'] for i in range(49))
            assert all(commits[i]['next_sigma'] == commits[i + 1]['sigma'] for i in range(49))
            assert commits[-1]['next_sigma'] == 0.
            for row in commits:
                assert row['source'] in ('draft', 'target') and (not row['bonus'] or row['source'] == 'target')
                assert row['actual_transition_indices']
                for index in row['actual_transition_indices']:
                    step = transitions[index]
                    for key in ('state_hash', 'direction_hash', 'timestep', 'raw_full_next_hash',
                                'projected_full_next_hash', 'projected_compact_next_hash'):
                        assert row[key] == step[key]
            accepted = Counter()
            for row in records:
                assert row['threshold'] == plan['threshold'] and row['effective_selector'] == plan['selector']
                chosen = select(row)
                assert (row['actual_parent'], row['actual_child'], row['actual_depth']) == (chosen[0], chosen[1], chosen[4])
                assert row['actual_depth'] == row['maximum_admissible_depth']
                accepted[row['actual_depth']] += 1
            source_counts = Counter(r['source'] for r in commits)
            bonus = sum(r['bonus'] for r in commits)
            assert source_counts['draft'] == accepted[1] + 2 * accepted[2]
            assert bonus == accepted[2]
            phase = chunk['phase']
            assert phase['committed_scheduler_update_count'] == 50 and phase['tree_count'] == len(records)
            assert all(phase.get('accepted_depth_' + str(d), 0) == accepted[d] for d in range(3))
            assert phase.get('bonus_target_tree_count', 0) == bonus
            assert phase['tree_scheduler_transition_count'] == len(transitions)
            assert phase.get('committed_draft_scheduler_update_count', 0) == source_counts['draft']
            assert phase.get('bonus_target_scheduler_update_count', 0) == bonus
            ordinary_steps = 50 - accepted[0] - 2 * accepted[1] - 3 * accepted[2]
            assert len(transitions) == 7 * len(records) + ordinary_steps
            kinds = Counter(r['kind'] for r in events)
            inputs = [r['value'] for r in events if r['kind'] == 'target_input']
            outputs = [r['value'] for r in events if r['kind'] == 'target_output']
            assert len(inputs) == len(outputs) == chunk['target_forwards']
            assert [r['index'] for r in inputs] == [r['index'] for r in outputs] == list(range(len(inputs)))
            assert sum(r['stage'] == 'policy' for r in inputs) == chunk['policy_queries']
            assert sum(r['stage'] == 'cache' for r in inputs) == chunk['cache_queries'] == 2
            assert [r['update'] for r in inputs if r['stage'] == 'cache'] == [2, 2]
            assert all(a['RNG'] == b['RNG'] and a['stage'] == b['stage'] for a, b in zip(inputs, outputs))
            candidates = [r for r in inputs if r['candidate']]
            assert len(candidates) == chunk['candidate_B14_queries'] == len(records)
            assert all(r['update'] == 0 and r['input_layout'][0][0]['noisy_latents']['shape'][0] == 14 for r in candidates)
            assert kinds['candidate_attention_input'] == kinds['candidate_attention_output'] == 30 * len(candidates) == chunk['actual_shared_layer_calls']
            assert [r['value'] for r in events if r['kind'] == 'actual_committed_scheduler_update'] == commits
            assert [r['value'] for r in events if r['kind'] == 'actual_action_scheduler_projection'] == transitions
            assert kinds['actual_action_scheduler_step'] == len(transitions)
            ordinary_queries = [r for r in inputs if r['update'] == 0 and not r['candidate']]
            assert len(scratch) == kinds['ordinary_original_scratch_effects'] == len(ordinary_queries)
            assert [r['value'] for r in events if r['kind'] == 'ordinary_original_scratch_effects'] == scratch
            for row in scratch:
                assert len(row['layers']) == 30
                for layer in row['layers']:
                    assert layer['valid_KV']['exact'] and not layer['invalid_KV_bytes_read']
                    assert layer['metadata']['mask']['different'] == 0
                    assert all(layer['metadata'][k]['different_at_original_valid'] == 0 for k in ('id', 'is_pred'))
                    scratch_versions[tuple(sorted(layer['version_deltas'].items()))] += 1
            assert snap.fingerprint(events) == chunk['events_hash'] and snap.fingerprint(records) == chunk['routes_hash']
            assert chunk['event_count'] == len(events)
            assert len(chunk['workspace']) == 1 and all(chunk['workspace'][0][k] for k in ('enabled', 'ok', 'closed'))
            assert chunk['online_lifecycle']['last_request_clean'] and not chunk['online_lifecycle']['request_active']
            raw = torch.load(folder / 'policy' / f'actions_{frame}.pt', map_location='cpu', weights_only=False)
            video = torch.load(folder / 'policy' / f'latents_{frame}.pt', map_location='cpu', weights_only=False)
            observed = torch.load(folder / 'policy' / f'obs_data_{frame}.pt', map_location='cpu', weights_only=False)
            assert bool(torch.isfinite(raw).all()) and bool(torch.isfinite(video).all())
            final = next(r['value'] for r in events if r['kind'] == 'policy_final_state')
            assert snap.fingerprint(raw) == final['normalized_action'] and snap.fingerprint(video) == final['final_video']
            original_obs = torch.load(next(r['path'] for r in plan['post_policy_observations'] if r['frame'] == frame),
                map_location='cpu', weights_only=False)
            assert snap.fingerprint(observed) == snap.fingerprint(original_obs)
            handoff = next(r['value'] for r in events if r['kind'] == 'actual_own_action_recorded_observation_cache_request')
            assert handoff['recorded_observation_hash'] == snap.fingerprint(observed)
            fixture = torch.load(Path(plan['qualified_fixture_root']) / label / 'real-verifier-fixture.pt',
                map_location='cpu', weights_only=False, mmap=True)
            for query in fixture['queries']:
                actual = candidates[query['index']]
                assert actual['input_hash'] == snap.fingerprint(((query['input'],), query['kwargs']))
                assert outputs[actual['index']]['output_hash'] == snap.fingerprint(query['output'])
            del fixture
            if arm == 'guarded':
                assert chunk['actual_guarded_groups'] and sum(chunk['actual_guarded_groups'].values()) == chunk['actual_shared_layer_calls']
                assert events == saved['native']['events']
                assert commits == saved['native']['commits'] and transitions == saved['native']['transitions']
                assert records == saved['native']['records'] and scratch == saved['native']['scratch']
                assert torch.equal(raw.contiguous().view(torch.uint8), saved['native']['raw'].contiguous().view(torch.uint8))
                assert torch.equal(video.contiguous().view(torch.uint8), saved['native']['video'].contiguous().view(torch.uint8))
                assert chunk['all_valid_full_state_hash'] == saved['native']['chunk']['all_valid_full_state_hash']
            saved[arm] = dict(events=events, commits=commits, transitions=transitions, records=records,
                scratch=scratch, chunk=chunk, raw=raw, video=video)
            total_queries += len(inputs)
            total_candidates += len(candidates)
            total_attention += chunk['actual_shared_layer_calls']
            total_commits += len(commits)
            total_transitions += len(transitions)
            summaries.append(dict(length=label, arm=arm, target_forwards=len(inputs),
                candidate_B14=len(candidates), shared_attention_layers=chunk['actual_shared_layer_calls'],
                actual_commits=50, actual_scheduler_transitions=len(transitions), accepted_depth=dict(accepted),
                bonus_steps=bonus, ordinary_scratch_queries=len(scratch),
                guarded_groups=chunk['actual_guarded_groups'], frozen_firstlast_actual=True,
                complete_policy_cache_exact=True))
    counter = read(job / 'COUNTER-CLOSED.json')
    assert counter['target_intents'] == counter['host_returned'] == total_queries + 18 == worker['total_target_forwards']
    assert counter['failed_forwards'] == counter['active_forwards'] == 0 and not counter['request_active']
    assert total_commits == 300
    new_bytes = sum(p.stat().st_size for p in job.rglob('*') if p.is_file())
    old_bytes = sum(p.stat().st_size for p in old.rglob('*') if p.is_file())
    assert new_bytes <= 116 * 2**20 and new_bytes + old_bytes <= 128 * 2**20
    assert terminal['GPU_wall_sec'] <= 840 and terminal['GPU_wall_sec'] + old_terminal['GPU_wall_sec'] <= 900
    assert worker['total_target_forwards'] <= 580 and worker['total_target_forwards'] + 19 <= 600
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS_INDEPENDENT_COMPLETE_S04_V2_TERMINAL_EVIDENCE_AUDIT', job=str(job),
        GPU_used=False, NVML_used=False, SSH_used=False, CPU_affinity=sorted(os.sched_getaffinity(0)),
        CPU_threads=torch.get_num_threads(), CUDA_initialized=False, elapsed_seconds=time.monotonic()-start,
        six_complete_policy_cache_chunks=True, actual_commits=300, exact_pairs=3,
        policy_cache_target_forwards=total_queries, middle_preparation_forwards=18,
        total_target_forwards=total_queries+18, candidate_B14_queries=total_candidates,
        actual_shared_attention_layer_calls=total_attention, scheduler_transition_calls=total_transitions,
        scratch_version_delta_patterns=[dict(pattern=dict(k), layer_queries=v) for k,v in scratch_versions.items()],
        rows=summaries, sourcefreeze_pass=True, firstlast_real_fixtures_verified=True,
        all_recorded_query_attention_scheduler_RNG_state_events_identical=True,
        saved_action_video_bytes_exact_and_finite=True, recorded_observations_exact=True,
        recorded_acceptance_rule_violations=0, cleanup_complete=True,
        v1_FAILED_preserved=True, v2_GPU_seconds=terminal['GPU_wall_sec'],
        cumulative_GPU_seconds=terminal['GPU_wall_sec']+old_terminal['GPU_wall_sec'],
        v1_bytes=old_bytes, v2_bytes=new_bytes, cumulative_bytes=old_bytes+new_bytes,
        no_new_environment_closed_loop=True, no_performance_result=True, no_TF_quality_certificate=True,
        limits=['per-query values are verified from actual frozen-source hash records; this audit does not execute GPU',
                'raw per-transition tensors are not stored separately; Euler math was not independently recomputed',
                'cost tables are logged at policy end; first recorded event difference stops execution'],
        receipts=receipts)
    out.mkdir(exist_ok=False)
    (out/'REVIEW-RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,default=JOB);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not (a.job/'SUPERVISOR-RESULT.json').exists():
        print('WAIT_REAL_TERMINAL_AND_CLEANUP');return
    result=audit(a.job,a.out)
    print(json.dumps({k:result[k] for k in ('status','actual_commits','total_target_forwards','v2_GPU_seconds','cumulative_GPU_seconds','cumulative_bytes')},ensure_ascii=False))


if __name__=='__main__':main()

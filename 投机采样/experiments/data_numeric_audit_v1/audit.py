"""CPU-only numerical audit of historical Long TRAIN194/CAL50 and same-query aux.

Read trusted local tensors; never start an environment, model, CUDA or network.
Historical target-only rows remain distinct from current verifier candidate labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert set(os.sched_getaffinity(0)) <= {14, 15}
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

ROOT = Path('/rjs/cxz/论文探索')
OUT = ROOT / '投机采样/results/runtime/p0-data-numeric-audit-v1-20261015-r1'
MANIFEST = ROOT / '.pipeline/research/speculative-sampling/evidence/old-targetonly-source-audit-20261015/MANIFEST.json'
AUX = Path('/var/tmp/cxz-wam-quality-20261014/p0-morning-long10-train-six-v1-20261015-r1/dataset.pt')
THRESHOLD = 0.007839680526450805
RESERVED = {(t, i) for t in (0, 1) for i in (6, 7, 8)}


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def tcheck(value, shape, dtype):
    require(isinstance(value, torch.Tensor), 'missing tensor')
    require(tuple(value.shape) == tuple(shape) and value.dtype == dtype, 'tensor shape/dtype differs')
    require(value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'nonfinite or non-CPU tensor')


def compact(tensor):
    return tensor[0, :7].float().reshape(112)


def equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def source_paths(e):
    base = ROOT / e['trace_root']
    return [(base / f'chunk_{i:06d}_frame_{4*j:06d}.pt', j)
            for j, i in enumerate(range(e['trace_chunk_start'], e['trace_chunk_end_exclusive']))]


def audit_history(manifest):
    selected = [e for e in manifest['episodes'] if e['split'] in ('train_candidate', 'cal_history')]
    n = sum(e['infer_chunks'] * 49 for e in selected if e['split'] == 'train_candidate')
    require(n == 179928, 'TRAIN194 expected row count changed')
    columns = {k: torch.empty((n, 112), dtype=torch.float32) for k in ('state', 'previous', 'target')}
    columns['scalars'] = torch.empty((n, 3), dtype=torch.float32)
    columns['delta'] = torch.empty(n, dtype=torch.float32)
    for k in ('task', 'init', 'episode', 'rollout_chunk', 'frame', 'step', 'source_id'):
        columns[k] = torch.empty(n, dtype=torch.int32)
    views = {'fit174': [], 'cal20': []}
    sources, episodes = [], []
    position = 0
    totals = Counter()
    maxes = defaultdict(float)
    prompts = {}
    full_mismatch = Counter()
    for ei, e in enumerate(selected):
        require((e['task'], e['init']) not in RESERVED, 'reserved DEV tensor selected')
        train = e['split'] == 'train_candidate'
        ep_count = Counter()
        ep_start = position
        for path, local_chunk in source_paths(e):
            stat = path.stat()
            trace = load(path)
            c = trace['chunk']
            require(trace['schema_version'] == 'wam-p0-trace/v3' and not trace['capture_errors'], 'trace schema/error differs')
            require(trace['draft_training_contract_version'] == 'wam-p0-action-draft/v2-transformer-output', 'wrong training trace contract')
            require(trace['run']['trace_mode'] == 'draft_train_v2_transformer_output', 'wrong trace mode')
            env = trace['run']['environment']
            require(env['LINGBOT_VA_LIBERO_CHECKPOINT_ID'] == 'robbyant/lingbot-va-posttrain-libero-long', 'wrong teacher checkpoint id')
            require(c['libero_benchmark'] == 'libero_10' and c['task_index'] == e['task'] and c['episode_index'] == e['init'], 'task/init trace binding mismatch')
            require(c['rollout_chunk'] == local_chunk and c['frame_st_id'] == local_chunk * 4, 'chunk/frame binding mismatch')
            require(c['cache_mode'] == 'original' and c['use_cfg'] is True, 'cache/CFG contract differs')
            act, video = c['action'], c['video']
            require(act['inference_updates'] == 50 and video['inference_updates'] == 20, 'action/video schedule differs')
            require(act['cfg_scale'] == 1. and video['cfg_scale'] == 5. and act['snr_shift'] == .05, 'CFG/shift differs')
            require(act['used_action_channel_ids'] == list(range(7)) and act['action_dim'] == 30 and act['action_per_frame'] == 4, 'action contract differs')
            condition = trace['condition']
            require(condition['prompt'] == c['prompt'] and condition['target_hidden_state'] == condition['target_kv_cache'] == 'not_captured', 'condition binding differs')
            if e['task'] in prompts:
                require(prompts[e['task']] == c['prompt'], 'task prompt changed')
            prompts[e['task']] = c['prompt']
            require(set(condition['observation']) == set(condition['observation_camera_keys']) and len(condition['observation']) == 2, 'camera condition differs')
            for image in condition['observation'].values():
                tcheck(image, (128, 128, 3), torch.uint8)
            prefix = condition['action_condition']
            if local_chunk == 0:
                tcheck(prefix, (1, 30, 1, 4, 1), torch.bfloat16)
                require(not bool(prefix.any()), 'initial observed action prefix is not zero')
            else:
                require(prefix is None, 'unexpected later action prefix')
            updates = trace['steps']
            require(len(updates) == 50 and [r['step_index'] for r in updates] == list(range(50)), 'not complete 50 action updates')
            if train:
                sid = len(sources)
                sources.append(dict(path=str(path), task=e['task'], init=e['init'], rollout_chunk=local_chunk,
                                    frame=local_chunk*4, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                    prompt=c['prompt'], row_start=position, row_count=49,
                                    checkpoint_id=env['LINGBOT_VA_LIBERO_CHECKPOINT_ID'],
                                    checkpoint_path=env.get('LINGBOT_VA_LIBERO_CHECKPOINT')))
            for i, row in enumerate(updates):
                require(row['kind'] == 'scheduler_update' and row['branch'] == 'action' and row['model_source'] == 'target', 'not target action scheduler update')
                require(row['teacher_label'] == 'cfg_combined_direction' and row['cfg_scale'] == 1. and row['cfg_active'] is False
                        and row['cache_update'] is False and row['scheduler_updated'] is True, 'direction label/CFG semantics differ')
                for key in ('x_i', 'd_cfg', 'x_next'):
                    tcheck(row[key], (1, 30, 4, 4, 1), torch.bfloat16)
                tcheck(row['transformer_output_raw'], (2, 16, 30), torch.bfloat16)
                raw = row['transformer_output_raw'][0:1].reshape(1, 4, 4, 30).permute(0, 3, 1, 2).unsqueeze(-1)
                require(equal(raw, row['d_cfg']), 'raw conditional output != recorded action CFG1 direction')
                sigma, nxt, timestep = (float(row[k]) for k in ('sigma', 'next_sigma', 'timestep'))
                require(all(math.isfinite(v) for v in (sigma, nxt, timestep)) and 0 <= nxt < sigma <= 1.00001, 'invalid schedule scalar')
                require(math.isclose(float(row['transition_scale']), nxt-sigma, rel_tol=1e-6, abs_tol=1e-8), 'transition step mismatch')
                require(math.isclose(sigma, float(act['scheduler_sigmas'][i]), abs_tol=1e-7)
                        and math.isclose(timestep, float(act['loop_timesteps'][i]), abs_tol=1e-5), 'actual schedule binding differs')
                maxes['timestep_minus_1000sigma_abs'] = max(maxes['timestep_minus_1000sigma_abs'], abs(timestep-1000*sigma))
                # FlowMatchScheduler: BF16 arithmetic at each operator, followed by observed-prefix clamp.
                predicted = row['x_i'] + row['d_cfg'] * (nxt-sigma)
                if prefix is not None:
                    predicted[:, :, :1] = prefix
                delta = (predicted.float()-row['x_next'].float()).abs()
                maxes['full_euler_reconstruction_abs'] = max(maxes['full_euler_reconstruction_abs'], float(delta.max()))
                if not equal(predicted, row['x_next']):
                    full_mismatch['euler_full_rows'] += 1
                compact_delta = delta[0, :7]
                maxes['compact_euler_reconstruction_abs'] = max(maxes['compact_euler_reconstruction_abs'], float(compact_delta.max()))
                if bool(compact_delta.any()):
                    full_mismatch['euler_compact_rows'] += 1
                if i:
                    left = updates[i-1]
                    require(equal(compact(left['x_next']), compact(row['x_i'])), 'used-channel transition discontinuity')
                    maxes['unused_channel_preparation_abs'] = max(maxes['unused_channel_preparation_abs'], float((left['x_next'].float()-row['x_i'].float()).abs().max()))
                    if not equal(left['x_next'], row['x_i']):
                        full_mismatch['full_transition_preparation_rows'] += 1
                    if train:
                        columns['state'][position] = compact(row['x_i'])
                        columns['previous'][position] = compact(left['d_cfg'])
                        columns['target'][position] = compact(row['d_cfg'])
                        columns['scalars'][position] = torch.tensor([sigma, nxt, timestep/1000.])
                        columns['delta'][position] = abs(nxt-sigma)
                        for key, value in dict(task=e['task'], init=e['init'], episode=e['task']*100+e['init'],
                                               rollout_chunk=local_chunk, frame=local_chunk*4, step=i, source_id=sid).items():
                            columns[key][position] = value
                        views['cal20' if e['init'] in (18, 19) else 'fit174'].append(position)
                        position += 1
                    ep_count['supervised_rows'] += 1
                ep_count['updates'] += 1
            action_file = ROOT / e['server_dir'] / f'actions_{4*local_chunk}.pt'
            actions = load(action_file)
            tcheck(actions, (1, 30, 4, 4, 1), torch.bfloat16)
            require(equal(compact(actions), compact(updates[-1]['x_next'])), 'used-channel server action endpoint != final recorded update')
            if not equal(actions, updates[-1]['x_next']):
                full_mismatch['endpoint_unused_channel_preparation_chunks'] += 1
            latent_file = ROOT / e['server_dir'] / f'latents_{4*local_chunk}.pt'
            latents = load(latent_file)
            tcheck(latents, (1, 48, 4, 8, 16), torch.bfloat16)
            require((path.stat().st_size, path.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns), 'input source mutated')
            ep_count['chunks'] += 1
            ep_count['source_trace_bytes'] += stat.st_size
        split = 'cal20' if train and e['init'] in (18, 19) else 'fit174' if train else 'cal50_audit_only'
        for key, value in ep_count.items():
            totals[split + ':' + key] += value
        episodes.append(dict(task=e['task'], init=e['init'], use=split, success=e['success'],
                             row_start=ep_start if train else None,
                             row_count=ep_count['supervised_rows'], chunks=ep_count['chunks']))
        if ei % 10 == 0 or ei+1 == len(selected):
            write('status.json', dict(status='RUNNING_HISTORY_AUDIT', at=stamp(), episodes=ei+1,
                                      total_episodes=len(selected), exported_rows=position, totals=dict(totals)))
    require(position == n and len(sources) == 3672, 'export cardinality differs')
    require(full_mismatch['euler_compact_rows'] == 0, 'Euler compact reconstruction failed')
    views = {k: torch.tensor(v, dtype=torch.int64) for k, v in views.items()}
    require(len({(e['task'], e['init']) for e in episodes if e['use']=='fit174'}) == 174 and
            len({(e['task'], e['init']) for e in episodes if e['use']=='cal20'}) == 20 and
            len({(e['task'], e['init']) for e in episodes if e['use']=='cal50_audit_only'}) == 50, 'split counts differ')
    pack = dict(schema='historical-target-only-long-train194-columnar/v1', columns=columns, views=views,
                sources=sources, episodes=[e for e in episodes if e['use']!='cal50_audit_only'],
                prompts=prompts, scalars_order=['sigma','next_sigma','timestep/1000'],
                state_source='actual same-step x_i', previous_source='same chunk previous step d_cfg',
                target_source='same-step actual target d_cfg, reconstructed bitwise from raw conditional output at CFG1',
                row_kind='historical_target_only', speculative_node_identity_available=False,
                current_CFG14_same_query_oracle=False, exact_historical_weight_bytes_recorded=False,
                normalization='not fitted; FIT174 only must fit future normalizers',
                split='FIT174 init0–17 excludes historical DEV task0/1 init6–8; CAL20 init18/19; CAL50 audited only; no quality tensors read')
    tmp = OUT / 'historical_train194_columnar.pt.tmp'
    torch.save(pack, tmp)
    tmp.replace(OUT / 'historical_train194_columnar.pt')
    back = load(OUT / 'historical_train194_columnar.pt')
    require(all(equal(v, back['columns'][k]) for k, v in columns.items()) and
            all(equal(v, back['views'][k]) for k, v in views.items()), 'columnar save/reload mismatch')
    summary = dict(status='PASS_COMPLETE_HISTORY_NUMERIC_AUDIT', totals=dict(totals),
                   numeric_maxima=dict(maxes), explained_mismatches=dict(full_mismatch), episodes=episodes,
                   exported_rows=n, exported_sources=len(sources), view_rows={k:len(v) for k,v in views.items()},
                   pack=str(OUT/'historical_train194_columnar.pt'), pack_bytes=(OUT/'historical_train194_columnar.pt').stat().st_size,
                   fields_finite=True, source_state_direction_dtype='BF16 [1,30,4,4,1]',
                   raw_output_dtype='BF16 [2,16,30]', raw_conditional_direction_bitwise_exact=True,
                   compact_transition_bitwise_exact=True, compact_euler_bitwise_exact=True,
                   source_writes=False, quality_or_reserved_dev_tensors_read=False,
                   historical_checkpoint_identity='recorded checkpoint id/path only; original weight-byte hash unavailable',
                   same_query_meaning='actual historical target query only; not candidate-induced current CFG14 oracle',
                   hidden_and_KV='not captured; arbitrary candidate teacher cannot be reconstructed by relabeling these rows')
    write('HISTORY-RESULT.json', summary)
    return summary


def row_key(r):
    return (r['task_index'], r['episode_index'], r['episode_attempt_id'], r['rollout_chunk'], r['tree_step_index'], r['node'])


def check_teacher_row(row):
    for key in ('state', 'previous', 'target'):
        tcheck(row[key], (112,), torch.float32)
    require(equal(row['state'], row['state'].to(torch.bfloat16).float()), 'query state is not BF16-representable')
    digest = hashlib.sha256(row['state'].contiguous().numpy().tobytes()).hexdigest()
    require(digest == row['state_sha256'] == row['target_query_state_sha256'], 'actual-query binding differs')
    require(all(math.isfinite(float(row[k])) for k in ('sigma','next_sigma','timestep')) and
            float(row['sigma']) > float(row['next_sigma']) >= 0., 'teacher scheduler scalar invalid')
    require(abs(float(row['timestep'])-1000*float(row['sigma'])) < 1e-3, 'teacher timestep differs from sigma schedule')
    require(row['teacher_label']=='actual_query_target_cfg_direction', 'not actual-query target label')


def audit_same_query():
    cache = load(AUX)
    train = cache['current']['train_rows']
    aux = cache['aux_rows']
    require(len(train)==3906 and len(aux)==11091 and len(cache['aux_sources'])==3697, 'same-query cache cardinality differs')
    # DEV rows are present in this historical cache; no access/evaluation/export occurs.
    groups = {'current_fit':train, 'aux_fit':aux}
    seen = set()
    by_task = {}
    by_key = {}
    for name, rows in groups.items():
        counts = Counter()
        for row in rows:
            check_teacher_row(row)
            key = row_key(row)
            require(key not in seen, 'same-query duplicate logical identity')
            seen.add(key)
            require(row['split']=='train' and (row['task_index'],row['episode_index']) not in RESERVED
                    and row['episode_index'] < 18, 'same-query current/aux leaks into CAL20')
            if name=='current_fit':
                require(row['task_index'] in (0,1) and row['episode_index'] in (0,1), 'current train split differs')
            else:
                require(row['task_index'] in range(10) and row['episode_index']==17, 'aux split differs')
            counts[(row['task_index'],row['episode_index'],row['node'])] += 1
            by_key[key] = row
        by_task[name] = [{'task':k[0],'init':k[1],'node':k[2],'rows':v} for k,v in sorted(counts.items())]
    aux_keys = {row_key(r) for r in aux}
    observed = set()
    generators = Counter()
    target_ids = Counter()
    source_contracts = Counter()
    sources = []
    for j, item in enumerate(cache['aux_sources']):
        path = Path(item['path'])
        payload = load(path)
        require(payload['schema_version']=='bonus-prefix-same-query-teacher/v2' and
                payload['extra_teacher_forwards']==0 and payload['captures_all_three_predictor_nodes'] is True,
                'aux source contract differs')
        contract = payload['contract']
        require(contract['used_action_channel_ids']==list(range(7)) and contract['cfg_scale']==1.
                and contract['cache_mode']=='original' and contract['action_updates_per_chunk']==50
                and contract['bonus_enabled'] is True and contract['mode']=='verifier_k2_b2_bonus'
                and contract['selector']=='prefix_then_cost' and contract['threshold']==THRESHOLD,
                'aux policy contract differs')
        require(len(payload['rows'])==3 and {r['node'] for r in payload['rows']}=={0,1,2}, 'aux source loses nodes')
        for row in payload['rows']:
            check_teacher_row(row)
            key = row_key(row)
            require(key in aux_keys and key not in observed, 'aux source index mismatch/duplicate')
            stored = by_key[key]
            require(row.keys()==stored.keys(), 'cache/source row field inventory differs')
            for k,v in row.items():
                require(equal(v,stored[k]) if isinstance(v,torch.Tensor) else v==stored[k], 'cache/source teacher field differs: '+k)
            observed.add(key)
        p = contract['provenance']
        generators[p['draft_sha256']] += 1
        target_ids[p['target_checkpoint_id']] += 1
        source_contracts[p['plan_sha256']] += 1
        sources.append(dict(path=str(path), task=p['task'], init=p['init'], plan=p['plan_sha256'],
                            draft=p['draft_sha256'], use='fit_only_init17'))
        if j % 500 == 0:
            write('status.json', dict(status='RUNNING_AUX_SOURCE_AUDIT', at=stamp(), files=j+1, total_files=len(cache['aux_sources'])))
    require(observed==aux_keys, 'aux source files do not cover cached rows')
    out = dict(status='PASS_COMPLETE_SAME_QUERY_AUX_NUMERIC_AUDIT', current_fit_rows=len(train), aux_fit_rows=len(aux),
               total_same_query_fit_rows=len(seen), aux_source_files=len(sources), aux_trees=len(sources),
               aux_source_rows_equal_cached_rows=True, actual_state_query_hash_matches=True,
               all_fields_finite=True, actual_teacher_queries=True, historical_target_only=False,
               generator_draft_inventory=dict(generators), target_checkpoint_ids=dict(target_ids), collection_plans=dict(source_contracts),
               coverage=by_task, whole_episode_overlap_current_aux=0,
               cal20_episode_overlap=0, cal50_episode_overlap=0,
               use='All current3906 + aux11091 are FIT; init17 cannot also be new CAL',
               caveat='Aux teacher is exact at states induced by the recorded research generator draft, not newly selected TF rollout states',
               original_weight_bytes='Some historical collections have content manifests; this audit does not retroactively certify absent manifests',
               source_path=str(AUX), source_files=sources,
               cached_dev_present_but_not_evaluated_or_exported=True,
               teacher_requery_for_new_TF='Needed only for new candidate-induced off-distribution queries, not for these existing valid labels')
    write('SAME-QUERY-RESULT.json', out)
    return out


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    write('PLAN.json', dict(schema='data-numeric-audit/v1', created_at=stamp(), history_manifest=str(MANIFEST),
                            history_train_episodes=194, history_cal_audit_only=50, aux=str(AUX),
                            CPU_affinity=sorted(os.sched_getaffinity(0)), threads=2, GPU=False,
                            source_modifications=False, current_same_query_train=3906, aux_same_query_train=11091,
                            quality_tensors_read=False, target_model_forwards=0, simulator_episodes=0))
    rng = torch.get_rng_state().clone()
    try:
        history = audit_history(json.loads(MANIFEST.read_text()))
        aux = audit_same_query()
        require(torch.equal(rng,torch.get_rng_state()), 'audit mutated CPU RNG')
        require(not torch.cuda.is_initialized(), 'audit initialized CUDA')
        result = dict(status='PASS_COMPLETE_NUMERIC_AUDIT', finished_at=stamp(), elapsed_sec=time.monotonic()-start,
                      history=dict(rows=history['exported_rows'], view_rows=history['view_rows'], pack=history['pack'], pack_bytes=history['pack_bytes']),
                      same_query=dict(current_rows=aux['current_fit_rows'], aux_rows=aux['aux_fit_rows'], source_files=aux['aux_source_files']),
                      CUDA_initialized=False, CPU_RNG_unchanged=True, source_writes=False,
                      networks_or_remote_used=False, target_model_forwards=0, simulator_episodes=0)
        write('RESULT.json', result)
        write('status.json', result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        write('FAILED.json', dict(status='FAILED_NUMERIC_AUDIT', at=stamp(), error=repr(exc), elapsed_sec=time.monotonic()-start,
                                  CUDA_initialized=torch.cuda.is_initialized(), partial_results_not_certified=True))
        raise


if __name__=='__main__':
    main()

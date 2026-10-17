"""Persistent frozen WAM for complete collection/DEV; independent entry only.

The action implementation is never replaced by a diagnostic AST. The new
collector observes the existing O3 instance and saves already-computed labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import types

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from tf_effect_campaign_v1.protocol import load
from tf_usable_quality_v1.legacy import server_helpers
from task4_direction_completion_v1.server_entry import fingerprint, snapshot


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Exclusive terminal/evidence writes; never overwrite a failed receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _source_origin(module, sources, *, expected=None):
    path = Path(module.__file__).resolve()
    require(str(path) in sources and (expected is None or path == expected),
        'loaded module origin lacks registered frozen provenance: ' + str(path))


def _full_episode_record(out, row):
    records = [json.loads(line) for line in (out / 'client/episodes.jsonl').read_text().splitlines() if line.strip()]
    require(len(records) == 1, 'one complete original client episode required')
    record = records[0]
    require(record.get('status') == 'completed' and record.get('task_index') == row['task']
        and record.get('episode_index') == row['init'] and record.get('libero_horizon') == 800
        and record.get('max_policy_chunks') is None and record.get('truncated_after_max_policy_chunks') is False
        and (record.get('success') is True or record.get('horizon_reached') is True),
        'full success/horizon terminal episode required; no truncated quality result')
    mapping = read(out / 'client/COMMAND-MAPPING.json')
    require(mapping.get('complete_episode') is True and mapping.get('env_steps') == record['env_steps']
        and mapping.get('success') == record['success']
        and len(mapping['submitted']) == len(mapping['environment']) == record['env_steps'],
        'complete submitted/environment command coverage required')
    return record


def main():
    job = Path(os.environ['TF_EFFECT_JOB']).resolve(strict=True)
    p = load(job)
    arm = os.environ['TF_EFFECT_ARM']
    q = p['reference_plan']
    require(isinstance(q, dict), 'reference_plan must embed the complete frozen M11r2 plan')
    require(arm in p['arms'] and arm in ('T', 'TF_CONT', 'TF_ON', 'MLP_ON')
        and os.getenv('CUDA_VISIBLE_DEVICES') == '0' and os.getenv('RESET_TEXT_BANK_ENABLED') == '1',
        'one registered local arm, CUDA0 and explicit reset bank required')
    require(not p['collect'] or (arm == 'TF_CONT' and p['arms'] == ['TF_CONT']),
        'new collection uses only unchanged current TF_CONT3000')
    require(not any(os.getenv(key) for key in ('WAM_P0_STATE_TEACHER_DIR', 'WAM_P0_TRACE_DIR', 'P0_NATIVE_DIAGNOSTIC_DIR')),
        'native trace/teacher and heavy diagnostic environments must remain disabled')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(len(os.sched_getaffinity(0)) <= 2, 'server must inherit its registered two CPU cores')
    validate_server, RamAccounting, modules = server_helpers()
    from wan_va import wan_va_server as source
    from reset_text_bank_v1.bank import TextBank, install as install_bank
    _source_origin(source, p['source_sha256'], expected=ROOT / '投机采样/wam-p0-worktree/wan_va/wan_va_server.py')
    for name in ('configs', 'configs.va_libero_cfg', 'distributed.fsdp', 'distributed.util',
            'modules.utils', 'modules.model', 'utils', 'utils.p0_k3_b2', 'utils.p0_trace', 'utils.p0_state_teacher'):
        module = sys.modules.get(name)
        require(module is not None and hasattr(module, '__file__'), 'missing frozen WAM helper: ' + name)
        _source_origin(module, p['source_sha256'])

    def loaded_native_origin():
        module = sys.modules.get('_shared_torch_v1')
        if module is not None:
            _source_origin(module, p['source_sha256'])

    original_run = source.run_async_server_mode

    def serve(s, local_rank, host, port):
        validate_server(s)
        channels = tuple(int(value) for value in s.job_config.used_action_channel_ids)
        require(local_rank == 0 and channels == tuple(range(7)) and s.job_config.action_per_frame == 4,
            'one-rank original compact seven-channel action contract required')
        mode = 'off' if arm == 'T' else p['mode']
        require(s._p0_action_draft_mode == mode and s._p0_action_draft_threshold == p['threshold']
            and p['mode'] == q['mode'] and p['selector'] == q['selector'] and p['threshold'] == q['threshold'],
            'original mode/selector/threshold differ from registered campaign')
        # These are the target-only initialization states, before any draft
        # construction, actual CAL, reset-bank lookup, or collector setup.
        target_initial_cpu = torch.get_rng_state().clone()
        target_initial_cuda = torch.cuda.get_rng_state(s.device).clone()
        initial_rng = dict(cpu=fingerprint(target_initial_cpu), cuda=fingerprint(target_initial_cuda))
        arm_root = job / 'arms' / arm
        require(arm_root.is_dir(), 'supervisor must create the owned arm directory')
        rng_path = arm_root / 'TARGET-INITIAL-RNG.pt'
        require(not rng_path.exists(), 'never overwrite prior target initialization RNG')
        torch.save(dict(CPU=target_initial_cpu, CUDA=target_initial_cuda), rng_path)
        s.action_per_frame = int(s.job_config.action_per_frame)
        proxy = online = collector = parity = None
        reused_CAL_rows = full_real_CAL_rows = new_real_CAL_rows = 0
        CAL_reference = None
        if arm == 'TF_CONT':
            require(p['selected'][arm]['checkpoint'] == q['selected']['TF_CONT']['checkpoint']
                and p['selected'][arm]['checkpoint_sha256'] == q['selected']['TF_CONT']['checkpoint_sha256'],
                'old current CAL must never certify different new checkpoint bytes')
            from task4_direction_completion_v1.observer import load_with_existing_cal
            predictor, parity = load_with_existing_cal(q, arm, s.device, channels, arm_root / 'CAL-REUSE.json')
            require(parity['rows'] == 20237 and parity['same_device_bitwise_exact']
                and parity['synthetic_queries'] == 0 and parity['complete_CAL']
                and parity['checkpoint_sha256'] == p['selected'][arm]['checkpoint_sha256'],
                'unchanged complete current actual GPU CAL required')
            reused_CAL_rows, CAL_reference = 20237, q['existing_CAL_certificate']
        elif arm != 'T':
            from tf_effect_train_v1.gpu_actual_cal import actual_cal_gpu_parity
            import tf_effect_train_v1.gpu_actual_cal as cal_module
            _source_origin(cal_module, p['source_sha256'])
            cal_job = Path(p['gpu_cal_jobs'][arm]).resolve(strict=True)
            cal_plan = read(cal_job / 'PLAN.json')
            require(cal_plan['condition'] == arm and cal_plan['architecture'] == p['selected'][arm]['architecture']
                and cal_plan['checkpoint'] == p['selected'][arm]['checkpoint']
                and cal_plan['checkpoint_sha256'] == p['selected'][arm]['checkpoint_sha256'],
                'new candidate requires its own exact prepared GPU CAL job')
            predictor, parity = actual_cal_gpu_parity(cal_job, device=s.device)
            require(parity['status'] == 'PASS_FULL_OLD_AND_NEW_REAL_CAL_SAME_DEVICE_PARITY'
                and parity['full_old_and_new_CAL'] is True and parity['CPU_rng_unchanged'] is True
                and parity['CUDA_rng_unchanged'] is True and parity['synthetic_queries'] == 0
                and parity['checkpoint_sha256'] == p['selected'][arm]['checkpoint_sha256']
                and all(row['complete_CAL'] and row['finite'] and row['same_device_bitwise_exact']
                    for row in parity['sets'].values()), 'new complete same-device CAL failed')
            full_real_CAL_rows = parity['sets']['old_CAL']['rows']
            new_real_CAL_rows = parity['sets']['new_CAL']['rows']
            require(full_real_CAL_rows == 20237 and new_real_CAL_rows > 0, 'both real CAL sets required')
            CAL_reference = str(cal_job / 'RESULT.json')
        if arm != 'T':
            require(predictor.architecture == p['selected'][arm]['architecture'], 'fixed draft architecture mismatch')
            s._p0_action_draft_b2_predictor = predictor
            s._p0_action_draft_b2_checkpoint = predictor.checkpoint_path
            proxy = modules['selection'].install(s, p['selector'])
            modules['buffers'].prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
            online = modules['measured_online'].control(s, 'O3', allow_cuda=True)
        bank = TextBank(p['bank'])
        install_bank(s, bank)

        def unexpected_text(_server, *args, **kwargs):
            raise RuntimeError('validated complete reset embeddings required; unexpected text-encoder forward')

        s._get_t5_prompt_embeds = types.MethodType(unexpected_text, s)
        arm_budget = int(os.environ['TF_EFFECT_ARM_BUDGET'])
        require(0 < arm_budget <= p['job_target_forward_cap'], 'actual remaining whole-job forward cap required')
        accounting = RamAccounting(s, cap=arm_budget)
        if p['collect']:
            from tf_effect_collector_v1.collector import Collector
            import tf_effect_collector_v1.collector as collector_module
            _source_origin(collector_module, p['source_sha256'])
            collector = Collector(s, proxy, job / 'teacher' / arm, dict(
                generator_checkpoint=p['selected'][arm]['checkpoint'],
                generator_checkpoint_sha256=p['selected'][arm]['checkpoint_sha256'],
                generator_condition='TF_CONT3000', generator_total_updates=3000,
                target_checkpoint_id=q['target_checkpoint_id'], target_weight_stats=q['target_weight_stats'],
                source_commit=p.get('commit', p.get('source_commit')),
                data_role_by_key={row['key']: row['role'] for row in p['cases']},
                case_manifest=[{name: row[name] for name in ('arm', 'task', 'init', 'key', 'role', 'rep')}
                    for row in p['cases']],
                storage_cap_bytes=p['collected_teacher_cap_bytes'],
                qualification_evidence=p.get('collector_qualification_evidence'),
                performance_claim=False))
            collector.install()

        # Keep the original function reference: the wrapper below intentionally
        # has no `executor` global. Drain the original queue, never wrapper globals.
        original_save = source.save_async
        executor = original_save.__globals__.get('executor')
        require(executor is not None and executor._max_workers == 1, 'original one-worker async save executor required')
        saves = []
        active = dict(key=None, row=None, request_start=0, forward_start=0, returned_start=0,
            tree_start=0, phase_start={}, bank_start=0, saves_start=0, O3_start=0,
            teacher_start=0, chunks=[], teacher=[], completed=set())
        gate = threading.RLock()

        def filtered_save(value, path, *args, **kwargs):
            path = Path(path)
            suppressed = path.name.startswith('obs_data_') and path.suffix == '.pt'
            require(active['key'] is not None, 'async output outside assigned complete episode')
            require(path.resolve().is_relative_to(job / 'episodes' / active['key'] / 'server_output'),
                'async output escaped its registered episode')
            saves.append(dict(path=str(path), name=path.name, suppressed=suppressed, key=active['key']))
            if suppressed:
                return None
            return original_save(value, path, *args, **kwargs)

        source.save_async = filtered_save
        original_reset, original_infer = s._reset, s.infer
        torch.set_rng_state(target_initial_cpu)
        torch.cuda.set_rng_state(target_initial_cuda, s.device)

        def reset(_server, prompt=None):
            row = active['row']
            require(row is not None, 'reset has no assigned full episode')
            before_cpu = torch.get_rng_state().clone()
            before_cuda = torch.cuda.get_rng_state(s.device).clone()
            torch.cuda.synchronize(s.device)
            start = time.monotonic()
            result = original_reset(prompt=prompt)
            torch.cuda.synchronize(s.device)
            wall = time.monotonic() - start
            require(torch.equal(before_cpu, torch.get_rng_state())
                and torch.equal(before_cuda, torch.cuda.get_rng_state(s.device)), 'bank reset consumed target RNG')
            for text, value in ((prompt, s.prompt_embeds), ('', s.negative_prompt_embeds)):
                require(torch.equal(value, bank.get(text, device=s.device, dtype=s.dtype, sequence_length=512)),
                    'actual cached complete reset embeddings differ')
            state = snapshot(s, empty=True)
            require(state['cpu_rng'] == initial_rng['cpu'] and state['cuda_rng'] == initial_rng['cuda'],
                'reset RNG differs from original target-only initialization')
            write(job / 'episodes' / row['key'] / 'RESET.json', dict(status='PASS_FULL_RESET_BANK_AND_EMPTY_STATE',
                row=row, state=state, RNG_unchanged=True, reset_wall_sec=wall, performance_claim=False))
            return result

        s._reset = types.MethodType(reset, s)

        def infer(_server, request):
            with gate:
                context = request.get('p0_trace_context') or {}
                key = context.get('parallel_worker_id')
                if request.get('reset'):
                    rows = [row for row in p['cases'] if row['key'] == key and row['arm'] == arm]
                    require(len(rows) == 1 and active['key'] is None and key not in active['completed'],
                        'one fresh reset only after complete prior episode')
                    row = rows[0]
                    require(context.get('task_index') == row['task'] and context.get('episode_index') == row['init'],
                        'actual task/init differs from registered episode')
                    torch.set_rng_state(target_initial_cpu)
                    torch.cuda.set_rng_state(target_initial_cuda, s.device)
                    active.update(key=key, row=row, request_start=len(accounting.requests),
                        forward_start=accounting.attempted, returned_start=accounting.returned,
                        tree_start=0 if proxy is None else len(proxy.records), phase_start=dict(accounting.phase),
                        bank_start=bank.hits, saves_start=len(saves), O3_start=len(modules['buffers'].EVENTS),
                        teacher_start=0 if collector is None else collector.total_trees, chunks=[], teacher=[])
                    accounting.cap = min(arm_budget, accounting.attempted + p['per_episode_target_forward_cap'])
                    s.save_root = str(job / 'episodes' / key / 'server_output')
                else:
                    require(active['key'] is not None and key == active['key']
                        and context.get('task_index') == active['row']['task']
                        and context.get('episode_index') == active['row']['init'],
                        'policy/cache request does not belong to active full episode')
                policy = not request.get('reset') and not request.get('compute_kv_cache')
                before_frame = getattr(s, 'frame_st_id', None) if request.get('reset') else int(s.frame_st_id)
                phase_before = dict(accounting.phase)
                tree_before = 0 if proxy is None else len(proxy.records)
                O3_before = len(modules['buffers'].EVENTS)
                forwards_before = accounting.attempted
                if collector is not None and policy:
                    collector.begin_chunk(context)
                try:
                    result = original_infer(request)
                    shard = collector.end_chunk() if collector is not None and policy else None
                    item = accounting.requests[-1]
                    require(item['trace_context'] == request.get('p0_trace_context'), 'RAM request identity changed')
                    names = ([f'actions_{before_frame}.pt', f'latents_{before_frame}.pt'] if policy else [])
                    item.update(server_frame_st_id_before=before_frame, server_frame_st_id_after=int(s.frame_st_id),
                        expected_saved_outputs=names,
                        suppressed_saved_outputs=([f'obs_data_{before_frame}.pt'] if item['kind'] == 'cache' else []))
                    if policy:
                        phases = {name: value-phase_before.get(name, 0) for name, value in accounting.phase.items()}
                        trees = 0 if proxy is None else len(proxy.records)-tree_before
                        events = modules['buffers'].EVENTS[O3_before:]
                        if arm != 'T':
                            require(trees == phases.get('tree_count', 0), 'policy acceptance/counter tree coverage differs')
                            require(len(events) == 1 and all(event.get('enabled') and event.get('ok') and event.get('closed')
                                for event in events), 'O3 must remain enabled and closed for every speculative policy chunk')
                            require(phases.get('committed_scheduler_update_count') == 50,
                                'original complete action scheduler did not commit all50 updates')
                        row = dict(context=dict(context), frame_st_id=before_frame, target_forwards=accounting.attempted-forwards_before,
                            phase_counters=phases, tree_count=trees, O3_events=events, performance_claim=False)
                        if shard is not None:
                            require(shard['trees'] == trees and shard['rows'] == 3*trees
                                and shard['extra_teacher_forwards'] == 0 and not collector.errors,
                                'teacher/counter tree coverage failed; shard is not eligible FIT/CAL')
                            proof = dict(status='PASS_SHARD_EXTERNAL_TREE_COUNTER_COVERAGE', arm=arm, key=key,
                                role=active['row']['role'], context=dict(context), shard=shard,
                                actual_phase_tree_count=phases.get('tree_count', 0), acceptance_tree_count=trees,
                                actual_O3_events=events, extra_teacher_forwards=0,
                                source_trace_and_teacher_disabled=s._p0_trace is None and s._p0_state_teacher is None,
                                performance_claim=False)
                            proof_path = Path(shard['path']).with_suffix('.COUNTER.json')
                            write(proof_path, proof)
                            active['teacher'].append(dict(**shard, counter_proof=str(proof_path)))
                            row['teacher_shard'] = shard
                        active['chunks'].append(row)
                    return result
                except BaseException as exc:
                    failure = dict(status='FAILED_REQUEST_OR_CAPTURE_NOT_FIT_CAL', arm=arm, key=key,
                        context=dict(context), error=repr(exc), traceback=traceback.format_exc(), performance_claim=False)
                    path = job / 'episodes' / key / 'SERVER-REQUEST-FAILED.json'
                    if not path.exists():
                        write(path, failure)
                    raise

        s.infer = types.MethodType(infer, s)

        def monitor():
            try:
                while True:
                    time.sleep(.2)
                    with gate:
                        key = active['key']
                        if key is None:
                            continue
                        out = job / 'episodes' / key
                        if not (out / 'FINISH_REQUESTED').exists():
                            continue
                        # The original executor's FIFO barrier drains all actions
                        # and latents even though source.save_async is wrapped.
                        executor.submit(lambda: None).result(timeout=90)
                        torch.cuda.synchronize(s.device)
                        loaded_native_origin()
                        require(not accounting.failed and not accounting.active and accounting.attempted == accounting.returned
                            and (proxy is None or proxy.pending is None), 'target/selection accounting did not close')
                        record = _full_episode_record(out, active['row'])
                        phases = {name: value-active['phase_start'].get(name, 0) for name, value in accounting.phase.items()}
                        acceptance = [] if proxy is None else proxy.records[active['tree_start']:]
                        require(proxy is None or len(acceptance) == phases.get('tree_count', 0), 'episode acceptance tree coverage differs')
                        requests = accounting.requests[active['request_start']:]
                        require(all(item['ok'] for item in requests)
                            and sum(item['kind'] == 'policy' for item in requests) == record['infer_chunks'] == len(active['chunks']),
                            'complete original policy chunk/terminal episode coverage differs')
                        saved = saves[active['saves_start']:]
                        suppressed = [item for item in saved if item['suppressed']]
                        kept = [item for item in saved if not item['suppressed']]
                        require(len(suppressed) == sum(item['kind'] == 'cache' for item in requests), 'obs suppression/cache coverage differs')
                        expected_names = [name for item in requests for name in item['expected_saved_outputs']]
                        require(sorted(item['name'] for item in kept) == sorted(expected_names), 'action/latent save call coverage differs')
                        require(len({item['path'] for item in kept}) == len(kept), 'saved policy artifact path reused within episode')
                        for item in kept:
                            path = Path(item['path'])
                            require(path.is_file() and path.stat().st_size > 0, 'async action/latent artifact missing after original drain')
                            value = torch.load(path, map_location='cpu', weights_only=True)
                            require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), 'invalid saved full action/latent tensor')
                        require(not any((out / 'server_output').rglob('obs_data_*.pt')), 'suppressed full visual data were still written')
                        events = modules['buffers'].EVENTS[active['O3_start']:]
                        if arm != 'T':
                            require(len(events) == record['infer_chunks'] and all(event.get('enabled') and event.get('ok') and event.get('closed')
                                for event in events), 'not all complete speculative chunks used unchanged O3')
                        teacher = dict(enabled=collector is not None, shards=active['teacher'], extra_teacher_forwards=0,
                            rows=sum(row['rows'] for row in active['teacher']), reached_rows=sum(row['reached_rows'] for row in active['teacher']),
                            trees=sum(row['trees'] for row in active['teacher']), performance_claim=False)
                        if collector is not None:
                            require(collector.active is None and collector.pending is None and not collector.errors
                                and teacher['trees'] == collector.total_trees-active['teacher_start'] == len(acceptance)
                                and len(teacher['shards']) == record['infer_chunks'], 'complete terminal teacher coverage failed')
                            require(not list((job / 'teacher' / arm).glob('*FAILED*')), 'failed capture source marker prevents collection eligibility')
                        write(out / 'TEACHER-INDEX.json', teacher)
                        write(out / 'FINAL-SERVER-STATE.json', snapshot(s, empty=False))
                        write(out / 'COUNTER.json', dict(arm=arm, key=key, row=active['row'],
                            target_forwards=accounting.attempted-active['forward_start'],
                            returned_forwards=accounting.returned-active['returned_start'], requests=requests,
                            phase_counters=phases, acceptance=acceptance, chunks=active['chunks'],
                            bank_hits=bank.hits-active['bank_start'], O3_events=events,
                            teacher=teacher, extra_teacher_forwards=0,
                            save_async=dict(original_executor_drained=True, obs_suppressed=len(suppressed), kept=len(kept),
                                kept_artifacts=[dict(**item, bytes=Path(item['path']).stat().st_size) for item in kept]),
                            persistent_model=True, permissions=p['permissions'], robot_quality_validated=False,
                            performance_claim=False))
                        finish = dict(status='PASS', arm=arm, key=key, async_drained=True,
                            full_episode_terminal=True, success=record['success'], env_steps=record['env_steps'],
                            teacher_complete=collector is not None, extra_teacher_forwards=0,
                            full_visual_obs_suppressed=len(suppressed), permissions=p['permissions'],
                            robot_quality_validated=False, performance_claim=False)
                        write(out / 'FINISH_SERVER.json', finish)
                        active['completed'].add(key)
                        active['key'] = active['row'] = None
                        write(out / 'FINISH_READY.json', finish)
            except BaseException as exc:
                path = job / f'{arm}-MONITOR-FAILED.json'
                if not path.exists():
                    write(path, dict(error=repr(exc), traceback=traceback.format_exc(), performance_claim=False))

        threading.Thread(target=monitor, daemon=True).start()
        torch.cuda.synchronize(s.device)
        loaded_native_origin()
        require(fingerprint(torch.get_rng_state()) == initial_rng['cpu']
            and fingerprint(torch.cuda.get_rng_state(s.device)) == initial_rng['cuda'], 'setup RNG restoration failed before READY')
        write(job / f'{arm}-READY.json', dict(status='READY', arm=arm, pid=os.getpid(), port=port,
            target_initial_rng=initial_rng, ready_rng=initial_rng, target_RNG_captured_before_draft_and_CAL=True,
            setup_RNG_restored=True, model_persistent=True, performance_claim=False, bank_explicitly_enabled=True,
            full_real_CAL_rows=full_real_CAL_rows, reused_real_CAL_rows=reused_CAL_rows,
            new_real_CAL_rows=new_real_CAL_rows, CAL_executed_query_rows=full_real_CAL_rows+new_real_CAL_rows,
            new_CAL_rows=new_real_CAL_rows, CAL_reference=CAL_reference,
            predictor_checkpoint=None if arm == 'T' else p['selected'][arm]['checkpoint'],
            selector='none' if arm == 'T' else p['selector'], O1_O2_O3=arm != 'T', synthetic_queries=0,
            arm_target_budget=arm_budget, per_episode_target_cap=p['per_episode_target_forward_cap'],
            collect=p['collect'], collector_installed=collector is not None,
            permissions=p['permissions'], robot_quality_validated=False,
            save_async_suppression='obs_data_*.pt only; original actions/latents and executor retained'))
        try:
            return original_run(s, local_rank, host, port)
        finally:
            source.save_async = original_save
            if collector is not None:
                collector.close()
            if online is not None:
                online.close()

    source.run_async_server_mode = serve
    source.init_logger()
    source.main()


if __name__ == '__main__':
    main()

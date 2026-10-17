"""One persistent owned WAM server per registered logical worker on two shared GPUs.

All checks execute inline with complete LIBERO episodes. There is no smoke,
standalone CAL/replay/test, old protocol loader, digest, or automatic launcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import threading
import time
import traceback
import types

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from tf_suite_pipeline_v1.protocol import WORKTREE, PERMISSIONS, episode_dir, row as phase_row, run_rows, split_role, load, read, write, require


def prefix_modules():
    """Bind only production helpers; do not import any legacy runner/protocol."""
    prefix, base = EXP/'prefix_acceptance_v1', EXP/'single_worker_chc_gpu_v4'
    for directory in (EXP, WORKTREE, WORKTREE/'wan_va', base, prefix):
        value = str(directory)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    modules = {}
    for name in ('selection', 'measured_online', 'buffers', 'prefix_kernel', 'experiment'):
        found = importlib.util.find_spec(name)
        require(found is not None and Path(found.origin).resolve() == prefix/(name+'.py'),
            'original prefix helper namespace collision: ' + name)
        modules[name] = importlib.import_module(name)
    return modules


def validate_server(s, p):
    cfg = s.job_config
    require(s._cache_mode == 'original' and len(s.transformer.blocks) == 30
        and s.streaming_vae_half is None and s._p0_trace is None
        and s._p0_state_teacher is None and s._nsys_profile_mode == 'off'
        and not s._phase_profile_enabled and cfg.action_num_inference_steps == 50
        and cfg.num_inference_steps == 20 and cfg.video_exec_step == -1 and cfg.action_guidance_scale == 1
        and cfg.guidance_scale == 5 and tuple(cfg.used_action_channel_ids) == tuple(range(7))
        and cfg.action_per_frame == 4 and cfg.frame_chunk_size == 4,
        'frozen original target/cache/CFG/scheduler geometry changed')
    require(s._p0_action_draft_mode == p['mode'] and s._p0_action_draft_threshold == p['threshold'],
        'fixed speculative mode/threshold required')


def terminal_record(out, row, arm):
    terminal = read(out/'EPISODE-TERMINAL.json')
    require(terminal['schema'] == 'tf-suite-full-episode-terminal/v1'
        and terminal['status'] == 'COMPLETE_RECORDED_FULL_EPISODE'
        and terminal['row_id'] == row['row_id'] and terminal['case_id'] == row['case_id'] and terminal['arm'] == arm
        and terminal['task'] == row['task'] and terminal['init'] == row['init']
        and terminal['complete_episode'] is True and terminal['quality_terminal'] is True
        and terminal['horizon'] == 800 and terminal['permissions'] == PERMISSIONS
        and (terminal['success'] is True or terminal['horizon_reached'] is True),
        'only original natural success/full horizon800 is a complete episode')
    records = [json.loads(line) for line in (out/'client/episodes.jsonl').read_text().splitlines() if line.strip()]
    require(len(records) == 1, 'exactly one original robot terminal record required')
    record = records[0]
    require(record['status'] == 'completed' and record['task_index'] == row['task']
        and record['episode_index'] == row['init'] and record['libero_horizon'] == 800
        and record.get('max_policy_chunks') is None
        and record.get('truncated_after_max_policy_chunks') is False
        and record['success'] == terminal['success'] and record['env_steps'] == terminal['env_steps']
        and record['infer_chunks'] == terminal['policy_chunks'], 'original client terminal mismatch')
    mapping = read(out/'client/COMMAND-MAPPING.json')
    require(mapping['complete_episode'] is True and mapping['success'] == terminal['success']
        and mapping['env_steps'] == terminal['env_steps']
        and len(mapping['submitted']) == len(mapping['environment']) == terminal['env_steps'],
        'all submitted/environment commands must cover the full episode')
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--worker-index', type=int, default=0, choices=(0,))
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--forward-cap', type=int, required=True)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    p = load(job)
    worker = job / 'server-worker0'
    worker.mkdir(parents=True, exist_ok=True)
    require(not (worker / 'READY.json').exists() and not (worker / 'FAILED.json').exists(),
        'one fresh suite worker lifecycle; no automatic retry')
    require(args.forward_cap == p['job_forward_attempt_cap'] > 0
        and args.port == p['port'] and args.master_port == p['master_port'], 'registered suite worker caps/ports differ')
    require(datetime.now(timezone.utc) < datetime.fromisoformat(p['absolute_work_stop'])
        and len(os.sched_getaffinity(0)) <= 2, 'suite start window or CPU affinity differs')
    require(os.getenv('RESET_TEXT_BANK_ENABLED') == '1'
        and os.getenv('CUDA_VISIBLE_DEVICES') == str(p['gpu_index']), 'explicit suite GPU and full reset bank required')
    require(not any(os.getenv(k) for k in ('WAM_P0_STATE_TEACHER_DIR', 'WAM_P0_TRACE_DIR', 'P0_NATIVE_DIAGNOSTIC_DIR')),
        'no independent trace/teacher/test entry')
    all_target = all(r['arm'] == 'T' for r in run_rows(p))
    predictor_path = str(job / 'UNUSED_T_ONLY_NO_SUITE_DRAFT') if all_target else str(Path(p['selected']['path']).resolve(strict=True))
    os.environ.update(MASTER_PORT=str(args.master_port), MASTER_ADDR='127.0.0.1', RANK='0', LOCAL_RANK='0',
        WORLD_SIZE='1', LOCAL_WORLD_SIZE='1', LINGBOT_SEED=str(p['evaluation_seed']),
        WAM_P0_ACTION_DRAFT_MODE=p['mode'], WAM_P0_ACTION_DRAFT_THRESHOLD=str(p['threshold']),
        WAM_P0_ACTION_DRAFT_SELECTOR='global', WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=predictor_path)
    # A reused T_POOL baseline and later C/P run in different processes. Bind
    # Python as well as native run()'s Torch/NumPy seed before model setup.
    random.seed(p['evaluation_seed'])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    modules = prefix_modules()
    from wan_va import wan_va_server as source
    require(Path(source.__file__).resolve() == WORKTREE / 'wan_va/wan_va_server.py', 'original native source path differs')
    from tf_fullsuite_eval_v2.controller import Controller
    from tf_fullsuite_eval_v2.evidence import ExecutionEvidence, atomic_current
    from tf_fullsuite_eval_v2.shared_prefix_no_digest import install_o3
    from tf_fullsuite_eval_v2.state import capture_rng, restore_rng, compare_raw
    from tf_suite_pipeline_v1.deployment import configure_suite_source, load_predictor, load_text_bank
    configured, config_receipt = configure_suite_source(source, p)
    write(worker / 'SUITE-CONFIG.json', config_receipt)
    shutdown = threading.Event()
    original_run = source.run_async_server_mode
    registered_rows = run_rows(p)

    def serve(s, local_rank, host, port):
        require(local_rank == 0 and int(port) == args.port and s.job_config is configured,
            'original native suite config/model endpoint differs')
        validate_server(s, p)
        require(s.job_config.norm_stat == p['suite_info']['norm_stat']
            and s.job_config.libero_eval_suite == p['suite']
            and s.job_config.libero_checkpoint_id == p['suite_info']['checkpoint_id'], 'actual suite model config differs')
        initial_rng = capture_rng(s)
        torch.save(initial_rng, worker / 'TARGET-INITIAL-RNG.pt')
        s.action_per_frame = int(s.job_config.action_per_frame)
        predictor, deployment = load_predictor(p, s)
        s._p0_action_draft_b2_predictor, s._p0_action_draft_b2_checkpoint = predictor, predictor.checkpoint_path
        proxy = modules['selection'].install(s, p['selector'])
        modules['buffers'].prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
        online = install_o3(s, original=modules['measured_online'], source_reference=p['target_source_reference'])
        bank, bank_receipt, restore_bank = load_text_bank(p, s)
        original_t5 = s._get_t5_prompt_embeds
        def unexpected_text(_server, *values, **keywords):
            raise RuntimeError('suite requires complete512-token bank; no slow text fallback')
        s._get_t5_prompt_embeds = types.MethodType(unexpected_text, s)
        evidence = ExecutionEvidence(s, shared=online, proxy=proxy, buffers=modules['buffers'],
            cap=args.forward_cap, worker_root=worker)
        def suite_pulse(_evidence):
            ep = _evidence.episode
            atomic_current(worker / 'LIVE-COUNTER.json', dict(schema='tf-suite-worker-live-counter/v1',
                suite=p['suite'], phase=p['phase'], target_forward_attempts=_evidence.attempted,
                target_forward_returned=_evidence.returned, target_forward_failed=_evidence.failed,
                active_target_forwards=_evidence.active, request_active=_evidence.request_active,
                episode=None if ep is None else dict(row_id=ep['case']['row_id'], case_id=ep['case']['case_id'],
                    arm=ep['arm'], out=str(ep['out'])), cap=_evidence.cap,
                rejected_before_forward_cap=_evidence.rejected_before_forward_cap,
                passive_isolation_failure=(worker / 'ISOLATION-FAILED.json').exists(), permissions=PERMISSIONS))
        evidence.pulse = types.MethodType(suite_pulse, evidence)
        properties = torch.cuda.get_device_properties(s.device)
        evidence.gpu_identity = dict(uuid=str(getattr(properties, 'uuid', None) or 'UNKNOWN'), name=str(properties.name),
            logical_device=str(s.device), CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),
            source='properties of actual initialized suite model device')
        controller = Controller(s, runtime=evidence, evidence=evidence,
            expected_checkpoint=predictor.checkpoint_path, expected_threshold=p['threshold'], periodic=0).install()
        collector = None
        if p['phase'] != 'FINAL_EVAL':
            from tf_suite_pipeline_v1.collector import Collector
            collector = Collector(s, proxy, p['collector_output_dir'], dict(suite=p['suite'],
                suite_short=p['suite'].removeprefix('libero_'), phase=p['phase'],
                suite_storage_root=p['suite_storage_root'], target_checkpoint_id=p['suite_info']['checkpoint_id'],
                publisher_revision=p['suite_info']['publisher_revision'], job=str(job), permissions=PERMISSIONS)).install()
        original_reset, original_infer, original_save = s._reset, s.infer, source.save_async
        executor = original_save.__globals__.get('executor')
        require(executor is not None and executor._max_workers == 1, 'original async save executor differs')
        gate = threading.RLock()
        active = dict(row=None, out=None, arm=None, saves=[], failed=False, completed=set(), collector_records=[], collector_errors=[])

        def filtered_save(value, path, *values, **keywords):
            path = Path(path)
            require(active['out'] is not None and path.resolve().is_relative_to(active['out'] / 'server_output'),
                'suite async output escaped original episode')
            suppressed = path.name.startswith('obs_data_') and path.suffix == '.pt'
            active['saves'].append(dict(path=str(path), name=path.name, suppressed=suppressed))
            if suppressed:
                return None
            require(path.name.startswith(('actions_', 'latents_')) and path.suffix == '.pt', 'unexpected native output kind')
            evidence.saved(path.name, value)
            return original_save(value, path, *values, **keywords)
        source.save_async = filtered_save

        def reset(_server, prompt=None):
            require(active['out'] is not None and prompt == p['suite_info']['public_task_prompts'][str(active['row']['task'])],
                'actual suite reset prompt differs')
            before, start = capture_rng(s), time.monotonic()
            result = original_reset(prompt=prompt)
            torch.cuda.synchronize(s.device)
            after = capture_rng(s)
            require(compare_raw(before, after)['equal'] and compare_raw(after, initial_rng)['equal'],
                'full suite reset must preserve original target RNG')
            for name in ('q01', 'q99'):
                expected = torch.tensor(p['suite_info']['norm_stat'][name], dtype=torch.float32).reshape(30, 1, 1)
                actual = getattr(s, 'actions_' + name)
                require(actual.device.type == 'cpu' and actual.dtype == torch.float32 and actual.shape == expected.shape
                    and torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)),
                    'actual original decode quantiles do not match suite norm: ' + name)
            require(s.action_norm_method == 'quantiles', 'native suite action normalization method differs')
            for text, value in ((prompt, s.prompt_embeds), ('', s.negative_prompt_embeds)):
                require(torch.equal(value, bank.get(text, device=s.device, dtype=s.dtype, sequence_length=512)),
                    'actual full suite text embedding differs')
            out = active['out']
            torch.save(after, out / 'RNG-RESET.pt')
            write(out / 'SERVER-RESET-RNG.json', dict(schema='tf-suite-server-reset-rng/v1',
                CPU=dict(dtype=str(after['CPU'].dtype), shape=list(after['CPU'].shape), values=after['CPU'].tolist()),
                CUDA=dict(dtype=str(after['CUDA'].dtype), shape=list(after['CUDA'].shape), values=after['CUDA'].tolist()),
                device_identity=evidence.gpu_identity, actual_initialized_server_rng=True, permissions=PERMISSIONS))
            write(out / 'RESET.json', dict(schema='tf-suite-reset/v1', status='PASS_ORIGINAL_SUITE_NORM_BANK_AND_RNG',
                row_id=active['row']['row_id'], case_id=active['row']['case_id'], arm=active['arm'], suite=p['suite'],
                q01=s.actions_q01.flatten().tolist(), q99=s.actions_q99.flatten().tolist(), quantile_dtype='torch.float32',
                suite_quantiles_effective_in_native_decode=True, RNG_unchanged=True, reset_wall_sec=time.monotonic() - start,
                no_SHA=True, permissions=PERMISSIONS))
            return result
        s._reset = types.MethodType(reset, s)

        def collector_error(where, exc):
            active['collector_errors'].append(dict(where=where, error=repr(exc)))

        def infer(_server, request):
            with gate:
                context = request.get('p0_suite_context')
                require(isinstance(context, dict) and set(context) == {'row_id', 'case_id', 'arm', 'out'}, 'new suite route required')
                r = phase_row(p, context['row_id'])
                out = episode_dir(job, r['case_id'], r['arm'])
                require(not r.get('reuse_source') and context == dict(row_id=r['row_id'], case_id=r['case_id'], arm=r['arm'], out=str(out)),
                    'actual suite phase row/arm/out differs or read-only T reuse was executed')
                kind = 'reset' if request.get('reset') else 'cache' if request.get('compute_kv_cache') else 'policy'
                if kind == 'reset':
                    require(active['out'] is None and len(active['completed']) < len(registered_rows)
                        and registered_rows[len(active['completed'])]['row_id'] == r['row_id']
                        and datetime.now(timezone.utc) < datetime.fromisoformat(p['absolute_work_stop'])
                        and not (job / 'STOP_REQUESTED').exists(), 'fresh ordered complete suite row required')
                    trace = request.get('p0_trace_context') or {}
                    require(trace.get('task_index') == r['task'] and trace.get('episode_index') == r['init']
                        and trace.get('libero_benchmark') == p['suite'], 'original client suite/task/init differs')
                    out.mkdir(parents=True, exist_ok=True)
                    require(not (out / 'COUNTER.json').exists(), 'no suite episode retry/overwrite')
                    active.update(row=r, out=out, arm=r['arm'], saves=[], failed=False, collector_records=[], collector_errors=[])
                    evidence.set_episode(case=r, arm=r['arm'], out=out)
                    controller.set_arm(r['arm'])
                    s.save_root = str(out / 'server_output')
                    restore_rng(s, initial_rng)
                else:
                    require(active['row'] is not None and active['row']['row_id'] == r['row_id']
                        and active['out'] == out and not active['failed'], 'request changed active suite episode')
                # Bookkeeping is consumed at this routing boundary. Original
                # WAM and C/P semantic comparisons receive original payloads.
                business = {k: v for k, v in request.items() if k != 'p0_suite_context'}
                if collector is not None and kind == 'policy':
                    trace = dict(request.get('p0_trace_context') or {})
                    try:
                        collector.begin_chunk(dict(row_id=r['row_id'], case_id=r['case_id'], task=r['task'], init=r['init'],
                            arm=r['arm'], out=str(out), rollout_chunk=trace['rollout_chunk'],
                            role=split_role(p['phase'], r['init']), original_trace=trace))
                    except Exception as exc:
                        collector_error('begin_chunk', exc)
                try:
                    result = original_infer(business)
                except BaseException as exc:
                    active['failed'] = True
                    if collector is not None and kind == 'policy':
                        try:
                            collector.abort_chunk(repr(exc))
                        except Exception as observation_exc:
                            collector_error('abort_chunk', observation_exc)
                    write(out / 'SERVER-REQUEST-FAILED.json', dict(schema='tf-suite-request-failure/v1', row_id=r['row_id'],
                        case_id=r['case_id'], arm=r['arm'], error=repr(exc), permissions=PERMISSIONS))
                    raise
                finally:
                    if evidence.requests:
                        evidence.requests[-1].update(schema='tf-suite-request/v1', row_id=r['row_id'], suite=p['suite'], phase=p['phase'])
                if collector is not None and kind == 'policy':
                    try:
                        receipt = collector.end_chunk()
                        active['collector_records'].append(receipt)
                        if str(receipt.get('status', '')).startswith('INVALID'):
                            collector_error('end_chunk_invalid', ValueError(str(receipt)))
                    except Exception as exc:
                        collector_error('end_chunk', exc)
                return result
        s.infer = types.MethodType(infer, s)

        def finish_episode():
            out, r, arm = active['out'], active['row'], active['arm']
            executor.submit(lambda: None).result(timeout=90)
            torch.cuda.synchronize(s.device)
            require(not active['failed'] and proxy.pending is None, 'native suite request or route failed')
            terminal = terminal_record(out, r, arm)
            kept = [x for x in active['saves'] if not x['suppressed']]
            require(len(kept) == 2 * terminal['policy_chunks'] and len({x['path'] for x in kept}) == len(kept),
                'all native suite action/latent outputs required')
            for item in kept:
                value = torch.load(item['path'], map_location='cpu', weights_only=True)
                require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), 'saved suite tensor invalid')
            counter, audit = evidence.end_episode(terminal)
            if p['phase'] != 'FINAL_EVAL':
                evidence.case_references.clear()
                audit['first2S']['collection_only_reference_status'] = audit['first2S']['status']
                audit['first2S']['status'] = 'NOT_A_PAIRED_COMPARISON_DATA_COLLECTION'
                audit['first2S']['not_a_quality_certificate'] = True
            if collector is not None:
                try:
                    require(len(active['collector_records']) == terminal['policy_chunks'], 'every original policy needs one compact shard')
                    for chunk, receipt in enumerate(active['collector_records']):
                        context = receipt['context']
                        require(receipt['status'] == 'COMPLETE_POLICY_CHUNK_PENDING_EPISODE_AUDIT'
                            and receipt['rows'] == receipt['reached_rows'] > 0 and receipt['extra_teacher_forwards'] == 0
                            and context['row_id'] == r['row_id'] and context['case_id'] == r['case_id']
                            and context['arm'] == arm and context['task'] == r['task'] and context['init'] == r['init']
                            and context['out'] == str(out) and context['rollout_chunk'] == chunk
                            and context['role'] == split_role(p['phase'], r['init'])
                            and Path(receipt['path']).resolve(strict=True).is_relative_to(Path(p['collector_output_dir']).resolve()),
                            'compact shard must cover the same complete natural episode and actual chunk')
                except Exception as exc:
                    collector_error('whole_episode_compact_qualification', exc)
            counter.update(schema='tf-suite-episode-counter/v1', row_id=r['row_id'], suite=p['suite'], phase=p['phase'],
                save_async=dict(original_executor_drained=True,
                    full_visual_obs_suppressed=sum(x['suppressed'] for x in active['saves']),
                    kept_artifacts=[dict(**x, bytes=Path(x['path']).stat().st_size) for x in kept]),
                collector_invalid=bool(active['collector_errors']), collector_records=active['collector_records'],
                collector_errors=active['collector_errors'])
            observer = getattr(predictor, 'inline_observer', None)
            counter['adapter_inline_observation'] = observer.summary() if observer is not None else 'NOT_APPLICABLE_T_ONLY'
            audit.update(schema='tf-suite-episode-audit/v1', row_id=r['row_id'], suite=p['suite'], phase=p['phase'],
                collector_invalid=bool(active['collector_errors']), collector_status='INVALID' if active['collector_errors'] else
                    'COMPLETE_POLICY_SHARDS_PENDING_DATASET_AUDIT' if collector is not None else 'NOT_COLLECTED_EVAL')
            write(out / 'COUNTER.json', counter)
            write(out / 'AUDIT.json', audit)
            if collector is not None:
                write(out / 'COLLECTION-AUDIT.json', dict(schema='tf-suite-whole-episode-collection/v1',
                    status='INVALID_COMPLETE_EPISODE_COLLECTION' if active['collector_errors'] else 'PASS_COMPLETE_EPISODE_COLLECTION',
                    row_id=r['row_id'], case_id=r['case_id'], suite=p['suite'], phase=p['phase'], task=r['task'], init=r['init'],
                    arm=arm, role=split_role(p['phase'], r['init']), full_episode_terminal=True,
                    success=terminal['success'], env_steps=terminal['env_steps'], policy_chunks=terminal['policy_chunks'],
                    shards=active['collector_records'], collector_errors=active['collector_errors'],
                    training_admissible=not active['collector_errors'],
                    terminal_source=str(out / 'EPISODE-TERMINAL.json'), counter_source=str(out / 'COUNTER.json'),
                    audit_source=str(out / 'AUDIT.json'), dataframe_content_audit='REQUIRED_BY_PACK_NOT_REPLACED_BY_THIS_RECEIPT',
                    no_extra_teacher_forwards=True, permissions=PERMISSIONS))
            component_pass = audit['status'] == 'PASS_COMPLETE_EPISODE_COMPONENTS'
            write(out / 'SERVER-TERMINAL.json', dict(schema='tf-suite-server-terminal/v1',
                status='COMPLETE_FULL_EPISODE_COMPONENTS' if component_pass else 'COMPLETE_EPISODE_COMPONENT_AUDIT_FAILED',
                row_id=r['row_id'], case_id=r['case_id'], arm=arm, suite=p['suite'], full_episode_terminal=True,
                success=terminal['success'], env_steps=terminal['env_steps'], async_drained=True, permissions=PERMISSIONS))
            active['completed'].add(r['row_id'])
            active.update(row=None, out=None, arm=None, saves=[], failed=False, collector_records=[], collector_errors=[])
            write(out / 'FINISH_READY.json', dict(status='PASS' if component_pass else 'FAILED_COMPONENT_AUDIT',
                row_id=r['row_id'], case_id=r['case_id'], arm=arm, full_episode_terminal=True, async_drained=True, permissions=PERMISSIONS))
            evidence.release_written_episode_observations()
            controller.records.clear()

        def monitor():
            try:
                while not shutdown.wait(.2):
                    with gate:
                        if active['out'] is not None:
                            if (active['out'] / 'FINISH_REQUESTED').exists():
                                finish_episode()
                            continue
                        if (worker / 'SHUTDOWN_REQUESTED').exists():
                            write(worker / 'SHUTDOWN-READY.json', dict(status='IDLE_READY_TO_EXIT', pid=os.getpid(), permissions=PERMISSIONS))
                            shutdown.set()
                            os.kill(os.getpid(), signal.SIGINT)
                            return
            except BaseException as exc:
                failure = dict(status='FAILED_INLINE_SUITE_AUDIT', error=repr(exc), traceback=traceback.format_exc(), permissions=PERMISSIONS)
                with gate:
                    if active['out'] is not None:
                        out = active['out']
                        if not (out / 'COUNTER.json').exists() and evidence.episode is not None:
                            partial = evidence.partial_episode_counter()
                            partial.update(schema='tf-suite-episode-counter/v1', row_id=active['row']['row_id'], suite=p['suite'], phase=p['phase'])
                            write(out / 'COUNTER.json', partial)
                        for name in ('AUDIT.json', 'SERVER-TERMINAL.json', 'FINISH_READY.json'):
                            if not (out / name).exists():
                                write(out / name, failure)
                    if not (worker / 'FAILED.json').exists():
                        write(worker / 'FAILED.json', failure)
                    atomic_current(job / 'WORKER-0-STOP.json', failure)
                shutdown.set()
                os.kill(os.getpid(), signal.SIGINT)

        restore_rng(s, initial_rng)
        require(compare_raw(capture_rng(s), initial_rng)['equal'], 'suite setup original target RNG restoration failed')
        write(worker / 'DEPLOYMENT.json', dict(schema='tf-suite-server-deployment/v1', suite_config=config_receipt, predictor=deployment, text_bank=bank_receipt,
            shared_prefix_source_admission=online.no_digest_admission_provenance, permissions=PERMISSIONS))
        threading.Thread(target=monitor, daemon=True).start()
        evidence.pulse()
        write(worker / 'READY.json', dict(schema='tf-suite-server-ready/v1', status='READY', pid=os.getpid(), worker_index=0, physical_GPU_index=p['gpu_index'],
            port=args.port, master_port=args.master_port, model_persistent=True, suite=p['suite'], phase=p['phase'],
            draft_checkpoint=predictor.checkpoint_path, no_draft_loaded=all_target, target_forward_cap=args.forward_cap,
            per_episode_forward_cap=4000, setup_RNG_restored=True, target_RNG_captured_before_draft=True,
            original_target_initial_RNG_file='TARGET-INITIAL-RNG.pt', O3_installed=True, complete_text_bank=True,
            suite_quantiles_config_bound=True, norm_effective_tensor_audit='ON_EACH_ACTUAL_RESET',
            GPU_identity=evidence.gpu_identity, no_SHA=True, independent_tests_executed=False, permissions=PERMISSIONS))
        try:
            return original_run(s, local_rank, host, port)
        finally:
            shutdown.set()
            if collector is not None:
                collector.close()
            observer = getattr(predictor, 'inline_observer', None)
            if observer is not None:
                observer.close()
            source.save_async, s.infer, s._reset, s._get_t5_prompt_embeds = original_save, original_infer, original_reset, original_t5
            controller.close()
            evidence.close()
            online.close()
            restore_bank()
            executor.submit(lambda: None).result(timeout=90)
            if not (worker / 'SERVER-EXIT.json').exists():
                write(worker / 'SERVER-EXIT.json', dict(status='SERVER_MAIN_RETURNED', pid=os.getpid(),
                    target_forward_attempts=evidence.attempted, active_target_forwards=evidence.active,
                    request_active=evidence.request_active, permissions=PERMISSIONS))

    source.run_async_server_mode = serve
    source.init_logger()
    sys.argv = [sys.argv[0], '--config-name', 'libero', '--port', str(args.port), '--save_root', str(worker / 'server_output')]
    try:
        return source.main()
    except KeyboardInterrupt:
        if not shutdown.is_set():
            raise
        return 2 if (worker / 'FAILED.json').exists() else 0
    except BaseException as exc:
        if not (worker / 'FAILED.json').exists():
            write(worker / 'FAILED.json', dict(status='FAILED_SUITE_SERVER_LIFECYCLE', error=repr(exc), traceback=traceback.format_exc(), permissions=PERMISSIONS))
        raise


if __name__ == '__main__':
    raise SystemExit(main())



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

from tf_fullsuite_eval_v2.protocol import WORKTREE, PERMISSIONS, episode_dir, case, load, read, write, require


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
    require(terminal['schema'] == 'tf-fullsuite-full-episode-terminal/v1'
        and terminal['status'] == 'COMPLETE_RECORDED_FULL_EPISODE'
        and terminal['case_id'] == row['case_id'] and terminal['arm'] == arm
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
    parser.add_argument('--worker-index', type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--forward-cap', type=int, required=True)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    p = load(job)
    worker = job/f'server-worker{args.worker_index}'
    worker.mkdir(parents=True, exist_ok=True)
    require(not (worker/'READY.json').exists() and not (worker/'FAILED.json').exists(),
        'one fresh persistent worker lifecycle; no automatic retry')
    require(type(p['worker_forward_caps'][str(args.worker_index)]) is int
        and args.forward_cap == p['worker_forward_caps'][str(args.worker_index)] > 0
        and sum(p['worker_forward_caps'].values()) + p['prior_target_forward_debit']
            <= p['job_forward_attempt_cap'] == 6001000,
        'exact worker cap and unchanged predecessor-debited series cap required')
    require(1 <= args.port <= 65535 and 1 <= args.master_port <= 65535
        and (args.port, args.master_port) == (19781 + args.worker_index, 29781 + args.worker_index),
        'registered distinct four-worker service/master ports required')
    require(datetime.now(timezone.utc) < datetime.fromisoformat(p['absolute_work_stop']),
        'registered absolute full-suite work window ended')
    require(len(os.sched_getaffinity(0)) <= 2, 'server must inherit its two registered CPU cores')
    require(os.getenv('RESET_TEXT_BANK_ENABLED') == '1'
        and os.getenv('CUDA_VISIBLE_DEVICES') == str(args.worker_index % 2),
        'one explicitly assigned visible GPU and reset bank required')
    require(not any(os.getenv(key) for key in ('WAM_P0_STATE_TEACHER_DIR', 'WAM_P0_TRACE_DIR',
        'P0_NATIVE_DIAGNOSTIC_DIR')), 'no independent teacher/trace/replay paths')
    require(os.getenv('LINGBOT_SEED', str(p['evaluation_seed'])) == str(p['evaluation_seed']),
        'registered original target evaluation seed differs')
    os.environ.update(MASTER_PORT=str(args.master_port), RANK='0', LOCAL_RANK='0', WORLD_SIZE='1')
    os.environ['LINGBOT_SEED'] = str(p['evaluation_seed'])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    modules = prefix_modules()
    from wan_va import wan_va_server as source
    require(Path(source.__file__).resolve() == WORKTREE/'wan_va/wan_va_server.py',
        'original native WAM source path changed')
    from tf_fullsuite_eval_v2.controller import Controller
    from tf_fullsuite_eval_v2.evidence import ExecutionEvidence, atomic_current
    from tf_fullsuite_eval_v2.deployment import load_predictor, load_text_bank
    from tf_fullsuite_eval_v2.shared_prefix_no_digest import install_o3
    from tf_fullsuite_eval_v2.state import capture_rng, restore_rng, capture_full_state, compare_raw

    original_run = source.run_async_server_mode
    shutdown_requested = threading.Event()

    def serve(s, local_rank, host, port):
        require(local_rank == 0 and int(port) == args.port, 'one-rank original server endpoint required')
        validate_server(s, p)
        # Capture original target-only initialization before any draft/bank setup.
        target_initial_rng = capture_rng(s)
        torch.save(target_initial_rng, worker/'TARGET-INITIAL-RNG.pt')
        s.action_per_frame = int(s.job_config.action_per_frame)
        predictor, deployment = load_predictor(p, s)
        s._p0_action_draft_b2_predictor = predictor
        s._p0_action_draft_b2_checkpoint = predictor.checkpoint_path
        proxy = modules['selection'].install(s, p['selector'])
        modules['buffers'].prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
        online = install_o3(s, original=modules['measured_online'],
            source_reference=p['target_source_reference'])
        bank, bank_receipt, restore_bank = load_text_bank(p, s)
        original_t5 = s._get_t5_prompt_embeds

        def unexpected_text(_server, *values, **keywords):
            raise RuntimeError('complete512-token bank required; no text encoder fallback')

        s._get_t5_prompt_embeds = types.MethodType(unexpected_text, s)
        evidence = ExecutionEvidence(s, shared=online, proxy=proxy, buffers=modules['buffers'],
            cap=args.forward_cap, worker_root=worker)
        properties = torch.cuda.get_device_properties(s.device)
        evidence.gpu_identity = dict(uuid=str(getattr(properties, 'uuid', None) or 'UNKNOWN'),
            name=str(properties.name), logical_device=str(s.device),
            CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),
            source='properties of the already initialized actual model device',
            metadata_query_is_not_independent_model_test=True)
        controller = Controller(s, runtime=evidence, evidence=evidence,
            expected_checkpoint=predictor.checkpoint_path, expected_threshold=p['threshold'], periodic=0).install()
        original_reset, original_infer = s._reset, s.infer
        original_save = source.save_async
        executor = original_save.__globals__.get('executor')
        require(executor is not None and executor._max_workers == 1, 'original save executor required')
        gate = threading.RLock()
        active = dict(row=None, arm=None, out=None, saves=[], failed=False,
            completed=set(), admitted_case=None)
        replay, replay_integration_gaps = None, []

        def replay_gap(stage, exc):
            # Fixture availability is separate from actual robot outcomes.
            # Never turn an optional capture failure into an algorithm failure.
            gap = dict(stage=stage, error=repr(exc), traceback=traceback.format_exc(),
                no_extra_forward=True, permissions=PERMISSIONS)
            if len(replay_integration_gaps) < 16:
                replay_integration_gaps.append(gap)
            path = worker/'REPLAY-CAPTURE-INTEGRATION-GAP.json'
            try:
                if not path.exists():
                    write(path, dict(schema='tf-fullsuite-replay-integration-gap/v1',
                        status='UNKNOWN_OPTIONAL_REPLAY_CAPTURE', **gap))
            except Exception:
                # Disk/resource guards still retain responsibility for limits.
                pass

        def observe_replay(name, *values, **keywords):
            if replay is None:
                return None
            try:
                return getattr(replay, name)(*values, **keywords)
            except Exception as exc:
                replay_gap(name, exc)
                return None

        try:
            from tf_fullsuite_eval_v2.replay_capture import ReplayCapture
            replay = ReplayCapture(server=s, worker_index=args.worker_index, job=job,
                source=source, runtime=evidence, proxy=proxy, metadata=dict(
                    scientific_commit=p['scientific_commit'],
                    source_code_commit=p['source_code_commit'],
                    runtime_mathematical_parent=p.get('prior_implementation_math_commit',
                        p.get('operational_math_commit', 'UNKNOWN')),
                    operational_math_commit=p.get('prior_implementation_math_commit',
                        p.get('operational_math_commit', 'UNKNOWN')),
                    target_checkpoint_id=p['target_checkpoint_id'],
                    selected_checkpoint=p['selected']['path'], selected_condition='TF_ON',
                    production_mode=p['mode'], production_selector=p['selector'],
                    production_threshold=p['threshold'], normalization_refit=False,
                    GPU_identity=evidence.gpu_identity, permissions=PERMISSIONS))
        except Exception as exc:
            replay_gap('construction', exc)

        def filtered_save(value, path, *values, **keywords):
            path = Path(path)
            require(active['out'] is not None and path.resolve().is_relative_to(active['out']/'server_output'),
                'original async output escaped its owned episode')
            suppressed = path.name.startswith('obs_data_') and path.suffix == '.pt'
            active['saves'].append(dict(path=str(path), name=path.name, suppressed=suppressed))
            if suppressed:
                return None
            require(path.name.startswith(('actions_', 'latents_')) and path.suffix == '.pt',
                'unexpected source output; do not silently suppress evidence')
            evidence.saved(path.name, value)
            observe_replay('saved', path.name, value)
            return original_save(value, path, *values, **keywords)

        source.save_async = filtered_save

        def reset(_server, prompt=None):
            require(active['out'] is not None, 'registered episode must own reset')
            before = capture_rng(s)
            start = time.monotonic()
            result = original_reset(prompt=prompt)
            torch.cuda.synchronize(s.device)
            after = capture_rng(s)
            check = compare_raw(before, after)
            require(check['equal'], 'bank reset consumed original target RNG')
            require(compare_raw(after, target_initial_rng)['equal'], 'reset target-only RNG differs')
            for text, value in ((prompt, s.prompt_embeds), ('', s.negative_prompt_embeds)):
                expected = bank.get(text, device=s.device, dtype=s.dtype, sequence_length=512)
                require(torch.equal(value, expected), 'actual complete text embedding differs')
            torch.save(after, active['out']/'RNG-RESET.pt')
            write(active['out']/'SERVER-RESET-RNG.json', dict(
                schema='tf-fullsuite-server-reset-rng/v1',
                CPU=dict(dtype=str(after['CPU'].dtype), shape=list(after['CPU'].shape),
                    values=after['CPU'].tolist()),
                CUDA=dict(dtype=str(after['CUDA'].dtype), shape=list(after['CUDA'].shape),
                    values=after['CUDA'].tolist()),
                device_identity=evidence.gpu_identity, actual_initialized_server_rng=True,
                permissions=PERMISSIONS))
            write(active['out']/'RESET.json', dict(schema='tf-fullsuite-reset/v1',
                status='PASS_ORIGINAL_FULL_RESET_BANK_AND_RNG', case_id=active['row']['case_id'],
                arm=active['arm'], target_RNG_before_draft=True, RNG_unchanged=True,
                raw_rng_file='RNG-RESET.pt', reset_wall_sec=time.monotonic()-start,
                remote_hashes_executed=False, permissions=PERMISSIONS))
            return result

        s._reset = types.MethodType(reset, s)

        def infer(_server, request):
            with gate:
                context = request.get('p0_fullsuite_context')
                require(isinstance(context, dict) and set(context) == {'case_id', 'arm', 'out'},
                    'every original request requires the separate registered full-suite context')
                row = case(p, context['case_id'])
                arm, out = context['arm'], episode_dir(job, row['case_id'], context['arm'])
                require(arm in ('T', 'C', 'P') and context['out'] == str(out)
                    and row['execution_worker_index'] == args.worker_index
                    and row['proposed_GPU_index'] == args.worker_index % 2,
                    'case/arm/output/physical-worker assignment changed')
                if request.get('reset'):
                    require(row['case_id'] not in p['prior_completed_case_ids']
                        and row['case_id'] not in p.get('blocked_prior_partial_case_ids', []),
                        'predecessor complete or partial cases cannot become new replacement episodes')
                    require(active['out'] is None and (row['case_id'], arm) not in active['completed'],
                        'only a fresh full episode can reset this persistent worker')
                    require(datetime.now(timezone.utc) < datetime.fromisoformat(p['absolute_work_stop'])
                        and not (job/'STOP_REQUESTED').exists(), 'no new episode after work stop')
                    require(active['admitted_case'] in (None, row['case_id']),
                        'all three arms of the admitted case must close before a new case')
                    order = row['within_case_order'].split('_')
                    done = [a for a in order if (row['case_id'], a) in active['completed']]
                    require(len(done) < 3 and arm == order[len(done)], 'within-case T/C/P order changed')
                    # This newly authorized descriptive full-suite campaign
                    # retains passive raw mismatches as INVALID comparison
                    # evidence and continues the remaining fixed cases. Actual
                    # API/route/math contracts and resource limits still raise.
                    out.mkdir(parents=True, exist_ok=True)
                    require(not (out/'COUNTER.json').exists() and not (out/'SERVER-TERMINAL.json').exists(),
                        'no receipt overwrite or replacement episode')
                    trace = request.get('p0_trace_context') or {}
                    require(trace.get('task_index') == row['task'] and trace.get('episode_index') == row['init'],
                        'real LIBERO task/init differs from registered case')
                    active.update(row=row, arm=arm, out=out, saves=[], failed=False,
                        admitted_case=row['case_id'], replay_observer_start=(
                            None if replay is None else replay.observer_host_sec))
                    evidence.set_episode(case=row, arm=arm, out=out)
                    controller.set_arm(arm)
                    s.save_root = str(out/'server_output')
                    restore_rng(s, target_initial_rng)
                else:
                    require(active['row'] == row and active['arm'] == arm and active['out'] == out,
                        'request escaped its active full episode')
                replay_token = observe_replay('begin', request, context=context, case=row)
                try:
                    result = original_infer(request)
                    observe_replay('end', replay_token, result=result)
                    return result
                except BaseException as exc:
                    observe_replay('fail', replay_token, repr(exc))
                    active['failed'] = True
                    failure = out/'SERVER-REQUEST-FAILED.json'
                    if not failure.exists():
                        write(failure, dict(schema='tf-fullsuite-request-failure/v1', status='FAILED',
                            case_id=row['case_id'], arm=arm, error=repr(exc),
                            traceback=traceback.format_exc(),
                            first2S=None if evidence.episode is None else evidence.episode['first2_results'],
                            permissions=PERMISSIONS))
                    raise

        s.infer = types.MethodType(infer, s)

        def finish_episode():
            out, row, arm = active['out'], active['row'], active['arm']
            executor.submit(lambda: None).result(timeout=90)
            torch.cuda.synchronize(s.device)
            require(not active['failed'] and proxy.pending is None, 'episode had failed request/route')
            terminal = terminal_record(out, row, arm)
            saved = active['saves']
            kept = [x for x in saved if not x['suppressed']]
            require(len(kept) == 2*terminal['policy_chunks']
                and len({x['path'] for x in kept}) == len(kept), 'all original actions/latents required')
            require(sum(x['suppressed'] for x in saved)
                == sum(r['kind'] == 'cache' for r in
                    evidence.requests[evidence.episode['start_requests']:]),
                'cache/obs suppression coverage differs')
            for item in kept:
                path = Path(item['path'])
                require(path.is_file() and path.stat().st_size > 0, 'drained original tensor artifact missing')
                value = torch.load(path, map_location='cpu', weights_only=True)
                require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()),
                    'original saved policy tensor is invalid')
            counter, audit = evidence.end_episode(terminal)
            counter['save_async'] = dict(original_executor_drained=True,
                full_visual_obs_suppressed=sum(x['suppressed'] for x in saved),
                kept_artifacts=[dict(**x, bytes=Path(x['path']).stat().st_size) for x in kept])
            counter['persistent_model'] = True
            counter['optional_replay_capture'] = dict(
                integration_status='UNKNOWN_GAPS_RECORDED' if replay_integration_gaps else
                    'UNKNOWN_NOT_INSTALLED' if replay is None else
                    'NOT_SELECTED_WORKER' if args.worker_index != 0 else
                    'UNKNOWN_DISABLED_CHECK_FIXTURE_INDEX' if replay.disabled else 'OBSERVER_INSTALLED',
                integration_gaps=list(replay_integration_gaps),
                fixture_index=str(job/'replay-fixtures'/'INDEX.json'),
                observer_host_seconds_cumulative=(
                    'UNKNOWN' if replay is None else replay.observer_host_sec),
                observer_host_seconds_this_episode=(
                    'UNKNOWN' if replay is None or active.get('replay_observer_start') is None else
                    replay.observer_host_sec-active['replay_observer_start']),
                overhead_scope='optional fixture observer only; original pairing audit remains separate',
                capture_overhead_is_not_formal_timing=True, extra_model_forwards=0)
            write(out/'COUNTER.json', counter)
            write(out/'AUDIT.json', audit)
            component_pass = audit['status'] == 'PASS_COMPLETE_EPISODE_COMPONENTS'
            write(out/'SERVER-TERMINAL.json', dict(schema='tf-fullsuite-server-terminal/v1',
                status=('COMPLETE_FULL_EPISODE_COMPONENTS' if component_pass else
                    'COMPLETE_EPISODE_COMPONENT_AUDIT_FAILED'), case_id=row['case_id'], arm=arm,
                full_episode_terminal=True, success=terminal['success'], env_steps=terminal['env_steps'],
                async_drained=True, permissions=PERMISSIONS))
            active['completed'].add((row['case_id'], arm))
            if all((row['case_id'], value) in active['completed'] for value in ('T', 'C', 'P')):
                require(not evidence.case_references, 'C/P first-two-S references did not close')
                active['admitted_case'] = None
            active.update(row=None, arm=None, out=None, saves=[], failed=False)
            write(out/'FINISH_READY.json', dict(status='PASS' if component_pass else 'FAILED_COMPONENT_AUDIT',
                case_id=row['case_id'], arm=arm,
                full_episode_terminal=True, async_drained=True, permissions=PERMISSIONS))
            # Full immutable receipts already contain these observations; keep
            # only the cross-arm raw reference until its same-case partner.
            evidence.release_written_episode_observations()
            controller.records.clear()

        def monitor():
            try:
                while not shutdown_requested.wait(.2):
                    with gate:
                        if active['out'] is not None:
                            if (active['out']/'FINISH_REQUESTED').exists():
                                finish_episode()
                            continue
                        if (worker/'SHUTDOWN_REQUESTED').exists():
                            write(worker/'SHUTDOWN-READY.json', dict(status='IDLE_READY_TO_EXIT',
                                pid=os.getpid(), target_forward_attempts=evidence.attempted,
                                permissions=PERMISSIONS))
                            shutdown_requested.set()
                            os.kill(os.getpid(), signal.SIGINT)
                            return
            except BaseException as exc:
                failure = dict(status='FAILED_INLINE_EPISODE_AUDIT', error=repr(exc),
                    traceback=traceback.format_exc(), permissions=PERMISSIONS)
                with gate:
                    if active['out'] is not None:
                        out = active['out']
                        if not (out/'COUNTER.json').exists() and evidence.episode is not None:
                            write(out/'COUNTER.json', evidence.partial_episode_counter())
                        if not (out/'AUDIT.json').exists():
                            write(out/'AUDIT.json', dict(schema='tf-fullsuite-episode-audit/v1', **failure,
                                case_id=active['row']['case_id'], arm=active['arm'],
                                first2S=dict(status='FAILED_OR_PARTIAL',
                                    records=[] if evidence.episode is None else evidence.episode['first2_results']),
                                numerical_isolation=dict(status='FAILED_OR_PARTIAL_INLINE_AUDIT',
                                    remote_SHA_executed=False, old_adapter_CAL_not_inherited=True)))
                        if not (out/'SERVER-TERMINAL.json').exists():
                            write(out/'SERVER-TERMINAL.json', failure)
                        if not (out/'FINISH_READY.json').exists():
                            write(out/'FINISH_READY.json', failure)
                    if not (worker/'FAILED.json').exists():
                        write(worker/'FAILED.json', failure)
                    atomic_current(job/f'WORKER-{args.worker_index}-STOP.json', failure)
                shutdown_requested.set()
                os.kill(os.getpid(), signal.SIGINT)

        restore_rng(s, target_initial_rng)
        require(compare_raw(capture_rng(s), target_initial_rng)['equal'], 'setup target RNG restoration failed')
        write(worker/'DEPLOYMENT.json', dict(predictor=deployment, text_bank=bank_receipt,
            shared_prefix_source_admission=online.no_digest_admission_provenance,
            no_independent_CAL_or_tests_executed=True, remote_SHA_executed=False))
        threading.Thread(target=monitor, daemon=True).start()
        evidence.pulse()
        write(worker/'READY.json', dict(status='READY', pid=os.getpid(), worker_index=args.worker_index,
            port=args.port, master_port=args.master_port, model_persistent=True,
            physical_GPU_index=args.worker_index % 2, GPU_shared_for_quality=True,
            TF_ON_checkpoint=predictor.checkpoint_path, periodic_schedule_per_episode=True,
            original_target_initial_RNG_file='TARGET-INITIAL-RNG.pt',
            setup_RNG_restored=True, target_RNG_captured_before_draft=True,
            target_forward_cap=args.forward_cap, per_episode_forward_cap=4000,
            O3_installed=True, complete_text_bank=True, remote_SHA_executed=False,
            independent_tests_executed=False, four_permissions=PERMISSIONS,
            permissions=PERMISSIONS, CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),
            logical_device=str(s.device), physical_GPU_name=torch.cuda.get_device_name(s.device),
            GPU_identity=evidence.gpu_identity,
            audit_events_in_task=True, performance_claim=False))
        try:
            return original_run(s, local_rank, host, port)
        finally:
            shutdown_requested.set()
            observe_replay('close')
            source.save_async = original_save
            s.infer = original_infer
            controller.close()
            evidence.close()
            online.close()
            restore_bank()
            s._get_t5_prompt_embeds = original_t5
            s._reset = original_reset
            executor.submit(lambda: None).result(timeout=90)
            path = worker/'SERVER-EXIT.json'
            if not path.exists():
                write(path, dict(status='SERVER_MAIN_RETURNED', pid=os.getpid(),
                    target_forward_attempts=evidence.attempted,
                    active_target_forwards=evidence.active, request_active=evidence.request_active,
                    permissions=PERMISSIONS))

    source.run_async_server_mode = serve
    source.init_logger()
    sys.argv = [sys.argv[0], '--config-name', 'libero', '--port', str(args.port),
        '--save_root', str(worker/'server_output')]
    try:
        return source.main()
    except KeyboardInterrupt:
        if not shutdown_requested.is_set():
            raise
        return 2 if (worker/'FAILED.json').exists() else 0
    except BaseException as exc:
        if not (worker/'FAILED.json').exists():
            write(worker/'FAILED.json', dict(status='FAILED_SERVER_LIFECYCLE', error=repr(exc),
                traceback=traceback.format_exc(), permissions=PERMISSIONS))
        raise


if __name__ == '__main__':
    raise SystemExit(main())

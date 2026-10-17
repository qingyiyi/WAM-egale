"""Registered CPU audit of the complete recorded-episode O3 collector proof.

No model, teacher, training, simulator, or CUDA operation is performed. The
qualification source must already have worker, supervisor, and clock terminal
receipts. This auditor never edits those receipts or their failed predecessor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = EXP.parent.parent
RT = ROOT / '投机采样/results/runtime'
COMMIT = 'b0dfac3a02f65de33a1217246b83003615166640'
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
                  noninferiority_established=False, formal_performance_permission=False)
SCHEMA = 'tf-effect-collector-qualification-cpu-audit/v1'
RECEIPTS = ('plan.json', 'PREREGISTRATION.json', 'WORKER-RESULT.json',
            'SUPERVISOR-RESULT.json', 'FINAL-OWNED-CLEANUP.json',
            'CLOCK-GUARD.json', 'CLOCK-RESULT.json', 'RUN-STARTED.json')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def write(path, value, *, replace=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(text)
        os.replace(tmp, path)
    else:
        with path.open('x') as handle:
            handle.write(text)


def runtime():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicitly hide CUDA for this CPU-only job')
    allowed = sorted(os.sched_getaffinity(0))
    require(bool(allowed), 'no allowed CPU cores')
    os.sched_setaffinity(0, set(allowed[:2]))
    require(len(os.sched_getaffinity(0)) <= 2, 'at most two CPU cores')


def alarm(registration):
    remaining = float(registration['deadline_epoch']) - time.time()
    require(remaining > 0, 'registered wall cap expired; retry must not reset it')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU audit wall cap')))
    signal.alarm(max(1, math.ceil(remaining)))


def storage(job, cap):
    size = sum(p.stat().st_size for p in job.rglob('*') if p.is_file())
    require(size <= cap, 'registered CPU audit output cap exceeded')
    return size


def prepare(args):
    runtime()
    job = args.job.resolve()
    qualification = args.qualification.resolve(strict=True)
    require(job.parent == RT and not job.exists(), 'fresh own runtime namespace required')
    require(300 <= args.wall_cap_seconds <= 1800 and 16 <= args.storage_cap_mib <= 128,
            'CPU cap must be 300..1800 seconds / 16..128 MiB')
    job.mkdir()
    start = time.time()
    registration = dict(schema=SCHEMA, goal='Full CPU verification of all50 qualified ON teacher shards',
        source_commit=COMMIT, qualification=str(qualification), created_at=datetime.now(timezone.utc).isoformat(),
        birth_epoch=start, deadline_epoch=start + args.wall_cap_seconds,
        wall_cap_seconds=args.wall_cap_seconds, storage_cap_bytes=args.storage_cap_mib * 2**20,
        cpu_threads=2, cpu_affinity=sorted(os.sched_getaffinity(0)), GPU=False,
        source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        terminal_status='REGISTERED_NOT_PREPARED', numerical_norm_atol=1e-6,
        cleanup_plan='single pinned CPU process; alarm and caller external timeout; no workers/services',
        evidence_path=str(job), permissions=PERMISSIONS, new_teacher_forwards=0,
        no_training=True, no_TEST=True, no_source_evidence_modification=True)
    write(job / 'PREREGISTRATION.json', registration)
    try:
        alarm(registration)
        require(subprocess.check_output(['git', '-C', str(ROOT / 'review_exports/WAM-egale'),
            'rev-parse', 'HEAD'], text=True).strip() == COMMIT, 'fixed scientific base differs')
        require(all((qualification / name).is_file() for name in RECEIPTS),
                'qualification is not terminal; refuse to audit an incomplete job')
        q = read(qualification / 'plan.json')
        require(q['commit'] == COMMIT and q['permissions'] == PERMISSIONS and q['kind'] == 'qualification',
                'qualification identity/permissions differ')
        ep = Path(q['source_episode']).resolve(strict=True)
        evidence = [qualification / name for name in RECEIPTS]
        evidence += [qualification / name for name in ('OFF-REQUEST-PROOF.json', 'ON-REQUEST-PROOF.json')]
        evidence += [ep / 'COUNTER.json']
        traces = sorted((ep / 'direction_trace').glob('chunk-*.pt'))
        teachers = sorted((qualification / 'teacher').rglob('chunk-*.pt'))
        require(len(traces) == len(teachers) == 50, 'all50 traces and ON teacher shards required')
        evidence += traces + teachers
        if q.get('predecessor'):
            old = Path(q['predecessor'])
            evidence += [old / name for name in ('plan.json', 'FAILED.json', 'FINAL-OWNED-CLEANUP.json')]
        require(all(p.is_file() for p in evidence), 'full terminal receipts and predecessor evidence required')
        plan = dict(**registration, source_episode=str(ep), traces=[str(p) for p in traces],
            teachers=[str(p) for p in teachers], evidence_stat={str(p): stat(p) for p in evidence},
            receipt_sha256={str(p): sha(p) for p in evidence if p.suffix == '.json'},
            preregistration_sha256=sha(job / 'PREREGISTRATION.json'))
        write(job / 'PLAN.json', plan)
        write(job / 'PREPARED.json', dict(status='PREPARED_CPU_ONLY_NO_MODEL_CALLS',
            total_teacher_shards=50, total_original_traces=50, elapsed_sec=time.time() - start,
            no_CUDA_import=True, permissions=PERMISSIONS))
        storage(job, registration['storage_cap_bytes'])
    except BaseException as exc:
        write(job / 'FAILED.json', dict(status='FAILED_CPU_PREPARATION', error=repr(exc),
            traceback=traceback.format_exc(), elapsed_sec=time.time() - start, permissions=PERMISSIONS))
        raise
    finally:
        signal.alarm(0)
    return plan


def same_bytes(torch, actual, expected, label):
    require(isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor), label + ': real tensors required')
    require(actual.device.type == expected.device.type == 'cpu' and actual.shape == expected.shape
            and actual.dtype == expected.dtype, label + ': shape/dtype/device mismatch')
    require(torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                        expected.contiguous().reshape(-1).view(torch.uint8)),
            label + ': bytes differ')


def live_pin(pin):
    """Read only: PID reuse is exit of the pinned lifetime, never a kill target."""
    path = Path('/proc') / str(pin['pid'])
    try:
        fields = (path / 'stat').read_text().rpartition(') ')[2].split()
        same = int(fields[19]) == int(pin['start_ticks']) and path.stat().st_uid == int(pin['uid'])
        return dict(pid=int(pin['pid']), start_ticks=int(pin['start_ticks']),
                    same_pinned_lifetime=same, live=same and fields[0] not in ('Z', 'X'), state=fields[0])
    except FileNotFoundError:
        return dict(pid=int(pin['pid']), start_ticks=int(pin['start_ticks']), same_pinned_lifetime=False, live=False)


def terminal_proof(qualification, q, counter):
    worker = read(qualification / 'WORKER-RESULT.json')
    supervisor = read(qualification / 'SUPERVISOR-RESULT.json')
    cleanup = read(qualification / 'FINAL-OWNED-CLEANUP.json')
    clock = read(qualification / 'CLOCK-RESULT.json')
    guard = read(qualification / 'CLOCK-GUARD.json')
    for name in ('FAILED.json', 'WORKER-FAILED.json', 'CLOCK-FAILED.json', 'CLOCK-STOP.json'):
        require(not (qualification / name).exists(), 'qualification retained failure/stop: ' + name)
    require(worker['status'] == 'PASS_FULL_RECORDED_EPISODE_COLLECTOR_EQUIVALENCE'
        and worker['permissions'] == PERMISSIONS and worker['new_robot_episodes'] == 0
        and worker['extra_teacher_forwards'] == 0 and worker['actual_o3_preserved'] is True
        and worker['all_valid_KV_RNG_routes_counts_exact'] is True
        and worker['all_original_actions_video_commands_exact'] is True,
        'complete original OFF/ON equivalence worker proof required')
    require(supervisor['status'] == 'COMPLETE' and supervisor['failure'] is None
        and supervisor['permissions'] == PERMISSIONS and cleanup['all_roots_reaped'] is True
        and supervisor['cleanup'] == cleanup, 'supervisor/cleanup must be complete and agree')
    require(clock['status'] == 'CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS'
        and clock['all_observed_owned_live_processes_exited'] is True
        and clock['error'] is None and clock['external_stop_reason'] is None
        and clock['no_foreign_signals'] is True and clock['no_workload_launched'] is True,
        'independent clean clock result required; it is not a robot quality pass')
    require(guard['clock']['plan_sha256'] == sha(qualification / 'plan.json')
        and guard['no_budget_reset'] is True, 'clock must bind actual frozen plan without budget reset')
    require(float(supervisor['elapsed_sec']) <= q['max_run_wall_sec']
        and float(clock['elapsed_quality_process_seconds']) <= q['max_run_wall_sec'] + 1,
        'registered wall cap exceeded')
    end = float(clock['clock']['process_birth_monotonic']) + float(clock['elapsed_quality_process_seconds'])
    require(end <= float(clock['clock']['hard_stop_monotonic']) + 1, 'original absolute deadline exceeded')
    passes = worker['passes']
    require(len(passes) == 2 and [p['collector'] for p in passes] == ['OFF', 'ON'], 'exact OFF/ON passes required')
    for item in passes:
        require(item['policy_chunks'] == 50 and item['target_forwards'] == counter['target_forwards']
            and item['accepted_trees'] == len(counter['acceptance'])
            and item['all_original_actions_latents_and_commands_exact'] is True,
            'full per-pass original target/tree/action conservation required')
    require(passes[1]['full_valid_KV_RNG_OFF_ON_exact'] is True, 'ON must match all OFF request boundaries')
    require(worker['target_accounting']['target_intents'] == 2 * counter['target_forwards']
        and worker['target_accounting']['failed'] == worker['target_accounting']['active'] == 0,
        'full target accounting mismatch')
    off = read(qualification / 'OFF-REQUEST-PROOF.json')
    on = read(qualification / 'ON-REQUEST-PROOF.json')
    require(len(off) == len(on) == len(counter['requests']), 'all original request receipts required')
    for a, b, old in zip(off, on, counter['requests']):
        require(a['index'] == b['index'] and a['kind'] == b['kind'] == old['kind']
            and a['forwards'] == b['forwards'] == old['target_forwards']
            and b['all_valid30layer_KV_RNG_exact'] is True, 'OFF/ON per-request proof mismatch')
    pins = list(supervisor.get('observed_pins', []))
    pins += [r['pin'] for r in clock.get('process_observations', [])]
    pins += [guard['supervisor_pin'], guard['guard_pin']]
    unique = {(int(p['pid']), int(p['start_ticks']), int(p['uid'])): p for p in pins}
    observations = [live_pin(p) for p in unique.values()]
    require(observations and not any(p['live'] for p in observations), 'a pinned owned process is still live')
    predecessor = None
    if q.get('predecessor'):
        old = Path(q['predecessor'])
        old_plan, old_failure, old_cleanup = (read(old / n) for n in
            ('plan.json', 'FAILED.json', 'FINAL-OWNED-CLEANUP.json'))
        require(q.get('original_absolute_budget_retained') is True
            and q['absolute_cleanup_deadline'] == old_plan['absolute_cleanup_deadline']
            and q['absolute_work_stop'] == old_plan['absolute_work_stop']
            and old_cleanup['all_roots_reaped'] is True, 'failed predecessor budget/cleanup was not preserved')
        predecessor = dict(job=str(old), original_failure_status=old_failure.get('status'),
            failed_receipt_retained=True, original_absolute_deadline_preserved=True)
    return dict(status='PASS_TERMINAL_AND_OWNED_PROCESS_AUDIT', source_status=worker['status'],
        source_supervisor_wall_sec=supervisor['elapsed_sec'], original_target_forwards_per_pass=counter['target_forwards'],
        total_replay_target_forwards=worker['target_accounting']['target_intents'],
        new_teacher_forwards=0, original_episode_failure_not_removed=worker['original_episode_was_failed'],
        process_observations=observations, predecessor=predecessor,
        clock_is_not_quality_certificate=True)


def run(job):
    runtime()
    job = job.resolve(strict=True)
    p = read(job / 'PLAN.json')
    require(p['schema'] == SCHEMA and p['source_commit'] == COMMIT and p['permissions'] == PERMISSIONS,
            'CPU audit identity/permissions differ')
    require(not (job / 'STARTED.json').exists() and not (job / 'FAILED.json').exists(), 'one attempt; preserve failures')
    registration = read(job / 'PREREGISTRATION.json')
    require(sha(job / 'PREREGISTRATION.json') == p['preregistration_sha256']
        and sha(__file__) == p['source_sha256'], 'CPU audit source/registration changed')
    start = time.time()
    write(job / 'STARTED.json', dict(status='RUNNING_CPU_ONLY', pid=os.getpid(), start_epoch=start,
        cpu_affinity=sorted(os.sched_getaffinity(0)), permissions=PERMISSIONS))
    try:
        alarm(registration)
        for path, expected in p['evidence_stat'].items():
            require(stat(path) == expected, 'frozen source evidence changed: ' + path)
        for path, expected in p['receipt_sha256'].items():
            require(sha(path) == expected, 'frozen receipt bytes changed: ' + path)
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized(), 'CPU auditor must have no CUDA context')
        qualification, ep = Path(p['qualification']), Path(p['source_episode'])
        q, counter = read(qualification / 'plan.json'), read(ep / 'COUNTER.json')
        terminal = terminal_proof(qualification, q, counter)
        total_trees = total_rows = total_reached = cpu_depth_flips = cpu_edge_flips = 0
        max_norm_error = 0.
        chunk_reports = []
        for chunk, (source_path, teacher_path) in enumerate(zip(p['traces'], p['teachers'])):
            original = torch.load(source_path, map_location='cpu', weights_only=False)
            pack = torch.load(teacher_path, map_location='cpu', weights_only=True)
            require(pack['schema'] == 'tf-effect-o3-same-query-columns/v1'
                and pack['status'] == 'COMPLETE_CHUNK_PENDING_EXTERNAL_COUNTER_CHECK', 'complete chunk collector schema required')
            require(pack['context'] == original['context'] and pack['context']['rollout_chunk'] == chunk
                and pack['context']['task_index'] == 4 and pack['context']['episode_index'] == 26,
                'only the fixed previously exposed DEV task4/init26 qualification is admitted')
            require(pack['metadata']['checkpoint'] == q['reference_plan']['selected']['TF_CONT']
                and pack['metadata']['target'] == q['reference_plan']['target_checkpoint_id']
                and pack['metadata']['data_role'] == 'previously_unblinded_DEV_qualification_never_FIT_CAL',
                'qualification generator/target/data role differs')
            trs = [r for r in original['rounds'] if r['kind'] == 'speculative_tree']
            n = len(trs)
            require(pack['tree_count'] == pack['candidate_calls'] == pack['execute_calls'] == n
                and pack['row_count'] == n*3 and len(original['commits']) == 50,
                'one-to-one complete tree/row/commit coverage required')
            require(pack['extra_teacher_forwards'] == 0 and pack['query_and_teacher_bytes_exact'] is True
                and pack['original_O3_active_at_every_capture'] is True
                and pack['original_results_returned_unchanged'] is True, 'collector must preserve actual target/O3 semantics')
            cols, tree_cols = pack['columns'], pack['trees']
            require(all(v.device.type == 'cpu' and len(v) == n*3 for v in cols.values()), 'full CPU row geometry required')
            require(all(len(v) == n for v in tree_cols.values()), 'full per-tree geometry required')
            reached_count, chunk_norm_error, chunk_flips, chunk_edge_flips = 0, 0., 0, 0
            acceptance = [a for a in counter['acceptance'] if a['context']['rollout_chunk'] == chunk]
            require(len(acceptance) == n, 'source COUNTER per-chunk tree coverage differs')
            for j, (tr, old_route) in enumerate(zip(trs, acceptance)):
                sigma, times = tr['sigma_path'], tr['target_timesteps']
                depth, parent, child = tr['accepted_depth'], tr['selected_parent'], tr['selected_child']
                require((depth, parent, child) == (old_route['actual_depth'], old_route['actual_parent'], old_route['actual_child']),
                    'old direction trace and COUNTER route differ')
                expected_tree = dict(sigma_path=torch.tensor(sigma, dtype=torch.float64),
                    target_timesteps=torch.tensor(times, dtype=torch.float64),
                    first_costs=torch.tensor(tr['first_costs'], dtype=torch.float32),
                    second_costs=torch.tensor(tr['second_costs'], dtype=torch.float32),
                    threshold=torch.tensor(tr['threshold'], dtype=torch.float64),
                    selected_parent=torch.tensor(parent, dtype=torch.int64), selected_child=torch.tensor(child, dtype=torch.int64),
                    accepted_depth=torch.tensor(depth, dtype=torch.int64), start_step=torch.tensor(tr['start_step'], dtype=torch.int64),
                    frame_st_id=torch.tensor(original['frame_st_id'], dtype=torch.int64),
                    O3_generation=torch.tensor(j+1, dtype=torch.int64))
                for key, value in expected_tree.items():
                    same_bytes(torch, tree_cols[key][j], value, f'chunk{chunk}/tree{j}/{key}')
                threshold = float(tr['threshold'])
                require(threshold == q['reference_plan']['threshold'], 'unchanged registered threshold required')
                require(tr['first_costs'] == old_route['first_costs'] and tr['second_costs'] == old_route['second_costs'],
                        'saved original six costs differ between source records')
                for node in range(3):
                    k, offset = j*3+node, 0 if node == 0 else 1
                    previous = tr['root_previous_direction'] if node == 0 else tr['first_directions'][node-1]
                    proposals = tr['first_directions'] if node == 0 else tr['second_directions'][node-1]
                    selected = node == 0 or node-1 == parent
                    reached = node == 0 or (selected and depth >= 1)
                    committed = (node == 0 and depth >= 1) or (node > 0 and selected and depth == 2)
                    expected_row = dict(state=tr['candidate_states'][node].float().reshape(112),
                        previous=previous.float().reshape(112), target=tr['teacher_target_directions'][node].float().reshape(112),
                        proposals=proposals.float().reshape(2,112),
                        scalars=torch.tensor([sigma[offset],sigma[offset+1],times[offset]/1000.], dtype=torch.float32),
                        delta=torch.tensor(abs(sigma[offset+1]-sigma[offset]), dtype=torch.float32),
                        task=torch.tensor(4, dtype=torch.int64), init=torch.tensor(26,dtype=torch.int64),
                        episode=torch.tensor(426,dtype=torch.int64), tree_index=torch.tensor(j,dtype=torch.int64),
                        tree_step_index=torch.tensor(tr['start_step'],dtype=torch.int64),
                        step_index=torch.tensor(tr['start_step']+offset,dtype=torch.int64), node=torch.tensor(node,dtype=torch.int64),
                        accepted_depth=torch.tensor(depth,dtype=torch.int64), selected_branch=torch.tensor(selected),
                        state_reached=torch.tensor(reached), draft_edge_committed=torch.tensor(committed))
                    for key, value in expected_row.items():
                        same_bytes(torch, cols[key][k], value, f'chunk{chunk}/tree{j}/node{node}/{key}')
                    reached_count += int(reached)
                # Both operands are converted separately before subtraction.
                # CPU/GPU reduction order is not claimed bitwise equivalent.
                proposals = cols['proposals'][j*3:j*3+3].float()
                labels = cols['target'][j*3:j*3+3].float()
                residual = proposals - labels[:,None,:]
                norms = torch.linalg.vector_norm(residual,dim=2) * cols['delta'][j*3:j*3+3,None]
                saved = torch.cat((tree_cols['first_costs'][j:j+1], tree_cols['second_costs'][j]),dim=0)
                error = float((norms-saved).abs().max())
                require(math.isfinite(error) and error <= p['numerical_norm_atol'], 'CPU FP32 six-norm error exceeds preregistered 1e-6')
                chunk_norm_error = max(chunk_norm_error,error)
                chunk_edge_flips += int(((norms <= threshold) != (saved <= threshold)).sum())
                cpu_depth = 0 if float(norms[0,parent]) > threshold else 1 if float(norms[1+parent,child]) > threshold else 2
                chunk_flips += int(cpu_depth != depth)
            require(reached_count == pack['reached_row_count'], 'actual reached-node count differs')
            total_trees += n
            total_rows += n*3
            total_reached += reached_count
            max_norm_error = max(max_norm_error,chunk_norm_error)
            cpu_depth_flips += chunk_flips
            cpu_edge_flips += chunk_edge_flips
            chunk_reports.append(dict(chunk=chunk,trees=n,rows=n*3,reached_rows=reached_count,
                all_saved_fields_bytes_exact=True,cpu_norm_max_abs=chunk_norm_error,
                cpu_reduction_depth_flips=chunk_flips,cpu_reduction_edge_gate_flips=chunk_edge_flips))
            write(job / 'status.json',dict(status='AUDITING_REAL_RECORDED_TEACHER_ROWS_CPU_ONLY',
                complete_chunks=chunk+1, trees=total_trees, rows=total_rows),replace=True)
            storage(job,p['storage_cap_bytes'])
        require(total_trees == len(counter['acceptance']) == 904 and total_rows == 2712 and total_reached == 1674,
                'full frozen task4 current-TF query counts differ')
        require(not torch.cuda.is_initialized(), 'CPU audit unexpectedly acquired CUDA context')
        for path, expected in p['evidence_stat'].items():
            require(stat(path) == expected, 'source evidence changed during audit: ' + path)
        result = dict(schema=SCHEMA,status='PASS_FULL_50_CHUNK_QUALIFICATION_AUDIT',source_commit=COMMIT,
            qualification=str(qualification),source_episode=str(ep),complete_chunks=50,
            actual_teacher_trees=total_trees,actual_teacher_rows=total_rows,root_rows=total_trees,
            actually_reached_rows=total_reached,counterfactual_parent_rows=total_rows-total_reached,
            all_state_previous_target_proposal_bytes_exact=True,all_sigma_time_step_route_cost_flags_exact=True,
            counter_coverage_checked=True,teacher_cpu_norm_max_abs=max_norm_error,
            cpu_norm_atol=p['numerical_norm_atol'],cpu_norm_bitwise_GPU_equivalence_claimed=False,
            cpu_norm_uses_separate_FP32_operands_before_subtraction=True,
            cpu_reduction_depth_flips=cpu_depth_flips,deployment_acceptance_uses_original_GPU_costs=True,
            cpu_reduction_edge_gate_flips=cpu_edge_flips,
            collector_equivalence_passed=True,data_role='previously_unblinded_DEV_not_new_FIT_CAL',
            new_robot_episodes=0,new_teacher_forwards=0,training_steps=0,GPU=False,CUDA_initialized=False,
            terminal_proof=terminal,chunk_reports=chunk_reports,permissions=PERMISSIONS,
            elapsed_run_sec=time.time()-start,elapsed_registered_wall_sec=time.time()-p['birth_epoch'],
            evidence=[dict(path=path,sha256=value) for path,value in p['receipt_sha256'].items()],
            performance_claim=False,source_evidence_modified=False)
        write(job / 'RESULT.json',result)
        report = ('# O3 teacher 完整记录重放的 CPU 验收\n\n'
            f'终态：`{result["status"]}`。50 个完整记录 chunk，904 棵真实树、2712 行；'
            '1674 行真实到达，1038 行反事实 parent。不是新增闭环机器人任务。\n\n'
            'state/previous/target/proposal、实际 sigma/time/step、route、原六 cost 和全部可达标志逐字段逐位对齐。'
            f'CPU 范数最大绝对差 {max_norm_error:.9g}，预登记容差 1e-6；'
            f'CPU reduction 接受深度翻转 {cpu_depth_flips}。双方 operand 先分别转 FP32 再相减，'
            '不声称 CPU/GPU reduction 逐位相同。生产接受器继续使用原 GPU score。\n\n'
            '完整 worker/supervisor/clock、原绝对预算和自有进程退出证明均已检查；旧失败保留。'
            '此证据只放行该版本旁路采集器，不放行 TF 机器人质量、独立 TEST、非劣或正式性能。\n')
        with (job/'RESULT.md').open('x') as handle:
            handle.write(report)
        storage(job,p['storage_cap_bytes'])
        return result
    except BaseException as exc:
        write(job / 'FAILED.json',dict(status='FAILED_CPU_FULL_QUALIFICATION_AUDIT',error=repr(exc),
            traceback=traceback.format_exc(),elapsed_run_sec=time.time()-start,
            elapsed_registered_wall_sec=time.time()-registration['birth_epoch'],permissions=PERMISSIONS))
        raise
    finally:
        signal.alarm(0)
        try:
            module = sys.modules.get('torch')
            write(job / 'CPU-CLEANUP.json',dict(status='NO_CHILDREN_OR_MODEL_CALLS_CREATED',pid=os.getpid(),
                CPU_threads=2,CUDA_initialized=False if module is None else module.cuda.is_initialized(),
                process_exits_after_return=True,no_signal_to_other_processes=True))
        except Exception as exc:
            print('CPU_AUDIT_CLEANUP_MARKER_ERROR',repr(exc),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--run',action='store_true')
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--wall-cap-seconds',type=int,default=900)
    parser.add_argument('--storage-cap-mib',type=int,default=64)
    args=parser.parse_args()
    if args.prepare:
        if args.qualification is None:
            parser.error('--qualification required for prepare')
        result=prepare(args)
        print(json.dumps(dict(status='PREPARED_CPU_ONLY',job=str(args.job.resolve()),
            deadline_epoch=result['deadline_epoch'],GPU=False),ensure_ascii=False))
    else:
        result=run(args.job)
        print(json.dumps(dict(status=result['status'],complete_chunks=result['complete_chunks'],
            trees=result['actual_teacher_trees'],rows=result['actual_teacher_rows'],GPU=False),ensure_ascii=False))


if __name__=='__main__':
    main()

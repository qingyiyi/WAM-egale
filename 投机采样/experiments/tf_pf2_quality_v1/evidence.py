"""Inline real-task accounting and raw first-two-S comparison, without digests.

No job, model, simulator, tensor replay, or independent test is started here.
CUDA event markers observe the existing stream; all model/scheduler calls are
delegated once. Source profiling remains disabled and old tokens are preserved.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import threading
import time
import types

from .protocol import PERMISSIONS, require, write
from .role_contract import audit_request
from .state import capture_full_state, compare_raw


def raw_copy(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: raw_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [raw_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(raw_copy(v) for v in value)
    return copy.deepcopy(value)


def semantic_context(context):
    # UUID, output path, arm and worker labels are bookkeeping, not WAM inputs.
    return {k: raw_copy(v) for k, v in context.items() if k not in (
        'episode_attempt_id', 'campaign_id', 'parallel_worker_id', 'worker_id',
        'arm', 'pair_id', 'out')}


def semantic_request(request):
    result = {k: raw_copy(v) for k, v in request.items() if k != 'p0_pf2_context'}
    if 'p0_trace_context' in result:
        result['p0_trace_context'] = semantic_context(result['p0_trace_context'] or {})
    return result


def input_layouts(request):
    """Retain real numeric layouts before CPU copies normalize their strides."""
    import numpy as np
    import torch
    result = {}

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            result[path] = dict(kind='torch', shape=list(value.shape), dtype=str(value.dtype),
                stride_elements=list(value.stride()), device=str(value.device))
        elif isinstance(value, np.ndarray):
            result[path] = dict(kind='numpy', shape=list(value.shape), dtype=str(value.dtype),
                stride_bytes=list(value.strides))
        elif isinstance(value, dict):
            for key, item in value.items():
                if key != 'p0_pf2_context':
                    visit(item, path+'/'+str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, path+'/'+str(index))

    visit(request, '$')
    return result


def semantic_routes(routes):
    return [{**raw_copy(r), 'context': semantic_context(r.get('context') or {})} for r in routes]


def without_lazy_scheduler(state):
    # _infer unconditionally sets both schedules before first use (native2206–2209).
    # Retain shift, sigma limits, num_train_timesteps and every other parameter:
    # set_timesteps consumes them rather than replacing them.
    result = dict(state)
    result['lazy_scheduler_state'] = {
        branch: {key: value for key, value in fields.items()
            if key not in ('sigmas', 'timesteps', 'training')}
        for branch, fields in state['lazy_scheduler_state'].items()}
    return result


def typed_calls(calls):
    return [dict(branch=k[0], update_cache=k[1], batch=k[2], candidate=k[3], **v)
        for k, v in sorted(calls.items())]


def atomic_current(path, value):
    """Only replace this job's current pulse; terminal receipts stay exclusive."""
    path = Path(path)
    temp = path.with_name(path.name + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class ExecutionEvidence:
    def __init__(self, server, *, shared, proxy, buffers, cap, worker_root):
        self.s, self.shared, self.proxy, self.buffers = server, shared, proxy, buffers
        self.cap, self.worker_root = int(cap), Path(worker_root)
        self.lock = threading.RLock()
        self.attempted = self.returned = self.failed = self.active = 0
        self.closed = False
        self.gpu_identity = dict(uuid='UNKNOWN', source='real service device properties not supplied')
        self.phase, self.calls, self.requests = Counter(), {}, []
        self.rejected_before_forward_cap = 0
        self.context = self.episode = None
        self.pair_references = {}
        self.completed = set()
        self._inflight = None
        self.original_counter = server._phase_profile_counter
        self.original_start, self.original_end = server._phase_profile_start, server._phase_profile_end
        self.scheduler = server.action_scheduler
        self.had_scheduler_step = 'step' in vars(self.scheduler)
        self.old_scheduler_step_value = vars(self.scheduler).get('step')
        self.original_scheduler_step = self.scheduler.step
        self.hooks = [server.transformer.register_forward_pre_hook(self.pre, with_kwargs=True),
            server.transformer.register_forward_hook(self.post, with_kwargs=True, always_call=True)]
        server._phase_profile_counter = types.MethodType(self.counter, server)
        server._phase_profile_start = types.MethodType(self.start, server)
        server._phase_profile_end = types.MethodType(self.end, server)
        self.scheduler.step = self.step

    @property
    def request_active(self):
        return self.context is not None

    def resource_reference(self):
        path = self.worker_root.parent/'GPU-RESOURCE-CURRENT.json'
        try:
            value = json.loads(path.read_text())
            sampled = value.get('sampled_at_unix')
            return dict(path=str(path), snapshot_id=value.get('snapshot_id', 'UNKNOWN'),
                sampled_at_unix=sampled,
                age_sec=max(0.,time.time()-sampled) if type(sampled) in (int,float) else 'UNKNOWN',
                status=value.get('status', 'UNKNOWN'), sampling_cannot_prove_continuous_exclusivity=True)
        except (OSError,ValueError,TypeError):
            return dict(status='UNKNOWN_NO_READABLE_RESOURCE_SNAPSHOT',
                sampling_cannot_prove_continuous_exclusivity=True)

    def set_episode(self, *, pair, arm, out):
        require(not self.request_active and self.episode is None and not self.closed,
            'prior full episode must close before a new reset')
        key = (pair['pair_id'], arm)
        require(key not in self.completed and self.cap-self.attempted >= 4000,
            'fresh full episode and complete4000 forward reservation required')
        require(not any(pair_id != pair['pair_id'] for pair_id in self.pair_references),
            'finish both arms of the current admitted pair before the next pair')
        self.episode = dict(pair=pair, arm=arm, out=Path(out), start_attempted=self.attempted,
            start_returned=self.returned, start_failed=self.failed,
            start_requests=len(self.requests), start_phase=Counter(self.phase),
            start_routes=len(self.proxy.records), start_events=len(self.buffers.EVENTS),
            start_calls=copy.deepcopy(self.calls), first2_results=[],
            start_wall=time.monotonic(), cap=min(self.cap, self.attempted+4000))

    def pre(self, _model, args, kwargs):
        with self.lock:
            require(self.context is not None and self.episode is not None and not self.closed,
                'target forward outside one owned real request')
            if self.attempted >= self.episode['cap']:
                self.rejected_before_forward_cap += 1
                raise RuntimeError('registered worker/per-episode target forward cap reached')
            require(self._inflight is None, 'nested target transformer call')
            inp = args[0] if args else kwargs['input_dict']
            key = ('action' if kwargs.get('action_mode', False) else 'video',
                int(kwargs.get('update_cache', 0)), int(inp['noisy_latents'].shape[0]),
                kwargs.get('candidate_cache_rows') is not None)
            self.attempted += 1
            self.active += 1
            row = self.calls.setdefault(key, dict(attempted=0, returned=0, failed=0))
            row['attempted'] += 1
            self._inflight = key

    def post(self, _model, _args, _kwargs, output):
        with self.lock:
            # always_call can also fire after a cap-rejecting pre-hook.
            if self._inflight is None:
                return
            key, self._inflight = self._inflight, None
            self.active -= 1
            if output is None:
                self.failed += 1
                self.calls[key]['failed'] += 1
            else:
                self.returned += 1
                self.calls[key]['returned'] += 1

    def counter(self, _server, name, value=1):
        result = self.original_counter(name, value)
        self.phase[name] += int(value)
        if name == 'committed_scheduler_update_count' and self.context is not None:
            require(self.context['role'] == 'S' and self.context['action'] is not None,
                'raw speculative commit outside S action interval')
            self.context['round']['g'] += int(value)
        return result

    def step(self, *args, **kwargs):
        value = self.original_scheduler_step(*args, **kwargs)
        if self.context is not None and self.context['role'] == 'T':
            require(self.context['action'] is not None and self.context['round'] is not None,
                'native target scheduler step outside action interval')
            self.context['target_action_updates'] += 1
            self.context['round']['g'] += 1
        return value

    def mark(self):
        import torch
        event = torch.cuda.Event(enable_timing=True)
        host = time.perf_counter()
        event.record(torch.cuda.current_stream(self.s.device))
        return dict(host=host, cuda=event)

    def close_round(self, mark):
        ctx = self.context
        row = ctx['round']
        require(row is not None, 'missing active action round')
        row['end'] = mark
        ctx['rounds'].append(row)
        ctx['round'] = None

    def start(self, _server, name):
        original = self.original_start(name)
        ctx = self.context
        if ctx is None or ctx['kind'] != 'policy':
            return original
        if name == 'action_total':
            require(ctx['action'] is None, 'nested action interval')
            mark = self.mark()
            ctx['action'] = dict(start=mark, complete=False)
            ctx['round'] = dict(start=mark, g=0)
            return dict(_pf2_phase=True, original=original, name=name)
        if name in ('action_tree_build', 'action_target_forward') and ctx['action'] is not None:
            # Disjoint boundaries: prelude belongs to the preceding round. The
            # final round includes the terminal cache call. Aggregates cover the
            # full action_total interval, with no guessed per-edge timing.
            if ctx['round']['g'] > 0:
                mark = self.mark()
                self.close_round(mark)
                ctx['round'] = dict(start=mark, g=0)
        if name in ('video_total', 'obs_encode', 'postprocess'):
            return dict(_pf2_phase=True, original=original, name=name, host=time.perf_counter())
        return original

    def end(self, _server, token):
        if not isinstance(token, dict) or not token.get('_pf2_phase'):
            return self.original_end(token)
        result = self.original_end(token['original'])
        ctx = self.context
        require(ctx is not None, 'phase token escaped its owned request')
        if token['name'] == 'action_total':
            mark = self.mark()
            self.close_round(mark)
            ctx['action'].update(end=mark, complete=True)
        else:
            ctx['major_host_spans'].append(dict(name=token['name'],
                seconds=time.perf_counter()-token['host']))
        return result

    def begin(self, *, kind, role, request, ordinal):
        require(self.episode is not None and self.context is None and not self.closed,
            'request must belong to one active complete episode')
        first = kind == 'policy' and ordinal < 2
        require(not first or role == 'S', 'both arms must start with two S chunks')
        ctx = dict(kind=kind, role=role, ordinal=ordinal,
            start_attempted=self.attempted, start_returned=self.returned,
            start_failed=self.failed, start_phase=Counter(self.phase),
            start_calls=copy.deepcopy(self.calls), start_routes=len(self.proxy.records),
            start_events=len(self.buffers.EVENTS), start_shared=self.shared.requests,
            action=None, round=None, rounds=[], target_action_updates=0,
            major_host_spans=[], saved_outputs={}, first2=first)
        ctx['resource_before'] = self.resource_reference()
        self.context = ctx
        if first:
            ctx['before_state'] = capture_full_state(self.s)
            ctx['semantic_input'] = semantic_request(request)
            ctx['input_layouts'] = input_layouts(request)
        return ctx

    def saved(self, name, value):
        if self.context is not None and self.context['first2']:
            label = name.split('_', 1)[0]
            require(label not in self.context['saved_outputs'], 'same real saved tensor repeated')
            self.context['saved_outputs'][label] = raw_copy(value)

    def synchronize_return(self):
        import torch
        torch.cuda.synchronize(self.s.device)

    def delta_calls(self, before):
        return {key: {name: row[name]-before.get(key, {}).get(name, 0)
            for name in ('attempted', 'returned', 'failed')}
            for key, row in self.calls.items()
            if row != before.get(key, dict(attempted=0, returned=0, failed=0))}

    def timing(self, ctx):
        action = ctx['action']
        if action is None:
            return dict(action_complete=False, sum_g=0, rounds=[])
        require(action['complete'] and ctx['round'] is None, 'incomplete action interval')
        rounds = [dict(g=row['g'], host_seconds=row['end']['host']-row['start']['host'],
            cuda_seconds=row['start']['cuda'].elapsed_time(row['end']['cuda'])/1000)
            for row in ctx['rounds']]
        return dict(action_complete=True,
            action_host_seconds=action['end']['host']-action['start']['host'],
            action_cuda_seconds=action['start']['cuda'].elapsed_time(action['end']['cuda'])/1000,
            sum_g=sum(row['g'] for row in rounds),
            sum_round_host_seconds=sum(row['host_seconds'] for row in rounds),
            sum_round_cuda_seconds=sum(row['cuda_seconds'] for row in rounds), rounds=rounds,
            scope='native_action_total_including_anchor_tail_fallback_bonus_terminal',
            audit_markers_included=True, performance_claim=False)

    def first2_compare(self, *, label, payload, exclude_lazy=False):
        ep = self.episode
        pair_id, arm = ep['pair']['pair_id'], ep['arm']
        ref = self.pair_references.setdefault(pair_id, dict(arm=arm, values={}))
        if ref['arm'] == arm:
            require(label not in ref['values'], 'first-arm reference overwritten')
            ref['values'][label] = payload
            answer = dict(status='REFERENCE_STORED', label=label,
                first_arm=arm, raw_all_fields_equal='NOT_COMPARED_YET')
        else:
            require(label in ref['values'], 'missing first-arm raw comparison material')
            expected = ref['values'][label]
            all_fields = compare_raw(expected, payload)
            if exclude_lazy:
                active = compare_raw(without_lazy_scheduler(expected), without_lazy_scheduler(payload))
                answer = dict(status='PASS_EXACT_ACTIVE_INPUTS' if active['equal'] else 'FAILED',
                    label=label, first_arm=ref['arm'], raw_all_fields_equal=all_fields['equal'],
                    active_input_comparison=active, raw_all_field_comparison=all_fields,
                    excluded_fields=[f'lazy_scheduler_state.{branch}.{key}'
                        for branch in ('video', 'action') for key in ('sigmas', 'timesteps', 'training')],
                    reason='native _infer unconditionally replaces20/50 timetables before first consumption')
            else:
                answer = dict(status='PASS_EXACT' if all_fields['equal'] else 'FAILED', label=label,
                    first_arm=ref['arm'], raw_all_fields_equal=all_fields['equal'], comparison=all_fields)
        ep['first2_results'].append(answer)
        if answer['status'] == 'FAILED':
            # A passive comparison cannot alter a returned action or truncate
            # the ongoing robot task. Preserve the failure and close new pairs
            # through the runner's barrier after admitted pairs finish.
            path = self.worker_root/'ISOLATION-FAILED.json'
            if not path.exists():
                write(path, dict(schema='tf-pf2-passive-isolation-failure/v1',
                    status='FAILED_FIRST_TWO_S_PAIRING', pair_id=pair_id, arm=arm,
                    evidence=answer, current_episode_must_finish=True,
                    current_admitted_pair_must_finish=True, no_new_pair=True,
                    permissions=PERMISSIONS))
        return answer

    def finish(self, ctx, *, row, result):
        require(ctx is self.context and self.active == 0 and self._inflight is None,
            'request/target accounting did not close')
        phase = {k: self.phase.get(k, 0)-ctx['start_phase'].get(k, 0)
            for k in set(self.phase) | set(ctx['start_phase'])}
        calls = self.delta_calls(ctx['start_calls'])
        routes = self.proxy.records[ctx['start_routes']:]
        events = self.buffers.EVENTS[ctx['start_events']:]
        timing = self.timing(ctx)
        audit = audit_request(kind=ctx['kind'], role=ctx['role'], phase=phase,
            routes=routes, events=events, shared_requests=self.shared.requests-ctx['start_shared'],
            target_counts=row.get('target_observed_counts', {}), calls=calls,
            target_action_updates=ctx['target_action_updates'], timing=timing)
        row.update(phase_counters=phase, target_forward_attempts=self.attempted-ctx['start_attempted'],
            target_forward_returned=self.returned-ctx['start_returned'],
            target_forward_failed=self.failed-ctx['start_failed'], typed_forward_calls=typed_calls(calls),
            O3_events=raw_copy(events), shared_action_requests=self.shared.requests-ctx['start_shared'],
            acceptance=raw_copy(routes), target_action_updates=ctx['target_action_updates'],
            timing=timing, audit=audit, GPU_identity=dict(self.gpu_identity),
            resource_before=ctx['resource_before'], resource_after=self.resource_reference(),
            timing_is_exploratory_AutoDL=True)
        if ctx['kind'] == 'reset':
            row['first2_reset'] = self.first2_compare(label='reset',
                payload=capture_full_state(self.s), exclude_lazy=True)
        elif ctx['first2']:
            ordinal = ctx['ordinal']
            before = ctx['before_state']
            if ordinal == 0:
                # Keep raw lazy differences explicit rather than pretending full equality.
                self.first2_compare(label='S0_before', payload=before, exclude_lazy=True)
                before = without_lazy_scheduler(before)
            after = capture_full_state(self.s)
            payload = dict(input=ctx['semantic_input'], input_layouts=ctx['input_layouts'],
                before=before, after=after,
                returned=raw_copy(result), saved_outputs=ctx['saved_outputs'],
                routes=semantic_routes(routes), phase_counters=phase,
                typed_forward_calls=typed_calls(calls))
            require(set(ctx['saved_outputs']) == {'actions', 'latents'},
                'first-two-S requires original normalized action and latent tensors')
            row['first2S'] = self.first2_compare(label='S' + str(ordinal), payload=payload)
        self.requests.append(row)
        self.context = None
        self.pulse()

    def fail(self, ctx, *, row, error):
        if self.context is not ctx:
            return
        if row is not None:
            row.update(error=error, partial=True,
                target_forward_attempts=self.attempted-ctx['start_attempted'],
                target_forward_returned=self.returned-ctx['start_returned'],
                target_forward_failed=self.failed-ctx['start_failed'],
                typed_forward_calls=typed_calls(self.delta_calls(ctx['start_calls'])))
            self.requests.append(row)
        self.context = None
        self.pulse()

    def pulse(self):
        atomic_current(self.worker_root/'LIVE-COUNTER.json', dict(
            schema='tf-pf2-worker-live-counter/v1', target_forward_attempts=self.attempted,
            target_forward_returned=self.returned, target_forward_failed=self.failed,
            active_target_forwards=self.active, request_active=self.request_active,
            episode=None if self.episode is None else dict(pair_id=self.episode['pair']['pair_id'],
                arm=self.episode['arm'], out=str(self.episode['out'])),
            cap=self.cap, rejected_before_forward_cap=self.rejected_before_forward_cap,
            passive_isolation_failure=(self.worker_root/'ISOLATION-FAILED.json').exists(),
            permissions=PERMISSIONS))

    def end_episode(self, terminal):
        ep = self.episode
        require(ep is not None and not self.request_active and self.active == 0
            and self.attempted == self.returned and self.failed == 0,
            'complete episode requires all real calls closed')
        requests = self.requests[ep['start_requests']:]
        policies = [r for r in requests if r['kind'] == 'policy']
        require(all(r['ok'] and r['audit']['status'] == 'PASS_COMPLETE_ROLE_REQUEST' for r in requests),
            'all original requests must pass independent role contracts')
        require(len(policies) == terminal['policy_chunks'] and all(
            r['successful_policy_ordinal'] == i for i, r in enumerate(policies)),
            'all complete policy chunks and ordinals required')
        expected_roles = ['T' if ep['arm'] == 'P' and i % 3 == 2 else 'S' for i in range(len(policies))]
        require([r['role'] for r in policies] == expected_roles, 'actual schedule differs from fixed C/P')
        g = sum(r['audit']['g'] for r in policies)
        require(g == 50*len(policies), 'all effective submitted scheduler updates must be counted')
        require(len(policies) >= 2, 'cannot certify first-two-S with fewer than two actual policy chunks')
        expected_labels = {'reset', 'S0_before', 'S0', 'S1'}
        require({r['label'] for r in ep['first2_results']} == expected_labels,
            'reset and both first-S raw comparison records required')
        second = self.pair_references[ep['pair']['pair_id']]['arm'] != ep['arm']
        isolation_failed = any(r['status'] == 'FAILED' for r in ep['first2_results'])
        first2 = dict(status=('FAILED' if isolation_failed else
                'PASS_EXACT' if second else 'REFERENCE_STORED'),
            compared_policy_chunks=2 if second else 0,
            reference_policy_chunks=2, records=ep['first2_results'],
            first_infer_lazy_scheduler_exclusion_explicit=True,
            post_first_S_and_all_second_S_fields_compared_without_exclusion=True)
        total_round = sum(r['audit']['sum_T_round_sec'] for r in policies)
        counter = dict(schema='tf-pf2-episode-counter/v1', pair_id=ep['pair']['pair_id'], arm=ep['arm'],
            task=ep['pair']['task'], init=ep['pair']['init'],
            target_forward_attempts=self.attempted-ep['start_attempted'],
            target_forward_returned=self.returned-ep['start_returned'],
            target_forward_failed=self.failed-ep['start_failed'], policy_chunks=len(policies),
            S_chunks=sum(r['role']=='S' for r in policies), T_chunks=sum(r['role']=='T' for r in policies),
            total_actual_g=g, phase_counters={k: self.phase.get(k,0)-ep['start_phase'].get(k,0)
                for k in set(self.phase)|set(ep['start_phase'])},
            typed_forward_calls=typed_calls(self.delta_calls(ep['start_calls'])),
            requests=requests, policy_records=policies,
            timing=dict(sum_T_round_sec=total_round, sum_g=g, round_coverage_complete=True,
                action_measurement_complete=True,
                action_total_sec=sum(r['timing']['action_cuda_seconds'] for r in policies),
                action_g=g,
                full_policy_elapsed_sec=sum(r['original_infer_host_seconds'] for r in policies),
                cost_per_valid_commit=total_round/g,
                sum_action_cuda_sec=sum(r['timing']['action_cuda_seconds'] for r in policies),
                sum_action_host_sec=sum(r['timing']['action_host_seconds'] for r in policies),
                sum_full_policy_host_sec=sum(r['original_infer_host_seconds'] for r in policies),
                episode_server_wall_sec=time.monotonic()-ep['start_wall'],
                clock='CUDA-event action rounds; host full-policy with explicit returned synchronization',
                scope='instrumented native action CUDA stream / complete policy host; exploratory AutoDL',
                nested_intervals_not_added=True, performance_claim=False),
            permissions=PERMISSIONS, robot_quality_validated=False)
        audit = dict(schema='tf-pf2-episode-audit/v1',
            status='FAILED_FIRST_TWO_S_PAIRING' if isolation_failed else 'PASS_COMPLETE_EPISODE_COMPONENTS',
            pair_id=ep['pair']['pair_id'], arm=ep['arm'], first2S=first2,
            numerical_isolation=dict(status=('FAILED_FIRST_TWO_S_PAIRING' if isolation_failed else
                'PASS_OBSERVED_ROUTE_STATE_AND_ROLE_CONTRACTS'),
                original_ObservedAPI_unchanged=True, raw_S_g_unchanged=True,
                manual_vs_scheduler12chunk_gate='NOT_EXECUTED_USER_NO_INDEPENDENT_TESTS',
                old_adapter_CAL_not_inherited=True, remote_SHA_executed=False),
            permissions=PERMISSIONS, engineering_quality_screen=False,
            independent_test_complete=False, noninferiority_established=False,
            formal_performance_permission=False)
        self.completed.add((ep['pair']['pair_id'], ep['arm']))
        if second:
            del self.pair_references[ep['pair']['pair_id']]
        self.episode = None
        self.pulse()
        return counter, audit

    def partial_episode_counter(self):
        """Retain known work after an inline failure; never label incomplete cost complete."""
        ep = self.episode
        require(ep is not None, 'no active episode for partial work receipt')
        requests = self.requests[ep['start_requests']:]
        policies = [r for r in requests if r['kind'] == 'policy']
        return dict(schema='tf-pf2-episode-counter/v1', status='PARTIAL_INLINE_AUDIT_FAILURE',
            pair_id=ep['pair']['pair_id'], arm=ep['arm'], task=ep['pair']['task'], init=ep['pair']['init'],
            target_forward_attempts=self.attempted-ep['start_attempted'],
            target_forward_returned=self.returned-ep['start_returned'],
            target_forward_failed=self.failed-ep['start_failed'],
            active_target_forwards=self.active,
            policy_chunks=len(policies), S_chunks=sum(r['role']=='S' for r in policies),
            T_chunks=sum(r['role']=='T' for r in policies), total_actual_g='UNKNOWN',
            requests=requests, policy_records=policies,
            typed_forward_calls=typed_calls(self.delta_calls(ep['start_calls'])),
            phase_counters={k: self.phase.get(k,0)-ep['start_phase'].get(k,0)
                for k in set(self.phase)|set(ep['start_phase'])},
            timing=dict(action_total_sec='UNKNOWN', action_g='UNKNOWN',
                full_policy_elapsed_sec='UNKNOWN', sum_T_round_sec='UNKNOWN', sum_g='UNKNOWN',
                round_coverage_complete=False, action_measurement_complete=False,
                scope='partial actual attempts retained; missing action timing is not zero cost'),
            permissions=PERMISSIONS, robot_quality_validated=False)

    def close(self):
        require(not self.request_active and self.active == 0, 'cannot close active accounting')
        for hook in self.hooks:
            hook.remove()
        self.s._phase_profile_counter = self.original_counter
        self.s._phase_profile_start, self.s._phase_profile_end = self.original_start, self.original_end
        if self.had_scheduler_step:
            self.scheduler.step = self.old_scheduler_step_value
        else:
            del self.scheduler.step
        self.closed = True

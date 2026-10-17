"""Default-OFF, per-episode spec/spec/original-target request wrapper.

Only the server instance's infer method and, during a target refresh request,
its action scheduler step observer are patched. The original request and infer
body generate the current video and complete 50-update action chunk. No state,
observation, cache, noise or action from another request is injected.
"""
from collections import Counter
from pathlib import Path
import sys
import threading
import types


ROOT = Path(__file__).resolve().parents[3]
FROZEN_AOFF = str(ROOT / '投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/training/A_off.pt')
FROZEN_THRESHOLD = 0.007839680526450805
SPEC_MODE = 'verifier_k2_b2_bonus'


class RefreshContractError(RuntimeError):
    pass


class PeriodicTargetRefresh:
    def __init__(self, server, *, enabled=False, runtime=None,
                 expected_checkpoint=FROZEN_AOFF,
                 expected_threshold=FROZEN_THRESHOLD):
        if type(enabled) is not bool:
            raise ValueError('enabled must be an explicit boolean')
        self.s, self.runtime = server, runtime
        self.enabled = enabled
        self.expected_checkpoint = str(Path(expected_checkpoint).resolve())
        self.expected_threshold = float(expected_threshold)
        self.installed = self.active = self.failed = False
        self.next_ordinal = 0
        self.episode_sequence = -1
        self.records = []
        self.lock = threading.RLock()
        self.original = None
        self._had_infer = False
        self._old_infer_value = None

    @staticmethod
    def target_ordinal(ordinal):
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError('nonnegative integer chunk ordinal required')
        return ordinal % 3 == 2

    def _frozen(self):
        if (self.s._p0_action_draft_mode != SPEC_MODE
                or str(Path(self.s._p0_action_draft_b2_checkpoint).resolve()) != self.expected_checkpoint
                or float(self.s._p0_action_draft_threshold) != self.expected_threshold
                or int(self.s.job_config.action_num_inference_steps) != 50):
            raise RefreshContractError('frozen A_off/threshold/bonus mode/full50 contract changed')
        if getattr(self.s, '_p0_state_teacher', None) is not None:
            raise RefreshContractError('periodic refresh requires teacher OFF')

    def _clean(self):
        if self.runtime is None:
            raise RefreshContractError('enabled refresh requires Runtime ownership/accounting')
        shared = self.runtime.shared
        if (shared.control is not None or shared.ctx is not None or shared.token is not None
                or shared.request_thread is not None or not shared.last_clean):
            raise RefreshContractError('shared-prefix request lease leaked')
        buffers = sys.modules.get('buffers')
        if buffers is not None and buffers.ACTIVE.get() is not None:
            raise RefreshContractError('O3 workspace leaked')

    def install(self):
        if not self.enabled:
            return self
        if self.installed or self.active:
            raise RefreshContractError('periodic refresh already installed/active')
        self._frozen()
        self._clean()
        if self.runtime.request_active or self.runtime.closed:
            raise RefreshContractError('install only on an idle open Runtime')
        self.original = self.s.infer
        self._had_infer = 'infer' in vars(self.s)
        self._old_infer_value = vars(self.s).get('infer')
        self.bound_infer = types.MethodType(self.infer, self.s)
        self.s.infer = self.bound_infer
        self.installed = True
        return self

    def _target(self, request, record):
        scheduler = self.s.action_scheduler
        had_step, old_step_value = 'step' in vars(scheduler), vars(scheduler).get('step')
        original_step = scheduler.step
        calls, hooks = Counter(), []

        def step(*args, **kwargs):
            value = original_step(*args, **kwargs)
            calls['action_scheduler_updates'] += 1
            return value

        def pre(_model, args, kwargs):
            inp = args[0] if args else kwargs['input_dict']
            branch = 'action' if kwargs.get('action_mode', False) else 'video'
            calls[branch + '_target_intents'] += 1
            if branch == 'action':
                calls['action_candidate_calls'] += int(kwargs.get('candidate_cache_rows') is not None)
                calls['action_terminal_calls'] += int(kwargs.get('update_cache', 0) == 1)
                calls['action_regular_calls'] += int(kwargs.get('update_cache', 0) == 0)
                calls['action_CFG2_calls'] += int(inp['noisy_latents'].shape[0] == 2)

        def post(_model, _args, kwargs, output):
            branch = 'action' if kwargs.get('action_mode', False) else 'video'
            calls[branch + ('_target_returned' if output is not None else '_target_failed')] += 1

        mode = self.s._p0_action_draft_mode
        self.s._p0_action_draft_mode = 'off'
        scheduler.step = step
        try:
            hooks.append(self.s.transformer.register_forward_pre_hook(pre, with_kwargs=True))
            hooks.append(self.s.transformer.register_forward_hook(post, with_kwargs=True, always_call=True))
            output = self.original(request)
            expected = dict(action_scheduler_updates=50, action_target_intents=51,
                action_target_returned=51, action_candidate_calls=0,
                action_terminal_calls=1, action_regular_calls=50, action_CFG2_calls=51)
            mismatch = {k: (calls[k], v) for k, v in expected.items() if calls[k] != v}
            if mismatch or calls['action_target_failed'] or calls['video_target_failed']:
                raise RefreshContractError('original full50 target accounting mismatch: ' + str(mismatch))
            return output
        finally:
            for hook in hooks:
                hook.remove()
            if had_step:
                scheduler.step = old_step_value
            else:
                del scheduler.step
            self.s._p0_action_draft_mode = mode
            record['target_observed_counts'] = dict(calls)
            record['scheduler_step_restored'] = scheduler.step == original_step
            record['mode_restored'] = self.s._p0_action_draft_mode == mode

    def infer(self, _server, request):
        with self.lock:
            if self.active:
                raise RefreshContractError('one request at a time required')
            self.active = True
        record = None
        try:
            self._clean()
            self._frozen()
            if request.get('reset'):
                output = self.original(request)
                self.next_ordinal = 0
                self.episode_sequence += 1
                self.failed = False
                return output
            if self.failed:
                raise RefreshContractError('failed episode requires an explicit reset')
            if request.get('compute_kv_cache'):
                return self.original(request)
            if any(request.get(k) for k in ('p0_counterfactual_set_initial_obs',
                    'p0_counterfactual_prime_video_cache', 'p0_counterfactual_target_directions',
                    'p0_counterfactual_restore_kv_cache')):
                raise RefreshContractError('periodic wrapper is for original online policy requests')
            context = request.get('p0_trace_context') or {}
            if not isinstance(context, dict):
                raise RefreshContractError('dict request context required')
            if 'rollout_chunk' in context and (type(context['rollout_chunk']) is not int
                    or context['rollout_chunk'] != self.next_ordinal):
                raise RefreshContractError('client chunk ordinal disagrees with successful policy requests')
            ordinal = self.next_ordinal
            target = self.target_ordinal(ordinal)
            counts = dict(self.runtime.phase)
            start_forwards = self.runtime.attempted
            start_shared_requests = self.runtime.shared.requests
            helper = self.s._infer_action_verifier_k2_b2
            record = dict(episode_sequence=self.episode_sequence, chunk_ordinal=ordinal,
                target_refresh=target, mode_before=self.s._p0_action_draft_mode,
                frame_before=int(self.s.frame_st_id), context=dict(context), ok=False,
                same_request_passed_to_original=True, periodic_schedule='spec,spec,target')
            output = self._target(request, record) if target else self.original(request)
            self._clean()
            self._frozen()
            if self.s._infer_action_verifier_k2_b2 != helper:
                raise RefreshContractError('O3 helper binding changed across request')
            self.next_ordinal += 1
            record['ok'] = True
            return output
        except BaseException as exc:
            if record is not None:
                self.failed = True
                record['error'] = repr(exc)
            raise
        finally:
            if record is not None:
                record['forward_intents'] = self.runtime.attempted - start_forwards
                record['shared_action_requests'] = self.runtime.shared.requests - start_shared_requests
                record['phase_deltas'] = {k: self.runtime.phase[k] - counts.get(k, 0)
                    for k in set(counts) | set(self.runtime.phase)
                    if self.runtime.phase[k] != counts.get(k, 0)}
                record['mode_after'] = self.s._p0_action_draft_mode
                record['frame_after'] = int(self.s.frame_st_id)
                record['shared_clean_after'] = (self.runtime.shared.control is None
                    and self.runtime.shared.ctx is None and self.runtime.shared.token is None
                    and self.runtime.shared.last_clean)
                self.records.append(record)
            with self.lock:
                self.active = False

    def summary(self):
        return dict(schema='periodic-target-refresh/v1', enabled=self.enabled,
            installed=self.installed, active=self.active, failed_episode=self.failed,
            episode_sequence=self.episode_sequence, next_chunk_ordinal=self.next_ordinal,
            frozen_checkpoint=self.expected_checkpoint, threshold=self.expected_threshold,
            action_updates=50, schedule='spec,spec,target', target_ordinals='2,5,8,...',
            records=self.records, raw_RT_phase_counters_modified=False,
            original_state_or_RNG_replayed=False)

    def close(self):
        if not self.installed:
            return
        if self.active or self.runtime.request_active:
            raise RefreshContractError('cannot close an active refresh request')
        self._clean()
        if self.s.infer != self.bound_infer:
            raise RefreshContractError('close wrappers in reverse install order')
        if self._had_infer:
            self.s.infer = self._old_infer_value
        else:
            del self.s.infer
        self.installed = False


def install(server, *, enabled=False, runtime=None, **kwargs):
    return PeriodicTargetRefresh(server, enabled=enabled, runtime=runtime, **kwargs).install()

"""One wrapper for continuous TF and SST; original S and T math stay untouched."""
from __future__ import annotations

import time

from .sealed_refresh import PeriodicTargetRefresh, RefreshContractError


class Controller(PeriodicTargetRefresh):
    def __init__(self, server, *, runtime, evidence, expected_checkpoint,
                 expected_threshold, periodic=0):
        super().__init__(server, enabled=True, runtime=runtime,
            expected_checkpoint=expected_checkpoint, expected_threshold=expected_threshold)
        self.evidence = evidence
        self.set_periodic(periodic)

    def set_periodic(self, periodic):
        if type(periodic) is not int or periodic not in (0, 2):
            raise ValueError('only C=0 and P=2 are registered')
        if self.active or (self.runtime is not None and self.runtime.request_active):
            raise RefreshContractError('change arm only at an idle full-episode boundary')
        self.periodic = periodic

    def target_ordinal(self, ordinal):
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError('nonnegative successful-policy ordinal required')
        return self.periodic == 2 and ordinal % 3 == 2

    def infer(self, _server, request):
        with self.lock:
            if self.active:
                raise RefreshContractError('one original request at a time required')
            self.active = True
        token = row = None
        try:
            self._clean()
            self._frozen()
            if any(request.get(key) for key in (
                    'p0_counterfactual_set_initial_obs', 'p0_counterfactual_prime_video_cache',
                    'p0_counterfactual_target_directions', 'p0_counterfactual_restore_kv_cache')):
                raise RefreshContractError('no old observation/state injection or replay')
            kind = ('reset' if request.get('reset') else
                'cache' if request.get('compute_kv_cache') else 'policy')
            if self.failed and kind != 'reset':
                raise RefreshContractError('failed episode requires explicit reset')
            context = request.get('p0_trace_context') or {}
            if not isinstance(context, dict):
                raise RefreshContractError('original trace context must be a dictionary')
            ordinal, role = None, None
            if kind == 'policy':
                if self.episode_sequence < 0:
                    raise RefreshContractError('successful original reset required first')
                ordinal = self.next_ordinal
                if 'rollout_chunk' in context and (
                        type(context['rollout_chunk']) is not int or context['rollout_chunk'] != ordinal):
                    raise RefreshContractError('client successful-policy ordinal differs')
                role = 'T' if self.target_ordinal(ordinal) else 'S'
            row = dict(schema='tf-pf2-request/v1', kind=kind, role=role,
                periodic=self.periodic, successful_policy_ordinal=ordinal,
                context=dict(context), frame_before=getattr(self.s, 'frame_st_id', None),
                ok=False, same_request_passed_once=True)
            helper = self.s._infer_action_verifier_k2_b2
            token = self.evidence.begin(kind=kind, role=role, request=request, ordinal=ordinal)
            started = time.perf_counter()
            try:
                result = self._target(request, row) if role == 'T' else self.original(request)
                self.evidence.synchronize_return()
            finally:
                row['original_infer_host_seconds'] = time.perf_counter() - started
            self._clean()
            self._frozen()
            if self.s._infer_action_verifier_k2_b2 != helper:
                raise RefreshContractError('original O3 helper binding changed')
            row['frame_after'] = int(self.s.frame_st_id)
            self.evidence.finish(token, row=row, result=result)
            token = None
            if kind == 'reset':
                self.next_ordinal = 0
                self.episode_sequence += 1
                self.failed = False
            elif kind == 'policy':
                # Even an entirely rejected S is one successful S policy chunk.
                self.next_ordinal += 1
            row['ok'] = True
            return result
        except BaseException as exc:
            self.failed = True
            if row is not None:
                row['error'] = repr(exc)
            if token is not None:
                self.evidence.fail(token, row=row, error=repr(exc))
            raise
        finally:
            if row is not None:
                self.records.append(row)
            with self.lock:
                self.active = False

    def summary(self):
        return dict(schema='tf-pf2-controller/v1', periodic=self.periodic,
            schedule='S' if self.periodic == 0 else 'S,S,T',
            installed=self.installed, active=self.active, failed_episode=self.failed,
            next_successful_policy_ordinal=self.next_ordinal,
            raw_S_phase_counters_modified=False, old_state_or_RNG_replayed=False,
            records=self.records)

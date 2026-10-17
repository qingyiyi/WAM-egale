"""Optional bounded observations from three real complete native policy calls.

No forward, replay, digest, simulator, file read of old artifacts, or CUDA
initialization is performed here. The caller already owns a live model. All
ordinary observation failures become UNKNOWN receipts, never robot actions.
Only worker0/C and the first encountered request in each fixed bucket qualify.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


CAP = 3 * 2**30
METADATA_RESERVE = 8 * 2**20
HISTORY_TO_KV = {144: 288, 1008: 1152, 2016: 2160}
CONFIG_KEYS = ('num_inference_steps', 'action_num_inference_steps', 'video_exec_step',
    'guidance_scale', 'action_guidance_scale', 'snr_shift', 'action_snr_shift',
    'action_dim', 'used_action_channel_ids', 'frame_chunk_size', 'patch_size',
    'attn_window', 'obs_cam_keys', 'action_norm_method', 'norm_stat')
NORMALIZATION = ('feature_mean', 'feature_std', 'residual_mean', 'residual_std')
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(value, reason):
    if not value:
        raise ValueError(reason)


class _BoundedWriter:
    """Sequential torch.save file object; a partial file also counts to the cap."""
    def __init__(self, stream, limit):
        self.stream, self.limit, self.count = stream, int(limit), 0

    def write(self, value):
        require(self.count + len(value) <= self.limit, 'three-GiB fixture byte limit')
        size = self.stream.write(value)
        self.count += size
        return size

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.count


class ReplayCapture:
    def __init__(self, *, server, worker_index, job, source, metadata=None,
                 runtime, proxy, cap_bytes=CAP):
        self.s, self.runtime, self.proxy = server, runtime, proxy
        self.worker_index, self.source = int(worker_index), source
        self.folder = Path(job)/'replay-fixtures'
        self.metadata = {} if metadata is None else dict(metadata)
        self.metadata.setdefault('operational_math_commit', 'UNKNOWN')
        self.cap, self.used, self.seq = min(int(cap_bytes), CAP), 0, 0
        self.active, self.handles = None, []
        self.attempted, self.completed, self.gaps = set(), {}, []
        self.disabled, self.closed, self.owns_folder = self.worker_index != 0, False, False
        self.observer_host_sec = 0.0
        if self.disabled:
            return
        try:
            require(self.cap > METADATA_RESERVE, 'positive bounded fixture allowance required')
            # One fresh lifecycle; never overwrite or silently reuse an old file.
            self.folder.mkdir(exist_ok=False)
            self.owns_folder = True
            json.dumps(self.metadata, allow_nan=False)
            from .state import capture_full_state, capture_rng, clone_raw
            self.capture_state, self.capture_rng, self.clone = capture_full_state, capture_rng, clone_raw
            self.handles.append(server.transformer.register_forward_pre_hook(self._pre, with_kwargs=True))
            self.handles.append(server.transformer.register_forward_hook(self._post, with_kwargs=True, always_call=True))
            self._event('READY_OPTIONAL_CAPTURE', expected_action_KV=list(HISTORY_TO_KV.values()),
                arm='C', zero_extra_model_calls=True)
        except Exception as exc:
            self.disabled = True
            self._gap('constructor', exc)

    def _event(self, status, **values):
        # Bounded, exclusive event receipts: no old terminal is replaced.
        if not self.owns_folder or not self.folder.is_dir():
            return
        self.seq += 1
        fixed = dict(schema='tf-fullsuite-optional-replay-capture-event/v1', at=stamp(),
            status=status, sequence=self.seq, worker_index=self.worker_index,
            capture_cap_bytes=self.cap, counted_data_bytes=self.used,
            performance_claim=False, permissions=PERMISSIONS)
        require(not set(fixed).intersection(values), 'capture event fields cannot override fixed provenance')
        data = {**fixed, **values}
        raw = (json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
        require(len(raw) <= 128*2**10 and self.seq <= 64, 'bounded capture metadata event')
        with (self.folder/('EVENT-%03d.json' % self.seq)).open('xb') as stream:
            stream.write(raw)
        self._index()

    def _index(self):
        data = dict(schema='tf-fullsuite-optional-replay-fixture-index/v1', at=stamp(),
            status=('ALL_THREE_REQUEST_FIXTURES_CAPTURED_UNQUALIFIED' if len(self.completed)==3
                else 'PARTIAL_OR_PENDING_CAPTURE_NO_REPLAY_QUALIFICATION'),
            worker_index=0, arm='C', fixture_cap_bytes=self.cap, counted_data_bytes=self.used,
            buckets={str(k):self.completed.get(k, dict(status='UNKNOWN_NOT_CAPTURED',
                attempted=k in self.attempted)) for k in HISTORY_TO_KV.values()},
            metadata=self.metadata, gaps=self.gaps, extra_model_calls=0,
            future_local_replay_qualified=False, physical_layout_restored_or_tested=False,
            no_verifier_KV_duplicate=True, performance_claim=False, permissions=PERMISSIONS)
        raw=(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
        require(len(raw)<=128*2**10, 'bounded fixture navigation index')
        pending=self.folder/'INDEX.json.pending'
        with pending.open('xb') as stream:
            stream.write(raw)
        pending.replace(self.folder/'INDEX.json')

    def _gap(self, where, error):
        item = dict(where=where, error=repr(error), coverage='UNKNOWN', at=stamp())
        if self.active is not None:
            item['action_KV'] = self.active['bucket']
            self.active['errors'].append(item)
        if len(self.gaps) < 32:
            self.gaps.append(item)
        try:
            self._event('UNKNOWN_CAPTURE_GAP', gap=item,
                current_robot_request_and_full_episode_must_continue=True)
        except Exception:
            pass

    def _size(self, value):
        import numpy as np
        import torch
        if isinstance(value, torch.Tensor):
            return value.numel()*value.element_size()
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(self._size(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(self._size(v) for v in value)
        if isinstance(value, (str, bytes)):
            return len(value)*4
        return 128

    def _layouts(self, value):
        import numpy as np
        import torch
        if isinstance(value, torch.Tensor):
            return dict(kind='tensor', shape=list(value.shape), stride_elements=list(value.stride()),
                storage_offset_elements=value.storage_offset(), dtype=str(value.dtype),
                device=str(value.device), contiguous=value.is_contiguous())
        if isinstance(value, np.ndarray):
            return dict(kind='ndarray', shape=list(value.shape), strides_bytes=list(value.strides),
                dtype=value.dtype.str, C_contiguous=bool(value.flags.c_contiguous))
        if isinstance(value, dict):
            return {str(k): self._layouts(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._layouts(v) for v in value]
        if value is None or type(value) in (str, bool, int, float):
            return dict(kind=type(value).__name__)
        return dict(kind=type(value).__name__, layout='UNKNOWN_UNSUPPORTED_OBJECT')

    def _histories(self):
        values = []
        for block in self.s.transformer.blocks:
            cache = block.attn1.attn_caches[self.s.cache_name]
            require(set(cache) == {'k', 'v', 'mask', 'id', 'is_pred'}, 'original complete cache schema differs')
            values.append(int(cache['mask'].sum().item()))
        require(len(values) == 30 and len(set(values)) == 1, 'thirty-layer valid-history coverage differs')
        return values

    def _live_inventory(self):
        names = ('init_latent', 'prompt_embeds', 'negative_prompt_embeds', 'action_mask',
            'actions_q01', 'actions_q99', 'action_norm_method', 'action_per_frame',
            'height', 'width', 'latent_height', 'latent_width')
        predictor = self.s._p0_action_draft_b2_predictor
        values = dict(mathematical={k:getattr(self.s, k) for k in names},
            VAE=self.s.streaming_vae.feat_cache,
            VAE_half=None if getattr(self.s, 'streaming_vae_half', None) is None
                else self.s.streaming_vae_half.feat_cache,
            scheduler=vars(self.s.scheduler), action_scheduler=vars(self.s.action_scheduler),
            draft_normalization={k:getattr(predictor, k) for k in NORMALIZATION})
        return values

    def _job_configuration(self):
        values, kinds, unknown = {}, {}, {}
        for key in CONFIG_KEYS:
            try:
                value=getattr(self.s.job_config,key)
                kinds[key]=type(value).__module__+'.'+type(value).__qualname__
                # The inspected native entry passes an EasyDict unchanged;
                # preserve explicit support if an authorized launcher instead
                # supplies OmegaConf containers. This affects metadata only.
                if type(value).__module__.startswith('omegaconf.'):
                    from omegaconf import OmegaConf
                    require(OmegaConf.is_config(value), 'unrecognized OmegaConf metadata value')
                    value=OmegaConf.to_container(value,resolve=True)
                values[key]=self.clone(value)
            except Exception as exc:
                values[key]='UNKNOWN'
                kinds.setdefault(key,'UNKNOWN_MISSING_FIELD')
                unknown[key]=repr(exc)
        return values,kinds,unknown

    def begin(self, request, *, context, case):
        if (self.disabled or self.closed or len(self.attempted) == len(HISTORY_TO_KV)
                or context.get('arm') != 'C' or self.worker_index != 0):
            return None
        if request.get('reset') or request.get('compute_kv_cache'):
            return None
        started = time.monotonic()
        try:
            require(self.active is None, 'nested optional request observation')
            require(case['proposed_GPU_index'] == 0 and case['execution_worker_index'] == 0
                and case['case_id'] == context['case_id'],
                'actual worker/case metadata differs')
            first_cache=self.s.transformer.blocks[0].attn1.attn_caches[self.s.cache_name]
            history=int(first_cache['mask'].sum().item())
            bucket = HISTORY_TO_KV.get(history)
            if bucket is None or bucket in self.attempted:
                return None
            self.attempted.add(bucket)
            require(self._histories()[0] == history, 'matched request history changed during observation')
            live = self._live_inventory()
            estimate = self._size(live)+self._size(request)
            for block in self.s.transformer.blocks:
                cache = block.attn1.attn_caches[self.s.cache_name]
                estimate += 2*cache['k'].shape[0]*history*cache['k'].shape[2]*cache['k'].shape[3]*cache['k'].element_size()
                estimate += sum(self._size(cache[k]) for k in ('mask','id','is_pred'))
            # Keep a conservative128MiB for one real query/oracle/serialization.
            reserve = int(estimate*1.02)+128*2**20
            if self.used+reserve > self.cap-METADATA_RESERVE:
                self._event('UNKNOWN_SKIPPED_PRECOPY_CAP', action_KV=bucket,
                    estimated_reserved_bytes=reserve, actual_request_continues=True)
                self._retire_if_done()
                return None
            token = object()
            config_values,config_types,config_unknown=self._job_configuration()
            self.active = dict(token=token, bucket=bucket, history=history, errors=[], saved={},
                query=None, query_inflight=False, query_count=0,
                before_request_count=len(self.runtime.requests),
                payload=dict(schema='tf-fullsuite-request-replay-fixture/v1',
                    source_policy=dict(case=self.clone(case), context=self.clone(context), arm='C',
                        draft_checkpoint=str(self.s._p0_action_draft_b2_checkpoint),
                        mode=str(self.s._p0_action_draft_mode), selector=str(self.s._p0_action_draft_selector),
                        threshold=float(self.s._p0_action_draft_threshold),
                        target_dtype=str(self.s.dtype), vae_device=str(self.s.vae_device),
                        cache_mode=str(getattr(self.s,'_cache_mode','UNKNOWN')),
                        draft_architecture=str(getattr(self.s._p0_action_draft_b2_predictor,
                            'architecture','UNKNOWN')),
                        draft_condition=str(getattr(self.s._p0_action_draft_b2_predictor,
                            'checkpoint_condition','UNKNOWN')),
                        target_source_file=str(getattr(self.source, '__file__', 'UNKNOWN')),
                        device_identity=dict(getattr(self.runtime, 'gpu_identity', {}))),
                    metadata=self.clone(self.metadata), request=self.clone(request),
                    request_pre_state=self.capture_state(self.s),
                    live_layouts=self._layouts(live), request_layouts=self._layouts(request),
                    draft_normalization=self.clone(live['draft_normalization']),
                    effective_job_config=config_values,
                    effective_job_config_source_types=config_types,
                    effective_job_config_unknown=config_unknown,
                    state_and_config_inventory_complete=not config_unknown,
                    request_history_tokens=history, expected_action_KV=bucket,
                    restoration_note='CPU tensor copies are contiguous values; restore recorded live stride/offset/layout separately. Preserve recognized observation wrappers outside numerical scheduler state.',
                    no_verifier_KV_duplicate=True, extra_model_calls=0,
                    future_replay_qualification='UNKNOWN_NOT_RUN', permissions=PERMISSIONS))
            self._event('CAPTURE_REQUEST_PRESTATE_IN_RAM', action_KV=bucket,
                context=self.clone(context), request_history_tokens=history,
                estimate_bytes=estimate, reservation_bytes=reserve)
            return token
        except Exception as exc:
            self.disabled = True
            self._gap('begin', exc)
            self.active = None
            self._retire_if_done()
            return None
        finally:
            self.observer_host_sec += time.monotonic()-started

    def _pre(self, _model, args, kwargs):
        active = self.active
        if active is None or active['errors'] or kwargs.get('candidate_cache_rows') is None:
            return
        started = time.monotonic()
        try:
            active['query_count'] += 1
            if active['query'] is not None:
                return
            inp = args[0] if args else kwargs['input_dict']
            histories = self._histories()
            q = int(inp['grid_id'].shape[1])
            require(q == 16 and histories[0]+q == active['bucket'], 'actual candidate action KV bucket differs')
            require(inp['noisy_latents'].shape[0] == 14 and kwargs.get('action_mode') is True
                and kwargs.get('update_cache',0) == 0, 'actual read-only B14 candidate contract differs')
            active['query'] = dict(input=self.clone(inp), kwargs=self.clone(kwargs),
                input_layouts=self._layouts(inp), kwargs_layouts=self._layouts(kwargs),
                history_tokens=histories, Q=q, total_KV=active['bucket'],
                history_values='NOT_DUPLICATED; recreate in RAM by full native policy replay')
            active['query_inflight'] = True
        except Exception as exc:
            self._gap('first_real_candidate_pre', exc)
        finally:
            self.observer_host_sec += time.monotonic()-started

    def _post(self, _model, _args, _kwargs, output):
        active = self.active
        if active is None or not active['query_inflight']:
            return
        started = time.monotonic()
        try:
            active['query_inflight'] = False
            require(output is not None, 'real candidate forward did not return')
            active['query']['output'] = self.clone(output)
            active['query']['output_layout'] = self._layouts(output)
        except Exception as exc:
            self._gap('first_real_candidate_post', exc)
        finally:
            self.observer_host_sec += time.monotonic()-started

    def saved(self, name, value):
        active = self.active
        if active is None or active['errors']:
            return
        started = time.monotonic()
        try:
            label = Path(name).name.split('_',1)[0]
            require(label in ('actions','latents') and label not in active['saved'],
                'one original normalized-action/video terminal tensor required')
            active['saved'][label] = self.clone(value)
            active['payload'].setdefault('saved_output_layouts', {})[label] = self._layouts(value)
        except Exception as exc:
            self._gap('saved_actual_output', exc)
        finally:
            self.observer_host_sec += time.monotonic()-started

    def end(self, token, *, result):
        if token is None:
            return None
        started, partial = time.monotonic(), None
        try:
            active = self.active
            require(active is not None and active['token'] is token, 'optional request token differs')
            require(not active['errors'], 'earlier capture errors retained; fixture UNKNOWN')
            require(len(self.runtime.requests) == active['before_request_count']+1,
                'one complete native policy request required')
            row = self.runtime.requests[-1]
            require(row['kind']=='policy' and row['role']=='S' and row['arm']=='C' and row['ok'] is True
                and row['audit']['status']=='PASS_COMPLETE_ROLE_REQUEST' and row['audit']['g']==50,
                'complete native fifty-update C chunk oracle missing')
            require(set(active['saved']) == {'actions','latents'} and active['query'] is not None
                and 'output' in active['query'] and not active['query_inflight'], 'complete real output/query oracle missing')
            payload = active['payload']
            payload.update(status=('CAPTURED_COMPLETE_NATIVE_POLICY_REQUEST_WITH_ORACLES'
                    if payload['state_and_config_inventory_complete'] else
                    'CAPTURED_NATIVE_POLICY_REQUEST_WITH_CONFIG_GAPS_UNQUALIFIED'),
                first_real_candidate=active['query'], saved_original_outputs=active['saved'],
                returned_physical_action=self.clone(result), final_RNG=self.capture_rng(self.s),
                policy_record=self.clone(row), verifier_calls_observed=active['query_count'],
                final_all_valid_KV_oracle='UNKNOWN_NOT_SERIALIZED; local reference-versus-new replay RAM comparison required',
                mathematical_tensor_values_copied=True, physical_strides_preserved_in_saved_tensors=False,
                captured_observer_host_sec=self.observer_host_sec,
                instrumented_capture_chunk_not_a_formal_timing_sample=True)
            estimate=self._size(payload)+4*2**20
            require(self.used+estimate <= self.cap-METADATA_RESERVE, 'prewrite fixture cap; no incomplete fixture emitted')
            import torch
            final=self.folder/('KV%04d-request.pt' % active['bucket'])
            partial=final.with_suffix('.pt.partial')
            require(not final.exists() and not partial.exists(), 'one immutable fixture per bucket')
            with partial.open('xb') as stream:
                bounded=_BoundedWriter(stream,self.cap-METADATA_RESERVE-self.used)
                torch.save(payload,bounded)
                bounded.flush()
            # Link is exclusive even if an unexpected producer created final.
            os.link(partial,final)
            partial.unlink()
            size=final.stat().st_size
            self.used += size
            receipt=dict(action_KV=active['bucket'], request_history_tokens=active['history'],
                path=str(final), bytes=size, source_policy=payload['source_policy'],
                complete_native_policy_request=True, replay_qualified=False,
                state_and_config_inventory_complete=payload['state_and_config_inventory_complete'],
                effective_job_config_unknown=payload['effective_job_config_unknown'],
                final_all_valid_KV_oracle='UNKNOWN_NOT_SERIALIZED',
                instrumented_capture_chunk_not_a_formal_timing_sample=True,
                no_new_teacher_or_model_calls=True)
            self.completed[active['bucket']] = receipt
            self._event('CAPTURED_COMPLETE_NATIVE_REQUEST_FIXTURE', fixture=receipt)
            return receipt
        except Exception as exc:
            if partial is not None and partial.is_file():
                # A failed partial is retained and consumes the same data cap.
                self.used += partial.stat().st_size
            self.disabled=True
            self._gap('end',exc)
            return None
        finally:
            self.active=None
            self.observer_host_sec += time.monotonic()-started
            self._retire_if_done()

    def fail(self, token, error):
        if token is None:
            return
        if self.active is not None and self.active['token'] is token:
            self._gap('original_policy_failed', error)
            self.active=None
            self._retire_if_done()

    def _retire_if_done(self):
        if self.active is not None or not self.handles or not (
                self.disabled or len(self.attempted) == len(HISTORY_TO_KV)):
            return
        for handle in self.handles:
            try:
                handle.remove()
            except Exception as exc:
                self._gap('remove_completed_observation_hook',exc)
        self.handles=[]
        try:
            self._event('FINITE_CAPTURE_OBSERVATION_HOOKS_REMOVED',
                attempted_action_KV=sorted(self.attempted), all_future_requests_unobserved=True)
        except Exception as exc:
            self._gap('retire_receipt',exc)

    def close(self):
        if self.closed:
            return
        for handle in self.handles:
            try:
                handle.remove()
            except Exception as exc:
                self._gap('remove_observation_hook',exc)
        self.handles=[]
        if self.worker_index == 0:
            try:
                self._event('CLOSED_OPTIONAL_CAPTURE', completed_action_KV=sorted(self.completed),
                    missing_action_KV={str(k):'UNKNOWN_NOT_CAPTURED' for k in HISTORY_TO_KV.values()
                        if k not in self.completed}, data_bytes=self.used,
                    observer_host_sec=self.observer_host_sec,
                    no_new_model_calls=True, future_local_replay_status='UNKNOWN_NOT_RUN')
            except Exception as exc:
                self._gap('close_receipt',exc)
        self.closed=True

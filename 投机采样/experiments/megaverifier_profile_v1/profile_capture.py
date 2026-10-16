"""Candidate-only profiling and separate real block-0 arithmetic captures.

Import is CPU-only: torch is imported only after the supervisor worker gate.
No profile duration is a benchmark result.  Existing model/cache math is kept.
"""
from pathlib import Path
import json
from functools import wraps

from mv_profile_contract import write, stamp, digest


TRACE_MAX_BYTES = 128 * 1024 * 1024


def layout(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                stride=list(tensor.stride()), storage_offset=tensor.storage_offset(),
                contiguous=tensor.is_contiguous(), device=str(tensor.device))


class CandidateObserver:
    def __init__(self, server, out, label, indices=(), mode='count'):
        import torch
        if mode not in ('count', 'capture', 'profile'):
            raise ValueError('unknown observation mode')
        self.torch, self.server, self.model = torch, server, server.transformer
        self.out, self.label, self.mode = Path(out), label, mode
        self.indices = frozenset(indices)
        self.count, self.active, self.current = 0, False, None
        self.rows, self.handles, self.open_ranges = [], [], []
        self.capture_tensors, self.capture_layout = {}, {}
        self.initial_forward = self.model.forward
        self.original_pack = server._repeat_action_candidates_for_cfg
        self.original = self.wrapped = None
        self.closed = False
        if mode == 'profile':
            self._profile_hooks()
        elif mode == 'capture':
            self._capture_hooks()
        @wraps(self.original_pack)
        def pack(*args, **kwargs):
            result = self.original_pack(*args, **kwargs)
            # Online O3 installs SharedPrefix inside the action call.  The
            # original packer has now installed the actual shared model wrapper.
            # Intercept only the immediately following candidate forward outside
            # that wrapper; release it before the action closes its lease.
            if self.original is not None:
                raise RuntimeError('candidate pack without consuming prior forward')
            self.original = self.model.forward
            @wraps(self.original)
            def candidate_forward(*forward_args, **forward_kwargs):
                try:
                    return self._forward(*forward_args, **forward_kwargs)
                finally:
                    self._release_forward()
            self.wrapped = candidate_forward
            self.model.forward = candidate_forward
            return result
        self.wrapped_pack = pack
        server._repeat_action_candidates_for_cfg = pack

    def _release_forward(self):
        if self.original is None:
            return
        if self.model.forward is not self.wrapped:
            raise RuntimeError('candidate observer forward ownership lost')
        self.model.forward = self.original
        self.original = self.wrapped = None

    def _profile_hooks(self):
        torch = self.torch
        for i, block in enumerate(self.model.blocks):
            modules = [('block', block)] + [(name, getattr(block, name)) for name in
                       ('norm1', 'attn1', 'norm2', 'attn2', 'norm3', 'ffn')]
            for name, module in modules:
                label = f'mv/block/{i:02d}/{name}'
                def pre(_module, _args, label=label):
                    if self.active:
                        scope = torch.profiler.record_function(label)
                        scope.__enter__()
                        self.open_ranges.append((_module, scope))
                def post(_module, _args, _result):
                    if self.active:
                        owner, scope = self.open_ranges.pop()
                        if owner is not _module:
                            raise RuntimeError('unbalanced module profile scope')
                        scope.__exit__(None, None, None)
                self.handles.append(module.register_forward_pre_hook(pre))
                self.handles.append(module.register_forward_hook(post, always_call=True))

    def _keep(self, name, value):
        if name in self.capture_tensors:
            raise RuntimeError('duplicate block0 capture: ' + name)
        self.capture_layout[name] = layout(value)
        self.capture_tensors[name] = value.detach().clone()

    def _capture_hooks(self):
        block = self.model.blocks[0]
        def block_pre(module, args, kwargs):
            if not self.active:
                return
            hidden = args[0] if len(args) > 0 else kwargs['hidden_states']
            temb = args[2] if len(args) > 2 else kwargs['temb']
            self._keep('self_residual_hidden', hidden)
            self._keep('temb', temb)
            self._keep('scale_shift_table', module.scale_shift_table)
            # Separate warmup only. This exactly repeats the original table
            # expression and does not intercept or replace its real execution.
            combined = module.scale_shift_table[None] + temb.float()
            for j, name in enumerate(('shift', 'scale', 'gate', 'c_shift', 'c_scale', 'c_gate')):
                self._keep(name, combined[:, :, j, :])
        self.handles.append(block.register_forward_pre_hook(block_pre, with_kwargs=True))
        def add_pre(module, name):
            def pre(_module, args):
                if self.active:
                    self._keep(name, args[0])
            self.handles.append(module.register_forward_pre_hook(pre))
        def add_post(module, name):
            def post(_module, _args, result):
                if self.active:
                    self._keep(name, result)
            self.handles.append(module.register_forward_hook(post))
        add_post(block.norm1, 'norm1_fp32')
        add_pre(block.attn1, 'self_input')
        add_post(block.attn1, 'self_attn_output')
        add_pre(block.norm3, 'ffn_residual_hidden_fp32')
        add_post(block.norm3, 'norm3_fp32')
        add_pre(block.ffn, 'ffn_input')
        add_post(block.ffn, 'ffn_output')
        add_post(block, 'block_output')

    def _save_capture(self, index):
        torch = self.torch
        required = {'self_residual_hidden', 'temb', 'scale_shift_table', 'shift', 'scale',
                    'gate', 'c_shift', 'c_scale', 'c_gate', 'norm1_fp32', 'self_input',
                    'self_attn_output', 'ffn_residual_hidden_fp32', 'norm3_fp32',
                    'ffn_input', 'ffn_output', 'block_output'}
        if set(self.capture_tensors) != required:
            raise RuntimeError('incomplete real block0 capture')
        tensors = {key: value.cpu() for key, value in self.capture_tensors.items()}
        assert tuple(tensors['self_residual_hidden'].shape) == (14, 16, 3072)
        assert tensors['self_residual_hidden'].dtype == torch.bfloat16
        assert tuple(tensors['temb'].shape) == (14, 16, 6, 3072)
        for key in ('norm1_fp32', 'norm3_fp32', 'shift', 'scale', 'gate', 'c_shift', 'c_scale', 'c_gate'):
            assert tensors[key].dtype == torch.float32
        for key in ('self_input', 'self_attn_output', 'ffn_input', 'ffn_output', 'block_output'):
            assert tensors[key].dtype == torch.bfloat16
        assert all(bool(torch.isfinite(value).all()) for value in tensors.values())
        folder = self.out / 'captures'
        folder.mkdir(exist_ok=True)
        path = folder / f'{self.label}-candidate-{index:02d}-block0.pt'
        if path.exists():
            raise FileExistsError(path)
        torch.save(dict(schema='mv-real-block0-arithmetic/v1', length=self.label,
            candidate_index=index, tensors=tensors, original_layout=self.capture_layout,
            table_parameters_source='recomputed original FP32 table+temb expression in separate warmup',
            actual_norm_attention_FFN_inputs_outputs=True), path)
        row = dict(mode='capture', candidate_index=index, path=str(path),
                   bytes=path.stat().st_size, sha256=digest(path),
                   original_layout=dict(self.capture_layout), at=stamp())
        write(path.with_suffix('.json'), row, exclusive=True, durable=True)
        self.rows.append(row)
        self.capture_tensors.clear()
        self.capture_layout.clear()

    def _save_profile(self, profiler, index):
        folder = self.out / 'profiles'
        folder.mkdir(exist_ok=True)
        stem = f'{self.label}-candidate-{index:02d}'
        trace = folder / (stem + '.trace.json')
        temporary = folder / (stem + '.trace.partial.json')
        if trace.exists() or temporary.exists():
            raise FileExistsError(trace)
        profiler.export_chrome_trace(str(temporary))
        size = temporary.stat().st_size
        temporary.rename(trace)
        if size > TRACE_MAX_BYTES:
            # Preserve this failed diagnostic artifact; the supervisor enforces
            # the separate whole-job disk limit. Never count it as a valid trace.
            write(folder / (stem + '.validation-failed.json'),
                  dict(status='FAIL_TRACE_SIZE', trace=str(trace), bytes=size,
                       limit=TRACE_MAX_BYTES), exclusive=True, durable=True)
            raise RuntimeError('trace exceeded fixed 128MiB per-file bound')
        document = json.loads(trace.read_text())
        events = document['traceEvents']
        kernels = [event for event in events if event.get('ph') == 'X'
                   and 'kernel' in str(event.get('cat', '')).split(',')]
        candidate_name = f'mv/candidate/{self.label}/{index:02d}'
        candidate_markers = sum(event.get('ph') == 'X' and event.get('name') == candidate_name
                                for event in events)
        block_markers = {f'mv/block/{i:02d}/block': 0 for i in range(30)}
        for event in events:
            if event.get('ph') == 'X' and event.get('name') in block_markers:
                block_markers[event['name']] += 1
        kernel_duration = sum(float(event.get('dur', 0)) for event in kernels)
        validation = dict(cuda_kernel_activity_count=len(kernels),
            cuda_kernel_duration_sum_us=kernel_duration,
            kernel_duration_sum_is_not_wall_time=True,
            candidate_marker_count=candidate_markers, block_marker_counts=block_markers,
            valid=(len(kernels) > 0 and kernel_duration > 0 and candidate_markers == 1
                   and all(count == 1 for count in block_markers.values())))
        del document, events, kernels
        averages = []
        for event in profiler.key_averages():
            averages.append(dict(name=event.key, count=event.count,
                self_cpu_time_us=event.self_cpu_time_total, cpu_time_us=event.cpu_time_total,
                self_device_time_us=getattr(event, 'self_device_time_total', 0.0),
                device_time_us=getattr(event, 'device_time_total', 0.0)))
        row = dict(mode='profile', candidate_index=index, trace=str(trace),
                   trace_bytes=size, trace_sha256=digest(trace), at=stamp(),
                   CPU_CUDA=True, record_shapes=False, profile_memory=False, with_stack=False,
                   nested_event_times_must_not_be_summed=True,
                   profiling_is_not_a_speed_measurement=True, activity_validation=validation,
                   status='PASS' if validation['valid'] else 'FAIL_MISSING_ACTIVITY_OR_MARKERS',
                   key_averages=averages)
        table = folder / (stem + '.key_averages.json')
        write(table, row, exclusive=True, durable=True)
        if not validation['valid']:
            raise RuntimeError('profile lacks CUDA kernel activity or complete unique module markers')
        self.rows.append({k: v for k, v in row.items() if k != 'key_averages'} | {'key_averages_path': str(table)})

    def _forward(self, *args, **kwargs):
        rows = kwargs.get('candidate_cache_rows')
        if rows is None and len(args) > 5:
            rows = args[5]
        if rows is None:
            raise RuntimeError('packed candidate was followed by non-candidate forward')
        index = self.count
        self.count += 1
        if index not in self.indices or self.mode == 'count':
            return self.original(*args, **kwargs)
        if self.active:
            raise RuntimeError('nested candidate observation')
        self.current = index
        torch = self.torch
        if self.mode == 'capture':
            self.active = True
            try:
                result = self.original(*args, **kwargs)
            finally:
                self.active = False
            self._save_capture(index)
            return result
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                profile_memory=False, with_stack=False) as profiler:
            self.active = True
            try:
                with torch.profiler.record_function(f'mv/candidate/{self.label}/{index:02d}'):
                    result = self.original(*args, **kwargs)
                torch.cuda.synchronize()
            finally:
                self.active = False
        if self.open_ranges:
            raise RuntimeError('unclosed module profile scope')
        self._save_profile(profiler, index)
        return result

    def close(self):
        if self.closed:
            return
        try:
            self._release_forward()
            if self.server._repeat_action_candidates_for_cfg is not self.wrapped_pack:
                raise RuntimeError('candidate observer packer ownership lost')
            self.server._repeat_action_candidates_for_cfg = self.original_pack
            if self.model.forward != self.initial_forward:
                raise RuntimeError('shared forward was not restored after request')
        finally:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            self.capture_tensors.clear()
            self.active = False
            while self.open_ranges:
                _, scope = self.open_ranges.pop()
                scope.__exit__(None, None, None)
            self.closed = True

    def summary(self):
        return dict(mode=self.mode, candidate_forward_count=self.count,
                    requested_indices=sorted(self.indices), observed_indices=[r['candidate_index'] for r in self.rows],
                    artifacts=list(self.rows), closed=self.closed)

"""Exact, unhashed, caller-owned RAM observations for the complete PF2 panel.

No capture runs at import.  Returned tensor/array storage belongs to the
snapshot, never to the running model.  This module writes no files and does
not initialize CUDA: the caller must already own the assigned server context.
"""
from __future__ import annotations

import random
import struct


def _device(server):
    import torch
    return torch.device(getattr(server, 'device', server))


def clone_raw(value):
    """Clone only an explicit tree of data; never deepcopy a module/model."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu').contiguous().clone()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('object arrays are not auditable raw state')
        return value.copy(order='C')
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, dict):
        if any(type(k) not in (str, int) for k in value):
            raise TypeError('raw state requires explicit string/integer keys')
        return {k: clone_raw(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_raw(x) for x in value]
    if isinstance(value, tuple):
        return tuple(clone_raw(x) for x in value)
    if value is None or type(value) in (str, bytes, bool, int, float):
        return value
    raise TypeError('unsupported raw state leaf: ' + type(value).__name__)


def capture_rng(server):
    import numpy as np
    import torch
    device = _device(server)
    if device.type == 'cuda' and not torch.cuda.is_initialized():
        raise RuntimeError('RNG observation requires the already-owned CUDA context')
    return dict(Python=clone_raw(random.getstate()), NumPy=clone_raw(np.random.get_state()),
        CPU=torch.get_rng_state().cpu().clone(),
        CUDA=torch.cuda.get_rng_state(device).cpu().clone() if device.type == 'cuda' else None)


def restore_rng(server, rng):
    import numpy as np
    import torch
    if set(rng) != {'Python', 'NumPy', 'CPU', 'CUDA'}:
        raise ValueError('complete Python/NumPy/CPU/CUDA RNG inventory required')
    device = _device(server)
    if device.type == 'cuda' and (not torch.cuda.is_initialized() or rng['CUDA'] is None):
        raise RuntimeError('restoring CUDA RNG requires the existing assigned context/state')
    if device.type != 'cuda' and rng['CUDA'] is not None:
        raise ValueError('cannot restore CUDA state into a CPU-only owner')
    for name in ('CPU', 'CUDA'):
        value = rng[name]
        if value is not None and (not isinstance(value, torch.Tensor)
                or value.device.type != 'cpu' or value.dtype != torch.uint8 or value.ndim != 1):
            raise TypeError('RNG bytes must be CPU uint8 vectors')
    random.setstate(clone_raw(rng['Python']))
    np.random.set_state(clone_raw(rng['NumPy']))
    torch.set_rng_state(rng['CPU'].clone())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(rng['CUDA'].clone(), device)


def _scheduler(scheduler):
    # The native FlowMatchScheduler has only scalar/tensor numerical fields.
    # Unknown future objects fail closed instead of being replaced by repr().
    state, exclusions = {}, []
    for key, value in vars(scheduler).items():
        function = getattr(value, '__func__', None)
        owner = getattr(value, '__self__', None)
        if (key == 'step' and function is not None and owner is not None
                and function.__module__ == 'tf_pf2_quality_v1.evidence'
                and function.__qualname__ == 'ExecutionEvidence.step'
                and type(owner).__module__ == 'tf_pf2_quality_v1.evidence'
                and type(owner).__name__ == 'ExecutionEvidence'
                and getattr(owner, 'scheduler', None) is scheduler):
            exclusions.append(dict(field='step', module=function.__module__,
                qualname=function.__qualname__,
                reason='recognized live observation wrapper; native numerical scheduler fields retained'))
            continue
        state[key] = clone_raw(value)
    # This audit annotation is stable across C/P; no observer counters, object
    # addresses or timings are misrepresented as mathematical scheduler state.
    state['_recognized_observer_exclusions'] = exclusions
    return state


def capture_full_state(server):
    """Capture all valid original-layout history KV and mathematical state.

    Invalid K/V allocation is intentionally not read.  Full visibility/index
    vectors and the original layout are retained.  lazy_scheduler_state is
    separate because native _reset does not rebuild the timestep tables;
    native _infer unconditionally replaces them before consuming them.  The
    evidence caller must record any narrowly scoped pre-infer exclusion,
    never describe such an exclusion as equality of every captured field.
    """
    import torch
    if _device(server).type == 'cuda' and not torch.cuda.is_initialized():
        raise RuntimeError('state observation cannot initialize a CUDA context')
    layers = []
    for index, block in enumerate(server.transformer.blocks):
        attn = block.attn1
        if getattr(attn, 'cache_mode', 'original') != 'original':
            raise ValueError('this panel captures the frozen original KV layout only')
        cache = attn.attn_caches[server.cache_name]
        if not isinstance(cache, dict) or set(cache) != {'k', 'v', 'mask', 'id', 'is_pred'}:
            raise ValueError('original complete KV cache inventory changed')
        k, v, mask = cache['k'], cache['v'], cache['mask']
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1
                or k.ndim != 4 or k.shape != v.shape or k.shape[1] != mask.numel()
                or cache['id'].shape != mask.shape or cache['is_pred'].shape != mask.shape):
            raise ValueError('complete KV mask/index/geometry changed')
        layers.append(dict(layer=index, mask=clone_raw(mask), id=clone_raw(cache['id']),
            is_pred=clone_raw(cache['is_pred']), k=clone_raw(k[:, mask]), v=clone_raw(v[:, mask]),
            layout=dict(full_shape=list(k.shape), k_stride=list(k.stride()),
                v_stride=list(v.stride()), dtype=str(k.dtype), valid_count=int(mask.sum().item()))))
    if len(layers) != 30:
        raise ValueError('all thirty frozen target layers must be observed')
    vae = server.streaming_vae
    half = getattr(server, 'streaming_vae_half', None)
    names = ('init_latent', 'prompt_embeds', 'negative_prompt_embeds', 'action_mask',
        'actions_q01', 'actions_q99', 'action_norm_method', 'action_per_frame',
        'height', 'width', 'latent_height', 'latent_width')
    mathematical = {name: clone_raw(getattr(server, name)) for name in names}
    return dict(schema='tf-pf2-full-valid-state-ram/v1', frame_st_id=int(server.frame_st_id),
        cache_name=str(server.cache_name), use_cfg=bool(server.use_cfg), KV=layers,
        mathematical=mathematical, VAE=clone_raw(vae.feat_cache),
        VAE_half=None if half is None else clone_raw(half.feat_cache),
        lazy_scheduler_state=dict(video=_scheduler(server.scheduler),
            action=_scheduler(server.action_scheduler)), RNG=capture_rng(server))


def compare_raw(reference, actual, *, max_mismatches=64):
    """Compare raw typed bytes; return a compact JSON-ready result, no digest."""
    import numpy as np
    import torch
    if not isinstance(max_mismatches, int) or max_mismatches < 1:
        raise ValueError('a positive mismatch display cap is required')
    mismatches, counts = [], dict(leaves=0, mismatches=0)

    def fail(path, reason, **details):
        counts['mismatches'] += 1
        if len(mismatches) < max_mismatches:
            mismatches.append(dict(path=path, reason=reason, **details))

    def visit(a, b, path):
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            counts['leaves'] += 1
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                fail(path, 'tensor/type differs'); return
            if a.dtype != b.dtype or a.shape != b.shape:
                fail(path, 'tensor dtype/shape differs', reference_dtype=str(a.dtype),
                    actual_dtype=str(b.dtype), reference_shape=list(a.shape), actual_shape=list(b.shape)); return
            aa = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            bb = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            if not torch.equal(aa, bb):
                fail(path, 'tensor raw bytes differ', dtype=str(a.dtype), shape=list(a.shape))
            return
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            counts['leaves'] += 1
            if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
                fail(path, 'array/type differs'); return
            if a.dtype.hasobject or b.dtype.hasobject:
                raise TypeError('object arrays cannot be compared as raw state')
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes(order='C') != b.tobytes(order='C'):
                fail(path, 'array dtype/shape/raw bytes differ', reference_dtype=str(a.dtype),
                    actual_dtype=str(b.dtype), reference_shape=list(a.shape), actual_shape=list(b.shape))
            return
        if isinstance(a, np.generic) or isinstance(b, np.generic):
            counts['leaves'] += 1
            if type(a) is not type(b) or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                fail(path, 'numpy scalar dtype/raw bytes differ')
            return
        if type(a) is not type(b):
            fail(path, 'type differs', reference_type=type(a).__name__, actual_type=type(b).__name__); return
        if isinstance(a, dict):
            if set(a) != set(b):
                fail(path, 'dictionary keys differ', reference_keys=sorted(map(str, a)),
                    actual_keys=sorted(map(str, b)))
            for key in a:
                if key in b:
                    visit(a[key], b[key], path + '/' + str(key))
            return
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                fail(path, 'sequence length differs', reference_length=len(a), actual_length=len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + '/' + str(i))
            return
        counts['leaves'] += 1
        if type(a) is float:
            if struct.pack('!d', a) != struct.pack('!d', b):
                fail(path, 'float64 raw bytes differ')
        elif a is None or type(a) in (str, bytes, bool, int):
            if a != b:
                fail(path, 'scalar value differs')
        else:
            raise TypeError('unsupported comparison leaf: ' + type(a).__name__)

    visit(reference, actual, '$')
    return dict(equal=counts['mismatches'] == 0, mismatches=mismatches,
        mismatch_count=counts['mismatches'], displayed_mismatches=len(mismatches),
        truncated=counts['mismatches'] > len(mismatches), compared_leaves=counts['leaves'],
        method='typed raw tensor/array/scalar bytes; no hashes or numerical tolerance')


# Explicit compatibility aliases; every variant returns independent CPU data.
capture_state = capture_full_state
compare_state = compare_raw
references = clone_raw


def equal(reference, actual):
    return compare_raw(reference, actual)['equal']

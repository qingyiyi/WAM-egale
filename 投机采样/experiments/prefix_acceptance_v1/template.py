"""O1 only: cache immutable Python code; bind fresh callbacks every install.

The frozen install/admit executable bodies are retained in private namespaces.
All attention math, KV ownership and invalidation methods remain inherited.
No tensor, model, controller, callback or function globals are cached.
"""
from dataclasses import dataclass
import threading
import types

from paths import BASE, S0, load

import adapter as base
from contract import MODEL_SHA256

frozen = load('chunk_template_frozen_prefix', S0 / 'prefix_adapter.py')
if base.__file__ != str(BASE / 'adapter.py'):
    raise ImportError('frozen base adapter required')


@dataclass(frozen=True)
class Template:
    original_code: types.CodeType
    private_code: types.CodeType
    defaults: tuple | None
    kwdefaults: tuple | None
    annotations: tuple
    doc: str | None


_lock = threading.RLock()
_templates = {}
_pinned = None
_counts = dict(AST_builds=0, bindings=0, pinned_code_builds=0)


def stats():
    with _lock:
        return {**_counts, 'templates': len(_templates)}


def reset_for_CPU_tests():
    global _pinned
    import os
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('cache reset is CPU-test-only')
    with _lock:
        _templates.clear()
        _pinned = None
        for key in _counts:
            _counts[key] = 0


def _cached_pinned_codes():
    global _pinned
    with _lock:
        if _pinned is None:
            _pinned = base._pinned_codes()
            _counts['pinned_code_builds'] += 1
        return _pinned


def _template_forward(original, callback):
    fn = original.__func__
    code = fn.__code__
    # Admission still checks the current source SHA, actual class/function,
    # executable structure and backend on every install, before this factory.
    key = (MODEL_SHA256, id(code))
    with _lock:
        template = _templates.get(key)
        if template is None:
            private = base._private_forward(original, None)
            if private.__closure__ is not None:
                raise base.LeaseError('unexpected template closure')
            template = Template(code, private.__code__, private.__defaults__,
                None if private.__kwdefaults__ is None else tuple(private.__kwdefaults__.items()),
                tuple(private.__annotations__.items()), private.__doc__)
            _templates[key] = template
            _counts['AST_builds'] += 1
        if template.original_code is not code:
            raise base.LeaseError('template code identity changed')
        _counts['bindings'] += 1
    # A new namespace/function is created for every layer and controller.
    # It cannot retain another chunk's callback or its historical KV leases.
    namespace = dict(fn.__globals__)
    namespace['__task_history'] = callback
    namespace['__name__'] = base.__name__
    result = types.FunctionType(template.private_code, namespace,
                                '_task_private_forward', template.defaults)
    result.__kwdefaults__ = None if template.kwdefaults is None else dict(template.kwdefaults)
    result.__annotations__ = dict(template.annotations)
    result.__doc__ = template.doc
    return result


def clone_function(fn, **overrides):
    """Retain bytecode/closure; isolate only named private helper bindings."""
    result = types.FunctionType(fn.__code__, {**fn.__globals__, **overrides},
                                fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    result.__annotations__ = dict(fn.__annotations__)
    result.__doc__ = fn.__doc__
    result.__qualname__ = fn.__qualname__
    return result


class TemplateSharedPrefix(frozen.SharedPrefix):
    install = clone_function(base.SingleWorkerCHC.install,
                             _private_forward=_template_forward)
    admit = clone_function(base.SingleWorkerCHC.admit,
                           _pinned_codes=_cached_pinned_codes)

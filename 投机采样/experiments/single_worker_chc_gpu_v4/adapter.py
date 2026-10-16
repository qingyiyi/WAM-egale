"""Default-off, instance-only C/HC historical KV adapter; original SDPA retained.

Only a source-pinned candidate HISTORY fragment is replaced. Projection,
normalization, RoPE, output processing, cross attention and public classes are
unchanged. This module neither imports a server nor starts workers/CUDA.
"""
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import inspect
from pathlib import Path
import sys
import threading
import types
import weakref

import torch

from contract import MODEL_SOURCE, MODEL_SHA256, ROWS, LAYER_COUNT, CACHE_BYTE_CAP, digest

_OWNERS = weakref.WeakKeyDictionary()
_LEGAL_MODULES = ('modules.model', 'wan_va.modules.model')


def _code_key(code):
    """Compare executable structure, not a spoofable name/filename alone."""
    return (code.co_code,
            tuple(_code_key(v) if isinstance(v, types.CodeType) else v for v in code.co_consts),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_flags, code.co_exceptiontable)


def _pinned_codes():
    # Compilation ONLY: no exec/import, model forward, tensor or CUDA operation.
    module = compile(MODEL_SOURCE.read_text(), str(MODEL_SOURCE), 'exec')
    cls = next(c for c in module.co_consts if isinstance(c, types.CodeType) and c.co_name == 'WanAttention')
    forward = next(c for c in cls.co_consts if isinstance(c, types.CodeType) and c.co_name == 'forward')
    sdpa = next(c for c in module.co_consts if isinstance(c, types.CodeType) and c.co_name == 'custom_sdpa')
    return _code_key(forward), _code_key(sdpa)


def _from_pinned_file(path):
    try:
        return Path(path).resolve() == MODEL_SOURCE.resolve()
    except (TypeError, ValueError, OSError):
        return False


class LeaseError(RuntimeError):
    pass


@dataclass
class Lease:
    request: str
    arm: str
    cache_name: str
    rows: torch.Tensor
    row_values: tuple
    row_version: int
    generation: int
    model_key: tuple
    revoked: bool = False
    reason: str = ''


@dataclass
class Entry:
    key: tuple
    k: torch.Tensor
    v: torch.Tensor
    payload_bytes: int


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError as exc:
        # Inference tensors can have no mutation counter: refuse rather than
        # falsely claiming that unobserved in-place writes are detectable.
        raise LeaseError('history/row tensor has no mutation version counter') from exc


def _tensor_key(t):
    if not isinstance(t, torch.Tensor):
        raise LeaseError('unknown history tensor schema')
    return (id(t), t.data_ptr(), _version(t), tuple(t.shape), tuple(t.stride()),
            t.storage_offset(), str(t.device), t.dtype)


def _private_forward(original, callback):
    """Mechanical AST splice; no public-file/class/global monkeypatch."""
    tree = ast.parse(MODEL_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanAttention')
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
    candidate = next(n for n in fn.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'candidate_cache_rows is not None')
    original_body = copy.deepcopy(candidate.body)
    if (len(original_body) != 7 or not isinstance(original_body[0], ast.If)
            or not isinstance(original_body[1], ast.If)):
        # Frozen source shape checked again even after SHA passes.
        raise LeaseError('candidate AST layout changed')
    candidate.body = original_body[:2] + ast.parse(
        'hidden_states = __task_history(self, query, key, value, kv_cache, '
        'cache_name, candidate_cache_rows, update_cache)').body
    # Normalize only the replaced fragment and compare the entire function.
    restored = copy.deepcopy(fn)
    restored_candidate = next(n for n in restored.body if isinstance(n, ast.If)
                              and ast.unparse(n.test) == 'candidate_cache_rows is not None')
    restored_candidate.body = original_body
    original_fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    if ast.dump(restored, include_attributes=False) != ast.dump(original_fn, include_attributes=False):
        raise LeaseError('AST changed outside candidate history')
    fn.name = '_task_private_forward'
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(original.__func__.__globals__)
    namespace['__task_history'] = callback
    namespace['__name__'] = __name__
    exec(compile(module, '<single-worker-C-HC-private-candidate-fragment>', 'exec'), namespace)
    return namespace[fn.name]


class SingleWorkerCHC:
    """Explicitly opt in; default construction does not patch any instance.

    `allow_cuda=False` is a separate fail-closed boundary. Enabling this task
    alone (or an old GPU flag) cannot execute a CUDA path. A future separately
    approved GPU runner must explicitly supply allow_cuda=True.
    """
    def __init__(self, model, *, enabled=False, expected_layers=LAYER_COUNT,
                 allow_cuda=False, max_cache_bytes=CACHE_BYTE_CAP,
                 diagnostic=False, expected_source_sha256=MODEL_SHA256):
        self.model, self.enabled = model, bool(enabled)
        self.expected_layers, self.allow_cuda = expected_layers, bool(allow_cuda)
        if type(max_cache_bytes) is not int or not 0 < max_cache_bytes <= CACHE_BYTE_CAP:
            raise ValueError('cache payload cap')
        self.max_cache_bytes, self.diagnostic = max_cache_bytes, diagnostic
        self.expected_source_sha256 = expected_source_sha256
        self.installed, self.owner_thread = False, threading.get_ident()
        self.epoch, self.generation, self.model_generation = 0, 0, 0
        self.active, self.entries, self.payload_bytes = None, {}, 0
        self.ops, self.patches, self.attentions = {}, [], []
        self.layer_ids = {}
        self.backends = {}

    def __enter__(self):
        return self.install()

    def __exit__(self, typ, value, tb):
        self.close()

    def _owner(self):
        if threading.get_ident() != self.owner_thread:
            raise LeaseError('single-worker adapter cannot be used from another thread')

    def _device(self, device):
        if torch.device(device).type != 'cpu' and not self.allow_cuda:
            raise LeaseError('CUDA not authorized for this CPU task')

    def _input_devices(self, value):
        if isinstance(value, torch.Tensor):
            self._device(value.device)
        elif isinstance(value, dict):
            for child in value.values():
                self._input_devices(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                self._input_devices(child)

    def _model_key(self):
        return (self.model_generation, self.model.training,
                tuple((id(p), _version(p), p.dtype, str(p.device)) for p in self.model.parameters()))

    def _live(self, rows, cache_name, update_cache=0, train_mode=False):
        token = self.active
        if token.revoked:
            raise LeaseError('lease revoked: ' + token.reason)
        if (rows is not token.rows or _version(rows) != token.row_version
                or cache_name != token.cache_name or update_cache != 0 or train_mode
                or self._model_key() != token.model_key):
            raise LeaseError('owned row/request/model generation context changed')

    def _patch(self, obj, name, value):
        had = name in obj.__dict__
        self.patches.append((obj, name, had, obj.__dict__.get(name)))
        setattr(obj, name, value)

    def _bump(self, layer, op):
        if self.diagnostic:
            row = self.ops.setdefault(layer, {})
            row[op] = row.get(op, 0) + 1

    def invalidate(self, reason='explicit external mutation'):
        self._owner()
        self.epoch += 1
        self.entries.clear()
        self.payload_bytes = 0
        if self.active is not None:
            self.active.revoked, self.active.reason = True, reason

    @contextmanager
    def mutation(self, reason):
        """Required for external snapshot restoration/uninstrumented writes."""
        self.invalidate(reason)
        try:
            yield
        finally:
            # Failed or reverted writes must not reactivate the old lease.
            self.invalidate(reason + ' finished (including exception)')

    def external_call(self, fn, *args, reason='snapshot restore', **kwargs):
        with self.mutation(reason):
            return fn(*args, **kwargs)

    def _mutation_wrapper(self, original, name):
        def wrapped(_instance, *args, **kwargs):
            with self.mutation(name):
                return original(*args, **kwargs)
        return wrapped

    def _attention_wrapper(self, layer, original, private):
        def forward(a, q, k, v, rotary_emb, update_cache=0, cache_name='pos',
                    candidate_cache_rows=None):
            self._owner()
            self._device(q.device)
            if candidate_cache_rows is None:
                with self.mutation('ordinary attention including update=0 rollback'):
                    return original(q, k, v, rotary_emb, update_cache, cache_name, None)
            if self.active is None or self.active.arm == 'R':
                return original(q, k, v, rotary_emb, update_cache, cache_name, candidate_cache_rows)
            try:
                self._precheck(layer, a, q, candidate_cache_rows, cache_name, update_cache)
                return private(a, q, k, v, rotary_emb, update_cache, cache_name, candidate_cache_rows)
            except BaseException:
                self.invalidate('candidate failure')
                raise
        return forward

    def install(self):
        if not self.enabled:
            return self
        self._owner()
        if self.installed or self.model in _OWNERS:
            raise LeaseError('adapter already installed on this model')
        self.admit()
        self.attentions = [b.attn1 for b in self.model.blocks]
        self.layer_ids = {id(a): layer for layer, a in enumerate(self.attentions)}
        self.backends = {id(a): a.attn_op for a in self.attentions}
        _OWNERS[self.model] = self
        try:
            for layer, a in enumerate(self.attentions):
                original = a.forward
                private = _private_forward(original, self._history)
                self._patch(a, 'forward', types.MethodType(self._attention_wrapper(layer, original, private), a))
                for name in ['allocate_slots', 'update_cache', 'restore_cache', 'clear_pred_cache',
                             'clear_cache', 'init_kv_cache']:
                    self._patch(a, name, types.MethodType(self._mutation_wrapper(getattr(a, name), name), a))
            original_model_forward = self.model.forward
            model_signature = inspect.signature(original_model_forward)
            def model_forward(_model, *args, **kwargs):
                self._owner()
                bound = model_signature.bind(*args, **kwargs)
                try:
                    self._input_devices(bound.arguments.get('input_dict'))
                    if bound.arguments.get('candidate_cache_rows') is None:
                        with self.mutation('non-candidate model forward'):
                            return original_model_forward(*args, **kwargs)
                    if self.active is not None and self.active.arm != 'R':
                        self._live(bound.arguments.get('candidate_cache_rows'),
                                   bound.arguments.get('cache_name', 'pos'),
                                   bound.arguments.get('update_cache', 0),
                                   bound.arguments.get('train_mode', False))
                    return original_model_forward(*args, **kwargs)
                except BaseException:
                    self.invalidate('model candidate failure')
                    raise
            self._patch(self.model, 'forward', types.MethodType(model_forward, self.model))
            # _apply covers model.to/device/dtype; train/load_state_dict are
            # conservative generation changes. Public methods are delegated.
            for name in ['_apply', 'train', 'load_state_dict']:
                original = getattr(self.model, name)
                def change(_model, *args, _original=original, _name=name, **kwargs):
                    self.model_generation += 1
                    with self.mutation('model generation ' + _name):
                        return _original(*args, **kwargs)
                self._patch(self.model, name, types.MethodType(change, self.model))
            self.installed = True
        except BaseException:
            self.close()
            raise
        return self

    def admit(self):
        """Read-only all-layer admission, usable BEFORE capture/cache reset.

        Exactly two known import spellings are accepted, each bound to the
        pinned public file, its actual exported classes, original executable
        forward and original SDPA. Does not install patches or need KV pools.
        """
        if not self.enabled:
            return dict(status='DISABLED', patched=False, target_forwards=0)
        self._owner()
        if digest(MODEL_SOURCE) != self.expected_source_sha256:
            raise LeaseError('public model source SHA changed')
        blocks = list(self.model.blocks)
        if len(blocks) != self.expected_layers:
            raise LeaseError('all-layer count mismatch')
        name = type(self.model).__module__
        module = sys.modules.get(name)
        if (name not in _LEGAL_MODULES or module is None
                or not _from_pinned_file(getattr(module, '__file__', None))
                or not _from_pinned_file(getattr(getattr(module, '__spec__', None), 'origin', None))
                or type(self.model) is not getattr(module, 'WanTransformer3DModel', None)):
            raise LeaseError('requires pinned legal WanAttention model module/class')
        if self.model.training:
            raise LeaseError('requires unmodified eval original WanAttention interface')
        forward_code, sdpa_code = _pinned_codes()
        for param in self.model.parameters():
            self._device(param.device)
        for layer, block in enumerate(blocks):
            a = block.attn1
            fn = getattr(a.forward, '__func__', None)
            if (type(a) is not getattr(module, 'WanAttention', None)
                    or type(a).__module__ != name or type(a).__name__ != 'WanAttention'
                    or type(a).__bases__ != (torch.nn.Module,)
                    or fn is None or fn is not type(a).forward
                    or fn is not type(a).__dict__.get('forward')
                    or fn.__globals__ is not module.__dict__
                    or not _from_pinned_file(inspect.getsourcefile(fn))
                    or not _from_pinned_file(fn.__code__.co_filename)
                    or _code_key(fn.__code__) != forward_code
                    or fn.__defaults__ != (0, 'pos', None) or fn.__kwdefaults__ is not None
                    or a.cache_mode != 'original' or a.training):
                raise LeaseError(f'requires unmodified eval original WanAttention interface (layer {layer})')
            backend = a.attn_op
            if (not inspect.isfunction(backend)
                    or backend is not getattr(module, 'custom_sdpa', None)
                    or backend is not fn.__globals__.get('custom_sdpa')
                    or backend.__globals__ is not module.__dict__
                    or not _from_pinned_file(backend.__code__.co_filename)
                    or _code_key(backend.__code__) != sdpa_code
                    or backend.__defaults__ is not None or backend.__kwdefaults__ is not None):
                raise LeaseError(f'requires original torch SDPA backend before install (layer {layer})')
        return dict(status='PASS', module=name, source=str(MODEL_SOURCE),
                    source_sha256=self.expected_source_sha256, layers=len(blocks),
                    all_layer_original_forward_and_SDPA=True, patched=False,
                    KV_pools_required=False, target_forwards=0)

    def close(self):
        self._owner()
        self.entries.clear()
        self.payload_bytes = 0
        if self.active is not None:
            self.active.revoked, self.active.reason = True, 'adapter closed'
            self.active = None
        for obj, name, had, old in reversed(self.patches):
            if had:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self.patches.clear()
        if _OWNERS.get(self.model) is self:
            del _OWNERS[self.model]
        self.installed = False

    @contextmanager
    def lease(self, *, request, arm, cache_name='pos', row_values=ROWS, device='cpu'):
        if not self.enabled:
            yield None
            return
        self._owner()
        if not self.installed or self.active is not None:
            raise LeaseError('install first; nested/reentrant leases are forbidden')
        if arm not in ('R', 'C', 'HC') or not isinstance(request, str) or not request:
            raise LeaseError('request/arm contract')
        row_values = tuple(row_values)
        if row_values != ROWS:
            raise LeaseError('only frozen seven conditional plus seven unconditional rows')
        self._device(device)
        # Owned row map MUST track mutation even in an outer inference context.
        with torch.inference_mode(False), torch.no_grad():
            rows = torch.tensor(row_values, dtype=torch.long, device=device)
        self.generation += 1
        token = Lease(request, arm, cache_name, rows, row_values, _version(rows), self.generation,
                      self._model_key())
        self.active = token
        self.entries.clear()
        self.payload_bytes = 0
        self.ops = {}
        try:
            yield token
        finally:
            self.entries.clear()
            self.payload_bytes = 0
            token.revoked, token.reason = True, token.reason or 'lease ended'
            self.active = None

    def _key(self, layer, a, q, cache):
        if set(cache) != {'k', 'v', 'mask', 'id', 'is_pred'}:
            raise LeaseError('unknown original cache schema')
        token = self.active
        return (token.request, token.generation, token.cache_name, layer,
                self.epoch, self.model_generation, id(cache), a.cache_mode, id(a.attn_op),
                a.heads, a.inner_dim, q.dtype, str(q.device),
                tuple(_tensor_key(cache[k]) for k in ['k', 'v', 'mask', 'id', 'is_pred']))

    def _precheck(self, layer, a, q, rows, cache_name, update_cache):
        token = self.active
        if token.revoked:
            raise LeaseError('lease revoked: ' + token.reason)
        if (a.training or a.cache_mode != 'original' or update_cache != 0
                or cache_name != token.cache_name or q.shape[0] != 14):
            raise LeaseError('read-only frozen geometry/cache context changed')
        if a.attn_op is not self.backends[id(a)]:
            raise LeaseError('original attention backend identity changed')
        if rows is not token.rows or _version(rows) != token.row_version:
            raise LeaseError('row map identity/version changed; bind the owned canonical rows')
        if rows.dtype != torch.long or rows.device != q.device or rows.ndim != 1:
            raise LeaseError('row map dtype/device/shape')
        cache = a.attn_caches.get(cache_name)
        if cache is None:
            raise LeaseError('existing original KV required')
        if set(cache) != {'k', 'v', 'mask', 'id', 'is_pred'}:
            raise LeaseError('unknown original cache schema')
        if (cache['k'].shape != cache['v'].shape or cache['k'].ndim != 4
                or cache['k'].shape[0] != 2 or cache['k'].dtype != q.dtype
                or cache['k'].device != q.device or cache['v'].dtype != q.dtype
                or cache['v'].device != q.device or cache['mask'].dtype != torch.bool
                or cache['mask'].device != q.device
                or cache['mask'].shape != (cache['k'].shape[1],)
                or cache['id'].dtype != torch.long or cache['id'].device != q.device
                or cache['id'].shape != cache['mask'].shape
                or cache['is_pred'].dtype != torch.bool or cache['is_pred'].device != q.device
                or cache['is_pred'].shape != cache['mask'].shape
                or cache['k'].shape[2:] != (a.heads, a.inner_dim // a.heads)):
            raise LeaseError('pool dtype/device/CFG geometry changed')
        key = self._key(layer, a, q, cache)
        entry = self.entries.get(layer)
        if entry is not None and entry.key != key:
            raise LeaseError('unannounced history/backend/model key change')

    def _valid(self, layer, cache):
        self._bump(layer, 'metadata')
        return cache['mask'].nonzero(as_tuple=False).squeeze(-1)

    def _rows_check(self, layer, rows, q, batch):
        self._bump(layer, 'row_check')
        if (rows.ndim != 1 or rows.numel() != q.shape[0] or rows.dtype != torch.long
                or rows.device != q.device or not bool(((rows >= 0) & (rows < batch)).all())):
            raise ValueError('invalid candidate_cache_rows for this KV cache')

    def _history(self, a, q, k, v, cache, cache_name, rows, update_cache):
        layer = self.layer_ids[id(a)]
        # Input projection does not modify history. Recheck after it so no
        # unexpected callback can mutate storage before the cached read.
        self._precheck(layer, a, q, rows, cache_name, update_cache)
        entry = self.entries.get(layer)
        if entry is None:
            valid = self._valid(layer, cache)
            need = 2 * rows.numel() * valid.numel() * cache['k'].shape[2] * cache['k'].shape[3] * cache['k'].element_size()
            if self.payload_bytes + need > self.max_cache_bytes:
                raise LeaseError('expanded history payload cap exceeded')
            self._bump(layer, 'gather')
            hk, hv = cache['k'][:, valid], cache['v'][:, valid]
            self._rows_check(layer, rows, q, hk.shape[0])
            self._bump(layer, 'expand')
            ek, ev = hk.index_select(0, rows), hv.index_select(0, rows)
            entry = Entry(self._key(layer, a, q, cache), ek, ev, need)
            self.entries[layer] = entry
            self.payload_bytes += need
        if self.active.arm == 'C':
            # Deliberately keep dynamic metadata cost. Epoch/version checks,
            # not this unused nonzero result, establish cache validity.
            self._valid(layer, cache)
            self._rows_check(layer, rows, q, cache['k'].shape[0])
        self._bump(layer, 'cat')
        full_k, full_v = torch.cat((entry.k, k), dim=1), torch.cat((entry.v, v), dim=1)
        self._bump(layer, 'SDPA')
        return a.attn_op(q, full_k, full_v)

"""Default-off instance-only S adapter, inheriting the audited v4 guards.

CFG stays batch2. This overrides ONLY the candidate history callback: the
public model, batch, projections, RoPE, cross attention and text are intact.
Unlike C/HC, S holds zero-copy references to the original two-row pools plus
valid-slot metadata, not gathered/expanded fourteen-row history tensors.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys

BASE = Path(__file__).resolve().parent.parent / 'single_worker_chc_gpu_v4'
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))
_base_spec=importlib.util.find_spec('adapter')
if _base_spec is None or Path(_base_spec.origin).resolve()!=(BASE/'adapter.py').resolve():
    raise ImportError('pinned v4 adapter import source required')
from adapter import SingleWorkerCHC, LeaseError
import adapter as _base_module
if Path(_base_module.__file__).resolve()!=(BASE/'adapter.py').resolve():
    raise ImportError('loaded adapter must be pinned v4 source')
from prefix_attention import shared_attention


@dataclass
class PrefixEntry:
    key: tuple
    k: object
    v: object
    payload_bytes: int
    slots: object
    slot_key: tuple


class SharedPrefix(SingleWorkerCHC):
    def __init__(self, model, *, cpu_reference_enabled=False, **kwargs):
        super().__init__(model, **kwargs)
        self.cpu_reference_enabled = bool(cpu_reference_enabled)
        self.prefix_peak_bytes = 0
        self.current_arm = None
        self.record_first = False
        self.first_attention = {}

    @contextmanager
    def lease(self, *, request, arm='S', **kwargs):
        if arm not in ('S', 'R', 'HC'):
            raise LeaseError('shared-prefix arms are R, frozen HC and S only')
        # HC supplies the frozen row ownership/host generation guards, not
        # the old HC history expansion callback (which is overridden below).
        with super().lease(request=request, arm='HC' if arm == 'S' else arm, **kwargs) as token:
            self.current_arm = arm
            try:
                yield token
            finally:
                self.current_arm = None

    def _history(self, a, q, k, v, cache, cache_name, rows, update_cache):
        if self.current_arm == 'HC':
            return super()._history(a, q, k, v, cache, cache_name, rows, update_cache)
        if self.current_arm != 'S':
            raise LeaseError('shared-prefix candidate outside S lease')
        layer = self.layer_ids[id(a)]
        self._precheck(layer, a, q, rows, cache_name, update_cache)
        entry = self.entries.get(layer)
        if entry is None:
            # Keep usable version counters even in the real outer inference
            # context, so accidental edits to our private slot map fail closed.
            import torch
            from adapter import _tensor_key
            with torch.inference_mode(False), torch.no_grad():
                slots = self._valid(layer, cache)
            self._rows_check(layer, rows, q, 2)
            need = slots.numel() * slots.element_size()
            if self.payload_bytes + need > self.max_cache_bytes:
                raise LeaseError('shared-prefix metadata payload cap exceeded')
            entry = PrefixEntry(self._key(layer, a, q, cache), cache['k'], cache['v'], need, slots, _tensor_key(slots))
            self.entries[layer] = entry
            self.payload_bytes += need
            self.prefix_peak_bytes = max(self.prefix_peak_bytes, self.payload_bytes)
        if entry.k is not cache['k'] or entry.v is not cache['v'] or entry.k.shape[0] != 2:
            raise LeaseError('zero-copy two-row prefix ownership changed')
        from adapter import _tensor_key
        if _tensor_key(entry.slots)!=entry.slot_key:
            raise LeaseError('private valid-slot index changed')
        self._bump(layer, 'shared_attention')
        out=shared_attention(q, entry.k, entry.v, k, v, entry.slots, rows,
                             enabled=True, allow_cuda=self.allow_cuda,
                             cpu_reference_enabled=self.cpu_reference_enabled, maps_validated=True)
        if self.record_first and layer not in self.first_attention:
            self.first_attention[layer]=dict(q=q.detach().cpu().clone(),k=k.detach().cpu().clone(),
                                            v=v.detach().cpu().clone(),out=out.detach().cpu().clone())
        return out

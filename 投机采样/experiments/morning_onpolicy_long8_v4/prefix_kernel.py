"""Unchanged compiled shared CUDA operator; new job's explicit GPU boundary."""
import os
from pathlib import Path
import sys
import hashlib
from functools import lru_cache

NATIVE_SHA='b63eef1c2ca3201533d057da8b325471d234b7f5a5585f57f083d8351cabac48'

@lru_cache(maxsize=1)
def load_native():
    native=Path(__file__).resolve().parent.parent/'shared_prefix_torch_v2'
    matches=list(native.glob('_shared_torch_v1*.so'))
    if len(matches)!=1 or hashlib.sha256(matches[0].read_bytes()).hexdigest()!=NATIVE_SHA:
        raise ValueError('frozen original shared-KV binary changed')
    if str(native) not in sys.path:sys.path.append(str(native))
    import _shared_torch_v1
    if Path(_shared_torch_v1.__file__).resolve()!=matches[0].resolve():
        raise ImportError('native shared-KV module path collision')
    return _shared_torch_v1


def run(q, pk, pv, ck, cv, slots, rows):
    from experiment import ENABLED, APPROVED
    if os.environ.get(ENABLED) != '1' or os.environ.get(APPROVED) != '1':
        raise ValueError('O3 CUDA experiment is default off')
    return load_native().shared(q, pk, pv, ck, cv, slots, rows)

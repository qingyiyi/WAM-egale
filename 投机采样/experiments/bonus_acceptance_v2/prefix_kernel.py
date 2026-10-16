"""Unchanged compiled shared CUDA operator; new job's explicit GPU boundary."""
import os
from pathlib import Path
import sys


def run(q, pk, pv, ck, cv, slots, rows):
    from experiment import ENABLED, APPROVED
    if os.environ.get(ENABLED) != '1' or os.environ.get(APPROVED) != '1':
        raise ValueError('O3 CUDA experiment is default off')
    native = Path(__file__).resolve().parent.parent / 'shared_prefix_torch_v2'
    if str(native) not in sys.path:
        sys.path.append(str(native))
    import _shared_torch_v1
    return _shared_torch_v1.shared(q, pk, pv, ck, cv, slots, rows)

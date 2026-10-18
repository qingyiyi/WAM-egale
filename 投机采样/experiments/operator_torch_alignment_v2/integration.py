"""Instance-only inference integration; explicit install/restore, no global default."""
from contextlib import contextmanager
from pathlib import Path
import sys
import torch

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
import _wam_torch_aligned_v2 as native

EXPECTED_TORCH_COMMIT = "0fabc3ba44823f257e70ce397d989c8de5e362c1"

def sdpa(q, k, v, scale=None):
    return native.sdpa(q, k, v, scale)

def install(model):
    if torch.version.git_version != EXPECTED_TORCH_COMMIT:
        raise RuntimeError("fixed Torch2.9 baseline required")
    blocks = list(model.blocks)
    if len(blocks) != 30:
        raise RuntimeError("expected real WAM 30-block model")
    targets = [a for b in blocks for a in (b.attn1, b.attn2)]
    if any(not hasattr(a, "attn_op") for a in targets):
        raise RuntimeError("unsupported WAM attention API")
    previous = [(a, a.attn_op) for a in targets]
    for attention in targets:
        attention.attn_op = sdpa
    return previous

def restore(previous):
    for attention, operation in previous:
        attention.attn_op = operation

@contextmanager
def installed(model):
    previous = install(model)
    try:
        yield model
    finally:
        restore(previous)

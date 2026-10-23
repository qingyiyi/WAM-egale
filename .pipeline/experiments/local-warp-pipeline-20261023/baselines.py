"""External comparison adapters; original baseline source remains unchanged."""
from contextlib import contextmanager
import torch

@contextmanager
def grouped_cross(block, compact_context, rows):
    """Original Linear/RMS, one packed call for distinct conditions, B14 gather.

    Gather costs are included. This is an available grouped Torch control, not
    a claim that every possible shared-context attention backend is covered.
    """
    attn = block.attn2
    k, v, norm = attn.to_k.forward, attn.to_v.forward, attn.norm_k.forward
    attn.to_k.forward = lambda _: k(compact_context)
    attn.to_v.forward = lambda _: v(compact_context).index_select(0, rows)
    attn.norm_k.forward = lambda key: norm(key).index_select(0, rows)
    try:
        yield
    finally:
        del attn.to_k.forward
        del attn.to_v.forward
        del attn.norm_k.forward

@contextmanager
def cached_cross(block, full_kv):
    """Warm cached baseline: K projection + RMS, V projection already built.

    full_kv uses original B14 shape, expanded outside warm timing. Cache build
    and its larger allocation are reported separately from warm execution.
    """
    attn=block.attn2
    attn.to_k.forward=lambda _:full_kv[0]
    attn.norm_k.forward=lambda _:full_kv[1]
    attn.to_v.forward=lambda _:full_kv[2]
    try:yield
    finally:
        del attn.to_k.forward
        del attn.norm_k.forward
        del attn.to_v.forward

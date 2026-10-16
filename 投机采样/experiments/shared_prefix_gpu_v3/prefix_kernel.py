"""Triton virtual-prefix attention. No cache writes and no global monkeypatch.

One CTA per candidate/head. This v1 shares STORAGE; it does not claim that
seven CTAs reuse the same prefix tile in on-chip shared memory.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q, PK, PV, CK, CV, SLOTS, ROWS, OUT,
               qs0: tl.constexpr, qs1: tl.constexpr, qs2: tl.constexpr, qs3: tl.constexpr,
               ps0: tl.constexpr, ps1: tl.constexpr, ps2: tl.constexpr, ps3: tl.constexpr,
               vs0: tl.constexpr, vs1: tl.constexpr, vs2: tl.constexpr, vs3: tl.constexpr,
               ks0: tl.constexpr, ks1: tl.constexpr, ks2: tl.constexpr, ks3: tl.constexpr,
               cs0: tl.constexpr, cs1: tl.constexpr, cs2: tl.constexpr, cs3: tl.constexpr,
               os0: tl.constexpr, os1: tl.constexpr, os2: tl.constexpr, os3: tl.constexpr,
               L: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    cfg = tl.load(ROWS + b)
    m = tl.arange(0, BM)
    d = tl.arange(0, D)
    n = tl.arange(0, BN)
    q = tl.load(Q + b*qs0 + m[:, None]*qs1 + h*qs2 + d[None, :]*qs3,
                mask=m[:, None] < N, other=0)
    max_prev = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(tl.cdiv(L + N, BN)):
        j = start*BN + n
        is_prefix = j < L
        slot = tl.load(SLOTS + j, mask=is_prefix, other=0)
        # Pointer selection is virtual concat. Each row reads only its own
        # current KV and the history row selected by its immutable CFG map.
        kpre = PK + cfg*ps0 + slot[None, :]*ps1 + h*ps2 + d[:, None]*ps3
        kcur = CK + b*ks0 + (j[None, :]-L)*ks1 + h*ks2 + d[:, None]*ks3
        kval = tl.load(tl.where(is_prefix[None, :], kpre, kcur),
                      mask=j[None, :] < L+N, other=0)
        logits = tl.dot(q, kval).to(tl.float32) * SCALE
        logits = tl.where(j[None, :] < L+N, logits, -float('inf'))
        max_now = tl.maximum(max_prev, tl.max(logits, 1))
        rescale = tl.exp2(max_prev - max_now)
        prob = tl.exp2(logits - max_now[:, None])
        denom = denom*rescale + tl.sum(prob, 1)
        vpre = PV + cfg*vs0 + slot[:, None]*vs1 + h*vs2 + d[None, :]*vs3
        vcur = CV + b*cs0 + (j[:, None]-L)*cs1 + h*cs2 + d[None, :]*cs3
        vval = tl.load(tl.where(is_prefix[:, None], vpre, vcur),
                      mask=j[:, None] < L+N, other=0)
        acc = acc*rescale[:, None] + tl.dot(prob.to(vval.dtype), vval)
        max_prev = max_now
    out = (acc / denom[:, None]).to(OUT.dtype.element_ty)
    tl.store(OUT + b*os0 + m[:, None]*os1 + h*os2 + d[None, :]*os3,
             out, mask=m[:, None] < N)


def run(q, pk, pv, ck, cv, slots, rows):
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _attention[(q.shape[0], q.shape[2])](
        q, pk, pv, ck, cv, slots, rows, out,
        *q.stride(), *pk.stride(), *pv.stride(), *ck.stride(), *cv.stride(), *out.stride(),
        slots.numel(), q.shape[1], q.shape[-1], 16, 64,
        (q.shape[-1] ** -0.5) * 1.4426950408889634,
        num_warps=4, num_stages=2)
    return out

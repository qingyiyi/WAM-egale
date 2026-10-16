"""Task-private virtual [prefix2, current14] attention, default disabled.

CPU is an explicit correctness reference, NEVER a CUDA fallback. The CUDA
kernel reads the existing two-row pool directly via valid-slot and row maps.
No historical batch14 tensor or concatenated fullKV14 is materialized.
Prefix and this candidate's current tokens have ONE joint softmax.
"""
import math
import torch


def validate(q, pk, pv, k, v, slots, rows):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('current Q/K/V must have matching BSHD shapes')
    if pk.ndim != 4 or pk.shape != pv.shape or pk.shape[0] != 2 or pk.shape[2:] != q.shape[2:]:
        raise ValueError('prefix must retain the two CFG rows')
    if q.shape[0] != 14 or q.shape[1] < 1:
        raise ValueError('frozen fourteen candidate rows and nonempty current')
    if any(x.dtype != q.dtype or x.device != q.device for x in (pk, pv, k, v)):
        raise ValueError('Q/K/V dtype/device mismatch')
    if (slots.dtype != torch.long or slots.ndim != 1 or rows.dtype != torch.long
            or rows.shape != (14,) or slots.device != q.device or rows.device != q.device):
        raise ValueError('slot/row map dtype/device/shape')
    if slots.stride()!=(1,) or rows.stride()!=(1,):
        raise ValueError('slot/row map must be contiguous')
    if any(x.requires_grad for x in (q, pk, pv, k, v)):
        raise ValueError('inference only, autograd forbidden')
    if q.device.type == 'cpu':
        if tuple(rows.tolist()) != (0,)*7 + (1,)*7:
            raise ValueError('canonical two CFG groups required')
        if (slots < 0).any() or (slots >= pk.shape[1]).any() or (slots[1:] <= slots[:-1]).any():
            raise ValueError('valid slots must be sorted unique in-range')


def cpu_reference(q, pk, pv, k, v, slots, rows):
    """Independent float64 joint-softmax reference; no history batch14 copy.

Loops over candidates, not for speed. Concatenation of ONE row is permitted
here as an oracle, not on the production CUDA shared-prefix path.
"""
    validate(q, pk, pv, k, v, slots, rows)
    if q.device.type != 'cpu':
        raise ValueError('CPU reference is CPU only')
    result = torch.empty_like(q)
    for b in range(14):
        kk = torch.cat((pk[int(rows[b]), slots], k[b]), dim=0).double().transpose(0, 1)
        vv = torch.cat((pv[int(rows[b]), slots], v[b]), dim=0).double().transpose(0, 1)
        qq = q[b].double().transpose(0, 1)
        p = torch.softmax((qq @ kk.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
        result[b] = (p @ vv).transpose(0, 1).to(q.dtype)
    return result


def shared_attention(q, pk, pv, k, v, slots, rows, *, enabled=False,
                     allow_cuda=False, cpu_reference_enabled=False, maps_validated=False):
    if not enabled:
        raise ValueError('shared-prefix attention is default disabled')
    validate(q, pk, pv, k, v, slots, rows)
    if q.device.type == 'cpu':
        if not cpu_reference_enabled:
            raise ValueError('explicit CPU reference required')
        return cpu_reference(q, pk, pv, k, v, slots, rows)
    if not allow_cuda or q.device.type != 'cuda':
        raise ValueError('CUDA shared-prefix attention not authorized')
    if q.dtype not in (torch.bfloat16, torch.float16) or q.shape[1:] != (16, 24, 128):
        raise ValueError('frozen CUDA N16/H24/D128 BF16/FP16 geometry required')
    if not maps_validated:
        # Direct callers do not have the adapter's owned canonical rows and
        # source-mask-derived slots. Fail BEFORE a possible out-of-range read.
        canonical=torch.tensor((0,)*7+(1,)*7,dtype=torch.long,device=q.device)
        if (not torch.equal(rows,canonical) or not bool(((slots>=0)&(slots<pk.shape[1])).all())
                or not bool((slots[1:]>slots[:-1]).all())):
            raise ValueError('unvalidated GPU row/slot map')
    # Lazy import: importing this wrapper on CPU does not initialize CUDA or
    # compile kernels. No SDPA fallback that could hide fourteen-row copies.
    from prefix_kernel import run
    return run(q, pk, pv, k, v, slots, rows)

"""Logical finite and byte comparisons; no padding/uninitialized-storage checks."""
import torch


def byte_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                            right.detach().contiguous().reshape(-1).view(torch.uint8)))


def compare(reference, candidate):
    same_shape = reference.shape == candidate.shape
    same_dtype = reference.dtype == candidate.dtype
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    row = dict(shape_match=same_shape, dtype_match=same_dtype, finite_pass=finite,
               shape=list(reference.shape), dtype=str(reference.dtype),
               bitwise_pass=False, tolerance_pass=False)
    if not same_shape or not same_dtype:
        return row
    a = reference.detach().contiguous().reshape(-1)
    b = candidate.detach().contiguous().reshape(-1)
    different = a.view(torch.uint8).reshape(-1, a.element_size()) != b.view(torch.uint8).reshape(-1, b.element_size())
    row["different_bytes"] = int(different.sum())
    row["different_elements"] = int(different.any(dim=1).sum())
    row["elements"] = a.numel()
    row["bitwise_pass"] = finite and row["different_bytes"] == 0
    # Tolerance is deliberately exact for CPU R0/B0; it never relaxes a GPU gate.
    row["tolerance_atol"] = row["tolerance_rtol"] = 0.0
    row["tolerance_pass"] = finite and bool(torch.equal(a, b))
    if row["bitwise_pass"]:
        # Exact finite bytes imply identically zero errors even for tensors
        # exceeding torch.quantile's 2**24-element implementation limit.
        row.update(max_abs=0.0, rel_l2=0.0, abs_p50_p90_max=[0.0, 0.0, 0.0],
                   quantile_status="EXACT_ALL_ZERO")
        return row
    if finite:
        delta = (b.to(torch.complex128) - a.to(torch.complex128)).abs() if a.is_complex() else (b.double() - a.double()).abs()
        norm = a.abs().double().square().sum().sqrt()
        row["max_abs"] = float(delta.max()) if delta.numel() else 0.0
        row["rel_l2"] = float(delta.square().sum().sqrt() / norm.clamp_min(1e-300))
        if delta.numel() > 2**24:
            row["abs_p50_p90_max"] = None
            row["quantile_status"] = "UNKNOWN_TORCH_QUANTILE_ELEMENT_LIMIT_NO_SAMPLING"
        else:
            row["abs_p50_p90_max"] = torch.quantile(delta, torch.tensor([0.5, 0.9, 1.], dtype=torch.float64)).tolist() if delta.numel() else [0., 0., 0.]
            row["quantile_status"] = "EXACT_TORCH_TYPE7"
    return row


def state_equal(reference, candidate):
    keys = ("slots", "id", "mask", "is_pred", "k", "v")
    comparisons = {key: byte_equal(reference[key], candidate[key]) for key in keys}
    finite = all(bool(torch.isfinite(state[key]).all())
                 for state in (reference, candidate) for key in ("k", "v"))
    return dict(pass_all=all(comparisons.values()) and finite, fields=comparisons,
                valid_KV_finite_pass=finite,
                semantics="all metadata and all logically valid KV; invalid KV excluded")

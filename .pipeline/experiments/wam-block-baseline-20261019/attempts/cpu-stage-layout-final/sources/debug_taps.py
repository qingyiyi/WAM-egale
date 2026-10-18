"""Observational hooks on the original reference; never feed gold into B0."""
from contextlib import contextmanager
import math

import torch

from .fixtures import tensor_description
from .metrics import compare


@contextmanager
def original_taps(block, sink):
    """Expose matching visible boundaries without rewriting R0 arithmetic.

    Private K immediately after RoPE has no original module hook; it is not
    fabricated from another implementation. The first-divergence report lists
    only actually observed common boundaries.
    """
    handles, originals = [], []
    def post(name, transform=lambda value: value):
        def hook(module, args, output):
            sink(name, transform(output))
            return None
        return hook
    def before(name, transform=lambda value: value):
        def hook(module, args):
            sink(name, transform(args[0]))
            return None
        return hook
    for name, module in (("self.norm_fp32", block.norm1), ("ffn.norm_fp32", block.norm3),
                         ("ffn.expansion", block.ffn.net[0].proj),
                         ("ffn.activation", block.ffn.net[0]),
                         ("ffn.contraction", block.ffn.net[2]), ("block.output", block)):
        handles.append(module.register_forward_hook(post(name)))
    handles.append(block.attn1.register_forward_pre_hook(before("self.modulated_input")))
    handles.append(block.attn2.register_forward_pre_hook(before("cross.normalized_input")))
    handles.append(block.ffn.register_forward_pre_hook(before("ffn.modulated_input")))
    # Original pre-norm receives .float(); reverse cast recovers the preceding
    # BF16 residual exactly for the registered BF16 block input contract.
    handles.append(block.norm2.register_forward_pre_hook(before("self.residual", lambda x: x.to(torch.bfloat16))))
    handles.append(block.norm3.register_forward_pre_hook(before("cross.residual", lambda x: x.to(torch.bfloat16))))
    for prefix, attention in (("self", block.attn1), ("cross", block.attn2)):
        for name in ("q", "k", "v"):
            handles.append(getattr(attention, "to_" + name).register_forward_hook(post(prefix + "." + name + "_projection")))
        for name in ("q", "k"):
            handles.append(getattr(attention, "norm_" + name).register_forward_hook(
                post(prefix + "." + name + "_norm", lambda x, heads=attention.heads: x.unflatten(2, (heads, -1)))))
        handles.append(attention.to_out[0].register_forward_hook(post(prefix + ".out_projection")))
        handles.append(attention.register_forward_hook(post(prefix + ".output")))
        operation = attention.attn_op
        originals.append((attention, operation))
        def observed(q, k, v, *, _operation=operation, _prefix=prefix):
            if _prefix == "self":
                sink("self.q_rope", q)
            result = _operation(q, k, v)
            sink(_prefix + ".sdpa", result)
            return result
        attention.attn_op = observed
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for attention, operation in originals:
            attention.attn_op = operation


def first_observed_divergence(reference, candidate):
    common = [name for name in reference if name in candidate]
    rows = {name: compare(reference[name], candidate[name]) for name in common}
    return dict(first_observed_boundary=next((name for name in common if not rows[name]["bitwise_pass"]), None),
                comparisons=rows, reference_only=[name for name in reference if name not in candidate],
                candidate_only=[name for name in candidate if name not in reference],
                claim="first observed common boundary only; not proof of an unobserved internal root cause")


@contextmanager
def linear_shape_ledger(block, rows):
    """Record real module input/output layouts and mathematical M/N/K.

    This is CPU-visible module geometry, not a claim about the eventual CUDA
    GEMM dispatch/kernel/traffic. No .contiguous() is added to the hot path.
    """
    handles = []
    for name, module in block.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        def capture(linear, args, output, *, _name=name):
            value = args[0]
            try:
                flattened = value.view(-1, value.shape[-1])
                view = dict(possible=True, stride=list(flattened.stride()),
                            same_storage=flattened.untyped_storage().data_ptr() == value.untyped_storage().data_ptr())
            except RuntimeError:
                view = dict(possible=False, required_copy="UNKNOWN_NOT_INSERTED")
            rows.append(dict(module=_name, input=tensor_description(value), output=tensor_description(output),
                             weight=tensor_description(linear.weight), bias=linear.bias is not None,
                             logical_M=math.prod(value.shape[:-1]), N=linear.out_features, K=linear.in_features,
                             flatten_view=view, actual_CUDA_GEMM="UNKNOWN_CPU_ONLY",
                             weights_shared_across_rows=True))
        handles.append(module.register_forward_hook(capture))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

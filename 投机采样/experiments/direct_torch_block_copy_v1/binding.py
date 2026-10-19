"""Reversible in-place replacement; parameters, buffers and cache stay put."""
from contextlib import contextmanager, ExitStack

from .block import WanAttention, WanTransformerBlock, custom_sdpa


@contextmanager
def copied_block(block):
    """Install the copied classes outside the timed forward, then restore.

    Torch-only eager modules are supported. This replaces the complete block
    and attention methods, rather than forwarding to the old block.forward.
    Existing norm/linear/FFN modules retain their original Torch implementations.
    No tensor is copied and no arithmetic wrapper is added to forward.
    """
    if block.attn_mode != "torch":
        raise ValueError("The direct baseline requires attn_mode='torch'")
    modules = (block, block.attn1, block.attn2)
    if any("forward" in module.__dict__ for module in modules):
        raise ValueError("Remove experimental instance forward overrides first")
    if any(attn.cache_mode != "original" for attn in modules[1:]):
        raise ValueError("The direct baseline requires the original cache protocol")
    if any(getattr(attn.attn_op, "__name__", None) != "custom_sdpa"
           for attn in modules[1:]):
        raise ValueError("The direct baseline requires the original Torch SDPA path")
    originals = tuple(type(module) for module in modules)
    attention_ops = tuple(attn.attn_op for attn in modules[1:])
    try:
        for module, replacement in zip(modules, (WanTransformerBlock, WanAttention, WanAttention)):
            module.__class__ = replacement
        for attention in modules[1:]:
            attention.attn_op = custom_sdpa
        yield block
    finally:
        for module, original in zip(modules, originals):
            module.__class__ = original
        for attention, operation in zip(modules[1:], attention_ops):
            attention.attn_op = operation


@contextmanager
def copied_model_blocks(model):
    """Model integration entry; validation scope is recorded separately."""
    with ExitStack() as stack:
        for block in model.blocks:
            stack.enter_context(copied_block(block))
        yield model

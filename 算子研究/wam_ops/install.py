"""只替换传入模型实例；不修改服务源码、参数、缓存或全局默认。"""
from contextlib import contextmanager, ExitStack


@contextmanager
def install_block(block, implementation="baseline", *, execution_mode=None, fused_ffn1=None):
    if implementation == "baseline":
        from .baseline.block import TransformerBlock as BaselineBlock
        from .baseline.attention import Attention as BaselineAttention
        from .optimized.attention import cuda_sdpa
        block_type, attention_type, operation = BaselineBlock, BaselineAttention, cuda_sdpa
    elif implementation == "optimized":
        from .optimized.block import TransformerBlock, Attention
        from .optimized.ops import cuda_sdpa
        block_type, attention_type, operation = TransformerBlock, Attention, cuda_sdpa
    else:
        raise ValueError("implementation必须是baseline或optimized")
    if block.attn_mode != "torch":
        raise ValueError("需要原Torch模型作为匹配基线")
    modules = (block, block.attn1, block.attn2)
    if any("forward" in module.__dict__ for module in modules):
        raise ValueError("原模型存在实验forward覆盖，不能混合安装")
    if any(attention.cache_mode != "original" for attention in modules[1:]):
        raise ValueError("仅支持original KV协议")
    classes = tuple(type(module) for module in modules)
    operations = tuple(attention.attn_op for attention in modules[1:])
    previous_native = block.__dict__.get('_native_block')
    try:
        for module, kind in zip(modules, (block_type, attention_type, attention_type)):
            module.__class__ = kind
        for attention in modules[1:]:
            attention.attn_op = operation
        if implementation == "baseline":
            from .baseline.native import NativeBlock
            if execution_mode is not None or fused_ffn1 is not None:
                raise ValueError("baseline is frozen; optimization flags are invalid")
            native = NativeBlock(block, fused_ffn1=block.native_fused_ffn1)
        else:
            from .optimized.native import NativeBlock
            native = NativeBlock(block,
                fused_ffn1=block.native_fused_ffn1 if fused_ffn1 is None else fused_ffn1,
                mode=execution_mode or block.execution_mode)
        object.__setattr__(block, '_native_block', native)
        yield block
    finally:
        for module, kind in zip(modules, classes):
            module.__class__ = kind
        for attention, operation in zip(modules[1:], operations):
            attention.attn_op = operation
        if previous_native is None:
            block.__dict__.pop('_native_block', None)
        else:
            object.__setattr__(block, '_native_block', previous_native)


@contextmanager
def install_model(model, implementation="baseline", **options):
    with ExitStack() as stack:
        for block in model.blocks:
            stack.enter_context(install_block(block, implementation, **options))
        yield model

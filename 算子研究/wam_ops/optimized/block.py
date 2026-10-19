"""完整native block；Q16启用保持BF16舍入的FFN1＋GELU原语。"""
from ..baseline.block import TransformerBlock as NativeBaseline
from ..baseline.attention import Attention


class TransformerBlock(NativeBaseline):
    native_fused_ffn1 = True

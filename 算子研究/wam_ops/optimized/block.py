"""当前优化候选：原block数学顺序＋已对齐的独立CUDA SDPA。"""
from ..baseline.block import TransformerBlock as TorchBlock
from ..baseline.attention import Attention as TorchAttention
from .attention import cuda_sdpa


class Attention(TorchAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_op = cuda_sdpa


class TransformerBlock(TorchBlock):
    attention_cls = Attention

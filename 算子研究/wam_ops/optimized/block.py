# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from torch import nn
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from .parameters import Attention
from .native import NativeBlock


class TransformerBlock(nn.Module):
    """参数/state_dict保持原结构；完整forward由独立C++/CUDA执行。

    nn.Linear/Norm子模块仅保存兼容参数，forward不调用其Torch数学算子。
    """
    attention_cls = Attention
    native_fused_ffn1 = False
    execution_mode = "sequential"

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "torch",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = self.attention_cls(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = self.attention_cls(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
        candidate_cache_rows=None,
    ) -> torch.Tensor:
        native = self.__dict__.get('_native_block')
        if native is None:
            native = NativeBlock(self, fused_ffn1=self.native_fused_ffn1, mode=self.execution_mode)
            object.__setattr__(self, '_native_block', native)
        return native(hidden_states, encoder_hidden_states, temb, rotary_emb,
                      update_cache, cache_name, candidate_cache_rows)

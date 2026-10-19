# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from torch import nn
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from .attention import Attention


class TransformerBlock(nn.Module):
    """完整block：self-attention、cross-attention、FFN及原舍入边界。"""
    attention_cls = Attention

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
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1. + scale_msa) +
                              shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 update_cache=update_cache,
                                 cache_name=cache_name,
                                 candidate_cache_rows=candidate_cache_rows)
        hidden_states = (hidden_states.float() +
                         attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(
            hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) *
                              (1. + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() +
                         ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states

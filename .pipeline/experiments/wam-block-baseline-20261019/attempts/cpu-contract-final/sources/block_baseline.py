# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Derived from WanAttention/WanTransformerBlock at the frozen REVIEW_COMMIT.
# Apache-2.0; original license copied in LICENSE.upstream.txt.
# Modifications: explicit stage boundaries, optional taps, strict supported-mode
# checks, and original-parameter sharing for an isolated sequential baseline.
"""B0_HYBRID_SEQUENTIAL: explicit stages, original Torch primitives, no fusion.

This is host-side execution. Neither Linear nor SDPA calls here are device-side
functions that a persistent CUDA kernel can call. Intermediate values produced
by B0 are passed to its subsequent stages, without reference-output injection.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
from torch import nn
from einops import rearrange

Tap = Callable[[str, torch.Tensor], None] | None


def _emit(tap: Tap, name: str, value: torch.Tensor):
    if tap is not None:
        tap(name, value)
    return value


def rotary_original(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Keep the original FP64 complex multiplication and final dtype boundary."""
    dtype = x.dtype
    complex_x = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:3], -1, 2))
    return torch.view_as_real(complex_x * freqs).flatten(3).to(dtype)


def attention_sequential(attention, q, k, v, rotary_emb, *, update_cache=0,
                         cache_name="pos", candidate_cache_rows=None,
                         tap: Tap = None, prefix="self"):
    """Original attention stages, including original cache allocation semantics.

    Supports the saved fixtures' original mask/index cache and private
    read-only candidate mapping. Other cache engines are not silently emulated.
    """
    if attention.cache_mode != "original":
        raise NotImplementedError("B0 v1 supports only the original mask/index cache")
    kv_cache = None
    if attention.attn_caches is not None and cache_name in attention.attn_caches:
        kv_cache = attention.attn_caches[cache_name]

    query = _emit(tap, prefix + ".q_projection", attention.to_q(q))
    key = _emit(tap, prefix + ".k_projection", attention.to_k(k))
    value = _emit(tap, prefix + ".v_projection", attention.to_v(v))
    query = _emit(tap, prefix + ".q_norm",
                  attention.norm_q(query).unflatten(2, (attention.heads, -1)))
    key = _emit(tap, prefix + ".k_norm",
                attention.norm_k(key).unflatten(2, (attention.heads, -1)))
    value = value.unflatten(2, (attention.heads, -1))
    if rotary_emb is not None:
        query = _emit(tap, prefix + ".q_rope", rotary_original(query, rotary_emb))
        key = _emit(tap, prefix + ".k_rope", rotary_original(key, rotary_emb))

    if candidate_cache_rows is not None:
        if update_cache != 0 or kv_cache is None:
            raise ValueError("private candidate KV requires an initialized read-only cache")
        mapping = candidate_cache_rows
        valid = kv_cache["mask"].nonzero(as_tuple=False).squeeze(-1)
        history_k, history_v = kv_cache["k"][:, valid], kv_cache["v"][:, valid]
        if (mapping.ndim != 1 or mapping.numel() != query.shape[0]
                or mapping.dtype != torch.long or mapping.device != query.device
                or not bool(((mapping >= 0) & (mapping < history_k.shape[0])).all())):
            raise ValueError("candidate cache mapping shape/dtype/device mismatch")
        # Identical to original B14: materialize selected history, then concat.
        # Candidates keep independent Q16 and do not see one another's suffix.
        selected_key = history_k.index_select(0, mapping)
        selected_value = history_v.index_select(0, mapping)
        attended = attention.attn_op(query, torch.cat((selected_key, key), dim=1),
                                    torch.cat((selected_value, value), dim=1))
    else:
        slots = None
        if kv_cache is not None and kv_cache["k"] is not None:
            slots = attention.update_cache(cache_name, key, value, is_pred=(update_cache == 1))
            valid = kv_cache["mask"].nonzero(as_tuple=False).squeeze(-1)
            key, value = kv_cache["k"][:, valid], kv_cache["v"][:, valid]
        attended = attention.attn_op(query, key, value)
        if update_cache == 0 and kv_cache is not None and kv_cache["k"] is not None:
            attention.restore_cache(cache_name, slots)
    attended = _emit(tap, prefix + ".sdpa", attended)
    attended = attended.flatten(2, 3).type_as(query)
    projected = _emit(tap, prefix + ".out_projection", attention.to_out[0](attended))
    del attended
    return _emit(tap, prefix + ".output", attention.to_out[1](projected))


class SequentialBlock(nn.Module):
    """Read-only original parameters; explicit unfused stage organization."""
    implementation = "B0_HYBRID_SEQUENTIAL"

    def __init__(self, original_block):
        super().__init__()
        self.original = original_block
        if original_block.attn_mode != "torch":
            raise NotImplementedError("B0 v1 requires the registered torch SDPA API")
        net = original_block.ffn.net
        if (len(net) != 3 or type(net[0]).__name__ != "GELU"
                or net[0].approximate != "tanh" or not isinstance(net[1], nn.Dropout)
                or not isinstance(net[2], nn.Linear)):
            raise TypeError("Unsupported FFN: expected original GELU(tanh)/dropout/linear")

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                update_cache=0, cache_name="pos", candidate_cache_rows=None,
                *, tap: Tap = None):
        block = self.original
        table = block.scale_shift_table[None] + temb.float()
        values = rearrange(table, "b l n c -> b n l c").chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            value.squeeze(1) for value in values)
        norm_hidden = block.norm1(hidden_states.float())
        _emit(tap, "self.norm_fp32", norm_hidden)
        norm_hidden = _emit(tap, "self.modulated_input", (
            norm_hidden * (1. + scale_msa) + shift_msa).type_as(hidden_states))
        self_output = attention_sequential(
            block.attn1, norm_hidden, norm_hidden, norm_hidden, rotary_emb,
            update_cache=update_cache, cache_name=cache_name,
            candidate_cache_rows=candidate_cache_rows, tap=tap, prefix="self")
        hidden_states = _emit(tap, "self.residual", (
            hidden_states.float() + self_output * gate_msa).type_as(hidden_states))
        del self_output

        norm_hidden = _emit(tap, "cross.normalized_input",
                            block.norm2(hidden_states.float()).type_as(hidden_states))
        cross_output = attention_sequential(
            block.attn2, norm_hidden, encoder_hidden_states, encoder_hidden_states,
            None, update_cache=0, cache_name=cache_name, tap=tap, prefix="cross")
        # Original cross residual is BF16 addition; no extra FP32 promotion.
        hidden_states = _emit(tap, "cross.residual", hidden_states + cross_output)
        del cross_output

        norm_hidden = block.norm3(hidden_states.float())
        _emit(tap, "ffn.norm_fp32", norm_hidden)
        norm_hidden = _emit(tap, "ffn.modulated_input", (
            norm_hidden * (1. + c_scale_msa) + c_shift_msa).type_as(hidden_states))
        expanded = _emit(tap, "ffn.expansion", block.ffn.net[0].proj(norm_hidden))
        activated = _emit(tap, "ffn.activation", block.ffn.net[0].gelu(expanded))
        del expanded
        dropped = block.ffn.net[1](activated)
        del activated
        ff_output = _emit(tap, "ffn.contraction", block.ffn.net[2](dropped))
        del dropped
        return _emit(tap, "block.output", (
            hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states))


@contextmanager
def aligned_attention(block, *, gpu_authorized: bool = False):
    """R1: reuse the existing v2 operation, without faking a 30-block model.

    Its model-level install() contract remains unchanged. This independent
    block fixture adapter sets/restores the same two operation slots only.
    Native import is delayed until explicit GPU authorization was checked.
    """
    if not gpu_authorized:
        raise PermissionError("No new GPU authorization: native attention is NOT_RUN")
    from experiments.operator_torch_alignment_v2.integration import sdpa, EXPECTED_TORCH_COMMIT
    if torch.version.git_version != EXPECTED_TORCH_COMMIT:
        raise RuntimeError("fixed Torch2.9 commit required by the existing v2 integration")
    old = [(attention, attention.attn_op) for attention in (block.attn1, block.attn2)]
    for attention, _ in old:
        attention.attn_op = sdpa
    try:
        yield block
    finally:
        for attention, operation in old:
            attention.attn_op = operation

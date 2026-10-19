# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Verbatim Torch-branch block/attention copy from fixed ca5f37e.

The three scientific definitions below are copied unchanged, including cache
helpers. Only import scaffolding is narrowed to the Torch execution branch.
Instantiate with attn_mode="torch". No CUDA extension or server import is needed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange


class FlexAttnFunc(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("This direct copy supports attn_mode='torch' only")


def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("This direct copy supports attn_mode='torch' only")


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)

class WanAttention(torch.nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.attn_caches = {} if cross_attention_dim_head is None else None
        # Opt-in C/D experiments. Original cache remains the default, including
        # training/server users. Configured once by the existing offline entry.
        self.cache_mode = "original"
        self.cache_validate = False
        self.cache_validation = None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        if self.cache_mode == "lifecycle":
            if cache['temporary_end'] != cache['committed_end']:
                raise RuntimeError("Cannot clear prediction KV while temporary KV is visible")
            cache['committed_end'] = cache['observed_end']
            cache['temporary_end'] = cache['observed_end']
            cache['stats']['prediction_clears'] += 1
            if self.cache_validate:
                ref_cache = self.attn_caches[cache_name + "__C_reference"]
                ref_cache['mask'][ref_cache['is_pred']] = False
            return
        if self.cache_mode != "original":
            raise NotImplementedError(
                "C cache supports offline append/rollback only, not observation replacement")
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None
        self.attn_caches.pop(cache_name + "__C_reference", None)

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is None:
            return
        if self.cache_mode == "original":
            self.attn_caches[cache_name] = self._original_cache(
                total_tolen, num_head, head_dim, device, dtype, batch_size)
            return
        if self.cache_mode not in ("contiguous", "paged", "lifecycle"):
            raise ValueError(f"Unknown cache mode: {self.cache_mode}")
        capacity = total_tolen
        if self.cache_mode == "paged":
            from flash_attn import flash_attn_with_kvcache
            self._paged_attention = flash_attn_with_kvcache
            page_size = 256
            pages = math.ceil(capacity / page_size)
            shape = (batch_size * pages, page_size, num_head, head_dim)
        else:
            shape = (batch_size, capacity, num_head, head_dim)
        cache = dict(k=torch.empty(shape, device=device, dtype=dtype),
                     v=torch.empty(shape, device=device, dtype=dtype),
                     capacity=capacity)
        if self.cache_mode == "lifecycle":
            cache.update(
                observed_end=0,
                committed_end=0,
                temporary_end=0,
                stats=dict(temporary_updates=0,
                           prediction_commits=0,
                           observation_commits=0,
                           prediction_clears=0,
                           temporary_tokens=0,
                           prediction_tokens=0,
                           observation_tokens=0))
        else:
            cache['length'] = 0
        if self.cache_mode == "paged":
            # A real page table, preallocated for this single logical request
            # (CFG has two sequences). No serving allocator or lifetime policy.
            cache['block_table'] = torch.arange(
                batch_size * pages, device=device, dtype=torch.int32).reshape(batch_size, pages)
            cache['seqlens'] = torch.zeros(batch_size, device=device, dtype=torch.int32)
            cache['batch_size'] = batch_size
        self.attn_caches[cache_name] = cache
        if self.cache_validate:
            self.attn_caches[cache_name + "__C_reference"] = self._original_cache(
                total_tolen, num_head, head_dim, device, dtype, batch_size)
            self.cache_validation = dict(
                # max_abs, error_sq, reference_sq, not_close, nonfinite,
                # cache_mismatches, not_bitwise
                metrics=torch.zeros(7, device=device, dtype=torch.float64),
                calls=0, elements=0, cache_checks=0)

    @staticmethod
    def _original_cache(total_tolen, num_head, head_dim, device, dtype, batch_size):
        return {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'id':
            torch.full((total_tolen, ), -1, device=device),
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def _c_attention(self, cache_name, query, key, value, update_cache):
        """Ordinary append + rollback; deliberately no eviction/three-state KV."""
        if update_cache not in (0, 1):
            raise NotImplementedError("C cache does not implement observation replacement")
        cache = self.attn_caches[cache_name]
        start = cache['length']
        end = start + key.shape[1]
        if end > cache['capacity']:
            raise NotImplementedError(
                f"C cache capacity {cache['capacity']} exceeded ({end}); eviction is not implemented")
        # Temporary denoising KV is still written on every call in BOTH C modes.
        # On rollback the storage stays dirty but the logical length is restored.
        if self.cache_mode == "contiguous":
            cache['k'][:, start:end].copy_(key)
            cache['v'][:, start:end].copy_(value)
            hidden_states = self.attn_op(query, cache['k'][:, :end], cache['v'][:, :end])
        else:
            cache['seqlens'].fill_(start)
            # Q/K already have the model's RoPE. Noncausal attention over all
            # visible history + this chunk; do NOT apply LLM causal masking.
            hidden_states = self._paged_attention(
                query, cache['k'], cache['v'], k=key, v=value,
                cache_seqlens=cache['seqlens'], block_table=cache['block_table'],
                causal=False)
        cache['length'] = end
        if self.cache_validate:
            self._validate_c_attention(
                cache_name, query, key, value, hidden_states, update_cache,
                visible_length=end)
        if update_cache == 0:
            cache['length'] = start
        return hidden_states

    def _lifecycle_attention(self, cache_name, query, key, value, update_cache):
        """Three-state contiguous KV: observed, committed prediction, temporary."""
        if update_cache not in (0, 1, 2):
            raise ValueError(f"Unknown lifecycle cache update mode: {update_cache}")
        cache = self.attn_caches[cache_name]
        start = cache['committed_end']
        if cache['temporary_end'] != start:
            raise RuntimeError("Lifecycle KV boundary is inconsistent before append")
        if update_cache == 2 and start != cache['observed_end']:
            raise RuntimeError(
                "Observation KV requires clear_pred_cache before replacement")

        key_size = key.shape[1]
        end = start + key_size
        if end > cache['capacity']:
            raise NotImplementedError(
                f"Lifecycle cache capacity {cache['capacity']} exceeded ({end}); "
                "eviction is not implemented in D0")

        cache['k'][:, start:end].copy_(key)
        cache['v'][:, start:end].copy_(value)
        cache['temporary_end'] = end
        hidden_states = self.attn_op(
            query, cache['k'][:, :end], cache['v'][:, :end])
        if self.cache_validate:
            self._validate_c_attention(
                cache_name, query, key, value, hidden_states, update_cache,
                visible_length=end)

        if update_cache == 0:
            cache['stats']['temporary_updates'] += 1
            cache['stats']['temporary_tokens'] += key_size
            cache['temporary_end'] = start
        elif update_cache == 1:
            cache['stats']['prediction_commits'] += 1
            cache['stats']['prediction_tokens'] += key_size
            cache['committed_end'] = end
        else:
            cache['stats']['observation_commits'] += 1
            cache['stats']['observation_tokens'] += key_size
            cache['observed_end'] = end
            cache['committed_end'] = end
        return hidden_states

    def _validate_c_attention(self, cache_name, query, key, value, output,
                              update_cache, visible_length):
        # Separate diagnostic run only. Independent ORIGINAL mask/index path on
        # real Q/K/V at selected blocks; no synthetic inputs or shadow model.
        ref_name = cache_name + "__C_reference"
        slots = self.update_cache(ref_name, key, value, is_pred=(update_cache == 1))
        ref_cache = self.attn_caches[ref_name]
        valid = ref_cache['mask'].nonzero(as_tuple=False).squeeze(-1)
        ref_k, ref_v = ref_cache['k'][:, valid], ref_cache['v'][:, valid]
        reference = custom_sdpa(query, ref_k, ref_v)
        error = output.float() - reference.float()
        stats = self.cache_validation
        metrics = stats['metrics']
        metrics[0] = torch.maximum(metrics[0], error.abs().max())
        metrics[1] += error.square().sum(dtype=torch.float64)
        metrics[2] += reference.float().square().sum(dtype=torch.float64)
        metrics[3] += (~torch.isclose(output, reference, atol=1e-2, rtol=1e-2)).sum()
        metrics[4] += (~torch.isfinite(output)).sum()
        metrics[6] += (output != reference).sum()
        stats['calls'] += 1
        stats['elements'] += output.numel()
        if update_cache in (1, 2):
            cache = self.attn_caches[cache_name]
            ck, cv = cache['k'], cache['v']
            if self.cache_mode == "paged":
                # Identity table maps each CFG sequence to disjoint pages.
                ck = ck.reshape(cache['batch_size'], -1, *ck.shape[-2:])
                cv = cv.reshape(cache['batch_size'], -1, *cv.shape[-2:])
            metrics[5] += (ck[:, :visible_length] != ref_k).sum()
            metrics[5] += (cv[:, :visible_length] != ref_v).sum()
            stats['cache_checks'] += 1
        if update_cache == 0:
            self.restore_cache(ref_name, slots)

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
        candidate_cache_rows=None,
    ):
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        if candidate_cache_rows is not None:
            # Speculative verification reads the committed history without ever
            # staging candidates in persistent KV slots. Rows map each candidate
            # to its conditional/unconditional history row (if CFG is active).
            if update_cache != 0 or kv_cache is None:
                raise ValueError("candidate verification requires existing read-only KV")
            if isinstance(self.attn_op, FlexAttnFunc):
                raise NotImplementedError(
                    "FlexAttnFunc currently assumes batch size 1; candidate batching is unsupported")
            if self.cache_mode == "original":
                valid = kv_cache['mask'].nonzero(as_tuple=False).squeeze(-1)
                history_key = kv_cache['k'][:, valid]
                history_value = kv_cache['v'][:, valid]
            elif self.cache_mode in ("contiguous", "lifecycle"):
                if (self.cache_mode == "lifecycle"
                        and kv_cache['temporary_end'] != kv_cache['committed_end']):
                    raise RuntimeError("candidate forward requires an idle lifecycle KV cache")
                end = (kv_cache['length'] if self.cache_mode == "contiguous"
                       else kv_cache['committed_end'])
                history_key = kv_cache['k'][:, :end]
                history_value = kv_cache['v'][:, :end]
            else:
                raise NotImplementedError("candidate read-only KV does not support paged cache")
            if (candidate_cache_rows.ndim != 1
                    or candidate_cache_rows.numel() != query.shape[0]
                    or candidate_cache_rows.dtype != torch.long
                    or candidate_cache_rows.device != query.device
                    or not bool(((candidate_cache_rows >= 0)
                                 & (candidate_cache_rows < history_key.shape[0])).all())):
                raise ValueError("invalid candidate_cache_rows for this KV cache")
            # F.scaled_dot_product_attention/flash attention sees independent
            # per-candidate histories; no token can attend across candidates.
            selected_key = history_key.index_select(0, candidate_cache_rows)
            selected_value = history_value.index_select(0, candidate_cache_rows)
            hidden_states = self.attn_op(
                query, torch.cat((selected_key, key), dim=1),
                torch.cat((selected_value, value), dim=1))
        elif kv_cache is not None and self.cache_mode == "lifecycle":
            hidden_states = self._lifecycle_attention(
                cache_name, query, key, value, update_cache)
        elif kv_cache is not None and self.cache_mode != "original":
            hidden_states = self._c_attention(
                cache_name, query, key, value, update_cache)
        else:
            slots = None
            if kv_cache is not None and kv_cache['k'] is not None:
                slots = self.update_cache(cache_name,
                                          key,
                                          value,
                                          is_pred=(update_cache == 1))
                key_pool = self.attn_caches[cache_name]['k']
                value_pool = self.attn_caches[cache_name]['v']
                mask = self.attn_caches[cache_name]['mask']
                valid = mask.nonzero(as_tuple=False).squeeze(-1)
                key = key_pool[:, valid]
                value = value_pool[:, valid]

            hidden_states = self.attn_op(query, key, value)

            if update_cache == 0:
                if kv_cache is not None and kv_cache['k'] is not None:
                    self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
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

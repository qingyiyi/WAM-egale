# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""原Torch attention：投影、RoPE、original KV协议与SDPA。"""
import torch
from torch.nn import functional as F


def torch_sdpa(q, k, v):
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return output.transpose(1, 2)


class Attention(torch.nn.Module):
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
        if attn_mode != "torch":
            raise ValueError("baseline只支持原Torch路径")
        self.attn_op = torch_sdpa

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
        self.cache_mode = "original"


    def clear_pred_cache(self, cache_name):
        if self.attn_caches is not None:
            cache = self.attn_caches[cache_name]
            cache["mask"][cache["is_pred"]] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is not None:
            self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is not None:
            self.attn_caches[cache_name] = self._original_cache(
                total_tolen, num_head, head_dim, device, dtype, batch_size)


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
        if self.cache_mode != "original":
            raise ValueError("仅复现original KV协议")
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
            valid = kv_cache['mask'].nonzero(as_tuple=False).squeeze(-1)
            history_key = kv_cache['k'][:, valid]
            history_value = kv_cache['v'][:, valid]
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

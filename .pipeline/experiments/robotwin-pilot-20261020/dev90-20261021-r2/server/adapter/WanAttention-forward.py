def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name='pos', candidate_cache_rows=None):
    kv_cache = self.attn_caches[cache_name] if self.attn_caches is not None and cache_name in self.attn_caches else None
    query, key, value = (self.to_q(q), self.to_k(k), self.to_v(v))
    query = self.norm_q(query)
    query = query.unflatten(2, (self.heads, -1))
    key = self.norm_k(key)
    key = key.unflatten(2, (self.heads, -1))
    value = value.unflatten(2, (self.heads, -1))
    if rotary_emb is not None:

        def apply_rotary_emb(x, freqs):
            x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
            x_out = torch.view_as_real(x_out * freqs).flatten(3)
            return x_out.to(x.dtype)
        query = apply_rotary_emb(query, rotary_emb)
        key = apply_rotary_emb(key, rotary_emb)
    if candidate_cache_rows is not None:
        hidden_states = self._robotwin_shared(query, key, value, kv_cache, candidate_cache_rows, update_cache)
    else:
        slots = None
        if kv_cache is not None and kv_cache['k'] is not None:
            slots = self.update_cache(cache_name, key, value, is_pred=update_cache == 1)
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

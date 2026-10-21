def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb, update_cache=0, cache_name='pos', candidate_cache_rows=None) -> torch.Tensor:
    temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
    shift_msa = shift_msa.squeeze(1)
    scale_msa = scale_msa.squeeze(1)
    gate_msa = gate_msa.squeeze(1)
    c_shift_msa = c_shift_msa.squeeze(1)
    c_scale_msa = c_scale_msa.squeeze(1)
    c_gate_msa = c_gate_msa.squeeze(1)
    norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = self.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states, rotary_emb, update_cache=update_cache, cache_name=cache_name, candidate_cache_rows=candidate_cache_rows)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, encoder_hidden_states, None, update_cache=0, cache_name=cache_name)
    hidden_states = hidden_states + attn_output
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
    return hidden_states

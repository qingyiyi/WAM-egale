def forward(self, input_dict, update_cache=0, cache_name='pos', action_mode=False, train_mode=False, candidate_cache_rows=None):
    """
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
    if train_mode:
        return self.forward_train(input_dict)
    if action_mode:
        latent_hidden_states = rearrange(input_dict['noisy_latents'], 'b c f h w -> b (f h w) c')
        latent_hidden_states = self.action_embedder(latent_hidden_states)
    else:
        latent_hidden_states = rearrange(input_dict['noisy_latents'], 'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)', p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])
        latent_hidden_states = self.patch_embedding_mlp(latent_hidden_states)
    text_hidden_states = self.condition_embedder.text_embedder(input_dict['text_emb'])
    latent_grid_id = input_dict['grid_id']
    rotary_emb = self.rope(latent_grid_id)[:, :, None]
    pach_scale_h, pach_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
    latent_time_steps = torch.repeat_interleave(input_dict['timesteps'], input_dict['noisy_latents'].shape[-2] // pach_scale_h * (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)
    current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
    temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=latent_hidden_states.dtype)
    timestep_proj = timestep_proj.unflatten(2, (6, -1))
    for block in self.blocks:
        latent_hidden_states = block(latent_hidden_states, text_hidden_states, timestep_proj, rotary_emb, update_cache=update_cache, cache_name=cache_name, candidate_cache_rows=candidate_cache_rows)
    temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
    shift, scale = rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(2, dim=1)
    shift = shift.to(latent_hidden_states.device).squeeze(1)
    scale = scale.to(latent_hidden_states.device).squeeze(1)
    latent_hidden_states = (self.norm_out(latent_hidden_states.float()) * (1.0 + scale) + shift).type_as(latent_hidden_states)
    if action_mode:
        latent_hidden_states = self.action_proj_out(latent_hidden_states)
    else:
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(latent_hidden_states, 'b l (n c) -> b (l n) c', n=math.prod(self.patch_size))
    return latent_hidden_states

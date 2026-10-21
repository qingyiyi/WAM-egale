def _infer(self, obs, frame_st_id=0):
    frame_chunk_size = self.job_config.frame_chunk_size
    if frame_st_id == 0:
        init_latent = self._encode_obs(obs)
        self.init_latent = init_latent
    latents = torch.randn(1, 48, frame_chunk_size, self.latent_height, self.latent_width, device=self.device, dtype=self.dtype)
    actions = torch.randn(1, self.job_config.action_dim, frame_chunk_size, self.action_per_frame, 1, device=self.device, dtype=self.dtype)
    video_inference_step = self.job_config.num_inference_steps
    action_inference_step = self.job_config.action_num_inference_steps
    video_step = self.job_config.video_exec_step
    self.scheduler.set_timesteps(video_inference_step)
    self.action_scheduler.set_timesteps(action_inference_step)
    timesteps = self.scheduler.timesteps
    action_timesteps = self.action_scheduler.timesteps
    timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)
    if video_step != -1:
        timesteps = timesteps[:video_step]
    action_timesteps = F.pad(action_timesteps, (0, 1), mode='constant', value=0)
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
            input_dict = self._prepare_latent_input(latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id)
            video_noise_pred = self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']), update_cache=1 if last_step else 0, cache_name=self.cache_name, action_mode=False)
            if not last_step or video_step != -1:
                video_noise_pred = data_seq_to_patch(self.job_config.patch_size, video_noise_pred, frame_chunk_size, self.latent_height, self.latent_width, batch_size=2 if self.use_cfg else 1)
                if self.job_config.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = self.scheduler.step(video_noise_pred, t, latents, return_dict=False)
            latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
        with self._dev_action_scope():
            if self._robotwin_spec:
                actions = self._infer_action_verifier_k2_b2(actions, action_timesteps, frame_st_id, action_inference_step, '', None, bonus_target_step=True)
            else:
                for i, t in enumerate(tqdm(action_timesteps)):
                    last_step = i == len(action_timesteps) - 1
                    action_cond = torch.zeros([1, self.job_config.action_dim, 1, self.action_per_frame, 1], device=self.device, dtype=self.dtype) if frame_st_id == 0 else None
                    input_dict = self._prepare_latent_input(None, actions, t, t, None, action_cond, frame_st_id=frame_st_id)
                    action_noise_pred = self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']), update_cache=1 if last_step else 0, cache_name=self.cache_name, action_mode=True)
                    if not last_step:
                        action_noise_pred = rearrange(action_noise_pred, 'b (f n) c -> b c f n 1', f=frame_chunk_size)
                        if self.job_config.action_guidance_scale > 1:
                            action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                        else:
                            action_noise_pred = action_noise_pred[:1]
                        actions = self.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
                    actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]
    actions[:, ~self.action_mask] *= 0
    save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
    save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))
    actions = self.postprocess_action(actions)
    torch.cuda.empty_cache()
    return (actions, latents)

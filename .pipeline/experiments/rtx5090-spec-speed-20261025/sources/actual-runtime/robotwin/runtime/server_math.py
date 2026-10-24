import torch
import numpy as np
from einops import rearrange
from mesh_helper import get_mesh_id
from bonus_helper import execute_verified_k2_bonus_step
def _prepare_action_candidate_input(self, candidate_states, candidate_timesteps, frame_st_id):
    """Build one model input row per speculative action state.

        Candidate tensors are in the exact action-latent layout used by
        ``_infer``: [K, C, F, action_per_frame, 1]. Timestep may be one scalar
        per candidate [K] or one value per candidate/action frame [K, F].
        """
    if not isinstance(candidate_states, torch.Tensor) or candidate_states.ndim != 5:
        raise ValueError('candidate_states must have shape [K,C,F,H,W]')
    count, channels, frames, height, width = candidate_states.shape
    if count < 1 or channels != self.job_config.action_dim:
        raise ValueError('candidate batch/action channel count is invalid')
    if frames != self.job_config.frame_chunk_size or height != self.action_per_frame or width != 1:
        raise ValueError('candidate action geometry must match configured [frame_chunk_size, action_per_frame, 1]')
    timesteps = torch.as_tensor(candidate_timesteps, dtype=torch.float32, device=self.device)
    if timesteps.ndim == 1 and timesteps.shape[0] == count:
        timesteps = timesteps[:, None].expand(count, frames).clone()
    elif timesteps.shape != (count, frames):
        raise ValueError('candidate_timesteps must have shape [K] or [K,F]')
    if not bool(torch.isfinite(timesteps).all()):
        raise ValueError('candidate_timesteps must be finite')
    noisy = candidate_states.to(device=self.device, dtype=self.dtype).clone()
    if frame_st_id == 0:
        noisy[:, :, 0:1] = 0
        timesteps[:, 0] = 0
    action_mask = self.action_mask.to(device=self.device)
    noisy[:, ~action_mask] = 0
    grid_id = get_mesh_id(frames, height, width, 1, 1, frame_st_id, action=True).to(self.device)
    grid_id = grid_id.unsqueeze(0).expand(count, -1, -1).clone()
    text_emb = self.prompt_embeds.to(device=self.device, dtype=self.dtype)
    text_emb = text_emb.expand(count, -1, -1).clone()
    return {'noisy_latents': noisy, 'text_emb': text_emb, 'grid_id': grid_id, 'timesteps': timesteps}

def _repeat_action_candidates_for_cfg(self, input_dict):
    """Pack candidates as [all conditional, all unconditional] for CFG.

        Returns the packed model input and a row map into the persistent
        target KV cache. The map is [0]*K + [1]*K for CFG, or [0]*K without
        CFG; candidate attention uses it read-only and never writes KV.
        """
    count = input_dict['noisy_latents'].shape[0]
    if count < 1:
        raise ValueError('candidate batch cannot be empty')
    if self.use_cfg:
        if self.negative_prompt_embeds is None:
            raise ValueError('CFG candidate forward requires negative prompt embeddings')
        conditional_text = input_dict['text_emb']
        if conditional_text.shape[0] != count:
            raise ValueError('candidate text embedding batch does not match states')
        negative_text = self.negative_prompt_embeds.to(device=self.device, dtype=self.dtype).expand(count, -1, -1)
        packed = {'noisy_latents': torch.cat([input_dict['noisy_latents'], input_dict['noisy_latents']], dim=0), 'text_emb': torch.cat([conditional_text, negative_text], dim=0), 'grid_id': torch.cat([input_dict['grid_id'], input_dict['grid_id']], dim=0), 'timesteps': torch.cat([input_dict['timesteps'], input_dict['timesteps']], dim=0)}
        cache_rows = torch.cat([torch.zeros(count, dtype=torch.long, device=self.device), torch.ones(count, dtype=torch.long, device=self.device)])
    else:
        packed = input_dict
        cache_rows = torch.zeros(count, dtype=torch.long, device=self.device)
    return (packed, cache_rows)

def _p0_action_channel_ids(self) -> tuple[int, ...]:
    """Return validated effective action channels for P0 compact latents."""
    channels = tuple((int(channel) for channel in self.job_config.used_action_channel_ids))
    action_dim = int(self.job_config.action_dim)
    if not channels:
        raise ValueError('P0 action compacting requires non-empty used_action_channel_ids')
    if len(set(channels)) != len(channels):
        raise ValueError('used_action_channel_ids must not contain duplicates')
    if any((channel < 0 or channel >= action_dim for channel in channels)):
        raise ValueError(f'used_action_channel_ids must lie in [0, {action_dim}), got {channels}')
    return channels

def _p0_action_compact(self, full_state: torch.Tensor) -> torch.Tensor:
    """Flatten effective channels from one *unbatched* action latent.

        The channel axis is axis 0 only for ``[C,F,H,W]``.  Reject a batched
        state instead of accidentally indexing its batch axis, which would be
        an asynchronous CUDA index-select failure when more than one action
        channel is effective.
        """
    if not isinstance(full_state, torch.Tensor):
        raise TypeError('full_state must be a torch.Tensor')
    expected_shape = (int(self.job_config.action_dim), int(self.job_config.frame_chunk_size), int(self.action_per_frame), 1)
    if tuple(full_state.shape) != expected_shape:
        raise ValueError(f'_p0_action_compact expects unbatched action state {expected_shape}, got {tuple(full_state.shape)}')
    channels = self._p0_action_channel_ids()
    return full_state.index_select(0, torch.tensor(channels, device=full_state.device)).reshape(-1)

def _p0_action_scatter_compact(self, compact_state: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Scatter one flattened compact state into a single-row full latent."""
    if not isinstance(compact_state, torch.Tensor) or compact_state.ndim != 1:
        raise ValueError('compact_state must have shape [D]')
    if not isinstance(template, torch.Tensor):
        raise TypeError('template must be a torch.Tensor')
    expected_template_shape = (1, int(self.job_config.action_dim), int(self.job_config.frame_chunk_size), int(self.action_per_frame), 1)
    if tuple(template.shape) != expected_template_shape:
        raise ValueError(f'_p0_action_scatter_compact expects single-row action template {expected_template_shape}, got {tuple(template.shape)}')
    channels = self._p0_action_channel_ids()
    expected = len(channels) * self.action_per_frame * self.job_config.frame_chunk_size
    if compact_state.numel() != expected:
        raise ValueError(f'compact action state has {compact_state.numel()} elements, expected {expected}')
    full = torch.zeros_like(template)
    full[0].index_copy_(0, torch.tensor(channels, device=full.device), compact_state.reshape(len(channels), self.job_config.frame_chunk_size, self.action_per_frame, 1).to(device=full.device, dtype=full.dtype))
    return full

def _p0_action_scatter_compact_batch(self, compact_states: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Scatter [N,D] compact candidates to [N,C,F,H,W] target rows."""
    if not isinstance(compact_states, torch.Tensor) or compact_states.ndim != 2:
        raise ValueError('compact_states must have shape [N,D]')
    if compact_states.shape[0] < 1:
        raise ValueError('compact_states must contain at least one target row')
    rows = []
    for compact in compact_states:
        rows.append(self._p0_action_scatter_compact(compact, template))
    return torch.cat(rows, dim=0)

def _p0_action_project_state(self, full_state: torch.Tensor, action_cond: torch.Tensor | None, frame_st_id: int) -> torch.Tensor:
    """Apply exactly the normal action post-step input preparation."""
    result = full_state.clone()
    if frame_st_id == 0 and action_cond is not None:
        result[:, :, 0:1] = action_cond
    result[:, ~self.action_mask] *= 0
    return result

@torch.no_grad()
def _infer_action_verifier_k2_b2(self, actions: torch.Tensor, action_timesteps: torch.Tensor, frame_st_id: int, action_update_count: int, nsys_base: str, p0_trace_chunk, *, bonus_target_step: bool=False) -> torch.Tensor:
    """Run online K=2/B=2 with one 7-node target batch per tree.

        The legacy mode commits at most two draft updates.  The explicit
        ``bonus_target_step`` mode consumes the already-verified target
        direction at a fully accepted leaf as a third, real target scheduler
        update.  The draft acceptance depth remains 0/1/2; the bonus is
        recorded separately and only occurs after a full two-edge acceptance.
        """
    api = self._load_p0_shadow_api()
    predictor = self._load_p0_action_draft_b2_predictor()
    threshold = float(self._p0_action_draft_threshold)
    channels = tuple((int(channel) for channel in self.job_config.used_action_channel_ids))
    if not channels:
        raise ValueError('verifier_k2_b2 requires non-empty used_action_channel_ids')
    action_cond = torch.zeros([1, self.job_config.action_dim, 1, self.action_per_frame, 1], device=self.device, dtype=self.dtype) if frame_st_id == 0 else None
    actions = self._p0_action_project_state(actions, action_cond, frame_st_id)
    previous_direction = None
    step_index = 0
    scheduler_transitions = []

    def compact_to_full(compact: torch.Tensor) -> torch.Tensor:
        return self._p0_action_scatter_compact(compact, actions)

    def scheduler_step(state: torch.Tensor, direction: torch.Tensor, timestep: float, offset: int) -> torch.Tensor:
        full_state = compact_to_full(state)
        full_direction = compact_to_full(direction)
        timestep_tensor = torch.tensor(float(timestep), device=self.device, dtype=torch.float32)
        with self._phase_profile_phase('action_tree_scheduler'):
            raw_full_next = self.action_scheduler.step(full_direction, timestep_tensor, full_state, return_dict=False)
        self._phase_profile_counter('tree_scheduler_transition_count')
        projected_full_next = self._p0_action_project_state(raw_full_next, action_cond, frame_st_id)
        projected = self._p0_action_compact(projected_full_next[0])
        if p0_trace_chunk is not None:
            scheduler_transitions.append({'timestep': float(timestep), 'raw_full_next': raw_full_next.detach().clone(), 'projected_full_next': projected_full_next.detach().clone()})
        return projected

    def record_update(*, step: int, state: torch.Tensor, direction: torch.Tensor, source: str, transition: dict | None, bonus_target_step_executed: bool=False) -> None:
        self._phase_profile_counter('committed_scheduler_update_count')
        if source == 'draft':
            self._phase_profile_counter('committed_draft_scheduler_update_count')
        elif bonus_target_step_executed:
            self._phase_profile_counter('bonus_target_scheduler_update_count')
        else:
            self._phase_profile_counter('ordinary_target_scheduler_update_count')
        if p0_trace_chunk is None:
            return
        if transition is None:
            raise RuntimeError('missing explicit scheduler raw/projected pair for K=2/B=2 trace update')
        expected_timestep = float(action_timesteps[step].detach().cpu().item())
        if not np.isclose(float(transition['timestep']), expected_timestep, rtol=0.0, atol=1e-06):
            raise RuntimeError(f"explicit K=2/B=2 trace transition timestep does not match step {step}: {transition['timestep']} vs {expected_timestep}")
        state_full = compact_to_full(state)
        direction_full = compact_to_full(direction)
        sigma = self.action_scheduler.sigmas[step]
        next_sigma = self.action_scheduler.sigmas[step + 1] if step + 1 < len(self.action_scheduler.sigmas) else 0.0
        p0_trace_chunk.record_scheduler_update(branch='action', model_source=source, step_index=step, timestep=action_timesteps[step], sigma=sigma, next_sigma=next_sigma, cfg_scale=self.job_config.action_guidance_scale, cfg_active=self.job_config.action_guidance_scale > 1, cache_update=False, x_i=state_full, d_cond=direction_full if source == 'target' else None, d_uncond=None, d_cfg=direction_full, x_next_raw=transition['raw_full_next'], x_next=transition['projected_full_next'], projection=self._p0_trace_projection(frame_st_id), input_preparation=self._p0_trace_input_preparation('action', frame_st_id))
    while step_index < action_update_count:
        remaining = action_update_count - step_index
        if previous_direction is None or remaining < (3 if bonus_target_step else 2):
            state = self._p0_action_compact(actions[0])
            timestep = action_timesteps[step_index]
            direction_full_input = self._prepare_latent_input(None, actions, timestep, timestep, None, action_cond, frame_st_id=frame_st_id)
            with self._phase_profile_phase('action_target_forward'):
                raw = self.transformer(self._repeat_input_for_cfg(direction_full_input['action_res_lst']), update_cache=0, cache_name=self.cache_name, action_mode=True)
            self._phase_profile_counter('target_forward_count')
            output = rearrange(raw, 'b (f n) c -> b c f n 1', f=self.job_config.frame_chunk_size)
            if self.use_cfg:
                cond, uncond = (output[:1], output[1:2])
                direction_full = uncond + self.job_config.action_guidance_scale * (cond - uncond) if self.job_config.action_guidance_scale > 1 else cond
            else:
                direction_full = output[:1]
            direction = self._p0_action_compact(direction_full[0])
            transition_start = len(scheduler_transitions)
            next_state = scheduler_step(state, direction, float(timestep), 0)
            transition = scheduler_transitions[transition_start] if p0_trace_chunk is not None else None
            record_update(step=step_index, state=state, direction=direction, source='target', transition=transition)
            actions = compact_to_full(next_state)
            previous_direction = direction.detach().clone()
            step_index += 1
            continue
        sigmas = self.action_scheduler.sigmas
        sigma_path = [float(sigmas[step_index].detach().cpu().item()), float(sigmas[step_index + 1].detach().cpu().item()), float(sigmas[step_index + 2].detach().cpu().item()) if step_index + 2 < len(sigmas) else 0.0]
        target_times = [float(action_timesteps[step_index].detach().cpu().item()), float(action_timesteps[step_index + 1].detach().cpu().item()), float(action_timesteps[step_index + 2].detach().cpu().item())]
        root_full = self._p0_action_project_state(actions, action_cond, frame_st_id)
        root = self._p0_action_compact(root_full[0])
        previous = torch.zeros_like(root) if previous_direction is None else previous_direction

        def draft_predictor(state: torch.Tensor, prev: torch.Tensor, sigma: float, next_sigma: float, timestep: float) -> torch.Tensor:
            with self._phase_profile_phase('action_draft_model'):
                result = predictor.predict(state, prev, sigma, next_sigma, timestep)
            self._phase_profile_counter('draft_forward_count')
            return result
        tree_transition_start = len(scheduler_transitions)
        with self._phase_profile_phase('action_tree_build'):
            tree = api.build_k2_b2_tree(root_state=root, previous_direction=previous, sigma_path=sigma_path, target_timesteps=target_times, draft_predictor=draft_predictor, scheduler_step=scheduler_step)
        self._phase_profile_counter('tree_count')
        if p0_trace_chunk is not None:
            tree_transitions = scheduler_transitions[tree_transition_start:]
            if len(tree_transitions) != 6:
                raise RuntimeError(f'K=2/B=2 tree must produce exactly six scheduler transitions, got {len(tree_transitions)}')
        else:
            tree_transitions = ()
        with self._phase_profile_phase('action_candidate_pack_scatter'):
            compact_states, packed_timesteps = api.pack_k2_b2_verifier_batch(tree)
            candidate_states = self._p0_action_scatter_compact_batch(compact_states, root_full)
        self._phase_profile_counter('logical_candidate_rows', int(candidate_states.shape[0]))
        self._phase_profile_counter('verifier_target_row_count', int(candidate_states.shape[0]))
        target_full = self._target_action_candidate_directions(candidate_states, packed_timesteps, frame_st_id)
        target_compact = torch.stack(tuple((self._p0_action_compact(target_full[row]) for row in range(target_full.shape[0]))), dim=0)
        with self._phase_profile_phase('action_error_score'):
            scores = api.score_k2_b2_tree(tree, target_compact)
        with self._phase_profile_phase('action_selector'):
            route = api.select_k2_b2_route(scores, selector=self._p0_action_draft_selector)
        with self._phase_profile_phase('action_route_execute'):
            execution = api.execute_k2_b2_route(tree, scores, route, threshold=threshold, scheduler_step=scheduler_step)
        self._phase_profile_counter('accepted_depth_' + str(int(execution.accepted_depth)))
        if self._p0_state_teacher is not None:
            self._p0_state_teacher.record(tree=tree, target_directions=target_compact, query_states=torch.stack(tuple((self._p0_action_compact(candidate_states[row]) for row in range(3))), dim=0), context=self._p0_teacher_context, step_index=step_index, route=route, depth=execution.accepted_depth, contract={'used_action_channel_ids': list(channels), 'frame_st_id': int(frame_st_id), 'use_cfg': bool(self.use_cfg), 'cfg_scale': float(self.job_config.action_guidance_scale), 'cache_mode': self._cache_mode, 'projection': self._p0_trace_projection(frame_st_id), 'input_preparation': self._p0_trace_input_preparation('action', frame_st_id), 'draft_checkpoint': self._p0_action_draft_b2_checkpoint, 'selector': self._p0_action_draft_selector, 'threshold': threshold})
        bonus_next_state = None
        bonus_transition = None
        if bonus_target_step and execution.accepted_depth == 2:
            if execution.prefetch_target_direction is None:
                raise RuntimeError('fully accepted K=2/B=2 route is missing leaf target prefetch')
            self._phase_profile_counter('bonus_target_tree_count')
            bonus_next_state = execute_verified_k2_bonus_step(accepted_depth=execution.accepted_depth, leaf_state=execution.next_state, leaf_target_direction=execution.prefetch_target_direction, timestep=target_times[2], scheduler_step=scheduler_step)
            if bonus_next_state is None:
                raise RuntimeError('fully accepted K=2/B=2 bonus did not commit its target step')
            if p0_trace_chunk is not None:
                bonus_index = tree_transition_start + 6
                if len(scheduler_transitions) != bonus_index + 1:
                    raise RuntimeError(f'K=2/B=2 bonus must add exactly one scheduler transition, got {len(scheduler_transitions) - tree_transition_start - 6}')
                bonus_transition = scheduler_transitions[bonus_index]
        if p0_trace_chunk is not None:
            fallback_transition = None
            if execution.accepted_depth < 2:
                fallback_index = tree_transition_start + 6
                if len(scheduler_transitions) != fallback_index + 1:
                    raise RuntimeError(f'K=2/B=2 fallback must add exactly one scheduler transition, got {len(scheduler_transitions) - tree_transition_start - 6}')
                fallback_transition = scheduler_transitions[fallback_index]
            trace_transitions = api.select_k2_b2_trace_transitions(tree_transitions, route=route, accepted_depth=execution.accepted_depth, fallback_transition=fallback_transition)
            p0_trace_chunk.record_k2_b2_verifier_tree(step_index=step_index, selector=self._p0_action_draft_selector, threshold=threshold, selected_parent=route.parent, selected_child=route.child, accepted_depth=execution.accepted_depth, bonus_target_step_executed=bool(bonus_target_step and execution.accepted_depth == 2), bonus_target_step_index=step_index + 2 if bonus_target_step and execution.accepted_depth == 2 else None, first_costs=scores.first_costs, second_costs=scores.second_costs, target_timesteps=target_times, root_state=tree.root_state, first_directions=tree.first_directions, first_states=tree.first_states, second_directions=tree.second_directions, leaf_states=tree.leaf_states, target_directions=scores.target_directions, cfg_multiplier=2 if self.use_cfg else 1)
        if execution.accepted_depth == 0:
            root_target = scores.target_directions[0]
            next_state = execution.next_state
            record_update(step=step_index, state=root, direction=root_target, source='target', transition=trace_transitions[0] if p0_trace_chunk is not None else None)
            actions = compact_to_full(next_state)
            previous_direction = root_target.detach().clone()
            step_index += 1
        elif execution.accepted_depth == 1:
            first = tree.first_directions[route.parent]
            first_state = tree.first_states[route.parent]
            record_update(step=step_index, state=root, direction=first, source='draft', transition=trace_transitions[0] if p0_trace_chunk is not None else None)
            second_target = scores.target_directions[1 + route.parent]
            next_state = execution.next_state
            record_update(step=step_index + 1, state=first_state, direction=second_target, source='target', transition=trace_transitions[1] if p0_trace_chunk is not None else None)
            actions = compact_to_full(next_state)
            previous_direction = second_target.detach().clone()
            step_index += 2
        else:
            first = tree.first_directions[route.parent]
            second = tree.second_directions[route.parent, route.child]
            first_state = tree.first_states[route.parent]
            leaf_state = tree.leaf_states[route.parent, route.child]
            record_update(step=step_index, state=root, direction=first, source='draft', transition=trace_transitions[0] if p0_trace_chunk is not None else None)
            record_update(step=step_index + 1, state=first_state, direction=second, source='draft', transition=trace_transitions[1] if p0_trace_chunk is not None else None)
            if bonus_target_step:
                bonus_direction = execution.prefetch_target_direction
                if bonus_direction is None or bonus_next_state is None:
                    raise RuntimeError('K=2/B=2 bonus state/direction was not prepared')
                record_update(step=step_index + 2, state=leaf_state, direction=bonus_direction, source='target', transition=bonus_transition, bonus_target_step_executed=True)
                actions = compact_to_full(bonus_next_state)
                previous_direction = bonus_direction.detach().clone()
                step_index += 3
            else:
                actions = compact_to_full(leaf_state)
                previous_direction = second.detach().clone()
                step_index += 2
    terminal_timestep = action_timesteps[action_update_count]
    terminal_input = self._prepare_latent_input(None, actions, terminal_timestep, terminal_timestep, None, action_cond, frame_st_id=frame_st_id)
    with self._phase_profile_phase('action_terminal_target_forward'):
        raw_terminal = self.transformer(self._repeat_input_for_cfg(terminal_input['action_res_lst']), update_cache=1, cache_name=self.cache_name, action_mode=True)
    self._phase_profile_counter('terminal_target_forward_count')
    if p0_trace_chunk is not None:
        p0_trace_chunk.record_terminal_forward(branch='action', step_index=action_update_count, timestep=terminal_timestep, sigma=0.0, cfg_scale=self.job_config.action_guidance_scale, cfg_active=self.job_config.action_guidance_scale > 1, raw_model_output=raw_terminal, x_i=actions, cache_update=True, input_preparation=self._p0_trace_input_preparation('action', frame_st_id))
    return actions

@torch.no_grad()
def _target_action_candidate_directions(self, candidate_states, candidate_timesteps, frame_st_id):
    """Evaluate speculative candidates in one read-only target forward.

        This is an opt-in primitive only; the normal LIBERO inference path does
        not call it. Output is the CFG-combined target direction [K,C,F,H,W].
        """
    with self._phase_profile_phase('verifier_prepare_input'):
        candidate_input = self._prepare_action_candidate_input(candidate_states, candidate_timesteps, frame_st_id)
        packed_input, cache_rows = self._repeat_action_candidates_for_cfg(candidate_input)
    self._phase_profile_counter('verifier_cfg_rows', int(packed_input['noisy_latents'].shape[0]))
    with self._phase_profile_phase('verifier_target_forward'):
        output = self.transformer(packed_input, update_cache=0, cache_name=self.cache_name, action_mode=True, candidate_cache_rows=cache_rows)
    self._phase_profile_counter('verifier_forward_count')
    with self._phase_profile_phase('verifier_score_postprocess'):
        output = rearrange(output, 'b (f n) c -> b c f n 1', f=self.job_config.frame_chunk_size)
    count = candidate_states.shape[0]
    if output.shape[0] != count * (2 if self.use_cfg else 1):
        raise ValueError('target candidate forward returned unexpected batch size')
    if self.use_cfg:
        conditional, unconditional = (output[:count], output[count:])
        if self.job_config.action_guidance_scale > 1:
            output = unconditional + self.job_config.action_guidance_scale * (conditional - unconditional)
        else:
            output = conditional
    return output

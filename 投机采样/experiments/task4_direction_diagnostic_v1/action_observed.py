@torch.no_grad()
def _infer_action_verifier_k2_b2(self, actions: torch.Tensor, action_timesteps: torch.Tensor, frame_st_id: int, action_update_count: int, nsys_base: str, p0_trace_chunk, *, bonus_target_step: bool=False) -> torch.Tensor:
    """Run online K=2/B=2 with one 7-node target batch per tree.

        The legacy mode commits at most two draft updates.  The explicit
        ``bonus_target_step`` mode consumes the already-verified target
        direction at a fully accepted leaf as a third, real target scheduler
        update.  The draft acceptance depth remains 0/1/2; the bonus is
        recorded separately and only occurs after a full two-edge acceptance.
        """
    self._direction_diag.begin_action(actions, action_timesteps, frame_st_id, action_update_count)
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
        self._direction_diag.projected(projected, timestep, offset)
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
        self._direction_diag.commit(step, state, direction, source, bonus_target_step_executed)
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
        self._direction_diag.begin_round(step_index, remaining)
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
            self._direction_diag.end_round(step_index)
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
            self._direction_diag.tree(tree, step_index)
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
            self._direction_diag.score(tree, scores, step_index)
        with self._phase_profile_phase('action_selector'):
            route = api.select_k2_b2_route(scores, selector=self._p0_action_draft_selector)
        with self._phase_profile_phase('action_route_execute'):
            execution = api.execute_k2_b2_route(tree, scores, route, threshold=threshold, scheduler_step=scheduler_step)
            self._direction_diag.executed(tree, scores, route, execution, step_index)
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
        self._direction_diag.end_round(step_index)
    terminal_timestep = action_timesteps[action_update_count]
    terminal_input = self._prepare_latent_input(None, actions, terminal_timestep, terminal_timestep, None, action_cond, frame_st_id=frame_st_id)
    with self._phase_profile_phase('action_terminal_target_forward'):
        raw_terminal = self.transformer(self._repeat_input_for_cfg(terminal_input['action_res_lst']), update_cache=1, cache_name=self.cache_name, action_mode=True)
    self._phase_profile_counter('terminal_target_forward_count')
    if p0_trace_chunk is not None:
        p0_trace_chunk.record_terminal_forward(branch='action', step_index=action_update_count, timestep=terminal_timestep, sigma=0.0, cfg_scale=self.job_config.action_guidance_scale, cfg_active=self.job_config.action_guidance_scale > 1, raw_model_output=raw_terminal, x_i=actions, cache_update=True, input_preparation=self._p0_trace_input_preparation('action', frame_st_id))
    self._direction_diag.end_action(actions)
    return actions

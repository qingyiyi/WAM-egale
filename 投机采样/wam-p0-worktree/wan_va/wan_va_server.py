# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import importlib.util
from contextlib import contextmanager
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from configs.va_libero_cfg import validate_libero_checkpoint_environment
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)
from utils.p0_k3_b2 import (
    build_k3_b2_tree,
    execute_k3_b2_route,
    execute_verified_k2_bonus_step,
    pack_k3_b2_verifier_batch,
    score_k3_b2_tree,
    select_k3_b2_route,
    select_k3_b2_trace_transitions,
)
from utils.p0_trace import P0TraceRecorder
from utils.p0_state_teacher import StateTeacher


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        # Keep text-encoder offload independent of VAE placement. A GPU VAE
        # avoids repeating CPU video encoding on every observation-cache update.
        self.enable_offload = getattr(job_config, 'enable_offload', True)
        self.vae_device = self._resolve_vae_device(job_config, self.device)

        # Optional Nsight Systems instrumentation.  The default is deliberately
        # off: no profiler API, NVTX range, hook, or synchronization is added to
        # an ordinary inference run.  NSYS_PROFILE_MIDDLE_CHUNKS=1 is retained
        # as a compatibility alias for the original profiling boundary.
        self._nsys_profile_mode = self._resolve_nsys_profile_mode()
        self._nsys_profile_active = False
        self._nsys_detail_phase = None
        self._nsys_detail_hook_handles = []
        self._nsys_detail_open_ranges = {}
        self._nsys_open_range_depth = 0
        self._cache_mode = os.getenv("LINGBOT_CACHE_MODE", "original").strip().lower()
        self._cache_validate = os.getenv("LINGBOT_CACHE_VALIDATE", "0") == "1"
        self._chunk_timing = os.getenv("LINGBOT_CHUNK_TIMING", "0") == "1"
        if self._cache_mode not in ("original", "contiguous", "paged", "lifecycle"):
            raise ValueError(
                "LINGBOT_CACHE_MODE must be original, contiguous, paged, or lifecycle")
        if self._cache_mode != "original":
            if (job_config.infer_mode != "i2va" or job_config.world_size != 1
                    or os.getenv("LINGBOT_DISABLE_FSDP", "0") != "1"
                    or os.getenv("LINGBOT_ATTN_MODE", "torch").strip().lower() != "torch"):
                raise ValueError(
                    "C/D cache requires offline i2va, single GPU, no FSDP, "
                    "torch cross-attention")
        if self._cache_validate and (self._cache_mode == "original"
                or self._chunk_timing or self._nsys_profile_mode != "off"):
            raise ValueError("Cache validation is a separate C-only run, without timing/profiling")
        logger.info("Cache experiment mode=%s validate=%s chunk_timing=%s",
                    self._cache_mode, self._cache_validate, self._chunk_timing)

        # Opt-in, structured phase profiling for one or more real policy
        # requests.  This is deliberately separate from the trajectory trace:
        # it synchronizes the CUDA stream at phase boundaries and therefore is
        # a diagnostic latency run, not a clean throughput benchmark.
        self._phase_profile_enabled = os.getenv("WAM_P0_PHASE_PROFILE", "0") == "1"
        self._phase_profile_path = os.getenv("WAM_P0_PHASE_PROFILE_PATH", "").strip()
        self._phase_profile_request_index = 0
        self._active_phase_profile = None
        self._phase_profile_wall_start = None
        if self._phase_profile_enabled and not self._phase_profile_path:
            raise ValueError(
                "WAM_P0_PHASE_PROFILE=1 requires WAM_P0_PHASE_PROFILE_PATH")
        if self._phase_profile_enabled:
            Path(self._phase_profile_path).parent.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "Structured phase profiling enabled at %s; CUDA synchronization "
                "is intentionally added and this run is diagnostic only",
                self._phase_profile_path)

        # P0 is opt-in and lives only in the isolated speculative-sampling
        # worktree.  A normal run has ``None`` here: no trace buffers, CPU
        # copies, or trace files are created.  Enabled trace runs are
        # diagnostic runs and must not be used for latency comparisons.
        self._p0_trace = P0TraceRecorder.from_env()
        self._p0_state_teacher = StateTeacher.from_env()
        shadow_depth = int(os.getenv("WAM_P0_SHADOW_CANDIDATE_DEPTH", "0"))
        if shadow_depth < 0:
            raise ValueError("WAM_P0_SHADOW_CANDIDATE_DEPTH must be >= 0")
        self._p0_shadow_config = None
        self._p0_shadow_api = None
        self._p0_shadow_predictor = None
        self._p0_action_draft_mode = os.getenv(
            "WAM_P0_ACTION_DRAFT_MODE", "off").strip().lower()
        if self._p0_action_draft_mode not in {
                "off", "alternate_k1", "verifier_k2_b2",
                "verifier_k2_b2_bonus", "verifier_k3_b2"}:
            raise ValueError(
                "WAM_P0_ACTION_DRAFT_MODE must be off, alternate_k1, verifier_k2_b2, "
                "verifier_k2_b2_bonus, or verifier_k3_b2")
        self._p0_action_draft_checkpoint = os.getenv(
            "WAM_P0_ACTION_DRAFT_CHECKPOINT", "").strip()
        self._p0_action_draft_predictor = None
        self._p0_action_draft_b2_checkpoint = os.getenv(
            "WAM_P0_ACTION_DRAFT_B2_CHECKPOINT", "").strip()
        self._p0_action_draft_b2_predictor = None
        self._p0_action_draft_selector = os.getenv(
            "WAM_P0_ACTION_DRAFT_SELECTOR", "global").strip().lower()
        threshold_value = os.getenv("WAM_P0_ACTION_DRAFT_THRESHOLD", "").strip()
        self._p0_action_draft_threshold = None
        if threshold_value:
            try:
                self._p0_action_draft_threshold = float(threshold_value)
            except ValueError as exc:
                raise ValueError(
                    "WAM_P0_ACTION_DRAFT_THRESHOLD must be a finite non-negative float"
                ) from exc
            if (not np.isfinite(self._p0_action_draft_threshold)
                    or self._p0_action_draft_threshold < 0):
                raise ValueError(
                    "WAM_P0_ACTION_DRAFT_THRESHOLD must be a finite non-negative float")
        if self._p0_action_draft_mode == "alternate_k1":
            if not self._p0_action_draft_checkpoint:
                raise ValueError(
                    "alternate_k1 requires WAM_P0_ACTION_DRAFT_CHECKPOINT")
            if self._cache_mode == "paged":
                raise ValueError(
                    "alternate_k1 does not support paged KV cache")
            logger.warning(
                "P0 online action draft enabled: mode=%s checkpoint=%s; "
                "target updates are on 1-based odd steps and draft updates "
                "are on 1-based even steps",
                self._p0_action_draft_mode, self._p0_action_draft_checkpoint)
        if self._p0_action_draft_mode in {
                "verifier_k2_b2", "verifier_k2_b2_bonus", "verifier_k3_b2"}:
            if not self._p0_action_draft_b2_checkpoint:
                raise ValueError(
                    f"{self._p0_action_draft_mode} requires WAM_P0_ACTION_DRAFT_B2_CHECKPOINT")
            if self._p0_action_draft_selector not in {"greedy", "global"}:
                raise ValueError(
                    f"{self._p0_action_draft_mode} requires "
                    "WAM_P0_ACTION_DRAFT_SELECTOR=greedy or global")
            if self._p0_action_draft_threshold is None:
                raise ValueError(
                    f"{self._p0_action_draft_mode} requires an explicit calibrated "
                    "WAM_P0_ACTION_DRAFT_THRESHOLD")
            if self._cache_mode == "paged":
                raise ValueError(
                    f"{self._p0_action_draft_mode} does not support paged KV cache")
            logger.warning(
                "P0 action verifier enabled: mode=%s checkpoint=%s selector=%s threshold=%g; "
                "each complete tree evaluates seven target states read-only",
                self._p0_action_draft_mode,
                self._p0_action_draft_b2_checkpoint,
                self._p0_action_draft_selector,
                self._p0_action_draft_threshold)
        if shadow_depth:
            predictor = os.getenv(
                "WAM_P0_SHADOW_PREDICTOR", "zoh_direction").strip().lower()
            if predictor not in {
                "zoh_direction", "sigma_linear", "zoh_displacement",
                "autoregressive_mlp",
            }:
                raise ValueError(
                    "WAM_P0_SHADOW_PREDICTOR must be zoh_direction, sigma_linear, "
                    "zoh_displacement, or autoregressive_mlp")
            anchor_stride = int(os.getenv("WAM_P0_SHADOW_ANCHOR_STRIDE", "1"))
            if anchor_stride < 1:
                raise ValueError("WAM_P0_SHADOW_ANCHOR_STRIDE must be >= 1")
            if predictor == "autoregressive_mlp" and not os.getenv(
                    "WAM_P0_SHADOW_CHECKPOINT", "").strip():
                raise ValueError(
                    "autoregressive_mlp shadow capture requires WAM_P0_SHADOW_CHECKPOINT")
            if self._p0_trace is None:
                raise ValueError(
                    "shadow candidate capture requires WAM_P0_TRACE_DIR")
            if self._p0_trace.mode != "full":
                raise ValueError(
                    "shadow candidate capture requires WAM_P0_TRACE_MODE=full")
            if self._cache_mode == "paged":
                raise ValueError(
                    "shadow candidate capture does not support paged KV cache")
            self._p0_shadow_config = {
                "depth": shadow_depth,
                "predictor": predictor,
                "anchor_stride": anchor_stride,
                "checkpoint": os.getenv("WAM_P0_SHADOW_CHECKPOINT", "").strip() or None,
            }
            logger.warning(
                "P0 shadow candidate capture enabled: %s; target behavior remains unchanged; "
                "this is neither an acceptance decision nor a latency benchmark",
                self._p0_shadow_config)
        if self._p0_trace is not None:
            logger.warning(
                "P0 trajectory trace enabled at %s (mode=%s); this run is not a latency benchmark",
                self._p0_trace.run_dir, self._p0_trace.mode)

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device=self.vae_device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        attn_mode = os.getenv("LINGBOT_ATTN_MODE", "torch").strip().lower()
        if attn_mode not in ("torch", "flashattn"):
            raise ValueError(
                "LINGBOT_ATTN_MODE must be one of: torch, flashattn"
            )
        logger.info("Transformer attention backend: %s", attn_mode)
        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode=attn_mode
        )

        disable_fsdp = (
            os.getenv("LINGBOT_DISABLE_FSDP", "0") == "1"
            and job_config.world_size == 1
        )
        if disable_fsdp:
            logger.info("Single-GPU inference: FSDP disabled by environment")
            shard_fn = lambda model: model
        else:
            shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        for block_id, block in enumerate(self.transformer.blocks):
            block.attn1.cache_mode = self._cache_mode
            block.attn1.cache_validate = self._cache_validate and block_id in (
                0, len(self.transformer.blocks) // 2, len(self.transformer.blocks) - 1)

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device=self.vae_device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        logger.info(
            "Component devices: transformer=%s vae=%s text_encoder=%s; "
            "enable_offload=%s (WAM_VAE_DEVICE overrides only VAE)",
            next(self.transformer.parameters()).device,
            next(self.vae.parameters()).device,
            next(self.text_encoder.parameters()).device, self.enable_offload,
        )

    @staticmethod
    def _resolve_vae_device(job_config, model_device):
        """Place VAE independently; ``cuda`` follows this worker's model rank.

        WAM_VAE_DEVICE=cpu is an explicit rollback. ``auto`` preserves the
        original enable_offload behavior for configs without a VAE override.
        An invalid policy fails rather than silently changing the workload.
        """
        policy = os.getenv("WAM_VAE_DEVICE", "").strip().lower()
        if not policy:
            policy = str(getattr(job_config, "vae_device", "auto")).strip().lower()
        if policy == "auto":
            return torch.device("cpu") if getattr(
                job_config, "enable_offload", True) else model_device
        if policy == "cpu":
            return torch.device("cpu")
        if policy == "cuda":
            if model_device.type != "cuda":
                raise ValueError("GPU VAE requires a CUDA model device")
            return model_device
        raise ValueError("WAM_VAE_DEVICE / vae_device must be auto, cpu, or cuda")

    @staticmethod
    def _resolve_nsys_profile_mode():
        mode = os.getenv("LINGBOT_NSYS_PROFILE", "").strip().lower()
        if not mode and os.getenv("NSYS_PROFILE_MIDDLE_CHUNKS") == "1":
            mode = "basic"
        if mode in ("", "off"):
            return "off"
        if mode not in ("basic", "detail"):
            raise ValueError(
                "LINGBOT_NSYS_PROFILE must be one of: off, basic, detail"
            )
        return mode

    def _phase_profile_sync(self):
        if getattr(self, "_phase_profile_enabled", False):
            torch.cuda.synchronize(self.device)

    def _phase_profile_begin(self, *, frame_st_id: int, obs: dict):
        if not self._phase_profile_enabled:
            return
        self._phase_profile_sync()
        self._phase_profile_request_index += 1
        self._active_phase_profile = {
            "schema_version": 1,
            "request_index": int(self._phase_profile_request_index),
            "started_at_unix": time.time(),
            "frame_st_id": int(frame_st_id),
            "draft_mode": self._p0_action_draft_mode,
            "draft_selector": self._p0_action_draft_selector,
            "target_checkpoint": os.getenv("LINGBOT_VA_LIBERO_CHECKPOINT_ID", ""),
            "draft_checkpoint": (
                self._p0_action_draft_b2_checkpoint
                or self._p0_action_draft_checkpoint
                or ""
            ),
            "phases": {},
            "counters": {},
            "notes": {
                "input_kind": "real LIBERO websocket policy request",
                "cuda_sync_at_boundaries": True,
                "timing_clock": "time.perf_counter after/before torch.cuda.synchronize",
            },
        }
        self._phase_profile_wall_start = time.perf_counter()

    def _phase_profile_record(self, name: str, elapsed: float):
        profile = getattr(self, "_active_phase_profile", None)
        if profile is None:
            return
        row = profile["phases"].setdefault(name, {"calls": 0, "wall_sec": 0.0})
        row["calls"] += 1
        row["wall_sec"] += float(elapsed)

    def _phase_profile_counter(self, name: str, value=1):
        profile = getattr(self, "_active_phase_profile", None)
        if profile is None:
            return
        counters = profile["counters"]
        counters[name] = counters.get(name, 0) + value

    def _phase_profile_start(self, name: str):
        if getattr(self, "_active_phase_profile", None) is None:
            return None
        self._phase_profile_sync()
        return name, time.perf_counter()

    def _phase_profile_end(self, token):
        if token is None:
            return
        name, started = token
        self._phase_profile_sync()
        self._phase_profile_record(name, time.perf_counter() - started)

    @contextmanager
    def _phase_profile_phase(self, name: str):
        token = self._phase_profile_start(name)
        try:
            yield
        finally:
            self._phase_profile_end(token)

    def _phase_profile_finish(self, *, status: str = "completed", error=None):
        profile = getattr(self, "_active_phase_profile", None)
        if profile is None:
            return
        self._phase_profile_sync()
        total = time.perf_counter() - self._phase_profile_wall_start
        profile["status"] = status
        if error is not None:
            profile["error"] = repr(error)
        profile["policy_total_wall_sec"] = float(total)
        phases = profile["phases"]
        top_level = ("obs_encode", "video_total", "action_total", "postprocess")
        profile["top_level_phase_sum_sec"] = float(sum(
            phases.get(name, {}).get("wall_sec", 0.0) for name in top_level))
        profile["server_non_phase_wall_sec"] = float(
            total - profile["top_level_phase_sum_sec"]
        )
        profile["finished_at_unix"] = time.time()
        path = Path(self._phase_profile_path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(profile, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        logger.info("P0_PHASE_PROFILE %s", json.dumps(profile, ensure_ascii=False))
        self._active_phase_profile = None
        self._phase_profile_wall_start = None

    def _nsys_push(self, label):
        """Push one NVTX range only while the selected capture is active."""
        if not self._nsys_profile_active:
            return False
        torch.cuda.nvtx.range_push(label)
        self._nsys_open_range_depth += 1
        return True

    def _nsys_pop(self, pushed):
        if not pushed:
            return
        torch.cuda.nvtx.range_pop()
        self._nsys_open_range_depth = max(0, self._nsys_open_range_depth - 1)

    def _install_nsys_detail_hooks(self):
        if self._nsys_profile_mode != "detail" or self._nsys_detail_hook_handles:
            return

        for block_id, block in enumerate(self.transformer.blocks):
            for module_name, module in (
                ("self_attention", block.attn1),
                ("cross_attention", block.attn2),
                ("ffn", block.ffn),
            ):
                def pre_hook(_module, _inputs, block_id=block_id,
                             module_name=module_name):
                    if (not self._nsys_profile_active
                            or self._nsys_detail_phase is None):
                        return
                    label = (
                        f"LBVA/detail/{self._nsys_detail_phase}/"
                        f"block={block_id:02d}/{module_name}")
                    torch.cuda.nvtx.range_push(label)
                    self._nsys_open_range_depth += 1
                    module_id = id(_module)
                    self._nsys_detail_open_ranges[module_id] = (
                        self._nsys_detail_open_ranges.get(module_id, 0) + 1)

                def post_hook(_module, _inputs, _output):
                    module_id = id(_module)
                    open_count = self._nsys_detail_open_ranges.get(module_id,
                                                                    0)
                    if open_count:
                        torch.cuda.nvtx.range_pop()
                        self._nsys_open_range_depth = max(
                            0, self._nsys_open_range_depth - 1)
                        if open_count == 1:
                            del self._nsys_detail_open_ranges[module_id]
                        else:
                            self._nsys_detail_open_ranges[module_id] = (
                                open_count - 1)

                self._nsys_detail_hook_handles.append(
                    module.register_forward_pre_hook(pre_hook))
                self._nsys_detail_hook_handles.append(
                    module.register_forward_hook(post_hook, always_call=True))

        logger.info(
            "Nsight detail hooks installed: %d transformer submodule hooks",
            len(self._nsys_detail_hook_handles),
        )

    def _remove_nsys_detail_hooks(self):
        for handle in self._nsys_detail_hook_handles:
            handle.remove()
        self._nsys_detail_hook_handles.clear()

        # This is normally zero because every forward hook closes its range.
        # The fallback also keeps an exception path from leaving NVTX ranges
        # open in the report.
        while self._nsys_detail_open_ranges:
            torch.cuda.nvtx.range_pop()
            self._nsys_open_range_depth = max(0,
                                              self._nsys_open_range_depth - 1)
            module_id, open_count = next(
                iter(self._nsys_detail_open_ranges.items()))
            if open_count == 1:
                del self._nsys_detail_open_ranges[module_id]
            else:
                self._nsys_detail_open_ranges[module_id] = open_count - 1

    def _stop_nsys_capture(self):
        self._nsys_detail_phase = None
        self._remove_nsys_detail_hooks()
        self._nsys_profile_active = False
        while self._nsys_open_range_depth:
            torch.cuda.nvtx.range_pop()
            self._nsys_open_range_depth -= 1
        torch.cuda.profiler.stop()

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]

    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_action_candidate_input(self, candidate_states, candidate_timesteps,
                                        frame_st_id):
        """Build one model input row per speculative action state.

        Candidate tensors are in the exact action-latent layout used by
        ``_infer``: [K, C, F, action_per_frame, 1]. Timestep may be one scalar
        per candidate [K] or one value per candidate/action frame [K, F].
        """
        if not isinstance(candidate_states, torch.Tensor) or candidate_states.ndim != 5:
            raise ValueError("candidate_states must have shape [K,C,F,H,W]")
        count, channels, frames, height, width = candidate_states.shape
        if count < 1 or channels != self.job_config.action_dim:
            raise ValueError("candidate batch/action channel count is invalid")
        if (frames != self.job_config.frame_chunk_size
                or height != self.action_per_frame or width != 1):
            raise ValueError(
                "candidate action geometry must match configured [frame_chunk_size, "
                "action_per_frame, 1]"
            )

        timesteps = torch.as_tensor(
            candidate_timesteps, dtype=torch.float32, device=self.device)
        if timesteps.ndim == 1 and timesteps.shape[0] == count:
            timesteps = timesteps[:, None].expand(count, frames).clone()
        elif timesteps.shape != (count, frames):
            raise ValueError("candidate_timesteps must have shape [K] or [K,F]")
        if not bool(torch.isfinite(timesteps).all()):
            raise ValueError("candidate_timesteps must be finite")

        noisy = candidate_states.to(device=self.device, dtype=self.dtype).clone()
        if frame_st_id == 0:
            # Match _prepare_latent_input's fixed first-frame conditioning.
            noisy[:, :, 0:1] = 0
            timesteps[:, 0] = 0
        action_mask = self.action_mask.to(device=self.device)
        noisy[:, ~action_mask] = 0

        grid_id = get_mesh_id(
            frames, height, width, 1, 1, frame_st_id, action=True
        ).to(self.device)
        grid_id = grid_id.unsqueeze(0).expand(count, -1, -1).clone()
        text_emb = self.prompt_embeds.to(device=self.device, dtype=self.dtype)
        text_emb = text_emb.expand(count, -1, -1).clone()
        return {
            'noisy_latents': noisy,
            'text_emb': text_emb,
            'grid_id': grid_id,
            'timesteps': timesteps,
        }

    def _repeat_action_candidates_for_cfg(self, input_dict):
        """Pack candidates as [all conditional, all unconditional] for CFG.

        Returns the packed model input and a row map into the persistent
        target KV cache. The map is [0]*K + [1]*K for CFG, or [0]*K without
        CFG; candidate attention uses it read-only and never writes KV.
        """
        count = input_dict['noisy_latents'].shape[0]
        if count < 1:
            raise ValueError("candidate batch cannot be empty")
        if self.use_cfg:
            if self.negative_prompt_embeds is None:
                raise ValueError("CFG candidate forward requires negative prompt embeddings")
            conditional_text = input_dict['text_emb']
            if conditional_text.shape[0] != count:
                raise ValueError("candidate text embedding batch does not match states")
            negative_text = self.negative_prompt_embeds.to(
                device=self.device, dtype=self.dtype).expand(count, -1, -1)
            packed = {
                'noisy_latents': torch.cat(
                    [input_dict['noisy_latents'], input_dict['noisy_latents']], dim=0),
                'text_emb': torch.cat([conditional_text, negative_text], dim=0),
                'grid_id': torch.cat([input_dict['grid_id'], input_dict['grid_id']], dim=0),
                'timesteps': torch.cat([input_dict['timesteps'], input_dict['timesteps']], dim=0),
            }
            cache_rows = torch.cat([
                torch.zeros(count, dtype=torch.long, device=self.device),
                torch.ones(count, dtype=torch.long, device=self.device),
            ])
        else:
            packed = input_dict
            cache_rows = torch.zeros(count, dtype=torch.long, device=self.device)
        return packed, cache_rows

    def _load_p0_shadow_api(self):
        """Load the research draft builder lazily, only for explicit shadow runs."""
        if self._p0_shadow_api is not None:
            return self._p0_shadow_api
        default_script = (
            Path(__file__).resolve().parents[2] / "scripts" / "p0_batched_verify.py"
        )
        script_path = Path(
            os.getenv("WAM_P0_VERIFY_SCRIPT", str(default_script))
        ).expanduser().resolve()
        if not script_path.is_file():
            raise FileNotFoundError(
                f"P0 draft/verifier API not found: {script_path}; set WAM_P0_VERIFY_SCRIPT")
        module_name = "_wam_p0_batched_verify_shadow"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load P0 draft/verifier API from {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        self._p0_shadow_api = module
        return module

    def _load_p0_shadow_predictor(self):
        """Load the optional trained AR action draft only on first use."""
        predictor = getattr(self, "_p0_shadow_predictor", None)
        if predictor is not None:
            return predictor
        config = self._p0_shadow_config or {}
        checkpoint = config.get("checkpoint") or os.getenv(
            "WAM_P0_SHADOW_CHECKPOINT", "").strip()
        if not checkpoint:
            raise ValueError(
                "autoregressive_mlp shadow capture requires WAM_P0_SHADOW_CHECKPOINT")
        api = self._load_p0_shadow_api()
        predictor = api.AutoregressiveActionPredictor.from_checkpoint(
            checkpoint, map_location="cpu"
        )
        self._p0_shadow_predictor = predictor
        return predictor

    def _load_p0_action_draft_b2_predictor(self):
        """Load the checkpoint-aligned two-head K=2/B=2 draft lazily."""
        predictor = getattr(self, "_p0_action_draft_b2_predictor", None)
        if predictor is not None:
            return predictor
        if self._p0_action_draft_mode not in {
                "verifier_k2_b2", "verifier_k2_b2_bonus", "verifier_k3_b2"}:
            raise RuntimeError("B=2 verifier draft requested while mode is disabled")
        api = self._load_p0_shadow_api()
        predictor = api.TwoHeadResidualActionPredictor.from_checkpoint(
            self._p0_action_draft_b2_checkpoint, map_location="cpu"
        )
        channels = tuple(int(channel) for channel in self.job_config.used_action_channel_ids)
        expected_dim = len(channels) * self.action_per_frame * self.job_config.frame_chunk_size
        if predictor.model.output_dim != expected_dim:
            raise ValueError(
                "K=2/B=2 checkpoint output dimension does not match compact WAM action "
                f"geometry: checkpoint={predictor.model.output_dim}, expected={expected_dim}"
            )
        self._p0_action_draft_b2_predictor = predictor
        logger.info(
            "Loaded K=2/B=2 action draft checkpoint=%s input_dim=%d output_dim=%d "
            "hidden_dim=%d heads=2 channels=%s",
            predictor.checkpoint_path,
            predictor.model.trunk[0].in_features,
            predictor.model.output_dim,
            predictor.model.trunk[0].out_features,
            channels,
        )
        return predictor

    def _load_p0_action_draft_predictor(self):
        """Load the online K=1 AR action draft lazily on first draft update."""
        predictor = self._p0_action_draft_predictor
        if predictor is not None:
            return predictor
        if self._p0_action_draft_mode != "alternate_k1":
            raise RuntimeError("online action draft requested while mode is disabled")
        api = self._load_p0_shadow_api()
        predictor = api.AutoregressiveActionPredictor.from_checkpoint(
            self._p0_action_draft_checkpoint, map_location="cpu"
        )
        self._p0_action_draft_predictor = predictor
        return predictor

    @torch.no_grad()
    def _predict_p0_action_draft(
            self, actions, previous_direction, step_index, action_timesteps):
        """Predict one compact action direction for an even 1-based update.

        The target direction from the preceding odd update is fed back as the
        autoregressive previous direction.  The predictor only covers WAM's
        seven effective action channels; padded channels remain zero.
        """
        if previous_direction is None:
            raise RuntimeError("K=1 draft update has no preceding target direction")
        channels = tuple(int(channel) for channel in self.job_config.used_action_channel_ids)
        if not channels:
            raise ValueError("alternate_k1 requires non-empty used_action_channel_ids")
        predictor = self._load_p0_action_draft_predictor()
        compact_state = actions[0].index_select(
            0, torch.tensor(channels, device=actions.device))
        compact_previous = previous_direction[0].index_select(
            0, torch.tensor(channels, device=previous_direction.device))
        sigmas = self.action_scheduler.sigmas
        if step_index >= len(sigmas):
            raise ValueError(
                f"draft scheduler update index {step_index} outside "
                f"sigma path length {len(sigmas)}")
        # FlowMatchScheduler has exactly one sigma per scheduler update.  Its
        # final update integrates from sigma[-1] to the scheduler's implicit
        # terminal sigma 0; the t=0 cache-only transformer forward is a
        # separate loop item and has no sigma entry.
        next_sigma = (
            sigmas[step_index + 1].detach().cpu().item()
            if step_index + 1 < len(sigmas) else 0.0
        )
        predicted_compact = predictor.predict(
            compact_state, compact_previous,
            sigmas[step_index].detach().cpu().item(),
            next_sigma,
            action_timesteps[step_index].detach().cpu().item(),
        )
        predicted = torch.zeros_like(actions)
        predicted[0].index_copy_(
            0, torch.tensor(channels, device=actions.device),
            predicted_compact.to(device=actions.device, dtype=actions.dtype))
        return predicted

    def _p0_action_channel_ids(self) -> tuple[int, ...]:
        """Return validated effective action channels for P0 compact latents."""
        channels = tuple(int(channel) for channel in self.job_config.used_action_channel_ids)
        action_dim = int(self.job_config.action_dim)
        if not channels:
            raise ValueError("P0 action compacting requires non-empty used_action_channel_ids")
        if len(set(channels)) != len(channels):
            raise ValueError("used_action_channel_ids must not contain duplicates")
        if any(channel < 0 or channel >= action_dim for channel in channels):
            raise ValueError(
                "used_action_channel_ids must lie in "
                f"[0, {action_dim}), got {channels}"
            )
        return channels

    def _p0_action_compact(self, full_state: torch.Tensor) -> torch.Tensor:
        """Flatten effective channels from one *unbatched* action latent.

        The channel axis is axis 0 only for ``[C,F,H,W]``.  Reject a batched
        state instead of accidentally indexing its batch axis, which would be
        an asynchronous CUDA index-select failure when more than one action
        channel is effective.
        """
        if not isinstance(full_state, torch.Tensor):
            raise TypeError("full_state must be a torch.Tensor")
        expected_shape = (
            int(self.job_config.action_dim), int(self.job_config.frame_chunk_size),
            int(self.action_per_frame), 1,
        )
        if tuple(full_state.shape) != expected_shape:
            raise ValueError(
                "_p0_action_compact expects unbatched action state "
                f"{expected_shape}, got {tuple(full_state.shape)}"
            )
        channels = self._p0_action_channel_ids()
        return full_state.index_select(
            0, torch.tensor(channels, device=full_state.device)
        ).reshape(-1)

    def _p0_action_scatter_compact(
            self, compact_state: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        """Scatter one flattened compact state into a single-row full latent."""
        if not isinstance(compact_state, torch.Tensor) or compact_state.ndim != 1:
            raise ValueError("compact_state must have shape [D]")
        if not isinstance(template, torch.Tensor):
            raise TypeError("template must be a torch.Tensor")
        expected_template_shape = (
            1, int(self.job_config.action_dim), int(self.job_config.frame_chunk_size),
            int(self.action_per_frame), 1,
        )
        if tuple(template.shape) != expected_template_shape:
            raise ValueError(
                "_p0_action_scatter_compact expects single-row action template "
                f"{expected_template_shape}, got {tuple(template.shape)}"
            )
        channels = self._p0_action_channel_ids()
        expected = len(channels) * self.action_per_frame * self.job_config.frame_chunk_size
        if compact_state.numel() != expected:
            raise ValueError(
                f"compact action state has {compact_state.numel()} elements, expected {expected}"
            )
        full = torch.zeros_like(template)
        full[0].index_copy_(
            0,
            torch.tensor(channels, device=full.device),
            compact_state.reshape(len(channels), self.job_config.frame_chunk_size,
                                 self.action_per_frame, 1).to(
                                     device=full.device, dtype=full.dtype),
        )
        return full

    def _p0_action_scatter_compact_batch(
            self, compact_states: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        """Scatter [N,D] compact candidates to [N,C,F,H,W] target rows."""
        if not isinstance(compact_states, torch.Tensor) or compact_states.ndim != 2:
            raise ValueError("compact_states must have shape [N,D]")
        if compact_states.shape[0] < 1:
            raise ValueError("compact_states must contain at least one target row")
        rows = []
        for compact in compact_states:
            rows.append(self._p0_action_scatter_compact(compact, template))
        return torch.cat(rows, dim=0)

    def _p0_action_project_state(
            self, full_state: torch.Tensor, action_cond: torch.Tensor | None,
            frame_st_id: int) -> torch.Tensor:
        """Apply exactly the normal action post-step input preparation."""
        result = full_state.clone()
        if frame_st_id == 0 and action_cond is not None:
            result[:, :, 0:1] = action_cond
        result[:, ~self.action_mask] *= 0
        return result

    @torch.no_grad()
    def _infer_action_verifier_k2_b2(
            self, actions: torch.Tensor, action_timesteps: torch.Tensor,
            frame_st_id: int, action_update_count: int, nsys_base: str,
            p0_trace_chunk, *, bonus_target_step: bool = False) -> torch.Tensor:
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
        channels = tuple(int(channel) for channel in self.job_config.used_action_channel_ids)
        if not channels:
            raise ValueError("verifier_k2_b2 requires non-empty used_action_channel_ids")
        action_cond = torch.zeros(
            [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
            device=self.device, dtype=self.dtype
        ) if frame_st_id == 0 else None
        actions = self._p0_action_project_state(actions, action_cond, frame_st_id)
        previous_direction = None
        step_index = 0
        # Every real/draft expansion uses the authoritative scheduler.  Keep
        # the raw/projected pair only for full-trace audit records; it is never
        # used to choose a route or to advance the policy state.
        scheduler_transitions = []

        def compact_to_full(compact: torch.Tensor) -> torch.Tensor:
            return self._p0_action_scatter_compact(compact, actions)

        def scheduler_step(
                state: torch.Tensor, direction: torch.Tensor,
                timestep: float, offset: int) -> torch.Tensor:
            full_state = compact_to_full(state)
            full_direction = compact_to_full(direction)
            timestep_tensor = torch.tensor(
                float(timestep), device=self.device, dtype=torch.float32
            )
            with self._phase_profile_phase("action_tree_scheduler"):
                raw_full_next = self.action_scheduler.step(
                    full_direction, timestep_tensor, full_state, return_dict=False
                )
            self._phase_profile_counter("tree_scheduler_transition_count")
            projected_full_next = self._p0_action_project_state(
                raw_full_next, action_cond, frame_st_id
            )
            # ``_p0_action_compact`` intentionally accepts only an unbatched
            # ``[C,F,H,W]`` state.  The scheduler keeps batch=1, so remove that
            # dimension before selecting effective channels.
            projected = self._p0_action_compact(projected_full_next[0])
            if p0_trace_chunk is not None:
                scheduler_transitions.append({
                    "timestep": float(timestep),
                    "raw_full_next": raw_full_next.detach().clone(),
                    "projected_full_next": projected_full_next.detach().clone(),
                })
            return projected

        def record_update(
                *, step: int, state: torch.Tensor, direction: torch.Tensor,
                source: str, transition: dict | None,
                bonus_target_step_executed: bool = False) -> None:
            self._phase_profile_counter("committed_scheduler_update_count")
            if source == "draft":
                self._phase_profile_counter("committed_draft_scheduler_update_count")
            elif bonus_target_step_executed:
                self._phase_profile_counter("bonus_target_scheduler_update_count")
            else:
                self._phase_profile_counter("ordinary_target_scheduler_update_count")
            if p0_trace_chunk is None:
                return
            if transition is None:
                raise RuntimeError(
                    "missing explicit scheduler raw/projected pair for K=2/B=2 trace update"
                )
            expected_timestep = float(action_timesteps[step].detach().cpu().item())
            if not np.isclose(float(transition["timestep"]), expected_timestep,
                              rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    "explicit K=2/B=2 trace transition timestep does not match "
                    f"step {step}: {transition['timestep']} vs {expected_timestep}"
                )
            state_full = compact_to_full(state)
            direction_full = compact_to_full(direction)
            sigma = self.action_scheduler.sigmas[step]
            next_sigma = (
                self.action_scheduler.sigmas[step + 1]
                if step + 1 < len(self.action_scheduler.sigmas) else 0.0
            )
            p0_trace_chunk.record_scheduler_update(
                branch="action",
                model_source=source,
                step_index=step,
                timestep=action_timesteps[step],
                sigma=sigma,
                next_sigma=next_sigma,
                cfg_scale=self.job_config.action_guidance_scale,
                cfg_active=self.job_config.action_guidance_scale > 1,
                cache_update=False,
                x_i=state_full,
                d_cond=direction_full if source == "target" else None,
                d_uncond=None,
                d_cfg=direction_full,
                x_next_raw=transition["raw_full_next"],
                x_next=transition["projected_full_next"],
                projection=self._p0_trace_projection(frame_st_id),
                input_preparation=self._p0_trace_input_preparation("action", frame_st_id),
            )

        while step_index < action_update_count:
            remaining = action_update_count - step_index
            # A two-edge tree needs two actual scheduler transitions.  The
            # final one-edge tail is target-only; the terminal t=0 cache update
            # is handled after all scheduler updates, exactly as normal _infer.
            if previous_direction is None or remaining < (3 if bonus_target_step else 2):
                state = self._p0_action_compact(actions[0])
                timestep = action_timesteps[step_index]
                direction_full_input = self._prepare_latent_input(
                    None, actions, timestep, timestep, None, action_cond,
                    frame_st_id=frame_st_id
                )
                with self._phase_profile_phase("action_target_forward"):
                    raw = self.transformer(
                        self._repeat_input_for_cfg(direction_full_input["action_res_lst"]),
                        update_cache=0,
                        cache_name=self.cache_name,
                        action_mode=True,
                    )
                self._phase_profile_counter("target_forward_count")
                output = rearrange(
                    raw, "b (f n) c -> b c f n 1", f=self.job_config.frame_chunk_size
                )
                if self.use_cfg:
                    cond, uncond = output[:1], output[1:2]
                    direction_full = (
                        uncond + self.job_config.action_guidance_scale * (cond - uncond)
                        if self.job_config.action_guidance_scale > 1 else cond
                    )
                else:
                    direction_full = output[:1]
                direction = self._p0_action_compact(direction_full[0])
                transition_start = len(scheduler_transitions)
                next_state = scheduler_step(state, direction, float(timestep), 0)
                transition = (
                    scheduler_transitions[transition_start]
                    if p0_trace_chunk is not None else None
                )
                record_update(
                    step=step_index, state=state, direction=direction,
                    source="target", transition=transition
                )
                actions = compact_to_full(next_state)
                previous_direction = direction.detach().clone()
                step_index += 1
                continue

            sigmas = self.action_scheduler.sigmas
            sigma_path = [
                float(sigmas[step_index].detach().cpu().item()),
                float(sigmas[step_index + 1].detach().cpu().item()),
                float(sigmas[step_index + 2].detach().cpu().item())
                if step_index + 2 < len(sigmas) else 0.0,
            ]
            target_times = [
                float(action_timesteps[step_index].detach().cpu().item()),
                float(action_timesteps[step_index + 1].detach().cpu().item()),
                float(action_timesteps[step_index + 2].detach().cpu().item()),
            ]
            root_full = self._p0_action_project_state(actions, action_cond, frame_st_id)
            root = self._p0_action_compact(root_full[0])
            previous = (
                torch.zeros_like(root)
                if previous_direction is None else previous_direction
            )

            def draft_predictor(
                    state: torch.Tensor, prev: torch.Tensor,
                    sigma: float, next_sigma: float, timestep: float) -> torch.Tensor:
                with self._phase_profile_phase("action_draft_model"):
                    result = predictor.predict(state, prev, sigma, next_sigma, timestep)
                self._phase_profile_counter("draft_forward_count")
                return result

            tree_transition_start = len(scheduler_transitions)
            with self._phase_profile_phase("action_tree_build"):
                tree = api.build_k2_b2_tree(
                    root_state=root,
                    previous_direction=previous,
                    sigma_path=sigma_path,
                    target_timesteps=target_times,
                    draft_predictor=draft_predictor,
                    scheduler_step=scheduler_step,
                )
            self._phase_profile_counter("tree_count")
            if p0_trace_chunk is not None:
                tree_transitions = scheduler_transitions[tree_transition_start:]
                if len(tree_transitions) != 6:
                    raise RuntimeError(
                        "K=2/B=2 tree must produce exactly six scheduler transitions, "
                        f"got {len(tree_transitions)}"
                    )
            else:
                tree_transitions = ()
            with self._phase_profile_phase("action_candidate_pack_scatter"):
                compact_states, packed_timesteps = api.pack_k2_b2_verifier_batch(tree)
                candidate_states = self._p0_action_scatter_compact_batch(compact_states, root_full)
            self._phase_profile_counter("logical_candidate_rows", int(candidate_states.shape[0]))
            self._phase_profile_counter("verifier_target_row_count", int(candidate_states.shape[0]))
            target_full = self._target_action_candidate_directions(
                candidate_states, packed_timesteps, frame_st_id
            )
            target_compact = torch.stack(tuple(
                self._p0_action_compact(target_full[row])
                for row in range(target_full.shape[0])
            ), dim=0)
            with self._phase_profile_phase("action_error_score"):
                scores = api.score_k2_b2_tree(tree, target_compact)
            with self._phase_profile_phase("action_selector"):
                route = api.select_k2_b2_route(
                    scores, selector=self._p0_action_draft_selector
                )
            with self._phase_profile_phase("action_route_execute"):
                execution = api.execute_k2_b2_route(
                    tree, scores, route, threshold=threshold,
                    scheduler_step=scheduler_step,
                )
            self._phase_profile_counter("accepted_depth_" + str(int(execution.accepted_depth)))
            if self._p0_state_teacher is not None:
                self._p0_state_teacher.record(
                    tree=tree, target_directions=target_compact,
                    query_states=torch.stack(tuple(
                        self._p0_action_compact(candidate_states[row])
                        for row in range(3)), dim=0),
                    context=self._p0_teacher_context,
                    step_index=step_index, route=route, depth=execution.accepted_depth,
                    contract={"used_action_channel_ids": list(channels),
                              "frame_st_id": int(frame_st_id),
                              "use_cfg": bool(self.use_cfg),
                              "cfg_scale": float(self.job_config.action_guidance_scale),
                              "cache_mode": self._cache_mode,
                              "projection": self._p0_trace_projection(frame_st_id),
                              "input_preparation": self._p0_trace_input_preparation("action", frame_st_id),
                              "draft_checkpoint": self._p0_action_draft_b2_checkpoint,
                              "selector": self._p0_action_draft_selector,
                              "threshold": threshold})
            bonus_next_state = None
            bonus_transition = None
            if bonus_target_step and execution.accepted_depth == 2:
                if execution.prefetch_target_direction is None:
                    raise RuntimeError("fully accepted K=2/B=2 route is missing leaf target prefetch")
                self._phase_profile_counter("bonus_target_tree_count")
                bonus_next_state = execute_verified_k2_bonus_step(
                    accepted_depth=execution.accepted_depth,
                    leaf_state=execution.next_state,
                    leaf_target_direction=execution.prefetch_target_direction,
                    timestep=target_times[2],
                    scheduler_step=scheduler_step,
                )
                if bonus_next_state is None:
                    raise RuntimeError("fully accepted K=2/B=2 bonus did not commit its target step")
                if p0_trace_chunk is not None:
                    bonus_index = tree_transition_start + 6
                    if len(scheduler_transitions) != bonus_index + 1:
                        raise RuntimeError(
                            "K=2/B=2 bonus must add exactly one scheduler transition, "
                            f"got {len(scheduler_transitions) - tree_transition_start - 6}"
                        )
                    bonus_transition = scheduler_transitions[bonus_index]
            if p0_trace_chunk is not None:
                fallback_transition = None
                if execution.accepted_depth < 2:
                    fallback_index = tree_transition_start + 6
                    if len(scheduler_transitions) != fallback_index + 1:
                        raise RuntimeError(
                            "K=2/B=2 fallback must add exactly one scheduler transition, "
                            f"got {len(scheduler_transitions) - tree_transition_start - 6}"
                        )
                    fallback_transition = scheduler_transitions[fallback_index]
                trace_transitions = api.select_k2_b2_trace_transitions(
                    tree_transitions, route=route,
                    accepted_depth=execution.accepted_depth,
                    fallback_transition=fallback_transition,
                )
                p0_trace_chunk.record_k2_b2_verifier_tree(
                    step_index=step_index,
                    selector=self._p0_action_draft_selector,
                    threshold=threshold,
                    selected_parent=route.parent,
                    selected_child=route.child,
                    accepted_depth=execution.accepted_depth,
                    bonus_target_step_executed=bool(
                        bonus_target_step and execution.accepted_depth == 2),
                    bonus_target_step_index=(step_index + 2 if (
                        bonus_target_step and execution.accepted_depth == 2) else None),
                    first_costs=scores.first_costs,
                    second_costs=scores.second_costs,
                    target_timesteps=target_times,
                    root_state=tree.root_state,
                    first_directions=tree.first_directions,
                    first_states=tree.first_states,
                    second_directions=tree.second_directions,
                    leaf_states=tree.leaf_states,
                    target_directions=scores.target_directions,
                    cfg_multiplier=2 if self.use_cfg else 1,
                )

            if execution.accepted_depth == 0:
                root_target = scores.target_directions[0]
                next_state = execution.next_state
                record_update(
                    step=step_index, state=root, direction=root_target,
                    source="target", transition=(trace_transitions[0]
                                                 if p0_trace_chunk is not None else None)
                )
                actions = compact_to_full(next_state)
                previous_direction = root_target.detach().clone()
                step_index += 1
            elif execution.accepted_depth == 1:
                first = tree.first_directions[route.parent]
                first_state = tree.first_states[route.parent]
                record_update(
                    step=step_index, state=root, direction=first,
                    source="draft", transition=(trace_transitions[0]
                                                if p0_trace_chunk is not None else None)
                )
                second_target = scores.target_directions[1 + route.parent]
                next_state = execution.next_state
                record_update(
                    step=step_index + 1, state=first_state, direction=second_target,
                    source="target", transition=(trace_transitions[1]
                                                  if p0_trace_chunk is not None else None)
                )
                actions = compact_to_full(next_state)
                previous_direction = second_target.detach().clone()
                step_index += 2
            else:
                first = tree.first_directions[route.parent]
                second = tree.second_directions[route.parent, route.child]
                first_state = tree.first_states[route.parent]
                leaf_state = tree.leaf_states[route.parent, route.child]
                record_update(
                    step=step_index, state=root, direction=first,
                    source="draft", transition=(trace_transitions[0]
                                                if p0_trace_chunk is not None else None)
                )
                record_update(
                    step=step_index + 1, state=first_state, direction=second,
                    source="draft", transition=(trace_transitions[1]
                                                if p0_trace_chunk is not None else None)
                )
                if bonus_target_step:
                    bonus_direction = execution.prefetch_target_direction
                    if bonus_direction is None or bonus_next_state is None:
                        raise RuntimeError("K=2/B=2 bonus state/direction was not prepared")
                    record_update(
                        step=step_index + 2, state=leaf_state, direction=bonus_direction,
                        source="target", transition=bonus_transition,
                        bonus_target_step_executed=True,
                    )
                    actions = compact_to_full(bonus_next_state)
                    previous_direction = bonus_direction.detach().clone()
                    step_index += 3
                else:
                    actions = compact_to_full(leaf_state)
                    previous_direction = second.detach().clone()
                    step_index += 2

        # The final t=0 transformer call updates the persistent target cache;
        # candidate verifier forwards above were all update_cache=0.
        terminal_timestep = action_timesteps[action_update_count]
        terminal_input = self._prepare_latent_input(
            None, actions, terminal_timestep, terminal_timestep, None, action_cond,
            frame_st_id=frame_st_id
        )
        with self._phase_profile_phase("action_terminal_target_forward"):
            raw_terminal = self.transformer(
                self._repeat_input_for_cfg(terminal_input["action_res_lst"]),
                update_cache=1,
                cache_name=self.cache_name,
                action_mode=True,
            )
        self._phase_profile_counter("terminal_target_forward_count")
        if p0_trace_chunk is not None:
            p0_trace_chunk.record_terminal_forward(
                branch="action",
                step_index=action_update_count,
                timestep=terminal_timestep,
                sigma=0.0,
                cfg_scale=self.job_config.action_guidance_scale,
                cfg_active=self.job_config.action_guidance_scale > 1,
                raw_model_output=raw_terminal,
                x_i=actions,
                cache_update=True,
                input_preparation=self._p0_trace_input_preparation("action", frame_st_id),
            )
        return actions

    @torch.no_grad()
    def _infer_action_verifier_k3_b2(
            self, actions: torch.Tensor, action_timesteps: torch.Tensor,
            frame_st_id: int, action_update_count: int, nsys_base: str,
            p0_trace_chunk) -> torch.Tensor:
        """Run online K=3/B=2 with a 7-row target verifier per three-edge tree.

        Target rows are root, the two depth-1 states, and the four depth-2
        states.  Thus every committed draft edge is explicitly verified, while
        the eight depth-3 states are draft-only leaves.  A first rejected edge
        is replaced by the matching target direction from that same batch.
        """
        predictor = self._load_p0_action_draft_b2_predictor()
        threshold = float(self._p0_action_draft_threshold)
        channels = tuple(int(channel) for channel in self.job_config.used_action_channel_ids)
        if not channels:
            raise ValueError("verifier_k3_b2 requires non-empty used_action_channel_ids")
        action_cond = torch.zeros(
            [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
            device=self.device, dtype=self.dtype
        ) if frame_st_id == 0 else None
        actions = self._p0_action_project_state(actions, action_cond, frame_st_id)
        previous_direction = None
        step_index = 0
        scheduler_transitions = []

        def compact_to_full(compact: torch.Tensor) -> torch.Tensor:
            return self._p0_action_scatter_compact(compact, actions)

        def scheduler_step(
                state: torch.Tensor, direction: torch.Tensor,
                timestep: float, offset: int) -> torch.Tensor:
            full_state = compact_to_full(state)
            full_direction = compact_to_full(direction)
            timestep_tensor = torch.tensor(
                float(timestep), device=self.device, dtype=torch.float32
            )
            with self._phase_profile_phase("action_tree_scheduler"):
                raw_full_next = self.action_scheduler.step(
                    full_direction, timestep_tensor, full_state, return_dict=False
                )
            self._phase_profile_counter("tree_scheduler_transition_count")
            projected_full_next = self._p0_action_project_state(
                raw_full_next, action_cond, frame_st_id
            )
            projected = self._p0_action_compact(projected_full_next[0])
            if p0_trace_chunk is not None:
                scheduler_transitions.append({
                    "timestep": float(timestep),
                    "raw_full_next": raw_full_next.detach().clone(),
                    "projected_full_next": projected_full_next.detach().clone(),
                })
            return projected

        def record_update(
                *, step: int, state: torch.Tensor, direction: torch.Tensor,
                source: str, transition: dict | None) -> None:
            self._phase_profile_counter("committed_scheduler_update_count")
            if source == "draft":
                self._phase_profile_counter("committed_draft_scheduler_update_count")
            else:
                self._phase_profile_counter("ordinary_target_scheduler_update_count")
            if p0_trace_chunk is None:
                return
            if transition is None:
                raise RuntimeError(
                    "missing explicit scheduler raw/projected pair for K=3/B=2 trace update"
                )
            expected_timestep = float(action_timesteps[step].detach().cpu().item())
            if not np.isclose(float(transition["timestep"]), expected_timestep,
                              rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    "explicit K=3/B=2 trace transition timestep does not match "
                    f"step {step}: {transition['timestep']} vs {expected_timestep}"
                )
            state_full = compact_to_full(state)
            direction_full = compact_to_full(direction)
            sigma = self.action_scheduler.sigmas[step]
            next_sigma = (
                self.action_scheduler.sigmas[step + 1]
                if step + 1 < len(self.action_scheduler.sigmas) else 0.0
            )
            p0_trace_chunk.record_scheduler_update(
                branch="action",
                model_source=source,
                step_index=step,
                timestep=action_timesteps[step],
                sigma=sigma,
                next_sigma=next_sigma,
                cfg_scale=self.job_config.action_guidance_scale,
                cfg_active=self.job_config.action_guidance_scale > 1,
                cache_update=False,
                x_i=state_full,
                d_cond=direction_full if source == "target" else None,
                d_uncond=None,
                d_cfg=direction_full,
                x_next_raw=transition["raw_full_next"],
                x_next=transition["projected_full_next"],
                projection=self._p0_trace_projection(frame_st_id),
                input_preparation=self._p0_trace_input_preparation("action", frame_st_id),
            )

        while step_index < action_update_count:
            remaining = action_update_count - step_index
            # The K=3 commitment consumes three real scheduler transitions.
            # Any shorter suffix falls back to ordinary target updates.
            if previous_direction is None or remaining < 3:
                state = self._p0_action_compact(actions[0])
                timestep = action_timesteps[step_index]
                direction_full_input = self._prepare_latent_input(
                    None, actions, timestep, timestep, None, action_cond,
                    frame_st_id=frame_st_id
                )
                with self._phase_profile_phase("action_target_forward"):
                    raw = self.transformer(
                        self._repeat_input_for_cfg(direction_full_input["action_res_lst"]),
                        update_cache=0,
                        cache_name=self.cache_name,
                        action_mode=True,
                    )
                self._phase_profile_counter("target_forward_count")
                output = rearrange(
                    raw, "b (f n) c -> b c f n 1", f=self.job_config.frame_chunk_size
                )
                if self.use_cfg:
                    cond, uncond = output[:1], output[1:2]
                    direction_full = (
                        uncond + self.job_config.action_guidance_scale * (cond - uncond)
                        if self.job_config.action_guidance_scale > 1 else cond
                    )
                else:
                    direction_full = output[:1]
                direction = self._p0_action_compact(direction_full[0])
                transition_start = len(scheduler_transitions)
                next_state = scheduler_step(state, direction, float(timestep), 0)
                transition = (
                    scheduler_transitions[transition_start]
                    if p0_trace_chunk is not None else None
                )
                record_update(
                    step=step_index, state=state, direction=direction,
                    source="target", transition=transition
                )
                actions = compact_to_full(next_state)
                previous_direction = direction.detach().clone()
                step_index += 1
                continue

            sigmas = self.action_scheduler.sigmas
            sigma_path = [
                float(sigmas[step_index].detach().cpu().item()),
                float(sigmas[step_index + 1].detach().cpu().item()),
                float(sigmas[step_index + 2].detach().cpu().item()),
                (float(sigmas[step_index + 3].detach().cpu().item())
                 if step_index + 3 < len(sigmas) else 0.0),
            ]
            target_times = [
                float(action_timesteps[step_index + offset].detach().cpu().item())
                for offset in range(4)
            ]
            root_full = self._p0_action_project_state(actions, action_cond, frame_st_id)
            root = self._p0_action_compact(root_full[0])
            previous = (
                torch.zeros_like(root)
                if previous_direction is None else previous_direction
            )

            def draft_predictor(
                    state: torch.Tensor, prev: torch.Tensor,
                    sigma: float, next_sigma: float, timestep: float) -> torch.Tensor:
                with self._phase_profile_phase("action_draft_model"):
                    result = predictor.predict(state, prev, sigma, next_sigma, timestep)
                self._phase_profile_counter("draft_forward_count")
                return result

            tree_transition_start = len(scheduler_transitions)
            with self._phase_profile_phase("action_tree_build"):
                tree = build_k3_b2_tree(
                    root_state=root,
                    previous_direction=previous,
                    sigma_path=sigma_path,
                    target_timesteps=target_times,
                    draft_predictor=draft_predictor,
                    scheduler_step=scheduler_step,
                )
            self._phase_profile_counter("tree_count")
            if p0_trace_chunk is not None:
                tree_transitions = scheduler_transitions[tree_transition_start:]
                if len(tree_transitions) != 14:
                    raise RuntimeError(
                        "K=3/B=2 tree must produce exactly fourteen scheduler transitions, "
                        f"got {len(tree_transitions)}"
                    )
            else:
                tree_transitions = ()
            with self._phase_profile_phase("action_candidate_pack_scatter"):
                compact_states, packed_timesteps = pack_k3_b2_verifier_batch(tree)
                candidate_states = self._p0_action_scatter_compact_batch(compact_states, root_full)
            self._phase_profile_counter("logical_candidate_rows", int(candidate_states.shape[0]))
            self._phase_profile_counter("verifier_target_row_count", int(candidate_states.shape[0]))
            target_full = self._target_action_candidate_directions(
                candidate_states, packed_timesteps, frame_st_id
            )
            target_compact = torch.stack(tuple(
                self._p0_action_compact(target_full[row])
                for row in range(target_full.shape[0])
            ), dim=0)
            with self._phase_profile_phase("action_error_score"):
                scores = score_k3_b2_tree(tree, target_compact)
            with self._phase_profile_phase("action_selector"):
                route = select_k3_b2_route(
                    scores, selector=self._p0_action_draft_selector
                )
            with self._phase_profile_phase("action_route_execute"):
                execution = execute_k3_b2_route(
                    tree, scores, route, threshold=threshold,
                    scheduler_step=scheduler_step,
                )
            self._phase_profile_counter("accepted_depth_" + str(int(execution.accepted_depth)))
            if p0_trace_chunk is not None:
                fallback_transition = None
                if execution.accepted_depth < 3:
                    fallback_index = tree_transition_start + 14
                    if len(scheduler_transitions) != fallback_index + 1:
                        raise RuntimeError(
                            "K=3/B=2 fallback must add exactly one scheduler transition, "
                            f"got {len(scheduler_transitions) - tree_transition_start - 14}"
                        )
                    fallback_transition = scheduler_transitions[fallback_index]
                trace_transitions = select_k3_b2_trace_transitions(
                    tree_transitions, route=route,
                    accepted_depth=execution.accepted_depth,
                    fallback_transition=fallback_transition,
                )
                p0_trace_chunk.record_k3_b2_verifier_tree(
                    step_index=step_index,
                    selector=self._p0_action_draft_selector,
                    threshold=threshold,
                    selected_parent=route.parent,
                    selected_child=route.child,
                    selected_grandchild=route.grandchild,
                    accepted_depth=execution.accepted_depth,
                    first_costs=scores.first_costs,
                    second_costs=scores.second_costs,
                    third_costs=scores.third_costs,
                    target_timesteps=target_times,
                    root_state=tree.root_state,
                    first_directions=tree.first_directions,
                    first_states=tree.first_states,
                    second_directions=tree.second_directions,
                    second_states=tree.second_states,
                    third_directions=tree.third_directions,
                    third_states=tree.third_states,
                    target_directions=scores.target_directions,
                    cfg_multiplier=2 if self.use_cfg else 1,
                )

            if execution.accepted_depth == 0:
                root_target = scores.target_directions[0]
                record_update(
                    step=step_index, state=root, direction=root_target,
                    source="target", transition=(trace_transitions[0]
                                                 if p0_trace_chunk is not None else None)
                )
                actions = compact_to_full(execution.next_state)
                previous_direction = root_target.detach().clone()
                step_index += 1
            elif execution.accepted_depth == 1:
                first = tree.first_directions[route.parent]
                first_state = tree.first_states[route.parent]
                first_target = scores.target_directions[1 + route.parent]
                record_update(
                    step=step_index, state=root, direction=first,
                    source="draft", transition=(trace_transitions[0]
                                                if p0_trace_chunk is not None else None)
                )
                record_update(
                    step=step_index + 1, state=first_state, direction=first_target,
                    source="target", transition=(trace_transitions[1]
                                                 if p0_trace_chunk is not None else None)
                )
                actions = compact_to_full(execution.next_state)
                previous_direction = first_target.detach().clone()
                step_index += 2
            elif execution.accepted_depth == 2:
                first = tree.first_directions[route.parent]
                second = tree.second_directions[route.parent, route.child]
                first_state = tree.first_states[route.parent]
                second_state = tree.second_states[route.parent, route.child]
                second_target = scores.target_directions[3 + 2 * route.parent + route.child]
                record_update(
                    step=step_index, state=root, direction=first,
                    source="draft", transition=(trace_transitions[0]
                                                if p0_trace_chunk is not None else None)
                )
                record_update(
                    step=step_index + 1, state=first_state, direction=second,
                    source="draft", transition=(trace_transitions[1]
                                                if p0_trace_chunk is not None else None)
                )
                record_update(
                    step=step_index + 2, state=second_state, direction=second_target,
                    source="target", transition=(trace_transitions[2]
                                                 if p0_trace_chunk is not None else None)
                )
                actions = compact_to_full(execution.next_state)
                previous_direction = second_target.detach().clone()
                step_index += 3
            else:
                first = tree.first_directions[route.parent]
                second = tree.second_directions[route.parent, route.child]
                third = tree.third_directions[route.parent, route.child, route.grandchild]
                first_state = tree.first_states[route.parent]
                second_state = tree.second_states[route.parent, route.child]
                record_update(
                    step=step_index, state=root, direction=first,
                    source="draft", transition=(trace_transitions[0]
                                                if p0_trace_chunk is not None else None)
                )
                record_update(
                    step=step_index + 1, state=first_state, direction=second,
                    source="draft", transition=(trace_transitions[1]
                                                if p0_trace_chunk is not None else None)
                )
                record_update(
                    step=step_index + 2, state=second_state, direction=third,
                    source="draft", transition=(trace_transitions[2]
                                                if p0_trace_chunk is not None else None)
                )
                actions = compact_to_full(execution.next_state)
                previous_direction = third.detach().clone()
                step_index += 3

        # Preserve the normal t=0 target cache update after all action updates.
        terminal_timestep = action_timesteps[action_update_count]
        terminal_input = self._prepare_latent_input(
            None, actions, terminal_timestep, terminal_timestep, None, action_cond,
            frame_st_id=frame_st_id
        )
        with self._phase_profile_phase("action_terminal_target_forward"):
            raw_terminal = self.transformer(
                self._repeat_input_for_cfg(terminal_input["action_res_lst"]),
                update_cache=1,
                cache_name=self.cache_name,
                action_mode=True,
            )
        self._phase_profile_counter("terminal_target_forward_count")
        if p0_trace_chunk is not None:
            p0_trace_chunk.record_terminal_forward(
                branch="action",
                step_index=action_update_count,
                timestep=terminal_timestep,
                sigma=0.0,
                cfg_scale=self.job_config.action_guidance_scale,
                cfg_active=self.job_config.action_guidance_scale > 1,
                raw_model_output=raw_terminal,
                x_i=actions,
                cache_update=True,
                input_preparation=self._p0_trace_input_preparation("action", frame_st_id),
            )
        return actions

    @torch.no_grad()
    def _capture_p0_action_shadow_candidates(
            self, trace_chunk, *, step_index, action_update_count, action_timesteps,
            anchor_state, anchor_direction, previous_direction, actual_next_state,
            frame_st_id, action_cond):
        """Capture one real-context candidate batch without affecting target actions."""
        config = self._p0_shadow_config
        if config is None or step_index % config["anchor_stride"] != 0:
            return
        future_count = action_update_count - step_index - 1
        if future_count <= 0:
            return

        predictor = config["predictor"]
        if predictor == "sigma_linear" and previous_direction is None:
            trace_chunk.record_shadow_candidate_skip(
                anchor_step_index=step_index, predictor=predictor,
                reason="sigma_linear requires a previous exact target direction")
            return

        candidate_count = min(config["depth"], future_count)
        try:
            scheduler_sigmas = self.action_scheduler.sigmas
            sigma_end = step_index + candidate_count + 2
            if sigma_end > len(scheduler_sigmas):
                raise ValueError(
                    "action scheduler sigma path is too short for candidate block: "
                    f"need {sigma_end}, have {len(scheduler_sigmas)}")
            sigma_path = scheduler_sigmas[step_index:sigma_end].detach().cpu().tolist()
            timestep_end = step_index + candidate_count + 1
            if timestep_end > len(action_timesteps):
                raise ValueError(
                    "action timestep path is too short for candidate block: "
                    f"need {timestep_end}, have {len(action_timesteps)}")
            candidate_timesteps = action_timesteps[
                step_index + 1:timestep_end].detach().cpu().tolist()

            action_mask = self.action_mask.to(device=anchor_state.device)
            fixed_action = None
            if frame_st_id == 0 and action_cond is not None:
                fixed_action = action_cond[0, :, 0:1].detach().clone()

            def transform_action_state(value, _offset):
                # The draft API operates on [C,F,H,W], without WAM's batch axis.
                if value.ndim != 4 or value.shape[0] != action_mask.numel():
                    raise ValueError("unexpected action state shape in P0 transform")
                if fixed_action is not None:
                    value[:, 0:1] = fixed_action.to(
                        device=value.device, dtype=value.dtype)
                value[~action_mask] = 0
                return value

            sigma_linear_kwargs = {}
            draft_predictor = None
            draft_channel_ids = None
            if predictor == "autoregressive_mlp":
                draft_predictor = self._load_p0_shadow_predictor()
                draft_channel_ids = tuple(
                    int(channel) for channel in self.job_config.used_action_channel_ids
                )
                if not draft_channel_ids:
                    raise ValueError(
                        "autoregressive_mlp requires non-empty used_action_channel_ids")
            if predictor == "sigma_linear":
                if step_index < 1:
                    raise ValueError("sigma_linear anchor has no previous scheduler sigma")
                sigma_linear_kwargs = {
                    "previous_direction": previous_direction[0],
                    "previous_sigma": float(
                        self.action_scheduler.sigmas[step_index - 1].detach().cpu().item()),
                }
            draft = self._load_p0_shadow_api().build_draft_candidate_block(
                anchor_state=anchor_state[0],
                anchor_direction=anchor_direction[0],
                sigma_path=sigma_path,
                candidate_timesteps=candidate_timesteps,
                predictor=predictor,
                state_transform=transform_action_state,
                draft_predictor=draft_predictor,
                draft_channel_ids=draft_channel_ids,
                **sigma_linear_kwargs,
            )
            target_directions = self._target_action_candidate_directions(
                draft.candidate_states, draft.target_timesteps, frame_st_id)

            expected_next = transform_action_state(actual_next_state[0].detach().clone(), -1)
            first_candidate_delta = (draft.candidate_states[0].float() - expected_next.float())
            first_candidate_max_abs = float(first_candidate_delta.abs().max().item())
            first_candidate_rel_l2 = float(
                torch.linalg.vector_norm(first_candidate_delta.reshape(-1))
                / torch.linalg.vector_norm(expected_next.float().reshape(-1)).clamp_min(1e-12)
            )
            cfg_active = self.job_config.action_guidance_scale > 1
            cache_rows = ([0] * candidate_count + [1] * candidate_count
                          if self.use_cfg else [0] * candidate_count)
            trace_chunk.record_shadow_candidate_block(
                anchor_step_index=step_index,
                predictor=predictor,
                draft_checkpoint=(
                    getattr(draft_predictor, "checkpoint_path", None)
                    if draft_predictor is not None else None),
                draft_channel_ids=draft_channel_ids,
                anchor_sigma=float(sigma_path[0]),
                previous_sigma=(
                    float(self.action_scheduler.sigmas[step_index - 1].detach().cpu().item())
                    if step_index > 0 else None),
                candidate_timesteps=draft.target_timesteps,
                candidate_sigmas=draft.candidate_sigmas,
                transition_scales=draft.transition_scales,
                cfg_scale=float(self.job_config.action_guidance_scale),
                cfg_active=bool(cfg_active),
                use_cfg=bool(self.use_cfg),
                cache_mode=self._cache_mode,
                cache_row_mapping=cache_rows,
                frame_st_id=int(frame_st_id),
                projection=self._p0_trace_projection(frame_st_id),
                input_preparation=self._p0_trace_input_preparation(
                    "action", frame_st_id),
                first_candidate_matches_live_successor_max_abs=first_candidate_max_abs,
                first_candidate_matches_live_successor_rel_l2=first_candidate_rel_l2,
                anchor_state=anchor_state[0],
                anchor_direction=anchor_direction[0],
                previous_direction=(
                    previous_direction[0] if previous_direction is not None else None),
                candidate_states=draft.candidate_states,
                draft_next_states=draft.draft_next_states,
                target_directions=target_directions,
            )
        except Exception as exc:
            trace_chunk.record_shadow_candidate_error(
                anchor_step_index=step_index, predictor=predictor, error=repr(exc))
            logger.exception(
                "P0 shadow candidate capture failed at action step %d; target action is retained",
                step_index)

    @torch.no_grad()
    def _target_action_candidate_directions(self, candidate_states,
                                            candidate_timesteps, frame_st_id):
        """Evaluate speculative candidates in one read-only target forward.

        This is an opt-in primitive only; the normal LIBERO inference path does
        not call it. Output is the CFG-combined target direction [K,C,F,H,W].
        """
        with self._phase_profile_phase("verifier_prepare_input"):
            candidate_input = self._prepare_action_candidate_input(
                candidate_states, candidate_timesteps, frame_st_id)
            packed_input, cache_rows = self._repeat_action_candidates_for_cfg(candidate_input)
        self._phase_profile_counter("verifier_cfg_rows", int(packed_input["noisy_latents"].shape[0]))
        with self._phase_profile_phase("verifier_target_forward"):
            output = self.transformer(
                packed_input,
                update_cache=0,
                cache_name=self.cache_name,
                action_mode=True,
                candidate_cache_rows=cache_rows,
            )
        self._phase_profile_counter("verifier_forward_count")
        with self._phase_profile_phase("verifier_score_postprocess"):
            output = rearrange(
                output, 'b (f n) c -> b c f n 1',
                f=self.job_config.frame_chunk_size)
        count = candidate_states.shape[0]
        if output.shape[0] != count * (2 if self.use_cfg else 1):
            raise ValueError("target candidate forward returned unexpected batch size")
        if self.use_cfg:
            conditional, unconditional = output[:count], output[count:]
            if self.job_config.action_guidance_scale > 1:
                output = unconditional + self.job_config.action_guidance_scale * (
                    conditional - unconditional)
            else:
                output = conditional
        return output

    @staticmethod
    def _p0_trace_scheduler_values(scheduler, step_index):
        """Return the scheduler's exact sigma pair for a trace record.

        ``_infer`` pads its timestep loop with a terminal zero to perform the
        cache update.  That final transformer forward has no matching entry in
        ``scheduler.sigmas`` and must not be counted as a denoising update.
        """
        if step_index < len(scheduler.sigmas):
            sigma = scheduler.sigmas[step_index]
            next_sigma = (scheduler.sigmas[step_index + 1]
                           if step_index + 1 < len(scheduler.sigmas) else 0.0)
            return sigma, next_sigma
        return 0.0, None

    @staticmethod
    def _p0_trace_projection(frame_st_id):
        """Describe the known post-step observation clamp for offline rollout."""
        if frame_st_id == 0:
            return {
                'kind': 'fixed_temporal_prefix',
                'axis': 2,
                'count': 1,
            }
        return {'kind': 'identity'}

    def _p0_trace_input_preparation(self, branch, frame_st_id):
        """Describe deterministic mutation before a branch's scheduler input.

        ``_prepare_latent_input`` intentionally writes directly into the latent
        tensors it receives.  In particular, the action branch zeroes model
        padding channels on every iteration.  That operation is part of the
        real denoising state transition; describing it lets offline rollouts
        reproduce the next scheduler input instead of treating it as drift.
        """
        operations = []
        if frame_st_id == 0:
            operations.append({
                'kind': 'fixed_temporal_prefix',
                'axis': 2,
                'count': 1,
            })
        if branch == 'action':
            zeroed_channel_ids = torch.where(~self.action_mask)[0].detach().cpu().tolist()
            if zeroed_channel_ids:
                operations.append({
                    'kind': 'zero_masked_channels',
                    'axis': 1,
                    'zeroed_channel_ids': [int(value) for value in zeroed_channel_ids],
                })
        if not operations:
            return {'kind': 'identity'}
        if len(operations) == 1:
            return operations[0]
        return {'kind': 'composite', 'operations': operations}

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,          #! 这里初始time step为 [t, t]因为flame 维度为2
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst']['noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]  #! 这里已经把第一帧替换成真实
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0     #! 这里把第一帧的time step设置为0，因为第一帧不需要预测

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        nsys_chunk = (f"chunk={frame_st_id // frame_chunk_size:02d}/"
                      f"frame_start={frame_st_id:02d}")
        nsys_base = f"LBVA/basic/{nsys_chunk}"
        if frame_st_id == 0:
            with self._phase_profile_phase("obs_encode"):
                init_latent = self._encode_obs(obs)   #! 这里已经执行了VAE，把首帧obs映射成latent
            self.init_latent = init_latent

        latents = torch.randn(1,        #! 初始化视频输入，后面用真实obs替换第一帧
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,        #! 初始化action输入
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

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

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        p0_trace_chunk = None
        if self._p0_state_teacher is not None:
            self._p0_teacher_context = dict(obs.get('p0_trace_context', {}))
            self._p0_state_teacher.begin_chunk()
        if self._p0_trace is not None:
            # The client may attach a small, JSON-safe rollout identity so
            # multi-chunk traces can be joined to LIBERO task/episode records
            # without relying on request order.  It is metadata only: it never
            # enters the model or changes inference/cache state.
            trace_context = obs.get('p0_trace_context', {})
            if not isinstance(trace_context, dict):
                trace_context = {'invalid_context_type': type(trace_context).__name__}
            p0_trace_chunk = self._p0_trace.begin_chunk(
                **trace_context,
                frame_st_id=frame_st_id,
                experiment_name=self.exp_name,
                config_name=getattr(self.job_config, '__name__', 'unknown'),
                cache_mode=self._cache_mode,
                use_cfg=bool(self.use_cfg),
                video=dict(
                    inference_updates=video_inference_step,
                    loop_timesteps=[float(value) for value in timesteps.cpu()],
                    scheduler_sigmas=[float(value) for value in self.scheduler.sigmas.cpu()],
                    cfg_scale=float(self.job_config.guidance_scale),
                    snr_shift=float(self.job_config.snr_shift),
                ),
                action=dict(
                    inference_updates=action_inference_step,
                    loop_timesteps=[float(value) for value in action_timesteps.cpu()],
                    scheduler_sigmas=[float(value) for value in self.action_scheduler.sigmas.cpu()],
                    cfg_scale=float(self.job_config.action_guidance_scale),
                    snr_shift=float(self.job_config.action_snr_shift),
                    action_dim=int(self.job_config.action_dim),
                    action_per_frame=int(self.action_per_frame),
                    used_action_channel_ids=list(self.job_config.used_action_channel_ids),
                    shadow_candidates=self._p0_shadow_config,
                    online_draft=dict(
                        mode=self._p0_action_draft_mode,
                        alternate_k1_checkpoint=self._p0_action_draft_checkpoint or None,
                        verifier_k2_b2_checkpoint=(
                            self._p0_action_draft_b2_checkpoint or None
                        ),
                        selector=self._p0_action_draft_selector,
                        threshold=self._p0_action_draft_threshold,
                    ),
                ),
            )

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            video_total_token = self._phase_profile_start("video_total")
            video_range = self._nsys_push(f"{nsys_base}/video")
            try:
                for i, t in enumerate(tqdm(timesteps)):   #! 这里的t就是噪声等级，但是非均匀，是从1000 到 0 非均匀分布
                    last_step = i == len(timesteps) - 1
                    step_range = self._nsys_push(
                        f"{nsys_base}/video/step={i:02d}/"
                        f"cache_update={int(last_step)}")
                    try:
                        latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None  #! 这里取第一段真实obs

                        prepare_range = self._nsys_push(
                            f"{nsys_base}/video/step={i:02d}/prepare_input")
                        try:
                            with self._phase_profile_phase("video_prepare_input"):
                                input_dict = self._prepare_latent_input(  # 在准备阶段替换完成
                                    latents,
                                    None,
                                    t,
                                    t,
                                    latent_cond,
                                    None,
                                    frame_st_id=frame_st_id)
                        finally:
                            self._nsys_pop(prepare_range)

                        if self._nsys_profile_mode == "detail" and self._nsys_profile_active:
                            self._nsys_detail_phase = (
                                f"{nsys_chunk}/video/step={i:02d}/transformer")
                        transformer_range = self._nsys_push(
                            f"{nsys_base}/video/step={i:02d}/transformer")
                        try:
                            with self._phase_profile_phase("video_target_forward"):
                                video_noise_pred = self.transformer(
                                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),   #? 这里复制成两份，一份是正常的，另一份是prompt为空的情况，每次transformer会产生两个方向
                                    update_cache=1 if last_step else 0,   #! 这里传入的update cache表示只有最后一轮才记录kv cache
                                    cache_name=self.cache_name,
                                    action_mode=False)          #! 这里是视频部分的循环推理  即 model.py 800行
                        finally:
                            self._nsys_pop(transformer_range)
                            self._nsys_detail_phase = None

                        did_video_scheduler_update = not last_step or video_step != -1
                        raw_video_noise_pred = video_noise_pred
                        if did_video_scheduler_update:
                            scheduler_range = self._nsys_push(
                                f"{nsys_base}/video/step={i:02d}/scheduler")
                            try:
                                with self._phase_profile_phase("video_scheduler"):
                                    video_noise_pred = data_seq_to_patch(   #! 把transformer输出还原成scheduler需要的latent网格，需要做逆patch
                                        self.job_config.patch_size, video_noise_pred,
                                        frame_chunk_size, self.latent_height,
                                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                                    video_d_cond = video_noise_pred[:1]
                                    video_d_uncond = (video_noise_pred[1:]
                                                       if video_noise_pred.shape[0] > 1 else None)
                                    if self.job_config.guidance_scale > 1:
                                        video_d_cfg = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])   #? CFG合成，这里是用有prompt的方向 - 无prompt方向，力求尽可能朝向有prompt方向去噪
                                    else:
                                        video_d_cfg = video_noise_pred[:1]
                                    video_x_i = latents
                                    latents = self.scheduler.step(video_d_cfg,   #! 这里scheduler更新整个模块
                                                                  t,
                                                                  latents,
                                                                  return_dict=False)
                                    # The observation clamp below mutates ``latents``
                                    # in place.  Preserve raw scheduler output only
                                    # for opt-in diagnostic tracing.
                                    video_x_next_raw = (
                                        latents.clone() if p0_trace_chunk is not None else None
                                    )
                            finally:
                                self._nsys_pop(scheduler_range)

                        latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
                        if p0_trace_chunk is not None:
                            video_sigma, video_next_sigma = self._p0_trace_scheduler_values(
                                self.scheduler, i)
                            if did_video_scheduler_update:
                                p0_trace_chunk.record_scheduler_update(
                                    branch='video',
                                    step_index=i,
                                    timestep=t,
                                    sigma=video_sigma,
                                    next_sigma=video_next_sigma,
                                    cfg_scale=self.job_config.guidance_scale,
                                    cfg_active=self.job_config.guidance_scale > 1,
                                    cache_update=last_step,
                                    x_i=video_x_i,
                                    d_cond=video_d_cond,
                                    d_uncond=video_d_uncond,
                                    d_cfg=video_d_cfg,
                                    x_next_raw=video_x_next_raw,
                                    x_next=latents,
                                    projection=self._p0_trace_projection(frame_st_id),
                                    input_preparation=self._p0_trace_input_preparation(
                                        'video', frame_st_id),
                                )
                            else:
                                p0_trace_chunk.record_terminal_forward(
                                    branch='video',
                                    step_index=i,
                                    timestep=t,
                                    sigma=video_sigma,
                                    cfg_scale=self.job_config.guidance_scale,
                                    cfg_active=self.job_config.guidance_scale > 1,
                                    raw_model_output=raw_video_noise_pred,
                                    x_i=latents,
                                    cache_update=last_step,
                                    input_preparation=self._p0_trace_input_preparation(
                                        'video', frame_st_id),
                                )
                    finally:
                        self._nsys_detail_phase = None
                        self._nsys_pop(step_range)
            finally:
                self._nsys_pop(video_range)
                self._phase_profile_end(video_total_token)

            action_total_token = self._phase_profile_start("action_total")
            action_range = self._nsys_push(f"{nsys_base}/action")
            try:
                if self._p0_action_draft_mode in {
                        "verifier_k2_b2", "verifier_k2_b2_bonus"}:
                    # Both K=2 variants use the same seven-row target batch.
                    # Only the explicit bonus mode consumes a fully accepted
                    # leaf target direction as a third committed target update.
                    actions = self._infer_action_verifier_k2_b2(
                        actions=actions,
                        action_timesteps=action_timesteps,
                        frame_st_id=frame_st_id,
                        action_update_count=action_inference_step,
                        nsys_base=nsys_base,
                        p0_trace_chunk=p0_trace_chunk,
                        bonus_target_step=(
                            self._p0_action_draft_mode == "verifier_k2_b2_bonus"),
                    )
                elif self._p0_action_draft_mode == "verifier_k3_b2":
                    actions = self._infer_action_verifier_k3_b2(
                        actions=actions,
                        action_timesteps=action_timesteps,
                        frame_st_id=frame_st_id,
                        action_update_count=action_inference_step,
                        nsys_base=nsys_base,
                        p0_trace_chunk=p0_trace_chunk,
                    )
                else:
                    # K=1 online convention: 1-based odd scheduler updates use the
                    # target; 1-based even updates use the draft conditioned on the
                    # immediately preceding target direction.  The t=0 terminal
                    # forward remains target-only because it maintains the KV cache.
                    previous_target_action_d_cfg = None
                    for i, t in enumerate(tqdm(action_timesteps)):
                        last_step = i == len(action_timesteps) - 1
                        did_action_scheduler_update = not last_step
                        use_action_draft = (
                            did_action_scheduler_update
                            and self._p0_action_draft_mode == "alternate_k1"
                            and i % 2 == 1
                        )
                        model_source = "draft" if use_action_draft else "target"
                        step_range = self._nsys_push(
                            f"{nsys_base}/action/step={i:02d}/"
                            f"cache_update={int(last_step)}/source={model_source}")
                        try:
                            action_cond = torch.zeros(
                                [
                                    1, self.job_config.action_dim, 1,
                                    self.action_per_frame, 1
                                ],
                                device=self.device,
                                dtype=self.dtype) if frame_st_id == 0 else None
                            if p0_trace_chunk is not None and i == 0:
                                # The compact training contract captures only
                                # re-encodable inputs: raw camera RGB, prompt,
                                # and the observed action prefix. It never
                                # copies target hidden/KV state.
                                p0_trace_chunk.record_draft_train_condition(
                                    prompt=obs.get('prompt'),
                                    observation=obs.get('obs'),
                                    action_condition=action_cond,
                                    observation_camera_keys=self.job_config.obs_cam_keys,
                                )

                            action_x_i = actions
                            action_d_cond = None
                            action_d_uncond = None
                            raw_action_noise_pred = None
                            if use_action_draft:
                                # Match the target path's deterministic next-input
                                # preparation before passing state to the action-only
                                # draft: fixed observed prefix and zero padded channels.
                                if action_cond is not None:
                                    actions[:, :, 0:1] = action_cond
                                actions[:, ~self.action_mask] *= 0
                                action_d_cfg = self._predict_p0_action_draft(
                                    actions, previous_target_action_d_cfg, i,
                                    action_timesteps)
                            else:
                                prepare_range = self._nsys_push(
                                    f"{nsys_base}/action/step={i:02d}/prepare_input")
                                try:
                                    with self._phase_profile_phase("action_prepare_input"):
                                        input_dict = self._prepare_latent_input(
                                            None,
                                            actions,
                                            t,
                                            t,
                                            None,
                                            action_cond,
                                            frame_st_id=frame_st_id)
                                finally:
                                    self._nsys_pop(prepare_range)

                                if self._nsys_profile_mode == "detail" and self._nsys_profile_active:
                                    self._nsys_detail_phase = (
                                        f"{nsys_chunk}/action/step={i:02d}/transformer")
                                transformer_range = self._nsys_push(
                                    f"{nsys_base}/action/step={i:02d}/transformer")
                                try:
                                    with self._phase_profile_phase(
                                            "action_terminal_target_forward"
                                            if last_step else "action_target_forward"):
                                        raw_action_noise_pred = self.transformer(
                                            self._repeat_input_for_cfg(input_dict['action_res_lst']),
                                            update_cache=1 if last_step else 0,
                                            cache_name=self.cache_name,
                                            action_mode=True)
                                finally:
                                    self._nsys_pop(transformer_range)
                                    self._nsys_detail_phase = None

                                if did_action_scheduler_update:
                                    action_noise_pred = rearrange(
                                        raw_action_noise_pred,
                                        'b (f n) c -> b c f n 1',
                                        f=frame_chunk_size)
                                    action_d_cond = action_noise_pred[:1]
                                    action_d_uncond = (
                                        action_noise_pred[1:]
                                        if action_noise_pred.shape[0] > 1 else None)
                                    if self.job_config.action_guidance_scale > 1:
                                        action_d_cfg = (
                                            action_noise_pred[1:]
                                            + self.job_config.action_guidance_scale
                                            * (action_noise_pred[:1] - action_noise_pred[1:]))
                                    else:
                                        action_d_cfg = action_noise_pred[:1]

                            if did_action_scheduler_update:
                                scheduler_range = self._nsys_push(
                                    f"{nsys_base}/action/step={i:02d}/scheduler")
                                try:
                                    with self._phase_profile_phase("action_scheduler"):
                                        actions = self.action_scheduler.step(
                                            action_d_cfg,
                                            t,
                                            actions,
                                            return_dict=False)
                                    # ``_prepare_latent_input`` will zero unused
                                    # action channels before the following step; the
                                    # prefix clamp below also mutates in place.
                                    # Keep the exact scheduler output separately for
                                    # an opt-in P0 contract audit.
                                    action_x_next_raw = (
                                        actions.clone() if p0_trace_chunk is not None else None
                                    )
                                finally:
                                    self._nsys_pop(scheduler_range)

                            actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]
                            if did_action_scheduler_update and model_source == "target":
                                # A following even (1-based) update consumes this
                                # exact target direction as its AR previous state.
                                previous_target_action_d_cfg = action_d_cfg.detach().clone()
                            if p0_trace_chunk is not None:
                                action_sigma, action_next_sigma = self._p0_trace_scheduler_values(
                                    self.action_scheduler, i)
                                if did_action_scheduler_update:
                                    p0_trace_chunk.record_scheduler_update(
                                        branch='action',
                                        model_source=model_source,
                                        step_index=i,
                                        timestep=t,
                                        sigma=action_sigma,
                                        next_sigma=action_next_sigma,
                                        cfg_scale=self.job_config.action_guidance_scale,
                                        cfg_active=self.job_config.action_guidance_scale > 1,
                                        cache_update=last_step,
                                        x_i=action_x_i,
                                        d_cond=action_d_cond,
                                        d_uncond=action_d_uncond,
                                        d_cfg=action_d_cfg,
                                        x_next_raw=action_x_next_raw,
                                        x_next=actions,
                                        transformer_output_raw=raw_action_noise_pred,
                                        projection=self._p0_trace_projection(frame_st_id),
                                        input_preparation=self._p0_trace_input_preparation(
                                            'action', frame_st_id),
                                    )
                                    # Candidate capture assumes a target anchor;
                                    # retain it only on exact target updates.
                                    if (self._p0_shadow_config is not None
                                            and model_source == "target"):
                                        self._capture_p0_action_shadow_candidates(
                                            p0_trace_chunk,
                                            step_index=i,
                                            action_update_count=action_inference_step,
                                            action_timesteps=action_timesteps,
                                            anchor_state=action_x_i,
                                            anchor_direction=action_d_cfg,
                                            previous_direction=previous_target_action_d_cfg,
                                            actual_next_state=actions,
                                            frame_st_id=frame_st_id,
                                            action_cond=action_cond,
                                        )
                                else:
                                    p0_trace_chunk.record_terminal_forward(
                                        branch='action',
                                        step_index=i,
                                        timestep=t,
                                        sigma=action_sigma,
                                        cfg_scale=self.job_config.action_guidance_scale,
                                        cfg_active=self.job_config.action_guidance_scale > 1,
                                        raw_model_output=raw_action_noise_pred,
                                        x_i=actions,
                                        cache_update=last_step,
                                        input_preparation=self._p0_trace_input_preparation(
                                            'action', frame_st_id),
                                    )
                        finally:
                            self._nsys_detail_phase = None
                            self._nsys_pop(step_range)

            finally:
                self._nsys_pop(action_range)
                self._phase_profile_end(action_total_token)

        if p0_trace_chunk is not None:
            try:
                trace_path = p0_trace_chunk.commit()
                logger.info("P0 trace committed: %s", trace_path)
            except Exception:
                # Trace failures must be visible but must not change the action
                # produced by this experimental inference request.
                logger.exception("P0 trace commit failed; target inference result is retained")

        postprocess_token = self._phase_profile_start("postprocess")
        postprocess_range = self._nsys_push(
            f"{nsys_base}/postprocess_and_async_save")
        try:
            actions[:, ~self.action_mask] *= 0

            save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
            save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

            actions = self.postprocess_action(actions)
            torch.cuda.empty_cache()
        finally:
            self._nsys_pop(postprocess_range)
            self._phase_profile_end(postprocess_token)
        if self._p0_state_teacher is not None:
            self._p0_state_teacher.end_chunk(self._p0_teacher_context)
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    # The following helpers are deliberately replay-only.  They are used to
    # measure the target model on recorded online-K=1 states, not to alter the
    # normal policy rollout.  In particular, a counterfactual client must
    # restore the same shared transformer / streaming-VAE history before it
    # sends a candidate action state to the read-only target forward.
    def _p0_counterfactual_require_frame(self, frame_st_id):
        requested = int(frame_st_id)
        if requested != int(self.frame_st_id):
            raise ValueError(
                "counterfactual replay cache desynchronized: "
                f"requested frame_st_id={requested}, server frame_st_id={self.frame_st_id}"
            )

    @torch.no_grad()
    def _p0_counterfactual_set_initial_obs(self, obs):
        """Prime the streaming VAE exactly as online chunk 0 does.

        This endpoint may be called once, immediately after an ordinary
        ``reset(prompt=...)``.  It encodes the initial, stabilized LIBERO
        observation and retains its VAE streaming state for the later recorded
        ``compute_kv_cache`` restores.  It intentionally performs no
        Transformer forward and creates no action/video prediction.
        """
        if int(self.frame_st_id) != 0 or self.init_latent is not None:
            raise ValueError(
                "counterfactual initial observation is valid only immediately after reset"
            )
        initial_latent = self._encode_obs({'obs': obs})
        if initial_latent is None:
            raise ValueError("counterfactual initial observation cannot be empty")
        self.init_latent = initial_latent
        return {
            'frame_st_id': int(self.frame_st_id),
            'initial_latent_shape': list(initial_latent.shape),
        }

    @torch.no_grad()
    def _p0_counterfactual_prime_video_cache(self, *, frame_st_id,
                                              video_terminal_state):
        """Commit the recorded terminal video prediction cache for one chunk.

        Normal ``_infer`` commits this cache with its final (t=0) video
        transformer forward before action denoising.  Recorded action states
        cannot be target-evaluated faithfully without it because video and
        action tokens share the persistent cache.  This is the same terminal
        forward with ``update_cache=1`` and the recorded online-K=1 latent.
        """
        self._p0_counterfactual_require_frame(frame_st_id)
        state = torch.as_tensor(video_terminal_state, device=self.device,
                                dtype=self.dtype)
        expected_shape = (
            1, 48, self.job_config.frame_chunk_size,
            self.latent_height, self.latent_width,
        )
        if tuple(state.shape) != expected_shape:
            raise ValueError(
                "counterfactual video terminal state shape mismatch: "
                f"got {tuple(state.shape)}, expected {expected_shape}"
            )
        if int(self.frame_st_id) == 0:
            if self.init_latent is None:
                raise ValueError(
                    "counterfactual chunk 0 requires p0_counterfactual_set_initial_obs"
                )
            latent_cond = self.init_latent[:, :, 0:1].to(self.dtype)
        else:
            latent_cond = None
        input_dict = self._prepare_latent_input(
            state.clone(),
            None,
            latent_t=0,
            action_t=0,
            latent_cond=latent_cond,
            action_cond=None,
            frame_st_id=int(self.frame_st_id),
        )
        # This mirrors the cache-only final video forward in _infer.  It is
        # deliberately write-enabled; the subsequent candidate forward is
        # read-only (update_cache=0).
        self.transformer(
            self._repeat_input_for_cfg(input_dict['latent_res_lst']),
            update_cache=1,
            cache_name=self.cache_name,
            action_mode=False,
        )
        return {
            'frame_st_id': int(self.frame_st_id),
            'video_terminal_state_shape': list(state.shape),
            'cache_committed': True,
        }

    @torch.no_grad()
    def _p0_counterfactual_target_directions(self, *, frame_st_id,
                                              candidate_states,
                                              candidate_timesteps):
        """Evaluate target action directions at recorded draft states only.

        ``_target_action_candidate_directions`` packs all candidates (and CFG
        rows, when active) into a single Transformer forward with
        ``update_cache=0``.  The persistent cache primed above therefore stays
        unchanged while all 25 K=1 draft/even states of a chunk are measured.
        """
        self._p0_counterfactual_require_frame(frame_st_id)
        states = torch.as_tensor(candidate_states, device=self.device,
                                 dtype=self.dtype)
        directions = self._target_action_candidate_directions(
            states, candidate_timesteps, int(self.frame_st_id))
        return {
            'frame_st_id': int(self.frame_st_id),
            'target_directions': directions.detach().float().cpu().numpy(),
        }

    @torch.no_grad()
    def _p0_counterfactual_restore_kv_cache(self, *, frame_st_id, obs,
                                             action_model_input):
        """Restore the inter-chunk cache from recorded observation/action artifacts.

        ``action_model_input`` is the saved normalized action tensor from the
        online server *before* ``postprocess_action`` (shape [1,C,F,H,1]).
        Accepting this exact artifact avoids a second float denormalize /
        renormalize round-trip in the replay client.  The body otherwise
        matches ``_compute_kv_cache``: clear previous prediction cache, encode
        the actual executed-observation window, update video then action cache,
        and advance ``frame_st_id``.
        """
        self._p0_counterfactual_require_frame(frame_st_id)
        self.transformer.clear_pred_cache(self.cache_name)
        latent_model_input = self._encode_obs({'obs': obs})
        if latent_model_input is None:
            raise ValueError("counterfactual cache restore observation cannot be empty")
        if int(self.frame_st_id) == 0:
            if self.init_latent is None:
                raise ValueError(
                    "counterfactual chunk 0 cache restore requires initial observation"
                )
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input], dim=2)

        action_input = torch.as_tensor(action_model_input, device=self.device,
                                       dtype=latent_model_input.dtype).clone()
        expected_shape = (
            1, self.job_config.action_dim, latent_model_input.shape[2],
            self.action_per_frame, 1,
        )
        if tuple(action_input.shape) != expected_shape:
            raise ValueError(
                "counterfactual normalized action artifact shape mismatch: "
                f"got {tuple(action_input.shape)}, expected {expected_shape}"
            )
        input_dict = self._prepare_latent_input(
            latent_model_input,
            action_input,
            frame_st_id=int(self.frame_st_id),
        )
        self.transformer(
            self._repeat_input_for_cfg(input_dict['latent_res_lst']),
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=False,
        )
        self.transformer(
            self._repeat_input_for_cfg(input_dict['action_res_lst']),
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=True,
        )
        restored_frame_count = int(latent_model_input.shape[2])
        self.frame_st_id += restored_frame_count
        torch.cuda.empty_cache()
        return {
            'frame_st_id_before': int(frame_st_id),
            'frame_count': restored_frame_count,
            'frame_st_id_after': int(self.frame_st_id),
            'cache_restored': True,
        }

    @torch.no_grad()
    def infer(self, obs):               #! 推理整体入口
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        # Explicit opt-in counterfactual-replay commands.  Existing rollout
        # requests do not carry these keys and keep the original control flow.
        replay_commands = (
            'p0_counterfactual_set_initial_obs',
            'p0_counterfactual_prime_video_cache',
            'p0_counterfactual_target_directions',
            'p0_counterfactual_restore_kv_cache',
        )
        requested_replay_commands = [
            command for command in replay_commands if obs.get(command, False)
        ]
        if len(requested_replay_commands) > 1:
            raise ValueError(
                "counterfactual replay request must contain exactly one command, got "
                f"{requested_replay_commands}"
            )

        if reset:
            if requested_replay_commands:
                raise ValueError("reset cannot be combined with a counterfactual replay command")
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif requested_replay_commands:
            command = requested_replay_commands[0]
            logger.info("################# Counterfactual replay: %s #################", command)
            if command == 'p0_counterfactual_set_initial_obs':
                return self._p0_counterfactual_set_initial_obs(obs['obs'])
            if command == 'p0_counterfactual_prime_video_cache':
                return self._p0_counterfactual_prime_video_cache(
                    frame_st_id=obs['frame_st_id'],
                    video_terminal_state=obs['video_terminal_state'],
                )
            if command == 'p0_counterfactual_target_directions':
                return self._p0_counterfactual_target_directions(
                    frame_st_id=obs['frame_st_id'],
                    candidate_states=obs['candidate_states'],
                    candidate_timesteps=obs['candidate_timesteps'],
                )
            if command == 'p0_counterfactual_restore_kv_cache':
                return self._p0_counterfactual_restore_kv_cache(
                    frame_st_id=obs['frame_st_id'],
                    obs=obs['obs'],
                    action_model_input=obs['action_model_input'],
                )
            raise AssertionError(f"unhandled counterfactual replay command: {command}")
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            profile_error = None
            if self._phase_profile_enabled:
                self._phase_profile_begin(frame_st_id=self.frame_st_id, obs=obs)
            try:
                action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
                return dict(action=action)
            except Exception as exc:
                profile_error = exc
                raise
            finally:
                if self._phase_profile_enabled:
                    self._phase_profile_finish(
                        status="error" if profile_error is not None else "completed",
                        error=profile_error,
                    )

    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video

    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs

    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        capture_started = False
        if self._chunk_timing:
            torch.cuda.reset_peak_memory_stats(self.device)
        if self._nsys_profile_mode != "off":
            logger.info(
                "Nsight Systems instrumentation mode=%s; capture chunks 4-6",
                self._nsys_profile_mode,
            )
        try:
            for chunk_id in range(self.job_config.num_chunks_to_infer):
                if self._nsys_profile_mode != "off" and chunk_id == 4:
                    torch.cuda.profiler.start()
                    capture_started = True
                    self._nsys_profile_active = True
                    if self._nsys_profile_mode == "detail":
                        self._install_nsys_detail_hooks()

                frame_st_id = chunk_id * self.job_config.frame_chunk_size
                if self._chunk_timing:
                    torch.cuda.synchronize(self.device)
                    chunk_start_time = time.perf_counter()
                chunk_range = self._nsys_push(
                    f"LBVA/basic/chunk={chunk_id:02d}/"
                    f"frame_start={frame_st_id:02d}")
                try:
                    actions, latents = self._infer(
                        init_obs, frame_st_id=frame_st_id)
                    if self._chunk_timing:
                        torch.cuda.synchronize(self.device)
                        elapsed = time.perf_counter() - chunk_start_time
                        logger.info("C_CHUNK_METRIC %s", json.dumps(dict(
                            mode=self._cache_mode, chunk=chunk_id, seconds=elapsed,
                            profile=self._nsys_profile_mode)))
                    actions = torch.from_numpy(actions)
                    pred_latent_lst.append(latents)
                    pred_action_lst.append(actions)
                finally:
                    self._nsys_pop(chunk_range)

                if (self._nsys_profile_mode != "off" and chunk_id == 6
                        and capture_started):
                    self._stop_nsys_capture()
                    capture_started = False
        finally:
            if capture_started:
                self._stop_nsys_capture()
        if self._cache_validate:
            for block_id, block in enumerate(self.transformer.blocks):
                stats = block.attn1.cache_validation
                if stats is not None:
                    m = stats['metrics'].cpu().tolist()
                    logger.info("C_CACHE_VALIDATION %s", json.dumps(dict(
                        mode=self._cache_mode, block=block_id, calls=stats['calls'],
                        cache_checks=stats['cache_checks'], max_abs=m[0],
                        relative_l2=(m[1] / max(m[2], 1e-30)) ** 0.5,
                        not_close_fraction=m[3] / stats['elements'],
                        nonfinite=int(m[4]), cache_mismatches=int(m[5]),
                        not_bitwise_fraction=m[6] / stats['elements'],
                        atol=1e-2, rtol=1e-2)))
        if self._chunk_timing:
            logger.info("C_MEMORY_METRIC %s", json.dumps(dict(
                mode=self._cache_mode,
                max_allocated_GiB=torch.cuda.max_memory_allocated(self.device) / 2**30,
                max_reserved_GiB=torch.cuda.max_memory_reserved(self.device) / 2**30)))
        if self._cache_mode == "lifecycle":
            selected_blocks = (0, len(self.transformer.blocks) // 2,
                               len(self.transformer.blocks) - 1)
            for block_id in selected_blocks:
                cache = self.transformer.blocks[block_id].attn1.attn_caches[
                    self.cache_name]
                logger.info("D_CACHE_STATE %s", json.dumps(dict(
                    block=block_id,
                    observed_end=cache['observed_end'],
                    committed_end=cache['committed_end'],
                    temporary_end=cache['temporary_end'],
                    capacity=cache['capacity'],
                    **cache['stats'])))
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()

        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)

        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):

    config = VA_CONFIGS[args.config_name]
    if args.config_name == 'libero':
        validate_libero_checkpoint_environment(config)
        logger.info(
            "Checkpoint-aligned LIBERO evaluation: suite=%s checkpoint=%s id=%s",
            config.libero_eval_suite,
            config.wan22_pretrained_model_name_or_path,
            config.libero_checkpoint_id,
        )
    seed_env = os.getenv("LINGBOT_SEED")
    if seed_env is not None:
        seed = int(seed_env)
        torch.manual_seed(seed)
        np.random.seed(seed)
        logger.info("Deterministic A/B seed: %d", seed)
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()

"""Opt-in P0 trace recording for WAM denoising trajectories.

This module is deliberately isolated from ordinary inference.  It is imported by
``wan_va_server.py`` but only allocates buffers or writes files when
``WAM_P0_TRACE_DIR`` is set.  Trace runs are diagnostic runs: CPU copies and
file I/O make them unsuitable for latency measurement.

The saved payload is a local, trusted PyTorch file with schema
``wam-p0-trace/v3``.  It captures the CFG-combined direction actually consumed
by ``FlowMatchScheduler.step`` as well as the conditional and unconditional
components where they exist.  For every update it distinguishes the scheduler
input ``x_i``, raw scheduler output ``x_next_raw``, post-update state
``x_next``, and the deterministic ``input_preparation`` applied before the
following model/scheduler input.  This makes fixed observations and masked
action channels auditable rather than looking like unexplained trajectory
discontinuities.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA_VERSION = "wam-p0-trace/v3"
TRACE_DIR_ENV = "WAM_P0_TRACE_DIR"
TRACE_MODE_ENV = "WAM_P0_TRACE_MODE"
TRACE_RUN_ID_ENV = "WAM_P0_TRACE_RUN_ID"

# ``full`` and ``stats`` are the established diagnostic contracts.  This mode
# is intentionally separate: it saves only the action teacher trajectory and
# raw, re-encodable conditioning needed by a small draft model.  It never saves
# the video denoising trajectory or target-transformer hidden states.
DRAFT_TRAIN_TRACE_MODE = "draft_train_v1"
DRAFT_TRAIN_TRANSFORMER_OUTPUT_TRACE_MODE = "draft_train_v2_transformer_output"
DRAFT_TRAIN_CONTRACT_VERSION = "wam-p0-action-draft/v1"
DRAFT_TRAIN_TRANSFORMER_OUTPUT_CONTRACT_VERSION = "wam-p0-action-draft/v2-transformer-output"
VALID_TRACE_MODES = frozenset({
    "full",
    "stats",
    DRAFT_TRAIN_TRACE_MODE,
    DRAFT_TRAIN_TRANSFORMER_OUTPUT_TRACE_MODE,
})


def _utc_timestamp() -> str:
    """Return a timezone-explicit timestamp suitable for trace metadata."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sanitize_component(value: str) -> str:
    """Keep run IDs and filenames portable without silently accepting empty IDs."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip(".-")
    return normalized or "p0-trace"


def _to_scalar(value: Any) -> float | int | bool | None:
    """Convert scalar tensor/number values to serialization-friendly scalars."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got shape={tuple(value.shape)}")
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    return float(value)


def _json_safe(value: Any) -> Any:
    """Convert small metadata trees to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _to_scalar(value)
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    """Write JSON atomically so a killed trace run cannot leave a fake manifest."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _copy_cpu_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Return a detached, non-aliased contiguous CPU tensor."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"trace input must be Tensor or None, got {type(tensor)!r}")
    # ``copy=True`` prevents an accidental alias if this recorder is used from
    # a CPU-only unit test. Trace runs may synchronize here by design.
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    """Compute a compact summary from an already-CPU tensor."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected Tensor or None, got {type(tensor)!r}")
    flattened = tensor.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": 0,
            "l2": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
        }
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(flattened.numel()),
        "l2": float(torch.linalg.vector_norm(flattened).item()),
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
        "max_abs": float(flattened.abs().max().item()),
    }


def _snapshot_condition_value(value: Any) -> Any:
    """Recursively copy trusted raw conditioning to CPU tensors.

    The compact draft-training contract persists only inputs that a future tiny
    draft can encode itself (prompt, raw camera RGB, and any explicit numeric
    condition).  It deliberately cannot serialize target-model hidden states.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _snapshot_condition_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_condition_value(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True).contiguous()
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(
            "draft-training condition must be JSON scalar, mapping/list, Tensor, or numeric array; "
            f"got {type(value)!r}"
        ) from exc
    return tensor.detach().to(device="cpu", copy=True).contiguous()


@dataclass
class P0TraceChunk:
    """In-memory trace for one WAM inference chunk.

    The source code only commits a chunk after the inference chunk returns
    normally.  An interrupted request therefore leaves no partially valid
    ``.pt`` trace that analysis could mistake for complete data.
    """

    recorder: "P0TraceRecorder"
    ordinal: int
    metadata: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    # Populated only by ``draft_train_v1``.  It is deliberately outside
    # ``chunk`` metadata because it contains CPU tensors rather than JSON.
    condition: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    _closed: bool = False

    def _snapshot(self, tensor: torch.Tensor | None) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
        """Copy an input tensor to CPU without mutating inference tensors."""
        cpu_tensor = _copy_cpu_tensor(tensor)
        if cpu_tensor is None:
            return None, None
        if self.recorder.mode == "full":
            return cpu_tensor, _tensor_summary(cpu_tensor)
        return None, _tensor_summary(cpu_tensor)

    def _append(self, record: dict[str, Any], tensors: Mapping[str, torch.Tensor | None]) -> None:
        """Attach snapshots and summaries; record capture failures but do not raise."""
        try:
            for name, tensor in tensors.items():
                snapshot, summary = self._snapshot(tensor)
                if self.recorder.mode == "full":
                    record[name] = snapshot
                record[f"{name}_summary"] = summary
            self.steps.append(record)
        except Exception as exc:  # diagnostic I/O must not corrupt inference
            self.errors.append(f"capture failure for {record.get('branch')}/{record.get('step_index')}: {exc!r}")

    def record_draft_train_condition(
        self,
        *,
        prompt: str | None,
        observation: Any,
        action_condition: torch.Tensor | None,
        observation_camera_keys: list[str] | tuple[str, ...],
    ) -> None:
        """Save re-encodable inputs for one compact action-draft sample.

        A target-only rollout supplies the teacher direction in each action
        update.  The saved condition is raw RGB plus prompt and the exact
        action-prefix condition; it does *not* claim that target KV cache or
        hidden states can be reconstructed from this trace.
        """
        if not self.recorder.is_draft_training:
            return
        if self.condition is not None:
            self.errors.append("duplicate draft-training condition capture ignored")
            return
        try:
            self.condition = {
                "contract_version": self.recorder.draft_training_contract_version,
                "kind": "raw_rgb_prompt_action_prefix",
                "prompt": prompt,
                "observation_camera_keys": [str(key) for key in observation_camera_keys],
                "observation": _snapshot_condition_value(observation),
                "action_condition": _snapshot_condition_value(action_condition),
                "target_hidden_state": "not_captured",
                "target_kv_cache": "not_captured",
            }
        except Exception as exc:  # trace must not perturb a real policy response
            self.errors.append(f"draft-training condition capture failure: {exc!r}")

    def _append_draft_train_action_update(
        self,
        record: dict[str, Any],
        *,
        x_i: torch.Tensor,
        d_cfg: torch.Tensor,
        x_next: torch.Tensor,
        transformer_output_raw: torch.Tensor | None,
    ) -> None:
        """Persist one compact action teacher transition.

        ``draft_train_v1`` stores only the scheduler teacher label.  The
        versioned v2 contract additionally stores the exact pre-reshape action
        Transformer output, allowing later experiments to train a draft on the
        Transformer head output itself and to reconstruct the CFG branches.
        """
        try:
            # Keep the transition tensor names shared with ``full`` so loaders
            # can operate on either contract.  Video tensors, terminal forwards,
            # and target hidden/KV state remain excluded in both compact modes.
            record["x_i"] = _copy_cpu_tensor(x_i)
            record["d_cfg"] = _copy_cpu_tensor(d_cfg)
            record["x_next"] = _copy_cpu_tensor(x_next)
            if any(record[name] is None for name in ("x_i", "d_cfg", "x_next")):
                raise ValueError("draft-training action transition requires x_i, d_cfg, and x_next")
            if self.recorder.is_transformer_output and transformer_output_raw is not None:
                raw = _copy_cpu_tensor(transformer_output_raw)
                if raw is None:
                    raise ValueError("v2 transformer-output transition requires raw output")
                record["transformer_output_raw"] = raw
                record["transformer_output_layout"] = "b_(f*n)_c"
                record["transformer_output_source"] = "action_transformer_before_layout_and_cfg"
            elif self.recorder.is_transformer_output:
                raise ValueError("v2 transformer-output transition requires raw action Transformer output")
            self.steps.append(record)
        except Exception as exc:  # diagnostic I/O must not corrupt inference
            self.errors.append(
                f"draft-training capture failure for action/{record.get('step_index')}: {exc!r}"
            )

    def record_scheduler_update(
        self,
        *,
        branch: str,
        step_index: int,
        timestep: Any,
        sigma: Any,
        next_sigma: Any,
        cfg_scale: float,
        cfg_active: bool,
        cache_update: bool,
        x_i: torch.Tensor,
        d_cond: torch.Tensor | None,
        d_uncond: torch.Tensor | None,
        d_cfg: torch.Tensor,
        x_next_raw: torch.Tensor,
        x_next: torch.Tensor,
        projection: Mapping[str, Any] | None,
        input_preparation: Mapping[str, Any] | None,
        model_source: str = "target",
        transformer_output_raw: torch.Tensor | None = None,
    ) -> None:
        """Record one real Euler/flow-matching update after CFG composition.

        ``x_next_raw`` is captured immediately after ``scheduler.step`` and
        before the server applies its known post-update observation clamp.
        ``x_next`` is the state retained by the server after that clamp.
        ``input_preparation`` describes the deterministic mutation made before
        this step's model/scheduler input (for example, action-channel masking).
        The next record's preparation can therefore be replayed against this
        record's ``x_next`` by the offline analyzer.
        """
        if model_source not in {"target", "draft"}:
            raise ValueError(f"model_source must be target or draft, got {model_source!r}")
        transition_scale = _to_scalar(next_sigma) - _to_scalar(sigma)
        if self.recorder.is_draft_training:
            # The training corpus is target-only by contract.  Skip video and
            # online-draft rows rather than blending them into teacher labels.
            if branch != "action" or model_source != "target":
                return
            compact_record: dict[str, Any] = {
                "kind": "scheduler_update",
                "branch": "action",
                "model_source": "target",
                "teacher_label": "cfg_combined_direction",
                "step_index": int(step_index),
                "timestep": _to_scalar(timestep),
                "sigma": _to_scalar(sigma),
                "next_sigma": _to_scalar(next_sigma),
                "transition_scale": float(transition_scale),
                "cfg_scale": float(cfg_scale),
                "cfg_active": bool(cfg_active),
                "cache_update": bool(cache_update),
                "scheduler_updated": True,
                "projection": _json_safe(dict(projection or {"kind": "identity"})),
                "input_preparation": _json_safe(
                    dict(input_preparation or {"kind": "identity"})
                ),
            }
            self._append_draft_train_action_update(
                compact_record,
                x_i=x_i,
                d_cfg=d_cfg,
                x_next=x_next,
                transformer_output_raw=transformer_output_raw,
            )
            return
        record: dict[str, Any] = {
            "kind": "scheduler_update",
            "branch": branch,
            "model_source": model_source,
            "step_index": int(step_index),
            "timestep": _to_scalar(timestep),
            "sigma": _to_scalar(sigma),
            "next_sigma": _to_scalar(next_sigma),
            "transition_scale": float(transition_scale),
            "cfg_scale": float(cfg_scale),
            "cfg_active": bool(cfg_active),
            "cache_update": bool(cache_update),
            "scheduler_updated": True,
            # This describes deterministic state projection performed by the
            # server after ``scheduler.step`` (e.g. fixed observed frame 0).
            "projection": _json_safe(dict(projection or {"kind": "identity"})),
            # This describes deterministic mutation already applied while
            # preparing this update's actual scheduler input ``x_i``.
            "input_preparation": _json_safe(
                dict(input_preparation or {"kind": "identity"})
            ),
        }
        self._append(
            record,
            {
                "x_i": x_i,
                "d_cond": d_cond,
                "d_uncond": d_uncond,
                "d_cfg": d_cfg,
                "x_next_raw": x_next_raw,
                "x_next": x_next,
            },
        )

    def record_terminal_forward(
        self,
        *,
        branch: str,
        step_index: int,
        timestep: Any,
        sigma: Any,
        cfg_scale: float,
        cfg_active: bool,
        raw_model_output: torch.Tensor,
        x_i: torch.Tensor,
        cache_update: bool,
        input_preparation: Mapping[str, Any] | None,
    ) -> None:
        """Record the terminal cache-update forward that has no scheduler update.

        Its raw transformer output is intentionally kept separate from
        ``d_cfg``: it was not reshaped/CFG-combined nor passed to the scheduler,
        so analyses must not count it as a skip-eligible denoising transition.
        """
        if self.recorder.is_draft_training:
            # The padded t=0 cache-maintenance forward has no scheduler target.
            return
        record: dict[str, Any] = {
            "kind": "terminal_forward",
            "branch": branch,
            "step_index": int(step_index),
            "timestep": _to_scalar(timestep),
            "sigma": _to_scalar(sigma),
            "next_sigma": None,
            "transition_scale": None,
            "cfg_scale": float(cfg_scale),
            "cfg_active": bool(cfg_active),
            "cache_update": bool(cache_update),
            "scheduler_updated": False,
            "input_preparation": _json_safe(
                dict(input_preparation or {"kind": "identity"})
            ),
        }
        self._append(record, {"x_i": x_i, "raw_model_output": raw_model_output})

    def record_shadow_candidate_block(
        self,
        *,
        anchor_step_index: int,
        predictor: str,
        anchor_sigma: float,
        previous_sigma: float | None,
        candidate_timesteps: tuple[float, ...] | list[float],
        candidate_sigmas: tuple[float, ...] | list[float],
        transition_scales: tuple[float, ...] | list[float],
        cfg_scale: float,
        cfg_active: bool,
        use_cfg: bool,
        cache_mode: str,
        cache_row_mapping: list[int],
        frame_st_id: int,
        projection: Mapping[str, Any],
        input_preparation: Mapping[str, Any],
        first_candidate_matches_live_successor_max_abs: float,
        first_candidate_matches_live_successor_rel_l2: float,
        anchor_state: torch.Tensor,
        anchor_direction: torch.Tensor,
        previous_direction: torch.Tensor | None,
        candidate_states: torch.Tensor,
        draft_next_states: torch.Tensor,
        target_directions: torch.Tensor,
        draft_channel_ids: list[int] | tuple[int, ...] | None = None,
        draft_checkpoint: str | None = None,
    ) -> None:
        """Persist one shadow-only candidate batch and its live target outputs."""
        candidate_count = int(candidate_states.shape[0])
        record: dict[str, Any] = {
            "kind": "shadow_candidate_block",
            "branch": "action",
            "step_index": int(anchor_step_index),
            "anchor_step_index": int(anchor_step_index),
            "predictor": str(predictor),
            "draft_checkpoint": None if draft_checkpoint is None else str(draft_checkpoint),
            "draft_channel_ids": (
                None if draft_channel_ids is None
                else [int(value) for value in draft_channel_ids]
            ),
            "anchor_sigma": float(anchor_sigma),
            "previous_sigma": (
                None if previous_sigma is None else float(previous_sigma)
            ),
            "candidate_timesteps": [float(value) for value in candidate_timesteps],
            "candidate_sigmas": [float(value) for value in candidate_sigmas],
            "transition_scales": [float(value) for value in transition_scales],
            "candidate_count": candidate_count,
            "target_batch_forward_calls": 1,
            "cfg_scale": float(cfg_scale),
            "cfg_active": bool(cfg_active),
            "use_cfg": bool(use_cfg),
            "cache_mode": str(cache_mode),
            "cache_row_mapping": [int(value) for value in cache_row_mapping],
            "frame_st_id": int(frame_st_id),
            "projection": _json_safe(dict(projection)),
            "input_preparation": _json_safe(dict(input_preparation)),
            "first_candidate_matches_live_successor_max_abs": float(
                first_candidate_matches_live_successor_max_abs
            ),
            "first_candidate_matches_live_successor_rel_l2": float(
                first_candidate_matches_live_successor_rel_l2
            ),
            "scheduler_updated": False,
            "acceptance_decision_made": False,
        }
        self._append(
            record,
            {
                "anchor_state": anchor_state,
                "anchor_direction": anchor_direction,
                "previous_direction": previous_direction,
                "candidate_states": candidate_states,
                "draft_next_states": draft_next_states,
                "target_directions": target_directions,
            },
        )

    def record_k2_b2_verifier_tree(
        self,
        *,
        step_index: int,
        selector: str,
        threshold: float,
        selected_parent: int,
        selected_child: int,
        accepted_depth: int,
        first_costs: list[float] | tuple[float, float],
        second_costs: list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
        target_timesteps: list[float] | tuple[float, ...],
        root_state: torch.Tensor,
        first_directions: torch.Tensor,
        first_states: torch.Tensor,
        second_directions: torch.Tensor,
        leaf_states: torch.Tensor,
        target_directions: torch.Tensor,
        bonus_target_step_executed: bool = False,
        bonus_target_step_index: int | None = None,
        cfg_multiplier: int = 1,
    ) -> None:
        """Persist one online K=2/B=2 7-node verifier decision.

        This is an audit record, not a scheduler update.  The two real updates
        (or target fallback) are separately written through
        :meth:`record_scheduler_update`, preserving the actual committed
        trajectory and source labels.  The seven target rows stay together so
        later analysis can prove that greedy/global selection consumed exactly
        one batched verifier result in the documented node order.
        """
        if selector not in {"greedy", "global"}:
            raise ValueError(f"K=2/B=2 selector must be greedy or global, got {selector!r}")
        if accepted_depth not in {0, 1, 2}:
            raise ValueError("K=2/B=2 accepted_depth must be 0, 1, or 2")
        if not (0 <= int(selected_parent) < 2 and 0 <= int(selected_child) < 2):
            raise ValueError("K=2/B=2 selected route must lie in the 2x2 tree")
        if bonus_target_step_executed and accepted_depth != 2:
            raise ValueError("K=2/B=2 bonus target step requires accepted_depth=2")
        if bonus_target_step_executed and bonus_target_step_index is None:
            raise ValueError("K=2/B=2 bonus target step requires its scheduler index")
        if not bonus_target_step_executed and bonus_target_step_index is not None:
            raise ValueError("K=2/B=2 bonus scheduler index requires an executed bonus step")
        if target_directions.shape[0] != 7:
            raise ValueError("K=2/B=2 verifier must persist exactly seven target rows")
        if cfg_multiplier not in {1, 2}:
            raise ValueError("K=2/B=2 CFG multiplier must be 1 or 2")
        fallback_used = accepted_depth < 2
        record: dict[str, Any] = {
            "kind": "k2_b2_verifier_tree",
            "branch": "action",
            "step_index": int(step_index),
            "selector": str(selector),
            "threshold": float(threshold),
            "selected_parent": int(selected_parent),
            "selected_child": int(selected_child),
            "accepted_depth": int(accepted_depth),
            "decision_schema_version": "wam-p0-verifier-decision/v1",
            "draft_tree_depth": 2,
            "accepted_prefix_depth": int(accepted_depth),
            "first_rejected_edge_index": int(accepted_depth) if fallback_used else None,
            "target_fallback_used": bool(fallback_used),
            "target_fallback_step_index": (
                int(step_index + accepted_depth) if fallback_used else None),
            "candidate_rows": {
                "logical": int(target_directions.shape[0]),
                "cfg_multiplier": int(cfg_multiplier),
                "transformer_input": int(target_directions.shape[0] * cfg_multiplier),
            },
            "bonus_target_step_executed": bool(bonus_target_step_executed),
            "bonus_target_step_index": (
                int(bonus_target_step_index)
                if bonus_target_step_index is not None else None),
            "first_costs": [float(value) for value in first_costs],
            "second_costs": [[float(value) for value in row] for row in second_costs],
            "target_timesteps": [float(value) for value in target_timesteps],
            "target_batch_rows": 7,
            "target_batch_forward_calls": 1,
            "scheduler_updated": False,
            "acceptance_decision_made": True,
        }
        self._append(
            record,
            {
                "root_state": root_state,
                "first_directions": first_directions,
                "first_states": first_states,
                "second_directions": second_directions,
                "leaf_states": leaf_states,
                "target_directions": target_directions,
            },
        )

    def record_k3_b2_verifier_tree(
        self,
        *,
        step_index: int,
        selector: str,
        threshold: float,
        selected_parent: int,
        selected_child: int,
        selected_grandchild: int,
        accepted_depth: int,
        first_costs: list[float] | tuple[float, float],
        second_costs: list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
        third_costs: list[list[list[float]]] | tuple[
            tuple[tuple[float, float], tuple[float, float]],
            tuple[tuple[float, float], tuple[float, float]],
        ],
        target_timesteps: list[float] | tuple[float, ...],
        root_state: torch.Tensor,
        first_directions: torch.Tensor,
        first_states: torch.Tensor,
        second_directions: torch.Tensor,
        second_states: torch.Tensor,
        third_directions: torch.Tensor,
        third_states: torch.Tensor,
        target_directions: torch.Tensor,
        cfg_multiplier: int = 1,
    ) -> None:
        """Persist one K=3/B=2 decision and its seven-row verifier contract.

        The 7 target rows are intentionally only root, depth-1, and depth-2
        states.  They verify all three draft edges; depth-3 states are not
        target-prefetched and are present only as draft commitment candidates.
        """
        if selector not in {"greedy", "global"}:
            raise ValueError(f"K=3/B=2 selector must be greedy or global, got {selector!r}")
        if accepted_depth not in {0, 1, 2, 3}:
            raise ValueError("K=3/B=2 accepted_depth must be 0, 1, 2, or 3")
        if not all(0 <= int(value) < 2 for value in (
                selected_parent, selected_child, selected_grandchild)):
            raise ValueError("K=3/B=2 selected route must lie in the 2x2x2 tree")
        if target_directions.shape[0] != 7:
            raise ValueError("K=3/B=2 verifier must persist exactly seven target rows")
        if cfg_multiplier not in {1, 2}:
            raise ValueError("K=3/B=2 CFG multiplier must be 1 or 2")
        fallback_used = accepted_depth < 3
        record: dict[str, Any] = {
            "kind": "k3_b2_verifier_tree",
            "branch": "action",
            "step_index": int(step_index),
            "selector": str(selector),
            "threshold": float(threshold),
            "selected_parent": int(selected_parent),
            "selected_child": int(selected_child),
            "selected_grandchild": int(selected_grandchild),
            "accepted_depth": int(accepted_depth),
            "decision_schema_version": "wam-p0-verifier-decision/v1",
            "draft_tree_depth": 3,
            "accepted_prefix_depth": int(accepted_depth),
            "first_rejected_edge_index": int(accepted_depth) if fallback_used else None,
            "target_fallback_used": bool(fallback_used),
            "target_fallback_step_index": (
                int(step_index + accepted_depth) if fallback_used else None),
            "candidate_rows": {
                "logical": int(target_directions.shape[0]),
                "cfg_multiplier": int(cfg_multiplier),
                "transformer_input": int(target_directions.shape[0] * cfg_multiplier),
            },
            "first_costs": [float(value) for value in first_costs],
            "second_costs": [[float(value) for value in row] for row in second_costs],
            "third_costs": [
                [[float(value) for value in child] for child in parent]
                for parent in third_costs
            ],
            "target_timesteps": [float(value) for value in target_timesteps],
            "target_batch_rows": 7,
            "target_batch_forward_calls": 1,
            "scheduler_updated": False,
            "acceptance_decision_made": True,
        }
        self._append(
            record,
            {
                "root_state": root_state,
                "first_directions": first_directions,
                "first_states": first_states,
                "second_directions": second_directions,
                "second_states": second_states,
                "third_directions": third_directions,
                "third_states": third_states,
                "target_directions": target_directions,
            },
        )

    def record_shadow_candidate_skip(
        self, *, anchor_step_index: int, predictor: str, reason: str
    ) -> None:
        """Record why a selected anchor could not form a candidate block."""
        self.steps.append({
            "kind": "shadow_candidate_skip",
            "branch": "action",
            "step_index": int(anchor_step_index),
            "anchor_step_index": int(anchor_step_index),
            "predictor": str(predictor),
            "reason": str(reason),
            "scheduler_updated": False,
            "acceptance_decision_made": False,
        })

    def record_shadow_candidate_error(
        self, *, anchor_step_index: int, predictor: str, error: str
    ) -> None:
        """Record an opt-in capture error without changing target inference."""
        self.steps.append({
            "kind": "shadow_candidate_error",
            "branch": "action",
            "step_index": int(anchor_step_index),
            "anchor_step_index": int(anchor_step_index),
            "predictor": str(predictor),
            "error": str(error),
            "scheduler_updated": False,
            "acceptance_decision_made": False,
        })

    def commit(self) -> Path | None:
        """Atomically persist this trace.  Calling it twice is an error."""
        if self._closed:
            raise RuntimeError("P0 trace chunk has already been committed")
        self._closed = True
        # Preserve the established ``full`` / ``stats`` payload shape exactly.
        # ``condition`` contains tensor-valued raw observations, so it belongs
        # only to the explicitly versioned compact training contract.
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_timestamp(),
            "run": self.recorder.run_metadata,
            "chunk": _json_safe(self.metadata),
            "steps": self.steps,
            "capture_errors": list(self.errors),
        }
        if self.recorder.is_draft_training:
            payload["draft_training_contract_version"] = self.recorder.draft_training_contract_version
            payload["condition"] = self.condition
        output_path = self.recorder.next_chunk_path(self.ordinal, self.metadata)
        temporary = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(payload, temporary)
            os.replace(temporary, output_path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return output_path


@dataclass
class P0TraceRecorder:
    """Factory and owner for opt-in P0 chunk traces."""

    root: Path
    run_id: str
    mode: str
    run_dir: Path
    run_metadata: dict[str, Any]
    _next_ordinal: int = 0

    @classmethod
    def from_env(cls) -> "P0TraceRecorder | None":
        """Return ``None`` unless a user explicitly enabled a trace directory."""
        root_value = os.getenv(TRACE_DIR_ENV, "").strip()
        if not root_value:
            return None
        mode = os.getenv(TRACE_MODE_ENV, "full").strip().lower()
        if mode not in VALID_TRACE_MODES:
            expected = ", ".join(sorted(VALID_TRACE_MODES))
            raise ValueError(f"{TRACE_MODE_ENV} must be one of {expected}, got {mode!r}")
        if mode in {DRAFT_TRAIN_TRACE_MODE, DRAFT_TRAIN_TRANSFORMER_OUTPUT_TRACE_MODE}:
            draft_mode = os.getenv("WAM_P0_ACTION_DRAFT_MODE", "off").strip().lower()
            if draft_mode not in {"", "off"}:
                raise ValueError(
                    f"{DRAFT_TRAIN_TRACE_MODE} requires target-only collection; "
                    f"WAM_P0_ACTION_DRAFT_MODE={draft_mode!r} is not permitted"
                )

        root = Path(root_value).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        default_run_id = f"p0-{time.strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}"
        run_id = _sanitize_component(os.getenv(TRACE_RUN_ID_ENV, default_run_id))
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        selected_environment = {
            name: os.getenv(name)
            for name in (
                "LINGBOT_SEED",
                "LINGBOT_DISABLE_FSDP",
                "LINGBOT_CACHE_MODE",
                "LINGBOT_ATTN_MODE",
                "LINGBOT_VA_LIBERO_SUITE",
                "LINGBOT_VA_LIBERO_CHECKPOINT",
                "LINGBOT_VA_LIBERO_CHECKPOINT_ID",
                "LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY",
                "WAM_P0_TRACE_DIR",
                "WAM_P0_TRACE_MODE",
                "WAM_P0_TRACE_RUN_ID",
                "WAM_P0_SHADOW_CANDIDATE_DEPTH",
                "WAM_P0_SHADOW_PREDICTOR",
                "WAM_P0_SHADOW_ANCHOR_STRIDE",
                "WAM_P0_VERIFY_SCRIPT",
                "WAM_P0_ACTION_DRAFT_MODE",
                "WAM_P0_ACTION_DRAFT_CHECKPOINT",
                "WAM_P0_ACTION_DRAFT_B2_CHECKPOINT",
                "WAM_P0_ACTION_DRAFT_SELECTOR",
                "WAM_P0_ACTION_DRAFT_THRESHOLD",
                "WAM_P0_TASK_ID",
                "WAM_P0_SOURCE_REVISION",
                "WAM_P0_TRAINING_CONFIG_HASH",
            )
            if os.getenv(name) is not None
        }
        run_metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_timestamp(),
            "run_id": run_id,
            "trace_mode": mode,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "rank": int(os.getenv("RANK", "0")),
            "environment": selected_environment,
        }
        manifest_path = run_dir / "run_metadata.json"
        if not manifest_path.exists():
            _atomic_json_dump(run_metadata, manifest_path)
        return cls(
            root=root,
            run_id=run_id,
            mode=mode,
            run_dir=run_dir,
            run_metadata=run_metadata,
        )

    @property
    def is_draft_training(self) -> bool:
        """Whether this recorder writes a compact action-teacher contract."""
        return self.mode in {
            DRAFT_TRAIN_TRACE_MODE,
            DRAFT_TRAIN_TRANSFORMER_OUTPUT_TRACE_MODE,
        }

    @property
    def is_transformer_output(self) -> bool:
        """Whether compact rows also persist raw action Transformer output."""
        return self.mode == DRAFT_TRAIN_TRANSFORMER_OUTPUT_TRACE_MODE

    @property
    def draft_training_contract_version(self) -> str:
        """Return the explicit version associated with the selected compact mode."""
        if self.mode == DRAFT_TRAIN_TRACE_MODE:
            return DRAFT_TRAIN_CONTRACT_VERSION
        if self.mode == DRAFT_TRAIN_TRANSFORMER_OUTPUT_TRACE_MODE:
            return DRAFT_TRAIN_TRANSFORMER_OUTPUT_CONTRACT_VERSION
        raise RuntimeError(f"mode {self.mode!r} is not a draft-training contract")

    def begin_chunk(self, **metadata: Any) -> P0TraceChunk:
        """Start a new trace chunk without touching model/cache state."""
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        materialized_metadata = {
            "ordinal": ordinal,
            "started_at": _utc_timestamp(),
            **_json_safe(metadata),
        }
        return P0TraceChunk(self, ordinal=ordinal, metadata=materialized_metadata)

    def next_chunk_path(self, ordinal: int, metadata: Mapping[str, Any]) -> Path:
        """Construct a readable, collision-resistant final filename."""
        frame_st_id = metadata.get("frame_st_id", "unknown")
        return self.run_dir / f"chunk_{ordinal:06d}_frame_{int(frame_st_id):06d}.pt"

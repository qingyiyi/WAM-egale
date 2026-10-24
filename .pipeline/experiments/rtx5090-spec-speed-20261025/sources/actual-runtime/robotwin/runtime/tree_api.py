#!/usr/bin/env python3
"""Pure scoring/acceptance contract for batched continuous-action verification.

This module deliberately does not run WAM.  The caller must supply the target
model's outputs evaluated at each draft candidate state.  A batch dimension is
not inferred or synthesized here, so these helpers cannot accidentally treat
a target-only trace output as a candidate-state verification result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Callable, Literal, Sequence

import torch
from torch import nn


EPSILON = 1e-12
Projection = Callable[[torch.Tensor, int], torch.Tensor]
StateTransform = Callable[[torch.Tensor, int], torch.Tensor]
PredictorName = Literal[
    "zoh_direction",
    "sigma_linear",
    "zoh_displacement",
    "autoregressive_mlp",
]
TargetBatchForward = Callable[[torch.Tensor, Sequence[float]], torch.Tensor]


class _AutoregressiveDirectionMLP(nn.Module):
    """Checkpoint-compatible copy of the tiny action-only AR draft network.

    The training pilot intentionally owns the training code.  Keeping this
    inference definition here avoids importing the pilot (and its CLI) from a
    live WAM server while preserving the checkpoint schema.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class AutoregressiveActionPredictor:
    """Lazy-loadable action-only AR predictor used by the shadow builder.

    A prediction is

    ``previous_direction + decoded_residual(current_state, previous_direction,
    sigma, next_sigma, timestep)``.

    The model is deliberately independent of WAM internals: it receives one
    unbatched action state and direction and returns one direction with the
    same shape.  It is a shadow-only draft; this class never evaluates or
    mutates the target model.
    """

    def __init__(
        self,
        model: _AutoregressiveDirectionMLP,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        residual_mean: torch.Tensor,
        residual_std: torch.Tensor,
        *,
        checkpoint_path: str | None = None,
    ) -> None:
        self.model = model.eval()
        self.feature_mean = feature_mean.detach().float().reshape(-1)
        self.feature_std = feature_std.detach().float().reshape(-1).clamp_min(1e-5)
        self.residual_mean = residual_mean.detach().float().reshape(-1)
        self.residual_std = residual_std.detach().float().reshape(-1).clamp_min(1e-5)
        self.checkpoint_path = checkpoint_path
        if self.feature_mean.numel() != model.net[0].in_features:
            raise ValueError("checkpoint feature statistics do not match model input_dim")
        if self.residual_mean.numel() != model.net[-1].out_features:
            raise ValueError("checkpoint residual statistics do not match model output_dim")

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "AutoregressiveActionPredictor":
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"autoregressive draft checkpoint not found: {path}")
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"autoregressive draft checkpoint is not a dict: {path}")
        config = payload.get("model_config")
        if not isinstance(config, dict):
            raise ValueError("autoregressive draft checkpoint has no model_config")
        # v1 checkpoints stored one ``parameterisation`` field.  The matched
        # direct-vs-residual pilot (v2) keeps both model descriptions in the
        # same checkpoint, so accept its explicit residual field as well.
        parameterisation = config.get("parameterisation")
        if parameterisation is None:
            parameterisation = config.get("residual_parameterisation")
        if parameterisation != "previous_direction + residual":
            raise ValueError(
                "unsupported autoregressive draft parameterisation: "
                f"{parameterisation!r}"
            )
        try:
            input_dim = int(config["input_dim"])
            output_dim = int(config["output_dim"])
            hidden_dim = int(config["hidden_dim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid autoregressive draft model_config") from exc
        if min(input_dim, output_dim, hidden_dim) <= 0:
            raise ValueError("autoregressive draft dimensions must be positive")
        model = _AutoregressiveDirectionMLP(input_dim, output_dim, hidden_dim)
        state_dict = payload.get("model")
        if not isinstance(state_dict, dict):
            raise ValueError("autoregressive draft checkpoint has no model state_dict")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"invalid autoregressive draft model weights: {path}") from exc
        required = ("feature_mean", "feature_std", "residual_mean", "residual_std")
        if any(key not in payload for key in required):
            raise ValueError(
                "autoregressive draft checkpoint is missing feature/residual statistics"
            )
        return cls(
            model,
            payload["feature_mean"],
            payload["feature_std"],
            payload["residual_mean"],
            payload["residual_std"],
            checkpoint_path=str(path),
        )

    @torch.no_grad()
    def predict(
        self,
        current_state: torch.Tensor,
        previous_direction: torch.Tensor,
        sigma: float,
        next_sigma: float,
        timestep: float,
    ) -> torch.Tensor:
        if not isinstance(current_state, torch.Tensor) or not isinstance(previous_direction, torch.Tensor):
            raise TypeError("autoregressive predictor inputs must be tensors")
        if current_state.shape != previous_direction.shape or current_state.numel() == 0:
            raise ValueError("current state and previous direction must have the same non-empty shape")
        values = (float(sigma), float(next_sigma), float(timestep))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("autoregressive predictor scalar inputs must be finite")
        if not bool(torch.isfinite(current_state).all()) or not bool(torch.isfinite(previous_direction).all()):
            raise ValueError("autoregressive predictor tensor inputs must be finite")
        output_dim = self.model.net[-1].out_features
        if current_state.numel() != output_dim:
            raise ValueError(
                "checkpoint output dimension does not match action state: "
                f"{output_dim} vs {current_state.numel()} elements"
            )
        device = current_state.device
        model_param = next(self.model.parameters())
        if model_param.device != device:
            self.model.to(device=device)
            model_param = next(self.model.parameters())
        model_dtype = model_param.dtype
        state = current_state.detach().reshape(1, -1).to(device=device, dtype=model_dtype)
        previous = previous_direction.detach().reshape(1, -1).to(device=device, dtype=model_dtype)
        scalars = torch.tensor(
            [[float(sigma), float(next_sigma), float(timestep) / 1000.0]],
            device=device, dtype=model_dtype,
        )
        features = torch.cat((state, previous, scalars), dim=1)
        if features.shape[1] != self.model.net[0].in_features:
            raise ValueError(
                "checkpoint input dimension does not match action state: "
                f"{features.shape[1]} vs {self.model.net[0].in_features}"
            )
        mean = self.feature_mean.to(device=device, dtype=model_dtype)
        std = self.feature_std.to(device=device, dtype=model_dtype)
        residual_mean = self.residual_mean.to(device=device, dtype=model_dtype)
        residual_std = self.residual_std.to(device=device, dtype=model_dtype)
        residual = self.model((features - mean) / std) * residual_std + residual_mean
        result = previous + residual
        return result.reshape(current_state.shape).to(device=device, dtype=current_state.dtype)


class _TwoHeadResidualMLP(nn.Module):
    """Checkpoint-compatible Phase-A K=2/B=2 continuous residual draft.

    The final projection emits a residual center and an antisymmetric residual
    offset.  Decoding therefore returns two genuine continuous proposals,
    ``center + delta`` and ``center - delta``; it is not a vocabulary top-2
    approximation.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, 2 * output_dim)
        self.output_dim = int(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        center, delta = self.head(self.trunk(features)).reshape(
            features.shape[0], 2, self.output_dim
        ).unbind(dim=1)
        return torch.stack((center + delta, center - delta), dim=1)


class TwoHeadResidualActionPredictor:
    """Lazy-loadable Phase-A B=2 action draft for online K=2 verification.

    It receives a single compact action state and the direction actually used
    on the preceding denoising edge.  It returns *two* directions of shape
    ``[2, *state.shape]``.  Each is parameterised as the preceding direction
    plus a independently decoded residual.  This exact interface makes the
    server's tree expansion autoregressive at every depth without allowing a
    draft-only suffix to bypass target verification.
    """

    PARAMETERISATION = "previous_direction + two residual heads"

    def __init__(
        self,
        model: _TwoHeadResidualMLP,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        residual_mean: torch.Tensor,
        residual_std: torch.Tensor,
        *,
        checkpoint_path: str | None = None,
    ) -> None:
        self.model = model.eval()
        self.feature_mean = feature_mean.detach().float().reshape(-1)
        self.feature_std = feature_std.detach().float().reshape(-1).clamp_min(1e-5)
        self.residual_mean = residual_mean.detach().float().reshape(-1)
        self.residual_std = residual_std.detach().float().reshape(-1).clamp_min(1e-5)
        self.checkpoint_path = checkpoint_path
        if self.feature_mean.numel() != model.trunk[0].in_features:
            raise ValueError("B=2 checkpoint feature statistics do not match model input_dim")
        if self.residual_mean.numel() != model.output_dim:
            raise ValueError("B=2 checkpoint residual statistics do not match model output_dim")
        if self.residual_std.numel() != model.output_dim:
            raise ValueError("B=2 checkpoint residual std does not match model output_dim")

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "TwoHeadResidualActionPredictor":
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"K=2/B=2 action draft checkpoint not found: {path}")
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"K=2/B=2 action draft checkpoint is not a dict: {path}")
        config = payload.get("model_config")
        if not isinstance(config, dict):
            raise ValueError("K=2/B=2 action draft checkpoint has no model_config")
        if config.get("parameterisation") != cls.PARAMETERISATION:
            raise ValueError(
                "unsupported K=2/B=2 draft parameterisation: "
                f"{config.get('parameterisation')!r}"
            )
        try:
            input_dim = int(config["input_dim"])
            output_dim = int(config["output_dim"])
            hidden_dim = int(config["hidden_dim"])
            heads = int(config["heads"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid K=2/B=2 action draft model_config") from exc
        if min(input_dim, output_dim, hidden_dim) <= 0 or heads != 2:
            raise ValueError("K=2/B=2 draft must have positive dimensions and exactly two heads")
        for key in ("model_state_dict", "feature_mean", "feature_std", "residual_mean", "residual_std"):
            if key not in payload:
                raise ValueError(f"K=2/B=2 action draft checkpoint missing {key}")
        model = _TwoHeadResidualMLP(input_dim, output_dim, hidden_dim)
        try:
            model.load_state_dict(payload["model_state_dict"], strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("K=2/B=2 action draft state dict is incompatible with model_config") from exc
        return cls(
            model,
            torch.as_tensor(payload["feature_mean"]),
            torch.as_tensor(payload["feature_std"]),
            torch.as_tensor(payload["residual_mean"]),
            torch.as_tensor(payload["residual_std"]),
            checkpoint_path=str(path),
        )

    @torch.no_grad()
    def predict(
        self,
        current_state: torch.Tensor,
        previous_direction: torch.Tensor,
        sigma: float,
        next_sigma: float,
        timestep: float,
    ) -> torch.Tensor:
        if not isinstance(current_state, torch.Tensor) or not isinstance(previous_direction, torch.Tensor):
            raise TypeError("K=2/B=2 predictor inputs must be tensors")
        if current_state.shape != previous_direction.shape or current_state.numel() == 0:
            raise ValueError("current state and previous direction must have the same non-empty shape")
        values = (float(sigma), float(next_sigma), float(timestep))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("K=2/B=2 predictor scalar inputs must be finite")
        if not bool(torch.isfinite(current_state).all()) or not bool(torch.isfinite(previous_direction).all()):
            raise ValueError("K=2/B=2 predictor tensor inputs must be finite")
        if current_state.numel() != self.model.output_dim:
            raise ValueError(
                "K=2/B=2 checkpoint output dimension does not match compact action state: "
                f"{self.model.output_dim} vs {current_state.numel()} elements"
            )
        device = current_state.device
        model_param = next(self.model.parameters())
        if model_param.device != device:
            self.model.to(device=device)
            model_param = next(self.model.parameters())
        dtype = model_param.dtype
        state = current_state.detach().reshape(1, -1).to(device=device, dtype=dtype)
        previous = previous_direction.detach().reshape(1, -1).to(device=device, dtype=dtype)
        scalars = torch.tensor(
            [[float(sigma), float(next_sigma), float(timestep) / 1000.0]],
            device=device, dtype=dtype,
        )
        features = torch.cat((state, previous, scalars), dim=1)
        if features.shape[1] != self.model.trunk[0].in_features:
            raise ValueError(
                "K=2/B=2 checkpoint input dimension does not match compact action state: "
                f"{features.shape[1]} vs {self.model.trunk[0].in_features}"
            )
        normalized = (features - self.feature_mean.to(device=device, dtype=dtype)) / self.feature_std.to(
            device=device, dtype=dtype
        )
        residual = self.model(normalized)
        residual = (
            residual * self.residual_std.to(device=device, dtype=dtype)[None, None, :]
            + self.residual_mean.to(device=device, dtype=dtype)[None, None, :]
        )
        result = previous.unsqueeze(1) + residual
        return result.reshape(2, *current_state.shape).to(device=device, dtype=current_state.dtype)


@dataclass(frozen=True)
class DraftCandidateBlock:
    """Draft states and their raw scheduler successors for one candidate batch.

    ``candidate_states[j]`` is already transformed into the exact state layout
    to be presented to the target at ``target_timesteps[j]``. In contrast,
    ``draft_next_states[j]`` is the *raw* scheduler successor; the verifier
    applies ``post_step_projection`` to both target and draft successors before
    comparing them.
    """

    predictor: str
    candidate_states: torch.Tensor
    draft_next_states: torch.Tensor
    target_timesteps: tuple[float, ...]
    candidate_sigmas: tuple[float, ...]
    transition_scales: tuple[float, ...]
    state_transform: StateTransform | None = field(default=None, repr=False, compare=False)
    draft_directions: torch.Tensor | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CandidateVerification:
    """Outputs from one draft proposal and one target candidate-batch call."""

    draft: DraftCandidateBlock
    target_directions: torch.Tensor
    scores: tuple[StepResidual, ...]
    decision: PrefixDecision


@dataclass(frozen=True)
class StepResidual:
    """One-step local target correction for a draft candidate."""

    step_offset: int
    state_rel_l2: float
    state_l2: float
    action_channel_max_rel_l2: float | None
    action_channel_max_rms: float | None
    action_component_abs_max: float | None
    action_channel_rel_l2: tuple[float, ...]
    action_channel_rms: tuple[float, ...]
    finite: bool


@dataclass(frozen=True)
class RiskThresholds:
    """Explicitly calibrated limits; no uncalibrated defaults are provided.

    ``action_channel_max_rms`` is measured after applying ``action_scales``.
    Those scales should be the linear scale used by the action postprocessor
    (e.g. (q99-q01+1e-6)/2) so the threshold is in decoded action units.
    """

    state_rel_l2: float
    action_channel_max_rms: float
    action_channel_max_rel_l2: float | None = None
    action_component_abs_max: float | None = None

    def __post_init__(self) -> None:
        values = [self.state_rel_l2, self.action_channel_max_rms]
        if self.action_channel_max_rel_l2 is not None:
            values.append(self.action_channel_max_rel_l2)
        if self.action_component_abs_max is not None:
            values.append(self.action_component_abs_max)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("risk thresholds must be finite and non-negative")


@dataclass(frozen=True)
class PrefixDecision:
    """Longest contiguous safe prefix; never a block-average decision."""

    accepted_length: int
    safe_steps: tuple[bool, ...]
    first_rejected_offset: int | None


@dataclass(frozen=True)
class K2B2Tree:
    """A two-step, two-branch continuous draft tree.

    The tree deliberately keeps the target-verifier rows separate from the
    draft edges.  The seven rows sent to the target are ordered as
    ``root, first[0], first[1], leaf[0,0], leaf[0,1], leaf[1,0], leaf[1,1]``.
    The four leaf target directions are a look-ahead cache for the next root;
    the current two-step route cost only contains the six draft edges.
    """

    root_state: torch.Tensor
    root_previous_direction: torch.Tensor
    first_directions: torch.Tensor
    first_states: torch.Tensor
    second_directions: torch.Tensor
    leaf_states: torch.Tensor
    sigma_path: tuple[float, float, float]
    target_timesteps: tuple[float, float, float]
    transition_scales: tuple[float, float]

    def verifier_states(self) -> torch.Tensor:
        """Return the exact seven-state target batch in a fixed order."""
        return torch.cat((
            self.root_state.unsqueeze(0),
            self.first_states,
            self.leaf_states.reshape(4, *self.root_state.shape),
        ), dim=0)

    def verifier_timesteps(self) -> tuple[float, ...]:
        """Return timesteps aligned with :meth:`verifier_states`."""
        t0, t1, t2 = self.target_timesteps
        return (t0, t1, t1, t2, t2, t2, t2)


@dataclass(frozen=True)
class K2B2Scores:
    """Verifier scores for one :class:`K2B2Tree`."""

    first_costs: tuple[float, float]
    second_costs: tuple[tuple[float, float], tuple[float, float]]
    target_directions: torch.Tensor

    @property
    def leaf_target_directions(self) -> torch.Tensor:
        """Target directions at leaves, shaped ``[2, 2, ...]``."""
        return self.target_directions[3:].reshape(2, 2, *self.target_directions.shape[1:])


@dataclass(frozen=True)
class K2B2Route:
    """A selected two-edge route before threshold fallback."""

    parent: int
    child: int
    first_cost: float
    second_cost: float
    total_cost: float
    selector: str


@dataclass(frozen=True)
class K2B2Execution:
    """The actual next state after selection and optional target fallback."""

    next_state: torch.Tensor
    next_previous_direction: torch.Tensor
    accepted_depth: int
    route: K2B2Route
    prefetch_target_direction: torch.Tensor | None


def build_k2_b2_tree(
    *,
    root_state: torch.Tensor,
    previous_direction: torch.Tensor,
    sigma_path: Sequence[float],
    target_timesteps: Sequence[float],
    draft_predictor: Callable[[torch.Tensor, torch.Tensor, float, float, float], torch.Tensor],
    scheduler_step: Callable[[torch.Tensor, torch.Tensor, float, int], torch.Tensor],
) -> K2B2Tree:
    """Expand the exact ``K=2, B=2`` continuous tree.

    ``scheduler_step`` is intentionally a callback rather than an inline Euler
    update.  The live WAM caller must bind it to the real scheduler plus the
    same deterministic prefix/mask projection used by normal inference.  This
    prevents the offline tree helper from silently drifting if scheduler
    behavior changes.

    ``draft_predictor`` receives one unbatched state/direction and must return
    two proposals with shape ``[2, *state.shape]``.  The first proposal pair
    creates two internal states; each internal state creates two leaf states.
    """
    if not isinstance(root_state, torch.Tensor) or not isinstance(previous_direction, torch.Tensor):
        raise TypeError("root_state and previous_direction must be tensors")
    if root_state.shape != previous_direction.shape or root_state.numel() == 0:
        raise ValueError("root_state and previous_direction must have the same non-empty shape")
    sigmas = tuple(float(value) for value in sigma_path)
    timesteps = tuple(float(value) for value in target_timesteps)
    if len(sigmas) != 3 or len(timesteps) != 3:
        raise ValueError("K=2/B=2 requires three sigmas and three target timesteps: root/internal/leaf")
    if not all(math.isfinite(value) for value in (*sigmas, *timesteps)):
        raise ValueError("sigma_path and target_timesteps must be finite")

    def predict(state: torch.Tensor, previous: torch.Tensor, sigma: float, next_sigma: float, timestep: float) -> torch.Tensor:
        result = draft_predictor(state, previous, sigma, next_sigma, timestep)
        if not isinstance(result, torch.Tensor) or result.shape != (2, *state.shape):
            raise ValueError(
                "draft_predictor must return [2, *state.shape], got "
                f"{getattr(result, 'shape', None)} for state={tuple(state.shape)}"
            )
        if not bool(torch.isfinite(result).all()):
            raise ValueError("draft_predictor returned non-finite values")
        return result

    def step(state: torch.Tensor, direction: torch.Tensor, timestep: float, offset: int) -> torch.Tensor:
        result = scheduler_step(state, direction, timestep, offset)
        if not isinstance(result, torch.Tensor) or result.shape != state.shape:
            raise ValueError("scheduler_step must preserve the state shape")
        if not bool(torch.isfinite(result).all()):
            raise ValueError("scheduler_step returned non-finite values")
        return result

    first_directions = predict(
        root_state, previous_direction, sigmas[0], sigmas[1], timesteps[0]
    )
    first_states = torch.stack(tuple(
        step(root_state, first_directions[parent], timesteps[0], 0)
        for parent in range(2)
    ), dim=0)

    second_rows: list[torch.Tensor] = []
    leaf_rows: list[torch.Tensor] = []
    for parent in range(2):
        second = predict(
            first_states[parent], first_directions[parent],
            sigmas[1], sigmas[2], timesteps[1],
        )
        second_rows.append(second)
        leaf_rows.append(torch.stack(tuple(
            step(first_states[parent], second[child], timesteps[1], 1)
            for child in range(2)
        ), dim=0))

    return K2B2Tree(
        root_state=root_state.detach().clone(),
        root_previous_direction=previous_direction.detach().clone(),
        first_directions=first_directions.detach().clone(),
        first_states=first_states.detach().clone(),
        second_directions=torch.stack(second_rows, dim=0).detach().clone(),
        leaf_states=torch.stack(leaf_rows, dim=0).detach().clone(),
        sigma_path=sigmas,
        target_timesteps=timesteps,
        transition_scales=(sigmas[1] - sigmas[0], sigmas[2] - sigmas[1]),
    )


def pack_k2_b2_verifier_batch(tree: K2B2Tree) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Pack the seven target rows and assert their documented alignment."""
    states = tree.verifier_states()
    timesteps = tree.verifier_timesteps()
    if states.shape[0] != 7 or len(timesteps) != 7:
        raise AssertionError("K=2/B=2 verifier batch must contain exactly seven rows")
    return states, timesteps


def _weighted_direction_l2(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scale: float,
    *,
    action_channel_ids: Sequence[int] | None = None,
    channel_axis: int = 0,
) -> float:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target direction shapes must match")
    if not math.isfinite(float(scale)):
        raise ValueError("transition scale must be finite")
    delta = predicted.float() - target.float()
    if action_channel_ids is not None:
        ids = tuple(int(value) for value in action_channel_ids)
        if not ids:
            raise ValueError("action_channel_ids must not be empty")
        axis = channel_axis if channel_axis >= 0 else delta.ndim + channel_axis
        if not 0 <= axis < delta.ndim:
            raise ValueError("channel_axis outside direction rank")
        if min(ids) < 0 or max(ids) >= delta.shape[axis]:
            raise ValueError("action_channel_ids outside direction channel axis")
        delta = delta.index_select(axis, torch.tensor(ids, device=delta.device))
    value = abs(float(scale)) * torch.linalg.vector_norm(delta.reshape(-1))
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("direction error is non-finite")
    return result


def score_k2_b2_tree(
    tree: K2B2Tree,
    target_directions: torch.Tensor,
    *,
    action_channel_ids: Sequence[int] | None = None,
    channel_axis: int = 0,
) -> K2B2Scores:
    """Score all six draft edges from one seven-row target verifier result.

    Rows 0/1..2/3..6 are root/internal/leaf target outputs.  Leaf rows are
    retained in the returned object for next-root prefetch, but are not added
    to the current K=2 route cost because no third draft edge was proposed.
    """
    expected = tree.verifier_states()
    if target_directions.shape != expected.shape:
        raise ValueError(
            "target_directions must match the seven packed verifier states: "
            f"{tuple(target_directions.shape)} vs {tuple(expected.shape)}"
        )
    root_target = target_directions[0]
    internal_targets = target_directions[1:3]
    first = tuple(_weighted_direction_l2(
        tree.first_directions[parent], root_target, tree.transition_scales[0],
        action_channel_ids=action_channel_ids, channel_axis=channel_axis,
    ) for parent in range(2))
    second = tuple(tuple(_weighted_direction_l2(
        tree.second_directions[parent, child], internal_targets[parent],
        tree.transition_scales[1], action_channel_ids=action_channel_ids,
        channel_axis=channel_axis,
    ) for child in range(2)) for parent in range(2))
    return K2B2Scores(first_costs=first, second_costs=second,
                      target_directions=target_directions.detach().clone())


def select_k2_b2_route(scores: K2B2Scores, *, selector: Literal["greedy", "global"]) -> K2B2Route:
    """Select a route using the same verified seven-row result."""
    if selector == "greedy":
        parent = min(range(2), key=lambda index: (scores.first_costs[index], index))
        child = min(range(2), key=lambda index: (scores.second_costs[parent][index], index))
    elif selector == "global":
        parent, child = min(
            ((p, c) for p in range(2) for c in range(2)),
            key=lambda pair: (
                scores.first_costs[pair[0]] + scores.second_costs[pair[0]][pair[1]], pair[0], pair[1]
            ),
        )
    else:
        raise ValueError(f"selector must be greedy or global, got {selector!r}")
    first = float(scores.first_costs[parent])
    second = float(scores.second_costs[parent][child])
    return K2B2Route(parent=parent, child=child, first_cost=first,
                     second_cost=second, total_cost=first + second,
                     selector=selector)


def execute_k2_b2_route(
    tree: K2B2Tree,
    scores: K2B2Scores,
    route: K2B2Route,
    *,
    threshold: float,
    scheduler_step: Callable[[torch.Tensor, torch.Tensor, float, int], torch.Tensor],
) -> K2B2Execution:
    """Commit prefix 0/1/2 with target correction on the first rejected edge.

    Prefix 2 uses the selected leaf exactly as generated by the draft and
    exposes that leaf's target direction as a *prefetch only*.  Prefix 0/1
    invalidate prefetch because the corrected state is not a draft leaf.
    """
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    root_target = scores.target_directions[0]
    internal_target = scores.target_directions[1 + route.parent]
    if route.first_cost > threshold:
        next_state = scheduler_step(tree.root_state, root_target, tree.target_timesteps[0], 0)
        return K2B2Execution(
            next_state=next_state, next_previous_direction=root_target.detach().clone(),
            accepted_depth=0, route=route, prefetch_target_direction=None,
        )
    if route.second_cost > threshold:
        next_state = scheduler_step(
            tree.first_states[route.parent], internal_target, tree.target_timesteps[1], 1
        )
        return K2B2Execution(
            next_state=next_state, next_previous_direction=internal_target.detach().clone(),
            accepted_depth=1, route=route, prefetch_target_direction=None,
        )
    return K2B2Execution(
        next_state=tree.leaf_states[route.parent, route.child].detach().clone(),
        next_previous_direction=tree.second_directions[route.parent, route.child].detach().clone(),
        accepted_depth=2, route=route,
        prefetch_target_direction=scores.target_directions[3 + route.parent * 2 + route.child].detach().clone(),
    )


def select_k2_b2_trace_transitions(
    tree_transitions: Sequence[object],
    *,
    route: K2B2Route,
    accepted_depth: int,
    fallback_transition: object | None,
) -> tuple[object, ...]:
    """Bind trace updates to known K=2/B=2 scheduler-call order.

    ``build_k2_b2_tree`` calls the live scheduler exactly six times in order
    ``first[0], first[1], leaf[0,0], leaf[0,1], leaf[1,0], leaf[1,1]``.
    A prefix-0 or prefix-1 rollback adds one distinct, target-corrected
    fallback call.  Returning the required transition objects directly keeps
    tracing independent of tensor-value equality (and hence of GPU sync).
    """
    if len(tree_transitions) != 6:
        raise ValueError(
            "K=2/B=2 trace binding requires exactly six tree transitions, got "
            f"{len(tree_transitions)}"
        )
    if accepted_depth not in (0, 1, 2):
        raise ValueError("accepted_depth must be 0, 1, or 2")
    if route.parent not in (0, 1) or route.child not in (0, 1):
        raise ValueError("K=2/B=2 route parent and child must be 0 or 1")
    if accepted_depth < 2:
        if fallback_transition is None:
            raise ValueError("prefix-0/1 trace binding requires fallback_transition")
        if accepted_depth == 0:
            return (fallback_transition,)
        return (tree_transitions[route.parent], fallback_transition)
    second_index = 2 + 2 * route.parent + route.child
    return (tree_transitions[route.parent], tree_transitions[second_index])


def build_draft_candidate_block(
    *,
    anchor_state: torch.Tensor,
    anchor_direction: torch.Tensor,
    sigma_path: Sequence[float],
    candidate_timesteps: Sequence[float],
    predictor: PredictorName,
    previous_direction: torch.Tensor | None = None,
    previous_sigma: float | None = None,
    draft_predictor: AutoregressiveActionPredictor | None = None,
    draft_channel_ids: Sequence[int] | None = None,
    state_transform: StateTransform | None = None,
) -> DraftCandidateBlock:
    """Roll out a cheap action-state draft from one exactly evaluated anchor.

    ``sigma_path`` contains ``[sigma_anchor, sigma_candidate_0, ...,
    sigma_candidate_(K-1), sigma_after_last]`` and therefore has K+2 values.
    The first transition uses the already-known exact ``anchor_direction``;
    the remaining K transitions use the selected draft. This mirrors a live
    denoising loop where the current target step has completed and only future
    steps are speculative. ``state_transform(state, offset)`` applies the real
    post-step projection plus next-input preparation (e.g. fixed-frame clamp
    and masked action channels). Offset -1 denotes the exact anchor transition;
    offsets 0..K-1 denote transitions out of candidate states.
    """
    if predictor not in {"zoh_direction", "sigma_linear", "zoh_displacement", "autoregressive_mlp"}:
        raise ValueError(f"unknown predictor: {predictor!r}")
    if predictor == "autoregressive_mlp" and draft_predictor is None:
        raise ValueError("autoregressive_mlp requires draft_predictor")
    if predictor != "autoregressive_mlp" and draft_predictor is not None:
        raise ValueError("draft_predictor is only valid with autoregressive_mlp")
    if draft_channel_ids is not None and predictor != "autoregressive_mlp":
        raise ValueError("draft_channel_ids is only valid with autoregressive_mlp")
    if not isinstance(anchor_state, torch.Tensor) or not isinstance(anchor_direction, torch.Tensor):
        raise TypeError("anchor_state and anchor_direction must be torch tensors")
    if anchor_state.shape != anchor_direction.shape or anchor_state.numel() == 0:
        raise ValueError("anchor state and direction must have the same non-empty shape")
    draft_channels = None
    if draft_channel_ids is not None:
        draft_channels = tuple(int(channel) for channel in draft_channel_ids)
        if not draft_channels:
            raise ValueError("draft_channel_ids must not be empty")
        if len(set(draft_channels)) != len(draft_channels):
            raise ValueError("draft_channel_ids must not contain duplicates")
        if min(draft_channels) < 0 or max(draft_channels) >= anchor_state.shape[0]:
            raise ValueError(
                "draft_channel_ids must index the unbatched action channel axis")

    sigmas = tuple(float(value) for value in sigma_path)
    timesteps = tuple(float(value) for value in candidate_timesteps)
    if len(sigmas) != len(timesteps) + 2:
        raise ValueError("sigma_path must contain K+2 values for K future candidates")
    if not timesteps:
        raise ValueError("candidate block must contain at least one future state")
    if not all(math.isfinite(value) for value in (*sigmas, *timesteps)):
        raise ValueError("sigma path and candidate timesteps must be finite")
    if predictor == "sigma_linear":
        if previous_direction is None or previous_sigma is None:
            raise ValueError("sigma_linear requires the previous real direction and sigma")
        if previous_direction.shape != anchor_state.shape:
            raise ValueError("previous_direction shape must match anchor_state")
        previous_sigma = float(previous_sigma)
        if not math.isfinite(previous_sigma) or sigmas[0] == previous_sigma:
            raise ValueError("sigma_linear requires distinct finite previous and anchor sigmas")

    anchor = anchor_state.detach().clone()
    direction = anchor_direction.detach().to(device=anchor.device, dtype=anchor.dtype).clone()
    if not bool(torch.isfinite(anchor).all()) or not bool(torch.isfinite(direction).all()):
        raise ValueError("anchor state and direction must be finite")
    previous = None
    if previous_direction is not None:
        if previous_direction.shape != anchor_state.shape:
            raise ValueError("previous_direction shape must match anchor_state")
        previous = previous_direction.detach().to(device=anchor.device, dtype=anchor.dtype).clone()
        if not bool(torch.isfinite(previous).all()):
            raise ValueError("previous_direction must be finite")

    def transform(value: torch.Tensor, offset: int) -> torch.Tensor:
        if state_transform is None:
            return value
        transformed = state_transform(value.clone(), offset)
        if not isinstance(transformed, torch.Tensor) or transformed.shape != value.shape:
            raise ValueError("state_transform must return a tensor with unchanged shape")
        if not bool(torch.isfinite(transformed).all()):
            raise ValueError("state_transform returned non-finite values")
        return transformed

    step_scales = tuple(sigmas[index + 1] - sigmas[index] for index in range(len(sigmas) - 1))
    latest_displacement = step_scales[0] * direction
    current = transform(anchor + step_scales[0] * direction, -1)
    candidates: list[torch.Tensor] = []
    draft_next: list[torch.Tensor] = []
    draft_directions: list[torch.Tensor] = []
    for offset, step_scale in enumerate(step_scales[1:]):
        candidates.append(current.clone())
        if predictor == "zoh_direction":
            predicted_direction = direction
        elif predictor == "autoregressive_mlp":
            assert draft_predictor is not None
            predictor_state = current
            predictor_direction = direction
            if draft_channels is not None:
                # The trained checkpoint covers only effective action channels
                # (7 x 4 x 4 = 112 in the current LIBERO layout), whereas WAM
                # carries a padded 30-channel state.  Predict compactly, then
                # scatter back to the full scheduler layout with zero padding.
                predictor_state = current.index_select(0, torch.tensor(
                    draft_channels, device=current.device))
                predictor_direction = direction.index_select(0, torch.tensor(
                    draft_channels, device=direction.device))
            predicted_compact = draft_predictor.predict(
                predictor_state, predictor_direction, sigmas[offset + 1],
                sigmas[offset + 2], timesteps[offset]
            )
            if draft_channels is None:
                predicted_direction = predicted_compact
            else:
                predicted_direction = torch.zeros_like(current)
                predicted_direction.index_copy_(
                    0, torch.tensor(draft_channels, device=current.device),
                    predicted_compact
                )
            if predicted_direction.shape != current.shape:
                raise ValueError("autoregressive draft predictor changed direction shape")
        elif predictor == "sigma_linear":
            assert previous is not None and previous_sigma is not None
            slope = (direction - previous) / (sigmas[0] - previous_sigma)
            predicted_direction = direction + slope * (sigmas[offset + 1] - sigmas[0])
        else:
            if step_scale == 0.0:
                raise ValueError("zoh_displacement cannot cross a zero-width scheduler transition")
            predicted_direction = latest_displacement / step_scale

        raw_successor = current + step_scale * predicted_direction
        draft_directions.append(predicted_direction.clone())
        draft_next.append(raw_successor.clone())
        # The last successor is only needed by the verifier. Earlier successors
        # become the next candidate input after the real deterministic transform.
        if offset + 1 < len(step_scales) - 1:
            current = transform(raw_successor, offset)
            if predictor == "autoregressive_mlp":
                direction = predicted_direction

    return DraftCandidateBlock(
        predictor=predictor,
        candidate_states=torch.stack(candidates),
        draft_next_states=torch.stack(draft_next),
        target_timesteps=timesteps,
        candidate_sigmas=sigmas[1:-1],
        transition_scales=tuple(step_scales[1:]),
        state_transform=state_transform,
        draft_directions=torch.stack(draft_directions),
    )


def verify_draft_candidate_block(
    *,
    draft: DraftCandidateBlock,
    target_batch_forward: TargetBatchForward,
    action_channel_ids: Sequence[int],
    thresholds: RiskThresholds,
    action_scales: torch.Tensor | Sequence[float] | float | None = None,
    state_mask: torch.Tensor | None = None,
    post_step_projection: Projection | None = None,
    channel_axis: int = 1,
) -> CandidateVerification:
    """Run exactly one candidate-state target batch and score every row.

    The callback must evaluate each row at its own candidate state/timestep
    under the live target weights, CFG, and matching observation/cache context.
    In particular, target-only trace outputs are not valid callback results.
    """
    target_directions = target_batch_forward(draft.candidate_states, draft.target_timesteps)
    if not isinstance(target_directions, torch.Tensor):
        raise TypeError("target_batch_forward must return a torch.Tensor")
    if target_directions.shape != draft.candidate_states.shape:
        raise ValueError(
            "target batch output shape must match candidate states; got "
            f"{tuple(target_directions.shape)} vs {tuple(draft.candidate_states.shape)}"
        )
    effective_projection = (
        post_step_projection if post_step_projection is not None else draft.state_transform
    )
    scores = score_candidate_block(
        candidate_states=draft.candidate_states,
        draft_next_states=draft.draft_next_states,
        target_directions=target_directions,
        transition_scales=draft.transition_scales,
        action_channel_ids=action_channel_ids,
        action_scales=action_scales,
        state_mask=state_mask,
        post_step_projection=effective_projection,
        channel_axis=channel_axis,
    )
    decision = accept_longest_safe_prefix(scores, thresholds)
    return CandidateVerification(
        draft=draft,
        target_directions=target_directions,
        scores=tuple(scores),
        decision=decision,
    )


def _float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.detach().to(dtype=torch.float64)


def _masked_flat(tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tensor.reshape(-1)
    try:
        expanded = torch.broadcast_to(mask.to(device=tensor.device, dtype=torch.bool), tensor.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to {tuple(tensor.shape)}"
        ) from exc
    if not bool(expanded.any()):
        raise ValueError("state_mask selects no effective elements")
    return tensor[expanded]


def _effective_mask(
    reference: torch.Tensor,
    action_channel_ids: Sequence[int],
    channel_axis: int,
    state_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Combine effective action channels with any caller-supplied state mask."""
    channel_mask = torch.zeros(
        reference.shape[channel_axis], dtype=torch.bool, device=reference.device
    )
    channel_mask[list(action_channel_ids)] = True
    shape = [1] * reference.ndim
    shape[channel_axis] = reference.shape[channel_axis]
    expanded = torch.broadcast_to(channel_mask.reshape(shape), reference.shape)
    if state_mask is None:
        return expanded
    try:
        supplied = torch.broadcast_to(
            state_mask.to(device=reference.device, dtype=torch.bool), reference.shape
        )
    except RuntimeError as exc:
        raise ValueError(
            f"state_mask shape {tuple(state_mask.shape)} is not broadcastable to {tuple(reference.shape)}"
        ) from exc
    combined = expanded & supplied
    if not bool(combined.any()):
        raise ValueError("state_mask selects no effective action elements")
    return combined


def _relative_l2(delta: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None) -> float:
    delta_values = _masked_flat(delta, mask)
    reference_values = _masked_flat(reference, mask)
    numerator = torch.linalg.vector_norm(delta_values)
    denominator = torch.linalg.vector_norm(reference_values).clamp_min(EPSILON)
    return float((numerator / denominator).item())


def _broadcast_action_scales(
    scales: torch.Tensor | Sequence[float] | float | None,
    reference: torch.Tensor,
    channel_axis: int,
) -> torch.Tensor:
    if scales is None:
        return torch.ones((), dtype=reference.dtype, device=reference.device)
    result = torch.as_tensor(scales, dtype=reference.dtype, device=reference.device)
    if result.ndim == 0:
        return result
    if result.ndim != 1 or result.numel() != reference.shape[channel_axis]:
        raise ValueError(
            "action_scales must be scalar or one scale per channel; "
            f"got {tuple(result.shape)} for channel count {reference.shape[channel_axis]}"
        )
    shape = [1] * reference.ndim
    shape[channel_axis] = result.numel()
    return result.reshape(shape)


def score_candidate_block(
    *,
    candidate_states: torch.Tensor,
    draft_next_states: torch.Tensor,
    target_directions: torch.Tensor,
    transition_scales: Sequence[float],
    action_channel_ids: Sequence[int],
    action_scales: torch.Tensor | Sequence[float] | float | None = None,
    state_mask: torch.Tensor | None = None,
    post_step_projection: Projection | None = None,
    channel_axis: int = 1,
) -> list[StepResidual]:
    """Score each candidate using a target output evaluated at that candidate.

    For candidate state ``xhat_j``, the target one-step result is
    ``P(xhat_j + h_j * F_target(xhat_j))``.  It is compared with the draft's
    corresponding next candidate state, after the same projection ``P``.
    This is a *local* correction score, not a guarantee on the final decoded
    action or the full target-only trajectory.

    The first tensor dimension is the proposal-block dimension: candidate
    states, draft next states, and target outputs must all be aligned there.
    ``target_directions[j]`` must come from the same batched target forward at
    ``candidate_states[j]`` under the same conditioning/cache state. A
    direction from the original target-only path is not valid for this API.
    """
    if not all(
        isinstance(value, torch.Tensor)
        for value in (candidate_states, draft_next_states, target_directions)
    ):
        raise TypeError("candidate, draft-next, and target outputs must be stacked tensors")
    if candidate_states.ndim < 2:
        raise ValueError("candidate tensors require a leading block dimension")
    if (
        candidate_states.shape != draft_next_states.shape
        or candidate_states.shape != target_directions.shape
    ):
        raise ValueError(
            "candidate, draft-next, and target batch shapes must match; "
            f"got {tuple(candidate_states.shape)}, {tuple(draft_next_states.shape)}, "
            f"{tuple(target_directions.shape)}"
        )
    block_length = candidate_states.shape[0]
    if block_length != len(transition_scales):
        raise ValueError("candidate batch length and step-size length must match")
    if not action_channel_ids:
        raise ValueError("action_channel_ids must identify at least one effective action channel")

    results: list[StepResidual] = []
    for offset, (candidate, draft_next, target_direction, step_size) in enumerate(
        zip(
            candidate_states.unbind(0),
            draft_next_states.unbind(0),
            target_directions.unbind(0),
            transition_scales,
        )
    ):
        candidate = _float_tensor(candidate, "candidate_state")
        draft_next = _float_tensor(draft_next, "draft_next_state")
        target_direction = _float_tensor(target_direction, "target_direction")
        if candidate.shape != draft_next.shape or candidate.shape != target_direction.shape:
            raise ValueError(
                f"shape mismatch at offset {offset}: candidate={tuple(candidate.shape)}, "
                f"draft_next={tuple(draft_next.shape)}, target={tuple(target_direction.shape)}"
            )
        if not math.isfinite(float(step_size)):
            raise ValueError(f"non-finite transition scale at offset {offset}")
        if channel_axis < 0:
            channel_axis += candidate.ndim
        if not 0 <= channel_axis < candidate.ndim:
            raise ValueError(f"channel_axis {channel_axis} outside tensor rank {candidate.ndim}")
        if min(action_channel_ids) < 0 or max(action_channel_ids) >= candidate.shape[channel_axis]:
            raise ValueError(f"action channel id outside axis size {candidate.shape[channel_axis]}")

        target_next = candidate + float(step_size) * target_direction
        if post_step_projection is not None:
            target_next = post_step_projection(target_next, offset)
            draft_next = post_step_projection(draft_next, offset)
        if target_next.shape != draft_next.shape:
            raise ValueError(f"projection changed shapes at offset {offset}")

        delta = target_next - draft_next
        effective_mask = _effective_mask(
            target_next, action_channel_ids, channel_axis, state_mask
        )
        state_rel = _relative_l2(delta, target_next, effective_mask)
        state_l2 = float(
            torch.linalg.vector_norm(_masked_flat(delta, effective_mask)).item()
        )

        scale = _broadcast_action_scales(action_scales, target_next, channel_axis)
        physical_delta = delta * scale
        physical_reference = target_next * scale
        channel_rel: list[float] = []
        channel_rms: list[float] = []
        for channel_id in action_channel_ids:
            channel_delta = physical_delta.select(channel_axis, int(channel_id))
            channel_reference = physical_reference.select(channel_axis, int(channel_id))
            channel_mask = effective_mask.select(channel_axis, int(channel_id))
            if not bool(channel_mask.any()):
                raise ValueError(
                    "state_mask selects no effective elements for action channel "
                    f"{channel_id} at offset {offset}"
                )
            selected_delta = channel_delta[channel_mask]
            selected_reference = channel_reference[channel_mask]
            channel_rel.append(
                float(
                    (
                        torch.linalg.vector_norm(selected_delta)
                        / torch.linalg.vector_norm(selected_reference).clamp_min(EPSILON)
                    ).item()
                )
            )
            channel_rms.append(float(torch.sqrt(torch.mean(selected_delta.square())).item()))
        # Padding/unused channels and masked state positions must not affect a
        # physical-action risk limit.
        component_abs_max = float(physical_delta[effective_mask].abs().amax().item())
        values = [state_rel, state_l2, *channel_rel, *channel_rms, component_abs_max]
        finite = all(torch.isfinite(torch.tensor(value)).item() for value in values)
        results.append(
            StepResidual(
                step_offset=offset,
                state_rel_l2=state_rel,
                state_l2=state_l2,
                action_channel_max_rel_l2=max(channel_rel),
                action_channel_max_rms=max(channel_rms),
                action_component_abs_max=component_abs_max,
                action_channel_rel_l2=tuple(channel_rel),
                action_channel_rms=tuple(channel_rms),
                finite=finite,
            )
        )
    return results


def accept_longest_safe_prefix(
    scores: Sequence[StepResidual], thresholds: RiskThresholds
) -> PrefixDecision:
    """Accept only the longest contiguous prefix whose every step is safe."""
    safe: list[bool] = []
    for score in scores:
        step_is_safe = (
            score.finite
            and score.state_rel_l2 <= thresholds.state_rel_l2
            and score.action_channel_max_rms is not None
            and score.action_channel_max_rms <= thresholds.action_channel_max_rms
        )
        if thresholds.action_channel_max_rel_l2 is not None:
            step_is_safe = (
                step_is_safe
                and score.action_channel_max_rel_l2 is not None
                and score.action_channel_max_rel_l2 <= thresholds.action_channel_max_rel_l2
            )
        if thresholds.action_component_abs_max is not None:
            step_is_safe = (
                step_is_safe
                and score.action_component_abs_max is not None
                and score.action_component_abs_max <= thresholds.action_component_abs_max
            )
        safe.append(bool(step_is_safe))

    accepted_length = 0
    for is_safe in safe:
        if not is_safe:
            break
        accepted_length += 1
    rejected = accepted_length if accepted_length < len(safe) else None
    return PrefixDecision(
        accepted_length=accepted_length,
        safe_steps=tuple(safe),
        first_rejected_offset=rejected,
    )

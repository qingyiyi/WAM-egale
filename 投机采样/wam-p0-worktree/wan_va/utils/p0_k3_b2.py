"""Pure K=3/B=2 speculative verifier-tree helpers.

The live WAM server supplies the trained two-head draft predictor and the
real scheduler callback.  Keeping expansion, scoring, selection, execution,
and trace binding here makes the K=3 contract CPU-testable without importing
the server or the external research helper.

One K=3/B=2 tree has 14 draft scheduler expansions and one 7-row target
verifier batch.  The verifier rows are ordered as ``root``, two depth-1
states, then four depth-2 states.  They verify the first, second, and third
draft edge respectively; depth-3 states are draft-only committed leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, Sequence

import torch


State = torch.Tensor
DraftPredictor = Callable[[State, State, float, float, float], torch.Tensor]
SchedulerStep = Callable[[State, State, float, int], State]


@dataclass(frozen=True)
class K3B2Tree:
    """Continuous binary draft tree through three scheduler edges."""

    root_state: torch.Tensor
    root_previous_direction: torch.Tensor
    first_directions: torch.Tensor
    first_states: torch.Tensor
    second_directions: torch.Tensor
    second_states: torch.Tensor
    third_directions: torch.Tensor
    third_states: torch.Tensor
    sigma_path: tuple[float, float, float, float]
    target_timesteps: tuple[float, float, float, float]
    transition_scales: tuple[float, float, float]

    def verifier_states(self) -> torch.Tensor:
        """Return target states in fixed root/depth-1/depth-2 order."""
        return torch.cat((
            self.root_state.unsqueeze(0),
            self.first_states,
            self.second_states.reshape(4, *self.root_state.shape),
        ), dim=0)

    def verifier_timesteps(self) -> tuple[float, ...]:
        """Return timesteps aligned with :meth:`verifier_states`."""
        t0, t1, t2, _ = self.target_timesteps
        return (t0, t1, t1, t2, t2, t2, t2)


@dataclass(frozen=True)
class K3B2Scores:
    """Per-edge weighted errors and the one batched target result."""

    first_costs: tuple[float, float]
    second_costs: tuple[tuple[float, float], tuple[float, float]]
    third_costs: tuple[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ]
    target_directions: torch.Tensor


@dataclass(frozen=True)
class K3B2Route:
    """A selected three-edge route before threshold fallback."""

    parent: int
    child: int
    grandchild: int
    first_cost: float
    second_cost: float
    third_cost: float
    total_cost: float
    selector: str


@dataclass(frozen=True)
class K3B2Execution:
    """Actual committed prefix after first-rejected-edge target correction."""

    next_state: torch.Tensor
    next_previous_direction: torch.Tensor
    accepted_depth: int
    route: K3B2Route


def _validate_tree_inputs(
    root_state: torch.Tensor,
    previous_direction: torch.Tensor,
    sigma_path: Sequence[float],
    target_timesteps: Sequence[float],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    if not isinstance(root_state, torch.Tensor) or not isinstance(previous_direction, torch.Tensor):
        raise TypeError("root_state and previous_direction must be tensors")
    if root_state.shape != previous_direction.shape or root_state.numel() == 0:
        raise ValueError("root_state and previous_direction must have the same non-empty shape")
    sigmas = tuple(float(value) for value in sigma_path)
    timesteps = tuple(float(value) for value in target_timesteps)
    if len(sigmas) != 4 or len(timesteps) != 4:
        raise ValueError("K=3/B=2 requires four sigmas and four target timesteps")
    if not all(math.isfinite(value) for value in (*sigmas, *timesteps)):
        raise ValueError("sigma_path and target_timesteps must be finite")
    return sigmas, timesteps  # type: ignore[return-value]


def _predict_pair(
    predictor: DraftPredictor,
    state: torch.Tensor,
    previous_direction: torch.Tensor,
    sigma: float,
    next_sigma: float,
    timestep: float,
) -> torch.Tensor:
    directions = predictor(state, previous_direction, sigma, next_sigma, timestep)
    expected_shape = (2, *state.shape)
    if not isinstance(directions, torch.Tensor) or tuple(directions.shape) != expected_shape:
        raise ValueError(
            "K=3/B=2 draft predictor must return exactly two directions shaped "
            f"{expected_shape}, got {getattr(directions, 'shape', None)}"
        )
    return directions


def build_k3_b2_tree(
    *,
    root_state: torch.Tensor,
    previous_direction: torch.Tensor,
    sigma_path: Sequence[float],
    target_timesteps: Sequence[float],
    draft_predictor: DraftPredictor,
    scheduler_step: SchedulerStep,
) -> K3B2Tree:
    """Expand a three-edge binary tree using the live scheduler callback."""
    sigmas, timesteps = _validate_tree_inputs(
        root_state, previous_direction, sigma_path, target_timesteps
    )

    first_directions = _predict_pair(
        draft_predictor, root_state, previous_direction, sigmas[0], sigmas[1], timesteps[0]
    )
    first_states = torch.stack(tuple(
        scheduler_step(root_state, first_directions[parent], timesteps[0], 0)
        for parent in range(2)
    ), dim=0)

    # Call the two-head predictor exactly once per draft node: 1 root, two
    # depth-1 nodes, and four depth-2 nodes.  The individual heads form the
    # outgoing binary edges; re-calling a predictor per child would silently
    # double the intended draft work and distort phase accounting.
    second_directions = torch.stack(tuple(
        _predict_pair(
            draft_predictor,
            first_states[parent],
            first_directions[parent],
            sigmas[1], sigmas[2], timesteps[1],
        )
        for parent in range(2)
    ), dim=0)
    second_states = torch.stack(tuple(torch.stack(tuple(
        scheduler_step(
            first_states[parent], second_directions[parent, child], timesteps[1], 1
        )
        for child in range(2)), dim=0) for parent in range(2)), dim=0)

    third_directions = torch.stack(tuple(torch.stack(tuple(
        _predict_pair(
            draft_predictor,
            second_states[parent, child],
            second_directions[parent, child],
            sigmas[2], sigmas[3], timesteps[2],
        )
        for child in range(2)
    ), dim=0) for parent in range(2)), dim=0)
    third_states = torch.stack(tuple(torch.stack(tuple(torch.stack(tuple(
        scheduler_step(
            second_states[parent, child],
            third_directions[parent, child, grandchild],
            timesteps[2],
            2,
        )
        for grandchild in range(2)), dim=0) for child in range(2)), dim=0)
        for parent in range(2)), dim=0)

    return K3B2Tree(
        root_state=root_state.detach().clone(),
        root_previous_direction=previous_direction.detach().clone(),
        first_directions=first_directions.detach().clone(),
        first_states=first_states.detach().clone(),
        second_directions=second_directions.detach().clone(),
        second_states=second_states.detach().clone(),
        third_directions=third_directions.detach().clone(),
        third_states=third_states.detach().clone(),
        sigma_path=sigmas,
        target_timesteps=timesteps,
        transition_scales=(sigmas[1] - sigmas[0], sigmas[2] - sigmas[1], sigmas[3] - sigmas[2]),
    )


def pack_k3_b2_verifier_batch(tree: K3B2Tree) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Pack the exact root/depth-1/depth-2 seven-row target batch."""
    return tree.verifier_states(), tree.verifier_timesteps()


def _weighted_direction_l2(
    draft_direction: torch.Tensor,
    target_direction: torch.Tensor,
    transition_scale: float,
) -> float:
    if draft_direction.shape != target_direction.shape:
        raise ValueError(
            "draft and target directions must share a shape, got "
            f"{tuple(draft_direction.shape)} vs {tuple(target_direction.shape)}"
        )
    if not math.isfinite(float(transition_scale)):
        raise ValueError("transition_scale must be finite")
    residual = (draft_direction.float() - target_direction.float()) * float(transition_scale)
    return float(torch.linalg.vector_norm(residual).item())


def score_k3_b2_tree(tree: K3B2Tree, target_directions: torch.Tensor) -> K3B2Scores:
    """Score all 14 draft edges from a seven-row target verifier result."""
    expected = tree.verifier_states()
    if tuple(target_directions.shape) != tuple(expected.shape):
        raise ValueError(
            "target_directions must match the seven packed verifier states: "
            f"{tuple(target_directions.shape)} vs {tuple(expected.shape)}"
        )
    root_target = target_directions[0]
    first_targets = target_directions[1:3]
    second_targets = target_directions[3:7].reshape(2, 2, *target_directions.shape[1:])
    first = tuple(
        _weighted_direction_l2(tree.first_directions[parent], root_target, tree.transition_scales[0])
        for parent in range(2)
    )
    second = tuple(tuple(
        _weighted_direction_l2(
            tree.second_directions[parent, child],
            first_targets[parent],
            tree.transition_scales[1],
        )
        for child in range(2)) for parent in range(2))
    third = tuple(tuple(tuple(
        _weighted_direction_l2(
            tree.third_directions[parent, child, grandchild],
            second_targets[parent, child],
            tree.transition_scales[2],
        )
        for grandchild in range(2)) for child in range(2)) for parent in range(2))
    return K3B2Scores(
        first_costs=first,
        second_costs=second,
        third_costs=third,
        target_directions=target_directions.detach().clone(),
    )


def select_k3_b2_route(scores: K3B2Scores, *, selector: Literal["greedy", "global"]) -> K3B2Route:
    """Select a route using the same verified per-edge rule as K=2/B=2."""
    if selector == "greedy":
        parent = min(range(2), key=lambda idx: (scores.first_costs[idx], idx))
        child = min(range(2), key=lambda idx: (scores.second_costs[parent][idx], idx))
        grandchild = min(
            range(2), key=lambda idx: (scores.third_costs[parent][child][idx], idx)
        )
    elif selector == "global":
        parent, child, grandchild = min(
            ((p, c, g) for p in range(2) for c in range(2) for g in range(2)),
            key=lambda route: (
                scores.first_costs[route[0]]
                + scores.second_costs[route[0]][route[1]]
                + scores.third_costs[route[0]][route[1]][route[2]],
                route[0], route[1], route[2],
            ),
        )
    else:
        raise ValueError(f"selector must be greedy or global, got {selector!r}")
    first = float(scores.first_costs[parent])
    second = float(scores.second_costs[parent][child])
    third = float(scores.third_costs[parent][child][grandchild])
    return K3B2Route(
        parent=parent,
        child=child,
        grandchild=grandchild,
        first_cost=first,
        second_cost=second,
        third_cost=third,
        total_cost=first + second + third,
        selector=selector,
    )


def execute_k3_b2_route(
    tree: K3B2Tree,
    scores: K3B2Scores,
    route: K3B2Route,
    *,
    threshold: float,
    scheduler_step: SchedulerStep,
) -> K3B2Execution:
    """Commit accepted draft prefix, replacing the first failed edge by target."""
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    root_target = scores.target_directions[0]
    first_target = scores.target_directions[1 + route.parent]
    second_target = scores.target_directions[3 + 2 * route.parent + route.child]
    if route.first_cost > threshold:
        next_state = scheduler_step(tree.root_state, root_target, tree.target_timesteps[0], 0)
        return K3B2Execution(
            next_state=next_state,
            next_previous_direction=root_target.detach().clone(),
            accepted_depth=0,
            route=route,
        )
    if route.second_cost > threshold:
        next_state = scheduler_step(
            tree.first_states[route.parent], first_target, tree.target_timesteps[1], 1
        )
        return K3B2Execution(
            next_state=next_state,
            next_previous_direction=first_target.detach().clone(),
            accepted_depth=1,
            route=route,
        )
    if route.third_cost > threshold:
        next_state = scheduler_step(
            tree.second_states[route.parent, route.child],
            second_target,
            tree.target_timesteps[2],
            2,
        )
        return K3B2Execution(
            next_state=next_state,
            next_previous_direction=second_target.detach().clone(),
            accepted_depth=2,
            route=route,
        )
    return K3B2Execution(
        next_state=tree.third_states[route.parent, route.child, route.grandchild].detach().clone(),
        next_previous_direction=(
            tree.third_directions[route.parent, route.child, route.grandchild].detach().clone()
        ),
        accepted_depth=3,
        route=route,
    )


def select_k3_b2_trace_transitions(
    tree_transitions: Sequence[object],
    *,
    route: K3B2Route,
    accepted_depth: int,
    fallback_transition: object | None,
) -> tuple[object, ...]:
    """Bind the actual committed updates to the 14 deterministic expansions."""
    if len(tree_transitions) != 14:
        raise ValueError(
            "K=3/B=2 trace binding requires exactly fourteen tree transitions, got "
            f"{len(tree_transitions)}"
        )
    if accepted_depth not in (0, 1, 2, 3):
        raise ValueError("accepted_depth must be 0, 1, 2, or 3")
    if any(value not in (0, 1) for value in (route.parent, route.child, route.grandchild)):
        raise ValueError("K=3/B=2 route must lie in the 2x2x2 tree")
    if accepted_depth < 3 and fallback_transition is None:
        raise ValueError("prefix-0/1/2 trace binding requires fallback_transition")
    first_index = route.parent
    second_index = 2 + 2 * route.parent + route.child
    third_index = 6 + 4 * route.parent + 2 * route.child + route.grandchild
    if accepted_depth == 0:
        return (fallback_transition,)  # type: ignore[return-value]
    if accepted_depth == 1:
        return (tree_transitions[first_index], fallback_transition)  # type: ignore[return-value]
    if accepted_depth == 2:
        return (
            tree_transitions[first_index],
            tree_transitions[second_index],
            fallback_transition,
        )  # type: ignore[return-value]
    return (
        tree_transitions[first_index],
        tree_transitions[second_index],
        tree_transitions[third_index],
    )


def execute_verified_k2_bonus_step(
    *,
    accepted_depth: int,
    leaf_state: torch.Tensor,
    leaf_target_direction: torch.Tensor | None,
    timestep: float,
    scheduler_step: SchedulerStep,
) -> torch.Tensor | None:
    """Consume a K=2 leaf target prefetch only after full draft acceptance.

    This intentionally has no K=2 tree dependency so the server can keep using
    the established external K=2 implementation while this commitment contract
    remains directly CPU-testable.  It returns ``None`` for prefix 0/1, when a
    target correction has invalidated the selected draft leaf.
    """
    if accepted_depth not in (0, 1, 2):
        raise ValueError("K=2/B=2 accepted_depth must be 0, 1, or 2")
    if accepted_depth < 2:
        return None
    if leaf_target_direction is None:
        raise ValueError("fully accepted K=2/B=2 route requires a leaf target direction")
    if leaf_state.shape != leaf_target_direction.shape:
        raise ValueError("K=2/B=2 leaf state and target direction must share a shape")
    return scheduler_step(leaf_state, leaf_target_direction, float(timestep), 2)

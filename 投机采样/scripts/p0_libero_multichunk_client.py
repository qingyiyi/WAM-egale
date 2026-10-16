#!/usr/bin/env python3
"""Run resumable multi-chunk LIBERO rollouts against the isolated P0 server.

This is a diagnostic/evaluation client, not a speculative sampler.  It uses the
same action layout and KV-cache handoff as the official WAM LIBERO client:
reset -> infer a chunk -> execute the returned actions in LIBERO -> hand the
key observations and executed action chunk back through ``compute_kv_cache``.
The server-side target path remains unchanged; P0 traces are only recorded by
the opt-in server recorder.

The default collection unit is one initial state per task.  LIBERO ships 50
initial states per task; collecting all 2,000 episodes across the four 10-task
suites is intentionally opt-in because a full 800-step rollout is expensive.
Every episode launched by this client runs the complete LIBERO horizon. Use
``--episodes-per-task 50`` only when the full cost of all init states is wanted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKTREE = RESEARCH_ROOT / "wam-p0-worktree"
BENCHMARKS = ("libero_10", "libero_goal", "libero_spatial", "libero_object")
LIBERO_HORIZON = 800
INIT_STABILIZATION_STEPS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=29256)
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help=(
            "torch.distributed master port used by the paired server launcher; "
            "recorded for parallel provenance only."
        ),
    )
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=["libero_10"])
    task_selection = parser.add_mutually_exclusive_group()
    task_selection.add_argument(
        "--task-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Contiguous task range [START, END). Default: 0 10.",
    )
    task_selection.add_argument(
        "--task-indices",
        type=int,
        nargs="+",
        metavar="TASK",
        help="Explicit disjoint task indices, used by independent parallel workers.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=1,
                        help="Initial states per task; LIBERO provides 50. Default: 1.")
    parser.add_argument("--episode-start", type=int, default=0,
                        help="Initial-state index for the first requested episode.")
    parser.add_argument(
        "--condition",
        default="target-only",
        help=(
            "Named evaluation condition written to run_metadata.json, e.g. "
            "target-only, direct-fewer-step, or speculative-k2."
        ),
    )
    parser.add_argument(
        "--campaign-note",
        default="",
        help="Optional human-readable note stored with the campaign metadata.",
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Parent campaign id for a parallel worker; omitted for a standalone run.",
    )
    parser.add_argument(
        "--trace-mode",
        choices=("on", "off", "unknown"),
        default="unknown",
        help=(
            "Server trace provenance recorded in run_metadata.json. This is supplied "
            "by a launcher; direct client use defaults to unknown rather than guessing."
        ),
    )
    parser.add_argument(
        "--latency-benchmark",
        action="store_true",
        help=(
            "Mark this as a trace-off latency condition in run_metadata.json. "
            "It does not alter the simulator or policy protocol."
        ),
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Independent worker id written to metadata and trace context.",
    )
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory for JSONL episode manifest and summary files.")
    parser.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    parser.add_argument("--save-video", action="store_true",
                        help="Save rendered observation videos; off by default to avoid disk bloat.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip completed manifest keys (default: true).")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on the first episode exception instead of recording and continuing.")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(obs["agentview_image"][::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }


def _resolve_task_indices(args: argparse.Namespace) -> tuple[int, ...]:
    if args.task_indices is not None:
        task_indices = tuple(int(value) for value in args.task_indices)
        if not task_indices:
            raise SystemExit("--task-indices requires at least one task")
        if len(set(task_indices)) != len(task_indices):
            raise SystemExit("--task-indices must not contain duplicates")
        if any(value < 0 for value in task_indices):
            raise SystemExit("--task-indices must be non-negative")
        return task_indices

    start, end = args.task_range or (0, 10)
    if start < 0 or end <= start:
        raise SystemExit("--task-range must satisfy 0 <= START < END")
    return tuple(range(start, end))


def _contiguous_task_range(task_indices: tuple[int, ...]) -> list[int] | None:
    if task_indices == tuple(range(task_indices[0], task_indices[-1] + 1)):
        return [task_indices[0], task_indices[-1] + 1]
    return None


def _append_jsonl(path: Path, payload: dict[str, Any], *, fsync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def _read_completed(path: Path) -> set[tuple[str, int, int]]:
    completed: set[tuple[str, int, int]] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "completed":
                completed.add((str(row.get("benchmark")), int(row["task_index"]), int(row["episode_index"])))
    return completed


def _construct_env(OffScreenRenderEnv, env_args: dict[str, Any]):
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            return OffScreenRenderEnv(**env_args)
        except Exception as exc:  # pragma: no cover - simulator-dependent retry
            last_error = exc
            print(f"[client] env construction failed (attempt {attempt + 1}/5): {exc}", flush=True)
            time.sleep(2)
    raise RuntimeError("failed to construct LIBERO environment") from last_error


def _simulator_timestep(env, fallback: int) -> int:
    observed = _observed_simulator_timestep(env)
    return int(fallback) if observed is None else observed


def _observed_simulator_timestep(env) -> int | None:
    """Return the simulator's observed timestep, never a synthesized fallback."""
    simulator = getattr(env, "env", env)
    try:
        return int(getattr(simulator, "timestep"))
    except (AttributeError, TypeError, ValueError):
        return None


def _finite_reward_scalar(value: Any) -> float | None:
    """Normalize a scalar reward without turning missing/non-finite data into 0."""
    if value is None:
        return None
    try:
        scalar = np.asarray(value)
        if scalar.size != 1:
            return None
        result = float(scalar.reshape(()))
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _summarize_rewards(rewards: list[Any]) -> dict[str, Any]:
    observed = [_finite_reward_scalar(value) for value in rewards]
    values = [value for value in observed if value is not None]
    return {
        "returned_count": len(rewards),
        "finite_scalar_count": len(values),
        "sum": float(sum(values)) if values else None,
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "nonzero_count": sum(value != 0.0 for value in values) if values else None,
    }


def _build_chunk_event(
    *,
    campaign_id: str | None,
    worker_id: str | None,
    benchmark_name: str,
    task_index: int,
    episode_index: int,
    episode_attempt_id: str,
    rollout_chunk: int,
    env_steps_before: int,
    env_steps_after: int,
    simulator_timestep_before: int | None,
    simulator_timestep_after: int | None,
    rewards: list[Any],
    done: bool,
    action_shape: list[int],
) -> dict[str, Any]:
    simulator_delta = (
        int(simulator_timestep_after) - int(simulator_timestep_before)
        if simulator_timestep_before is not None and simulator_timestep_after is not None
        else None
    )
    return {
        "event_schema_version": "wam-p0-chunk-event/v1",
        "event_id": f"{episode_attempt_id}:chunk:{int(rollout_chunk):05d}",
        "recorded_at": _utc_now(),
        "campaign_id": campaign_id,
        "parallel_worker_id": worker_id,
        "benchmark": str(benchmark_name),
        "task_index": int(task_index),
        "episode_index": int(episode_index),
        "episode_attempt_id": str(episode_attempt_id),
        "rollout_chunk": int(rollout_chunk),
        "action_shape": [int(value) for value in action_shape],
        "env_steps_before": int(env_steps_before),
        "env_steps_after": int(env_steps_after),
        "env_steps_delta": int(env_steps_after) - int(env_steps_before),
        "simulator_timestep_before": simulator_timestep_before,
        "simulator_timestep_after": simulator_timestep_after,
        "simulator_timestep_delta": simulator_delta,
        "simulator_timestep_source": (
            "env.env.timestep" if simulator_timestep_before is not None
            and simulator_timestep_after is not None else None
        ),
        "reward_summary": _summarize_rewards(rewards),
        "done_after_chunk": bool(done),
    }


def _init_env(env, init_state: Any) -> tuple[dict[str, np.ndarray], int]:
    env.reset()
    env.set_init_state(init_state)
    raw_obs = None
    for _ in range(INIT_STABILIZATION_STEPS):
        raw_obs, _, _, _ = env.step([0.0] * 7)
    assert raw_obs is not None
    return _extract_obs(raw_obs), _simulator_timestep(env, INIT_STABILIZATION_STEPS)


def _episode_key(benchmark_name: str, task_index: int, episode_index: int) -> tuple[str, int, int]:
    return benchmark_name, int(task_index), int(episode_index)


def _run_episode(
    *,
    model,
    benchmark_name: str,
    benchmark_instance,
    task_index: int,
    episode_index: int,
    save_video: bool,
    video_root: Path,
    campaign_id: str | None = None,
    worker_id: str | None = None,
    episode_attempt_id: str | None = None,
    chunk_events_path: Path | None = None,
) -> dict[str, Any]:
    task = benchmark_instance.get_task(task_index)
    prompt = task.language
    init_states = benchmark_instance.get_task_init_states(task_index)
    if not 0 <= episode_index < int(init_states.shape[0]):
        raise IndexError(f"episode index {episode_index} outside [0, {init_states.shape[0]})")
    if episode_attempt_id is None:
        episode_attempt_id = uuid.uuid4().hex

    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_index),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = _construct_env(type(model)._p0_offscreen_env_cls, env_args)
    context_base = {
        "libero_benchmark": benchmark_name,
        "task_index": int(task_index),
        "episode_index": int(episode_index),
        "episode_attempt_id": episode_attempt_id,
        "prompt": prompt,
    }
    if campaign_id is not None:
        context_base["campaign_id"] = campaign_id
    if worker_id is not None:
        context_base["parallel_worker_id"] = worker_id
    # Opt-in sidecar only: the observer never changes the transport payload.
    native_observer = None
    if os.getenv("P0_NATIVE_DIAGNOSTIC_DIR", "").strip():
        from p0_native_observer import NativeObserver
        native_observer = NativeObserver.from_env(context_base)
    started = time.monotonic()
    started_at = _utc_now()
    obs = None
    full_obs: list[dict[str, np.ndarray]] = []
    infer_chunks = 0
    env_steps = 0
    done = False
    first = True
    action_shapes: list[list[int]] = []
    inference_timing = {
        "reset_infer_wall_sec": 0.0,
        "policy_infer_wall_sec": 0.0,
        "cache_infer_wall_sec": 0.0,
        "reset_infer_calls": 0,
        "policy_infer_calls": 0,
        "cache_infer_calls": 0,
    }

    def timed_infer(kind: str, request: dict[str, Any]):
        key_seconds = f"{kind}_infer_wall_sec"
        key_calls = f"{kind}_infer_calls"
        if key_seconds not in inference_timing or key_calls not in inference_timing:
            raise ValueError(f"unsupported timed inference kind {kind!r}")
        call_started = time.monotonic()
        result = model.infer(request)
        inference_timing[key_seconds] += time.monotonic() - call_started
        inference_timing[key_calls] += 1
        return result

    try:
        obs, simulator_timestep_start = _init_env(env, init_states[episode_index])
        if native_observer is not None:
            native_observer.capture(env, env_step=0)
        timed_infer("reset", {"reset": True, "prompt": prompt, "p0_trace_context": context_base})

        while _simulator_timestep(env, simulator_timestep_start + env_steps) < LIBERO_HORIZON:
            simulator_timestep_before = _simulator_timestep(
                env, simulator_timestep_start + env_steps
            )
            simulator_timestep_before_observed = _observed_simulator_timestep(env)
            env_steps_before = int(env_steps)
            rollout_chunk = int(infer_chunks)
            context = {
                **context_base,
                "rollout_chunk": int(infer_chunks),
                "env_steps_before": int(env_steps),
                "simulator_timestep_before": int(simulator_timestep_before),
            }
            result = timed_infer("policy", {
                "obs": obs,
                "prompt": prompt,
                "p0_trace_context": context,
            })
            action = np.asarray(result["action"])
            if action.ndim != 3 or action.shape[2] % 4 != 0:
                raise ValueError(f"unexpected action shape {action.shape}; expected (C,F,H) with H divisible by 4")
            action_shapes.append([int(v) for v in action.shape])
            action_per_frame = int(action.shape[2] // 4)
            start_frame = 1 if first else 0
            key_frame_list: list[dict[str, np.ndarray]] = []
            chunk_rewards: list[Any] = []

            for frame_index in range(start_frame, int(action.shape[1])):
                for action_index in range(int(action.shape[2])):
                    ee_action = np.asarray(action[:, frame_index, action_index], dtype=np.float32)
                    raw_obs, reward, done, _info = env.step(ee_action)
                    chunk_rewards.append(reward)
                    env_steps += 1
                    if native_observer is not None:
                        native_observer.capture(
                            env, env_step=env_steps, command=ee_action,
                            reward=reward, done=done, raw_obs=raw_obs,
                            rollout_chunk=rollout_chunk, frame_index=frame_index,
                            action_index=action_index)
                    if done:
                        break
                    frame_obs = _extract_obs(raw_obs)
                    if (action_index + 1) % action_per_frame == 0:
                        full_obs.append(frame_obs)
                        key_frame_list.append(frame_obs)
                if done:
                    break

            simulator_timestep_after_observed = _observed_simulator_timestep(env)
            if chunk_events_path is not None:
                chunk_event = _build_chunk_event(
                    campaign_id=campaign_id,
                    worker_id=worker_id,
                    benchmark_name=benchmark_name,
                    task_index=task_index,
                    episode_index=episode_index,
                    episode_attempt_id=episode_attempt_id,
                    rollout_chunk=rollout_chunk,
                    env_steps_before=env_steps_before,
                    env_steps_after=env_steps,
                    simulator_timestep_before=simulator_timestep_before_observed,
                    simulator_timestep_after=simulator_timestep_after_observed,
                    rewards=chunk_rewards,
                    done=done,
                    action_shape=[int(value) for value in action.shape],
                )
                # Flush each event for recovery, but avoid per-chunk fsync so
                # diagnostic logging does not impose a durability sync on every
                # policy chunk. Such runs are not clean latency benchmarks.
                _append_jsonl(chunk_events_path, chunk_event, fsync=False)

            infer_chunks += 1
            if done:
                break
            if not key_frame_list:
                raise RuntimeError("model returned no complete action frame before the LIBERO horizon")

            # This call only advances the server's observation/action KV cache;
            # it does not create another denoising trace chunk.
            timed_infer("cache", {
                "obs": key_frame_list,
                "compute_kv_cache": True,
                "imagine": False,
                "state": action,
                "p0_trace_context": {
                    **context_base,
                    "rollout_chunk": int(infer_chunks - 1),
                    "env_steps_after": int(env_steps),
                    "simulator_timestep_after": int(
                        _simulator_timestep(env, simulator_timestep_start + env_steps)
                    ),
                },
            })
            first = False
            obs = key_frame_list[-1]

        simulator_timestep_final = _simulator_timestep(
            env, simulator_timestep_start + env_steps
        )
        horizon_reached = not done and simulator_timestep_final >= LIBERO_HORIZON

        record: dict[str, Any] = {
            "status": "completed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "benchmark": benchmark_name,
            "task_index": int(task_index),
            "episode_index": int(episode_index),
            "episode_attempt_id": episode_attempt_id,
            "init_state_count": int(init_states.shape[0]),
            "prompt": prompt,
            "success": bool(done),
            "horizon_reached": bool(horizon_reached),
            # Retained as a fixed compatibility field for historical manifest
            # consumers.  The executable client has no truncation option and
            # always runs until task success or the complete LIBERO horizon.
            "max_policy_chunks": None,
            "truncated_after_max_policy_chunks": False,
            "libero_horizon": LIBERO_HORIZON,
            "horizon_basis": "env.env.timestep",
            "stabilization_env_steps": INIT_STABILIZATION_STEPS,
            "simulator_timestep_start": int(simulator_timestep_start),
            "simulator_timestep_final": int(simulator_timestep_final),
            "env_steps": int(env_steps),
            "env_steps_semantics": "policy actions after stabilization; excludes reset/no-op steps",
            "infer_chunks": int(infer_chunks),
            "action_shapes": action_shapes,
            "elapsed_sec": float(time.monotonic() - started),
            "timing_clock": "client time.monotonic around each websocket model.infer call",
            **inference_timing,
            "model_request_wall_sec": float(
                inference_timing["reset_infer_wall_sec"]
                + inference_timing["policy_infer_wall_sec"]
                + inference_timing["cache_infer_wall_sec"]
            ),
            "trace_context": context_base,
        }
        if native_observer is not None:
            record["native_diagnostic"] = native_observer.summary()
        if save_video and full_obs:
            _save_video(full_obs, video_root / benchmark_name / f"task{task_index}" / f"episode{episode_index}.mp4")
            record["video_path"] = str(video_root / benchmark_name / f"task{task_index}" / f"episode{episode_index}.mp4")
        return record
    finally:
        env.close()


def _save_video(observations: list[dict[str, np.ndarray]], path: Path) -> None:
    import cv2
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    names = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
    frames = []
    base_h, base_w = observations[0][names[0]].shape[:2]
    for observation in observations:
        frames.append(np.hstack([
            cv2.resize(observation[name], (base_w, base_h)) for name in names
        ]).astype(np.uint8))
    imageio.mimsave(path, frames, fps=60)


def _checkpoint_alignment_metadata(args: argparse.Namespace) -> dict[str, str] | None:
    """Validate the launcher-provided suite/checkpoint contract for a client run."""
    names = (
        "LINGBOT_VA_LIBERO_SUITE",
        "LINGBOT_VA_LIBERO_CHECKPOINT",
        "LINGBOT_VA_LIBERO_CHECKPOINT_ID",
        "LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY",
    )
    values = {name: os.getenv(name, "").strip() for name in names}
    if not any(values.values()):
        return None
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit(
            "partial LIBERO checkpoint alignment environment: missing "
            + ", ".join(missing)
        )
    if args.benchmarks != [values["LINGBOT_VA_LIBERO_SUITE"]]:
        raise SystemExit(
            "checkpoint-aligned client must evaluate exactly its resolved suite "
            f"{values['LINGBOT_VA_LIBERO_SUITE']!r}, not {args.benchmarks!r}"
        )
    return {
        "suite": values["LINGBOT_VA_LIBERO_SUITE"],
        "local_path": values["LINGBOT_VA_LIBERO_CHECKPOINT"],
        "official_checkpoint": values["LINGBOT_VA_LIBERO_CHECKPOINT_ID"],
        "registry": values["LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY"],
    }


def _configure_trusted_local_libero_loads() -> None:
    # The bundled LIBERO init-state archives are trusted local experiment
    # inputs.  PyTorch 2.6+ defaults torch.load(..., weights_only=True),
    # which rejects these NumPy-backed archives.  Scope the compatibility
    # switch to this CPU client process; never export it from the GPU server
    # launcher or use it for untrusted files.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def _install_transport(worktree: Path):
    remote_root = worktree.resolve() / "wan_va" / "utils" / "Simple_Remote_Infer"
    if not remote_root.is_dir():
        raise SystemExit(f"transport client not found under {worktree}")
    sys.path.insert(0, str(remote_root))
    from deploy.websocket_client_policy import WebsocketClientPolicy
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    # Keep the simulator class local to this client process without changing
    # the transport API.  It avoids importing the CUDA WAM package.
    WebsocketClientPolicy._p0_offscreen_env_cls = OffScreenRenderEnv
    return WebsocketClientPolicy, benchmark


def main() -> int:
    args = parse_args()
    if args.episodes_per_task < 1 or args.episode_start < 0:
        raise SystemExit("episodes-per-task must be positive and episode-start non-negative")
    if not 1 <= args.port <= 65535:
        raise SystemExit(f"port must be in [1, 65535], got {args.port}")
    if args.master_port is not None and not 1 <= args.master_port <= 65535:
        raise SystemExit(f"master-port must be in [1, 65535], got {args.master_port}")
    task_indices = _resolve_task_indices(args)
    checkpoint_alignment = _checkpoint_alignment_metadata(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "episodes.jsonl"
    chunk_events_path = args.out_dir / "chunk_events.jsonl"
    completed = _read_completed(manifest_path) if args.resume else set()
    run_meta = args.out_dir / "run_metadata.json"
    if not run_meta.exists():
        run_meta.write_text(json.dumps({
            "created_at": _utc_now(),
            "script": str(Path(__file__).resolve()),
            "benchmarks": args.benchmarks,
            "task_range": _contiguous_task_range(task_indices),
            "task_indices": list(task_indices),
            "episodes_per_task": args.episodes_per_task,
            "episode_start": args.episode_start,
            # Historical schema compatibility only; no bounded-rollout CLI
            # entry exists in this formal client.
            "max_policy_chunks": None,
            "libero_horizon": LIBERO_HORIZON,
            "horizon_basis": "env.env.timestep",
            "stabilization_env_steps": INIT_STABILIZATION_STEPS,
            "condition": args.condition,
            "campaign_note": args.campaign_note,
            "trace_mode": args.trace_mode,
            "server_trace_mode": os.getenv("WAM_P0_TRACE_MODE") if args.trace_mode == "on" else None,
            "seed": os.getenv("LINGBOT_SEED"),
            "source_revision": os.getenv("WAM_P0_SOURCE_REVISION"),
            "training_config_hash": os.getenv("WAM_P0_TRAINING_CONFIG_HASH"),
            "latency_benchmark": bool(args.latency_benchmark),
            "campaign_id": args.campaign_id,
            "parallel_worker_id": args.worker_id,
            "server_port": args.port,
            "master_port": args.master_port,
            "checkpoint_alignment": checkpoint_alignment,
            "trace_mode_note": ("Server traces are diagnostic and record the server-selected "
                                "action source per update; this client only executes returned actions."),
            "chunk_event_schema_version": "wam-p0-chunk-event/v1",
            "chunk_event_path": str(chunk_events_path),
            "chunk_event_logging_note": (
                "Per-policy-chunk diagnostic events; flushes without per-event fsync; "
                "do not use this instrumented run as a clean latency benchmark."
            ),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _configure_trusted_local_libero_loads()
    WebsocketClientPolicy, benchmark = _install_transport(args.worktree)
    model = WebsocketClientPolicy(port=args.port)
    # Make the env constructor available to _run_episode without importing
    # simulator modules in the CUDA server process.
    for benchmark_name in args.benchmarks:
        benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
        invalid = [value for value in task_indices if value >= benchmark_instance.get_num_tasks()]
        if invalid:
            raise SystemExit(
                f"task indices {invalid} exceed {benchmark_name} task count "
                f"{benchmark_instance.get_num_tasks()}"
            )
        print(
            f"[client] benchmark={benchmark_name} tasks={list(task_indices)} "
            f"episodes={args.episodes_per_task}",
            flush=True,
        )
        for task_index in task_indices:
            for offset in range(args.episodes_per_task):
                episode_index = args.episode_start + offset
                key = _episode_key(benchmark_name, task_index, episode_index)
                if key in completed:
                    print(f"[client] skip completed {key}", flush=True)
                    continue
                print(f"[client] start benchmark={benchmark_name} task={task_index} episode={episode_index}", flush=True)
                episode_attempt_id = uuid.uuid4().hex
                try:
                    record = _run_episode(
                        model=model,
                        benchmark_name=benchmark_name,
                        benchmark_instance=benchmark_instance,
                        task_index=task_index,
                        episode_index=episode_index,
                        save_video=args.save_video,
                        video_root=args.out_dir / "videos",
                        campaign_id=args.campaign_id,
                        worker_id=args.worker_id,
                        episode_attempt_id=episode_attempt_id,
                        chunk_events_path=chunk_events_path,
                    )
                except Exception as exc:
                    record = {
                        "status": "error",
                        "started_at": _utc_now(),
                        "finished_at": _utc_now(),
                        "benchmark": benchmark_name,
                        "task_index": int(task_index),
                        "episode_index": int(episode_index),
                        "episode_attempt_id": episode_attempt_id,
                        "campaign_id": args.campaign_id,
                        "parallel_worker_id": args.worker_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    if args.fail_fast:
                        _append_jsonl(manifest_path, record)
                        raise
                _append_jsonl(manifest_path, record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    print(f"[client] manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

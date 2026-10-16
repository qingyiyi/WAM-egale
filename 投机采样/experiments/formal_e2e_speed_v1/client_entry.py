"""Time the unmodified full-horizon LIBERO client loop for one local pair arm.

The historical lean E2E wrapper supplies identical seeding and initial-state
snapshots. This wrapper adds only monotonic timers around simulator operations;
all hashes/files are written after the original episode timer returns.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from speed_contract import EXP, ROOT, validate, write


def _lean_wrapper():
    path = EXP / "speculative_e2e_v3/client_entry.py"
    spec = importlib.util.spec_from_file_location("formal_speed_frozen_lean_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(client, out: Path, lean=None):
    lean = lean or _lean_wrapper()
    lean.install(client, out)
    construct, init, episode = client._construct_env, client._init_env, client._run_episode
    timers = {}

    def timed_construct(cls, args):
        started = time.monotonic()
        env = construct(cls, args)
        timers["environment_construct_sec"] = time.monotonic() - started
        timers["simulator_step_stabilization_sec"] = 0.
        timers["simulator_step_policy_sec"] = 0.
        timers["simulator_step_stabilization_calls"] = 0
        timers["simulator_step_policy_calls"] = 0
        timers["stage"] = "stabilization"
        original_step, original_close = env.step, env.close

        def step(action):
            start = time.monotonic()
            result = original_step(action)
            stage = timers["stage"]
            timers[f"simulator_step_{stage}_sec"] += time.monotonic() - start
            timers[f"simulator_step_{stage}_calls"] += 1
            return result

        def close():
            start = time.monotonic()
            try:
                return original_close()
            finally:
                timers["environment_close_sec"] = time.monotonic() - start

        env.step, env.close = step, close
        return env

    def timed_init(env, init_state):
        start = time.monotonic()
        result = init(env, init_state)
        timers["local_reset_stabilization_sec"] = time.monotonic() - start
        timers["stage"] = "policy"
        return result

    def timed_episode(**kwargs):
        timers.clear()
        record = episode(**kwargs)
        request_sec = sum(record[k] for k in (
            "reset_infer_wall_sec", "policy_infer_wall_sec", "cache_infer_wall_sec"))
        disjoint = (timers["local_reset_stabilization_sec"] + request_sec
                    + timers["simulator_step_policy_sec"])
        residual = record["elapsed_sec"] - disjoint
        if residual < -0.005:
            raise ValueError("phase timers overlap the base episode timer")
        if timers["simulator_step_policy_calls"] != record["env_steps"]:
            raise ValueError("policy simulator-step timer misses executed commands")
        timing = dict(
            environment_construct_sec=timers["environment_construct_sec"],
            local_reset_stabilization_sec=timers["local_reset_stabilization_sec"],
            simulator_step_stabilization_sec=timers["simulator_step_stabilization_sec"],
            simulator_step_stabilization_calls=timers["simulator_step_stabilization_calls"],
            reset_model_request_sec=record["reset_infer_wall_sec"],
            policy_model_request_sec=record["policy_infer_wall_sec"],
            cache_model_request_sec=record["cache_infer_wall_sec"],
            simulator_step_policy_sec=timers["simulator_step_policy_sec"],
            simulator_step_policy_calls=timers["simulator_step_policy_calls"],
            client_other_sec=max(0., residual),
            original_episode_elapsed_sec=record["elapsed_sec"],
            environment_close_sec=timers.get("environment_close_sec", 0.),
            outer_episode_wall_sec=record["outer_episode_wall_sec"],
            disjoint_phase_sum_sec=disjoint + max(0., residual),
            phase_semantics="local reset includes stabilization simulator steps; model-request timers include websocket transport; model loading and EGL construction are outside original episode elapsed",
        )
        record["formal_speed_phases"] = timing
        write(out / "PHASE-TIMING.json", timing)
        return record

    client._construct_env = timed_construct
    client._init_env = timed_init
    client._run_episode = timed_episode


def main():
    job = Path(os.environ["FORMAL_SPEED_JOB"]).resolve()
    plan = validate(job)
    key = os.environ["FORMAL_SPEED_KEY"]
    rows = [x for x in plan["schedule"] if x["key"] == key]
    if len(rows) != 1:
        raise ValueError("client schedule key mismatch")
    row = rows[0]
    if (os.getenv("MUJOCO_GL") != "egl" or os.getenv("PYOPENGL_PLATFORM") != "egl"
            or os.getenv("MUJOCO_EGL_DEVICE_ID") != "0"
            or os.getenv("CUDA_VISIBLE_DEVICES") != "0"
            or os.getenv("P0_NATIVE_DIAGNOSTIC_DIR")
            or os.getenv("WAM_P0_STATE_TEACHER_DIR")):
        raise ValueError("formal local EGL/trace/teacher boundary mismatch")
    sys.path.insert(0, str(ROOT / "投机采样/scripts"))
    import p0_libero_multichunk_client as client
    args = client.parse_args()
    expected_out = job / "episodes" / key / "client"
    if (args.benchmarks != ["libero_10"] or args.task_indices != [row["task"]]
            or args.episode_start != row["init"] or args.episodes_per_task != 1
            or args.condition != row["arm"] or args.out_dir.resolve() != expected_out
            or args.trace_mode != "off" or not args.latency_benchmark
            or args.resume or args.save_video):
        raise ValueError("formal one-episode client flags differ from frozen schedule")
    install(client, expected_out)
    return client.main()


if __name__ == "__main__":
    raise SystemExit(main())

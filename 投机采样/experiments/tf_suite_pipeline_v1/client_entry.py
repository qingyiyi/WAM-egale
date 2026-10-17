"""Original complete LIBERO episode with new no-digest raw T/C/P observation.

The original episode loop and its action/environment semantics remain in use.
No retry, shortened episode, synthetic query or independent test is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import uuid

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from tf_suite_pipeline_v1.protocol import (CAMERAS, EXP, ROOT, WORKTREE, PERMISSIONS,
    episode_dir, load, row as phase_row, require, write, split_role)


def utc():
    return datetime.now(timezone.utc).isoformat()


def seed(value):
    import numpy as np
    random.seed(value)
    np.random.seed(value)


def raw_rng():
    import numpy as np
    n = np.random.get_state()
    return dict(schema="python-numpy-raw-rng/v1", python_state=random.getstate(),
        numpy_state=dict(algorithm=n[0], keys=n[1].tolist(), keys_dtype=str(n[1].dtype),
            pos=n[2], has_gauss=n[3], cached_gaussian=n[4]), no_digest_computed=True)


def physical_state(env):
    import numpy as np
    e, sim = env.env, env.env.sim
    data = sim.data._data
    s = {"flattened": np.array(sim.get_state().flatten(), copy=True)}
    for name in ("qacc_warmstart", "ctrl", "act", "qacc", "qfrc_applied", "xfrc_applied",
            "mocap_pos", "mocap_quat", "qpos", "qvel"):
        s[name] = np.array(getattr(data, name), copy=True)
    for name in ("body_xpos", "body_xquat", "site_xpos", "site_xmat"):
        s[name] = np.array(getattr(sim.data, name), copy=True)
    controller = e.robots[0].controller
    for name in ("goal_pos", "goal_ori", "joint_pos", "joint_vel", "initial_joint"):
        if getattr(controller, name, None) is not None:
            s["controller_" + name] = np.array(getattr(controller, name), copy=True)
    require(all(bool(np.isfinite(v).all()) for v in s.values()), "nonfinite physical/controller state")
    return s


def install(client, model, context, row, plan):
    """Copy observations; forward every original business payload unchanged."""
    import numpy as np
    original_init, original_infer = client._init_env, model.infer
    memory = dict(commands=[], submitted=[], environment=[], chunks=[],
        active_chunk=None, command_offset=0, frames=None, construction_attempts=0)

    def construct(cls, args):
        memory["construction_attempts"] += 1
        require(memory["construction_attempts"] == 1, "one EGL construction attempt; no retry")
        seed(plan["evaluation_seed"])
        return cls(**args)

    def init(env, init_state):
        seed(plan["evaluation_seed"])
        env.seed(plan["evaluation_seed"])
        obs, timestep = original_init(env, init_state)
        require(set(obs) == set(CAMERAS), "both original camera observations required")
        images = {k: np.array(obs[k], copy=True) for k in CAMERAS}
        require(all(v.dtype == np.uint8 and v.shape == (128, 128, 3) for v in images.values()),
            "original initial image geometry/dtype differs")
        memory.update(initial=physical_state(env), images=images, initial_rng=raw_rng(),
            init_state=np.array(init_state, copy=True), xml=env.env.sim.model.get_xml(), initial_timestep=timestep)
        outer_step, inner_step, original_close = env.step, env.env.step, env.close

        def internal(command, *args, **kwargs):
            value = np.array(command, dtype=np.float32, copy=True)
            result = inner_step(command, *args, **kwargs)
            memory["environment"].append(dict(chunk=memory["active_chunk"], command=value.tolist(),
                env_index=len(memory["environment"]), done=bool(result[2])))
            return result

        def step(command):
            value = np.array(command, dtype=np.float32, copy=True)
            frames = memory["frames"]
            require(frames is not None, "environment action before original policy output")
            offset, chunk = memory["command_offset"], memory["active_chunk"]
            frame = (1 if chunk == 0 else 0) + offset // frames.shape[2]
            subaction = offset % frames.shape[2]
            require(value.tobytes() == np.asarray(frames[:, frame, subaction], dtype=np.float32).tobytes(),
                "physical transport to submitted action differs")
            memory["submitted"].append(dict(chunk=chunk, command_offset=offset, frame=frame,
                subaction=subaction, command=value.tolist(), env_index=len(memory["submitted"])))
            memory["commands"].append(value)
            memory["command_offset"] += 1
            return outer_step(command)

        def close():
            memory["final"], memory["final_rng"] = physical_state(env), raw_rng()
            return original_close()

        env.env.step, env.step, env.close = internal, step, close
        return obs, timestep

    def infer(request):
        require("p0_suite_context" not in request, "original business request unexpectedly has new routing field")
        routed = dict(request, p0_suite_context=dict(context))
        result = original_infer(routed)
        if not request.get("reset") and not request.get("compute_kv_cache"):
            trace = dict(request["p0_trace_context"])
            chunk = trace["rollout_chunk"]
            frames = np.array(result["action"], copy=True)
            require(frames.shape == (7, 4, 4) and frames.dtype == np.float32 and bool(np.isfinite(frames).all()),
                "actual physical action shape/dtype/finiteness differs")
            memory.update(active_chunk=chunk, command_offset=0, frames=frames)
            memory["chunks"].append(dict(chunk=chunk, context=trace, p0_suite_context=dict(context),
                received_action=frames.tolist(), dtype=str(frames.dtype), shape=list(frames.shape)))
        return result

    client._construct_env, client._init_env, model.infer = construct, init, infer
    return memory


def save_raw(folder, memory, record, context, seed_value):
    import numpy as np
    def npz(name, values):
        with (folder / name).open("xb") as f:
            np.savez(f, **values)
    def npy(name, values):
        with (folder / name).open("xb") as f:
            np.save(f, values, allow_pickle=False)
    if "initial" in memory:
        npz("initial_state.npz", memory["initial"])
        npz("initial_observations.npz", memory["images"])
        npy("init_state.npy", memory["init_state"])
        with (folder / "model.xml").open("x") as f:
            f.write(memory["xml"])
        write(folder / "RNG-INITIAL.json", memory["initial_rng"])
    if "final" in memory:
        npz("final_state.npz", memory["final"])
        write(folder / "RNG-FINAL.json", memory["final_rng"])
    commands = np.asarray(memory["commands"], dtype=np.float32).reshape(-1, 7)
    npy("commands.npy", commands)
    complete = record is not None
    require(not complete or len(commands) == len(memory["submitted"]) == len(memory["environment"]) == record["env_steps"],
        "full original submitted/environment command coverage differs")
    for a, b in zip(memory["submitted"], memory["environment"]):
        require(a["env_index"] == b["env_index"] and a["chunk"] == b["chunk"] and a["command"] == b["command"],
            "submitted action differs at underlying LIBERO environment boundary")
    mapping = dict(schema="physical-action-to-environment-command/v1", context=context,
        chunks=memory["chunks"], submitted=memory["submitted"], environment=memory["environment"],
        complete_episode=complete, env_steps=len(commands), success=record["success"] if complete else None,
        termination_reason=("success" if record["success"] else "full_horizon") if complete else "PARTIAL_EXCEPTION",
        performance_claim=False, environment_command_boundary="argument to original underlying env.env.step, not actuator ctrl")
    write(folder / "COMMAND-MAPPING.json", mapping)
    write(folder / "repro_metadata.json", dict(schema="tf-suite-raw-client-reproduction/v1", context=context,
        seed=seed_value, actual_EGL_initialization_attempts=memory["construction_attempts"],
        initial_timestep=memory.get("initial_timestep"), command_count=len(commands),
        no_policy_chunk_truncation=True, no_retry=True, no_digest_computed=True,
        raw_initial_rng="RNG-INITIAL.json", raw_final_rng="RNG-FINAL.json", raw_model_xml="model.xml",
        physical_state_scope="saved flattened simulator and listed arrays/controller fields; unsaved internals UNKNOWN",
        original_full_episode_loop=True, physical_transport_submission_environment_exact=complete,
        observation_artifact_writes_after_episode_timer=True, permissions=PERMISSIONS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--row-id", type=int, required=True)
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--arm", choices=("T", "C", "P"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--master-port", type=int)
    args = parser.parse_args()
    started, started_mono = time.time(), time.monotonic()
    job, out = args.job.resolve(strict=True), args.out.resolve()
    p = load(job)
    row = phase_row(p, args.row_id)
    require(not row.get("reuse_source"), "a read-only reused T row cannot start a new episode")
    require(args.case_id in (None, row["case_id"]) and args.arm in (None, row["arm"]), "row/case/arm differs")
    args.case_id, args.arm = row["case_id"], row["arm"]
    require(out == episode_dir(job, args.case_id, args.arm) and args.worker_index == row["worker_index"] == 0 and args.port == p["port"]
        and 1 <= args.port <= 65535, "registered fullsuite case/out/worker/port differs")
    require(os.getenv("MUJOCO_GL") == "egl" and os.getenv("PYOPENGL_PLATFORM") == "egl"
        and not os.getenv("P0_NATIVE_DIAGNOSTIC_DIR") and not os.getenv("WAM_P0_STATE_TEACHER_DIR")
        and len(os.sched_getaffinity(0)) <= p["client_threads_per_worker"], "original EGL/lean client boundary differs")
    require(not (out / "EPISODE-TERMINAL.json").exists() and not (out / "client").exists(), "no retry/overwrite")
    folder = out / "client"
    folder.mkdir(parents=True, exist_ok=False)
    context = dict(row_id=args.row_id, case_id=args.case_id, arm=args.arm, out=str(out))
    write(out / "CLIENT-STARTED.json", dict(schema="tf-suite-client-start/v1", pid=os.getpid(), context=context,
        task=row["task"], init=row["init"], at=utc(), worker_index=args.worker_index, permissions=PERMISSIONS))
    deadline = min(started + p["per_episode_wall_sec"], datetime.fromisoformat(p["absolute_work_stop"]).timestamp())
    require(deadline > time.time(), "whole-episode work window already exhausted")
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("registered complete episode deadline")))
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    memory, record = None, None
    try:
        sys.path.insert(0, str(ROOT / "投机采样/scripts"))
        import p0_libero_multichunk_client as client
        require(Path(client.__file__).resolve() == ROOT / "投机采样/scripts/p0_libero_multichunk_client.py"
            and client.LIBERO_HORIZON == 800, "original complete client source/horizon differs")
        client._configure_trusted_local_libero_loads()
        policy_cls, benchmark = client._install_transport(WORKTREE)
        transport = sys.modules[policy_cls.__module__]
        def connect_once(self):
            # The old waiting loop catches every Exception and can swallow the
            # single SIGALRM. Readiness admission belongs to the runner; this
            # actual client connection is attempted once, without a retry.
            headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
            conn = transport.websockets.sync.client.connect(self._uri, compression=None,
                max_size=None, additional_headers=headers, ping_interval=None,
                close_timeout=10, open_timeout=30)
            return conn, transport.unpackb(conn.recv(timeout=30))
        policy_cls._wait_for_server = connect_once
        model = policy_cls(port=args.port)
        memory = install(client, model, context, row, p)
        seed(p["evaluation_seed"])
        benchmark_instance = benchmark.get_benchmark_dict()[p["suite"]]()
        require(benchmark_instance.get_task(row["task"]).language == p["suite_info"]["public_task_prompts"][str(row["task"])], "actual suite prompt differs from target bank manifest")
        attempt = uuid.uuid4().hex
        record = client._run_episode(model=model, benchmark_name=p["suite"], benchmark_instance=benchmark_instance,
            task_index=row["task"], episode_index=row["init"], save_video=False, video_root=folder / "unused-videos",
            campaign_id=job.name, worker_id=f"row-{args.row_id:04d}-case-{args.case_id:04d}-{args.arm}", episode_attempt_id=attempt,
            chunk_events_path=None)
        require(record["status"] == "completed" and (record["success"] or record["horizon_reached"] is True)
            and record["max_policy_chunks"] is None and record["truncated_after_max_policy_chunks"] is False,
            "original full natural success/horizon terminal required")
        save_raw(folder, memory, record, context, p["evaluation_seed"])
        with (folder / "episodes.jsonl").open("x") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        terminal = dict(schema="tf-suite-full-episode-terminal/v1", status="COMPLETE_RECORDED_FULL_EPISODE",
            row_id=args.row_id, case_id=args.case_id, arm=args.arm, task=row["task"], init=row["init"], context=context, suite=p["suite"], phase=p["phase"], dataset_role=split_role(p["phase"], row["init"]),
            success=record["success"], env_steps=record["env_steps"],
            failure800_penalty=record["env_steps"] if record["success"] else 800,
            horizon=800, horizon_reached=record["horizon_reached"], policy_chunks=record["infer_chunks"],
            elapsed_sec=record["elapsed_sec"], client_infer_timing={k: record[k] for k in ("reset_infer_wall_sec", "policy_infer_wall_sec", "cache_infer_wall_sec", "reset_infer_calls", "policy_infer_calls", "cache_infer_calls")}, client_total_wall_sec=time.monotonic() - started_mono,
            complete_episode=True, quality_terminal=True, episode_attempt_id=attempt,
            termination_reason="success" if record["success"] else "full_horizon",
            command_mapping_exact=True, no_digest_computed=True, no_retry=True,
            transport_connection_attempts=1,
            original_elapsed_boundary="after environment construction, before reset; includes reset/policy/cache/environment, excludes initial model load and environment construction",
            permissions=PERMISSIONS, performance_claim=False, finished_at=utc())
        write(out / "EPISODE-TERMINAL.json", terminal)
        print(json.dumps(terminal, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        artifact_error = None
        if memory is not None and not (folder / "commands.npy").exists():
            try:
                save_raw(folder, memory, None, context, p["evaluation_seed"])
            except BaseException as artifact_exc:
                artifact_error = repr(artifact_exc)
        terminal = dict(schema="tf-suite-full-episode-terminal/v1", status="PARTIAL_CLIENT_EXCEPTION",
            row_id=args.row_id, case_id=args.case_id, arm=args.arm, task=row["task"], init=row["init"], context=context, suite=p["suite"], phase=p["phase"], dataset_role=split_role(p["phase"], row["init"]),
            success=None, env_steps=len(memory["commands"]) if memory is not None else 0,
            failure800_penalty=None, complete_episode=False, quality_terminal=False,
            exception_is_not_algorithm_failure=True, error=repr(exc), traceback=traceback.format_exc(),
            raw_artifact_error=artifact_error,
            recorded_robot_outcome_if_already_returned={k: record[k] for k in ("success", "env_steps", "horizon_reached")} if record is not None else None,
            client_total_wall_sec=time.monotonic() - started_mono, permissions=PERMISSIONS, performance_claim=False, finished_at=utc())
        if not (out / "EPISODE-TERMINAL.json").exists():
            write(out / "EPISODE-TERMINAL.json", terminal)
        print(json.dumps(terminal, ensure_ascii=False), flush=True)
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == "__main__":
    raise SystemExit(main())

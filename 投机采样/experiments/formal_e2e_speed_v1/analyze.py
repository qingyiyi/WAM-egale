"""CPU-only audit and paired summary for local natural E2E speed timing."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

from speed_contract import validate, read, sha, write


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * fraction
    i = int(k)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (k - i)


def paired_bootstrap_ratio(rows, base_key, candidate_key, draws=5000):
    if not rows:
        return None
    rng = random.Random(20260922)
    ratios = []
    for _ in range(draws):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        ratios.append(sum(x[base_key] for x in sample) / sum(x[candidate_key] for x in sample))
    return [percentile(ratios, .025), percentile(ratios, .975)]


def validate_episode(job, plan, row):
    out = job / "episodes" / row["key"]
    ready = read(out / "READY.json")
    run = read(out / "RUN-TIMING.json")
    phase = read(out / "client/PHASE-TIMING.json")
    meta = read(out / "client/repro_metadata.json")
    counter = read(out / "COUNTER-CLOSED.json")
    acceptance = read(out / "ACCEPTANCE-TREES.json")
    records = lines(out / "client/episodes.jsonl")
    if len(records) != 1:
        raise ValueError("exactly one natural episode required: " + row["key"])
    episode = records[0]
    if (ready["status"] != "READY" or ready["key"] != row["key"]
            or ready["arm"] != row["arm"] or ready["task"] != row["task"]
            or ready["init"] != row["init"]
            or ready["draft_sha256"] != plan["draft_sha256"]
            or ready["target_checkpoint_id"] != plan["target_checkpoint_id"]
            or run["row"] != row or run["client_status"] != "completed"):
        raise ValueError("model/run provenance mismatch: " + row["key"])
    if (episode.get("status") != "completed" or episode.get("benchmark") != "libero_10"
            or episode.get("task_index") != row["task"]
            or episode.get("episode_index") != row["init"]
            or episode.get("libero_horizon") != 800
            or episode.get("max_policy_chunks") is not None
            or episode.get("truncated_after_max_policy_chunks") is not False
            or not (episode.get("success") or episode.get("horizon_reached"))
            or not 0 < episode.get("env_steps", 0) <= 800
            or episode.get("infer_chunks", 0) <= 0):
        raise ValueError("incomplete or truncated natural episode: " + row["key"])
    if (meta["seed"] != plan["seed"] or meta["actual_EGL_initialization_attempts"] != 1
            or meta["command_count"] != episode["env_steps"]
            or not meta["no_policy_chunk_truncation"]
            or meta["per_step_file_IO"]
            or meta["chunk_event_logging"]):
        raise ValueError("EGL/reproduction/latency client contract mismatch: " + row["key"])
    if (counter["status"] != "CLOSED" or counter["failed"] or counter["active"]
            or counter["target_intents"] != counter["returned"]
            or not counter["per_request_file_io"] is False):
        raise ValueError("target-forward accounting incomplete: " + row["key"])
    requests = counter["requests"]
    for kind, expected in (("reset", episode["reset_infer_calls"]),
                           ("policy", episode["policy_infer_calls"]),
                           ("cache", episode["cache_infer_calls"])):
        if sum(x["kind"] == kind for x in requests) != expected:
            raise ValueError("client/server request-count mismatch: " + row["key"])
    if any(not request["ok"] for request in requests):
        raise ValueError("failed model request: " + row["key"])
    if row["arm"] == "T":
        if ready["mode"] != "off" or acceptance["records"]:
            raise ValueError("target baseline unexpectedly speculative")
    else:
        phases = counter["phase_counters"]
        if (ready["mode"] != "verifier_k2_b2_bonus"
                or ready["selector"] != "prefix_then_cost"
                or not ready["O1_O2_O3"]
                or len(acceptance["records"]) != phases.get("tree_count", -1)
                or phases.get("committed_scheduler_update_count") != 50 * episode["infer_chunks"]):
            raise ValueError("current A_off bonus/prefix/O3 coverage mismatch")
    if (phase["simulator_step_policy_calls"] != episode["env_steps"]
            or not math.isclose(phase["original_episode_elapsed_sec"], episode["elapsed_sec"],
                                rel_tol=0, abs_tol=1e-6)
            or not math.isclose(phase["disjoint_phase_sum_sec"], episode["elapsed_sec"],
                                rel_tol=0, abs_tol=.005)
            or phase["simulator_step_stabilization_calls"] != meta["actual_EGL_initialization_attempts"] * episode["stabilization_env_steps"]):
        raise ValueError("phase timers do not cover original episode: " + row["key"])
    if any(not math.isfinite(value) or value < 0 for key, value in phase.items()
           if key.endswith("_sec")):
        raise ValueError("nonfinite/negative phase timing: " + row["key"])
    return dict(row=row, episode=episode, phase=phase, meta=meta, run=run,
                target_intents=counter["target_intents"],
                accepted_depths={i: counter["phase_counters"].get(f"accepted_depth_{i}", 0)
                                 for i in (0, 1, 2)})


def summarize(pairs):
    all_elapsed = sum(x["T_elapsed_sec"] for x in pairs) / sum(x["A_OFF_elapsed_sec"] for x in pairs)
    both = [x for x in pairs if x["T_success"] and x["A_OFF_success"]]
    ratios = [x["episode_speedup"] for x in pairs]
    task_summary = {}
    for task in sorted({x["task"] for x in pairs}):
        subset = [x for x in pairs if x["task"] == task]
        task_summary[str(task)] = dict(
            pairs=len(subset), T_successes=sum(x["T_success"] for x in subset),
            A_OFF_successes=sum(x["A_OFF_success"] for x in subset),
            complete_episode_speedup=sum(x["T_elapsed_sec"] for x in subset) /
                                    sum(x["A_OFF_elapsed_sec"] for x in subset),
            T_env_steps=sum(x["T_env_steps"] for x in subset),
            A_OFF_env_steps=sum(x["A_OFF_env_steps"] for x in subset))
    return dict(pairs=len(pairs), all_episode_total_speedup=all_elapsed,
                both_success_pairs=len(both),
                both_success_total_speedup=(sum(x["T_elapsed_sec"] for x in both) /
                                            sum(x["A_OFF_elapsed_sec"] for x in both)) if both else None,
                per_pair_speedup_P50=percentile(ratios, .5),
                per_pair_speedup_P90=percentile(ratios, .9),
                paired_bootstrap_95pct_ratio=paired_bootstrap_ratio(
                    pairs, "T_elapsed_sec", "A_OFF_elapsed_sec"),
                T_successes=sum(x["T_success"] for x in pairs),
                A_OFF_successes=sum(x["A_OFF_success"] for x in pairs),
                task_summary=task_summary)


def analyze(job: Path, plan: dict, *, require_collection=True):
    if require_collection:
        collected = read(job / "GPU-COLLECTION.json")
        if (collected["status"] != "PASS_LOCAL_FORMAL_E2E_COLLECTION"
                or collected["completed"] != [x["key"] for x in plan["schedule"]]
                or collected["expected"] != len(plan["schedule"])):
            raise ValueError("all planned formal episodes required")
    seen = {row["key"]: validate_episode(job, plan, row) for row in plan["schedule"]}
    telemetry = lines(job / "GPU-TELEMETRY.jsonl")
    if not telemetry or any(row["foreign"] for row in telemetry):
        raise ValueError("missing or foreign-contaminated GPU telemetry")
    phase_seen = {row["stage"] for row in telemetry}
    if "idle_admission" not in phase_seen or "natural_episode" not in phase_seen:
        raise ValueError("GPU idle and workload telemetry both required")
    paired = []
    for i, pair in enumerate(plan["pairs"]):
        rows = [seen[x["key"]] for x in plan["schedule"] if x["pair"] == i]
        if len(rows) != 2 or {x["row"]["arm"] for x in rows} != {"T", "A_OFF"}:
            raise ValueError("paired T/A_off coverage incomplete")
        by_arm = {x["row"]["arm"]: x for x in rows}
        t, a = by_arm["T"], by_arm["A_OFF"]
        for name in ("model_xml_sha256", "init_state_sha256", "initial_state_hashes",
                     "initial_obs_hashes", "initial_rng"):
            if t["meta"][name] != a["meta"][name]:
                raise ValueError("paired initial simulator/prompt state differs: " + pair + "/" + name)
        result = dict(pair=i, task=t["row"]["task"], init=t["row"]["init"],
                      order=[x["row"]["arm"] for x in rows],
                      T_success=t["episode"]["success"], A_OFF_success=a["episode"]["success"],
                      T_env_steps=t["episode"]["env_steps"], A_OFF_env_steps=a["episode"]["env_steps"],
                      T_chunks=t["episode"]["infer_chunks"], A_OFF_chunks=a["episode"]["infer_chunks"],
                      T_elapsed_sec=t["episode"]["elapsed_sec"],
                      A_OFF_elapsed_sec=a["episode"]["elapsed_sec"],
                      T_cold_launch_to_client_exit_sec=t["run"]["cold_server_launch_to_client_exit_sec"],
                      A_OFF_cold_launch_to_client_exit_sec=a["run"]["cold_server_launch_to_client_exit_sec"],
                      T_model_load_sec=t["run"]["model_init_sec"],
                      A_OFF_model_load_sec=a["run"]["model_init_sec"],
                      phases={arm: by_arm[arm]["phase"] for arm in ("T", "A_OFF")},
                      target_intents={arm: by_arm[arm]["target_intents"] for arm in ("T", "A_OFF")},
                      A_OFF_accepted_depths=a["accepted_depths"])
        result["episode_speedup"] = result["T_elapsed_sec"] / result["A_OFF_elapsed_sec"]
        result["cold_launch_to_exit_speedup"] = (
            result["T_cold_launch_to_client_exit_sec"] /
            result["A_OFF_cold_launch_to_client_exit_sec"])
        paired.append(result)
    return dict(status="PASS_LOCAL_FORMAL_E2E_PAIRED_AUDIT", job=str(job),
                source_plan_sha256=sha(job / "plan.json"),
                label=plan["label"], schedule=plan["schedule"], paired=paired,
                summary=summarize(paired), GPU_telemetry_samples=len(telemetry),
                GPU_foreign_processes=0,
                caveat="Fresh model for each episode: natural client episode elapsed, model loading, and cold server-launch-to-client-exit are distinct. No warm episode or AutoDL latency claim. Historic global/S0 1.0817x is a different candidate and must not be pooled.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    args = ap.parse_args()
    plan = validate(args.job)
    result = analyze(args.job.resolve(), plan)
    write(args.job / "FINAL-ANALYSIS.json", result)
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

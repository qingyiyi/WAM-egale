"""No-digest descriptive report for the authorized Long500 T/C/P coverage.

Component observations never upgrade the four frozen permissions or old NO_GO.
Absolute/instrumented costs include complete failures and retain partial attempts.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from tf_fullsuite_eval_v1.protocol import (ARMS, CAMERAS, PERMISSIONS, episode_dir, load, read, require)

UNKNOWN = "UNKNOWN"
REPORT_CAP = 32 * 2**20


def write(path, obj):
    path = Path(path)
    raw = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    used = sum(x.stat().st_size for x in path.parent.iterdir() if x.is_file())
    require(used + len(raw) <= REPORT_CAP, "32MiB report cap before write")
    with path.open("xb") as f:
        f.write(raw)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def quantiles(values):
    require(all(number(x) for x in values), "finite nonnegative metric required")
    values = sorted(values)
    def at(q):
        if not values:
            return UNKNOWN
        x = (len(values) - 1) * q
        a, b = math.floor(x), math.ceil(x)
        return values[a] + (values[b] - values[a]) * (x - a)
    return dict(n=len(values), sum=sum(values), mean=sum(values) / len(values) if values else UNKNOWN,
        p50=at(.5), p90=at(.9), max=values[-1] if values else UNKNOWN, method="linear, index=(N-1)*q")


def safe(path):
    if not path.is_file():
        return None, "MISSING"
    try:
        return read(path), None
    except (OSError, ValueError, TypeError) as exc:
        return None, repr(exc)


def row(plan, job, spec, arm):
    out = episode_dir(job, spec["case_id"], arm)
    names = ("EPISODE-TERMINAL.json", "COUNTER.json", "AUDIT.json", "FINISH_READY.json")
    docs = {name: safe(out / name) for name in names}
    terminal, counter, audit, finish = [docs[n][0] for n in names]
    gaps = [{"file": n, "error": error} for n, (_, error) in docs.items() if error is not None]
    started = (out / "CLIENT-STARTED.json").is_file() or any(docs[n][0] is not None for n in names)
    r = dict(case_id=spec["case_id"], data_exposure=spec.get("data_exposure", "UNKNOWN"), task=spec["task"], init=spec["init"], arm=arm,
        within_case_order=spec["within_case_order"], source_dir=str(out), started=started,
        status="NOT_STARTED" if not started else "PARTIAL", complete_episode=False, success=UNKNOWN,
        actual_env_steps=UNKNOWN, failure800_penalty=UNKNOWN, policy_chunks=UNKNOWN, S_chunks=UNKNOWN, T_chunks=UNKNOWN,
        target_forward_attempts=UNKNOWN, target_forward_returned=UNKNOWN, target_forward_failed=UNKNOWN,
        action_total_sec=UNKNOWN, action_g=UNKNOWN, full_policy_elapsed_sec=UNKNOWN, sum_action_host_sec=UNKNOWN,
        C_update=UNKNOWN, C_chunk=UNKNOWN, action_per_chunk=UNKNOWN, sum_T_round_sec=UNKNOWN, sum_g=UNKNOWN,
        cost_per_legal_round_commit=UNKNOWN, client_episode_wall_sec=UNKNOWN, client_total_wall_sec=UNKNOWN,
        all_attempt_cost_covered=False, component_audit_status=UNKNOWN, first2S=UNKNOWN,
        numerical_isolation=UNKNOWN, evidence_gaps=gaps, no_quality_certificate=True, permissions=PERMISSIONS)
    if terminal is not None:
        try:
            require(terminal["schema"] == "tf-fullsuite-full-episode-terminal/v1" and terminal["case_id"] == spec["case_id"]
                and terminal["arm"] == arm and terminal["task"] == spec["task"] and terminal["init"] == spec["init"]
                and terminal["context"] == dict(case_id=spec["case_id"], arm=arm, out=str(out))
                and terminal["permissions"] == PERMISSIONS and terminal["performance_claim"] is False,
                "client terminal identity/permissions differ")
            if terminal["complete_episode"] is True:
                require(terminal["status"] == "COMPLETE_RECORDED_FULL_EPISODE" and terminal["quality_terminal"] is True
                    and type(terminal["success"]) is bool and type(terminal["env_steps"]) is int
                    and 0 < terminal["env_steps"] <= 800 and terminal["horizon"] == 800
                    and (terminal["success"] or terminal["horizon_reached"] is True)
                    and terminal["failure800_penalty"] == (terminal["env_steps"] if terminal["success"] else 800)
                    and terminal["policy_chunks"] > 0, "full client success/horizon/penalty terminal differs")
                r.update(status="COMPLETE_RECORDED_FULL_EPISODE", complete_episode=True, success=terminal["success"],
                    actual_env_steps=terminal["env_steps"], failure800_penalty=terminal["failure800_penalty"],
                    policy_chunks=terminal["policy_chunks"], client_episode_wall_sec=terminal["elapsed_sec"],
                    client_total_wall_sec=terminal["client_total_wall_sec"], client_infer_timing=terminal.get("client_infer_timing", UNKNOWN))
            else:
                r.update(status=terminal["status"], actual_env_steps=terminal.get("env_steps", UNKNOWN),
                    partial_exception=terminal.get("error"), partial_is_not_algorithm_failure=True,
                    recorded_robot_outcome_if_already_returned=terminal.get("recorded_robot_outcome_if_already_returned"),
                    client_total_wall_sec=terminal.get("client_total_wall_sec", UNKNOWN))
        except (KeyError, ValueError, TypeError) as exc:
            r["evidence_gaps"].append(dict(file="EPISODE-TERMINAL.json", error=repr(exc)))
    finish_valid = False
    if finish is not None:
        try:
            require(finish["status"] in ("PASS", "FAILED_COMPONENT_AUDIT")
                and finish["case_id"] == spec["case_id"] and finish["arm"] == arm
                and finish["full_episode_terminal"] is True and finish["async_drained"] is True
                and finish["permissions"] == PERMISSIONS, "original finish/drain component receipt differs")
            finish_valid = True
            r["finish_component_status"] = finish["status"]
        except (KeyError, ValueError, TypeError) as exc:
            r["evidence_gaps"].append(dict(file="FINISH_READY.json", error=repr(exc)))
    if counter is not None:
        try:
            require(counter["schema"] == "tf-fullsuite-episode-counter/v1" and counter["case_id"] == spec["case_id"]
                and counter["arm"] == arm and counter["task"] == spec["task"] and counter["init"] == spec["init"]
                and counter["permissions"] == PERMISSIONS, "counter identity/permissions differ")
            attempts, returned, failed = [counter[k] for k in
                ("target_forward_attempts", "target_forward_returned", "target_forward_failed")]
            active = counter.get("active_target_forwards", 0)
            require(all(type(x) is int and x >= 0 for x in (attempts, returned, failed, active))
                and attempts == returned + failed + active and attempts <= plan["per_episode_forward_attempt_cap"],
                "actual target forward attempt accounting differs")
            if r["complete_episode"]:
                require(counter["policy_chunks"] == r["policy_chunks"], "policy chunk coverage differs")
            r.update(target_forward_attempts=attempts, target_forward_returned=returned, target_forward_failed=failed,
                active_target_forwards_at_receipt=active,
                S_chunks=counter["S_chunks"], T_chunks=counter["T_chunks"],
                counter_policy_chunks=counter["policy_chunks"], total_actual_g=counter.get("total_actual_g", UNKNOWN),
                typed_forward_calls=counter.get("typed_forward_calls", UNKNOWN))
            policies = counter.get("policy_records", [])
            r["GPU_identity"] = policies[0].get("GPU_identity", UNKNOWN) if policies else UNKNOWN
            r["policy_resource_references"] = dict(source=str(out / "COUNTER.json"),
                policy_records=len(policies), no_policy_metadata_duplication=True)
            timing = counter["timing"]
            for name in ("action_total_sec", "action_g", "full_policy_elapsed_sec", "sum_action_host_sec"):
                if number(timing.get(name)):
                    r[name] = timing[name]
            require(r["action_g"] == UNKNOWN or type(r["action_g"]) is int, "actual action g must be an integer")
            chunks = counter["policy_chunks"]
            if number(r["action_total_sec"]) and type(r["action_g"]) is int and r["action_g"] > 0:
                r["C_update"] = r["action_total_sec"] / r["action_g"]
            if type(chunks) is int and chunks > 0:
                if number(r["full_policy_elapsed_sec"]):
                    r["C_chunk"] = r["full_policy_elapsed_sec"] / chunks
                if number(r["action_total_sec"]):
                    r["action_per_chunk"] = r["action_total_sec"] / chunks
            # A sum of action timers is not a complete T_round partition.
            if (timing.get("round_coverage_complete") is True and number(timing.get("sum_T_round_sec"))
                    and type(timing.get("sum_g")) is int and timing["sum_g"] > 0):
                r.update(sum_T_round_sec=timing["sum_T_round_sec"], sum_g=timing["sum_g"],
                    cost_per_legal_round_commit=timing["sum_T_round_sec"] / timing["sum_g"])
            r["all_attempt_cost_covered"] = (finish_valid and number(r["action_total_sec"])
                and type(r["action_g"]) is int and number(r["full_policy_elapsed_sec"])
                and timing.get("action_measurement_complete") is True and active == 0)
            if not r["all_attempt_cost_covered"]:
                r.update(C_update=UNKNOWN, C_chunk=UNKNOWN, action_per_chunk=UNKNOWN)
            r["timing_scope"] = timing.get("scope", "instrumented shared AutoDL exploratory; not formal performance")
        except (KeyError, ValueError, TypeError) as exc:
            r["evidence_gaps"].append(dict(file="COUNTER.json", error=repr(exc)))
            r["all_attempt_cost_covered"] = False
    if audit is not None:
        r["component_audit_status"] = audit.get("status", UNKNOWN)
        r["recorded_component_error"] = audit.get("error")
        try:
            require(audit["schema"] == "tf-fullsuite-episode-audit/v1" and audit["case_id"] == spec["case_id"]
                and audit["arm"] == arm and audit["permissions"] == PERMISSIONS, "component audit identity differs")
            r.update(component_audit_status=audit["status"], first2S=audit["first2S"],
                numerical_isolation=audit["numerical_isolation"])
        except (KeyError, ValueError, TypeError) as exc:
            r["evidence_gaps"].append(dict(file="AUDIT.json", error=repr(exc)))
    if r["complete_episode"]:
        try:
            r["raw_command_component"] = client_commands(out, r)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            r["evidence_gaps"].append(dict(file="client raw commands", error=repr(exc)))
    if started and (counter is None or type(r["target_forward_attempts"]) is not int):
        r["unknown_forward_attempt_debit"] = dict(actual=UNKNOWN, lower_known=0,
            upper_debited=plan["per_episode_forward_attempt_cap"], reason="missing/invalid complete counter; not zero work")
    return r


def raw_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        return {k: np.array(arrays[k], copy=True) for k in arrays.files}


def exact_arrays(a, b):
    return set(a) == set(b) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
        and a[k].tobytes() == b[k].tobytes() for k in a)


def client_commands(out, terminal):
    """Read actual raw commands; no digest or changed postprocessing formula."""
    import numpy as np
    folder = out / "client"
    commands = np.load(folder / "commands.npy", allow_pickle=False)
    mapping = read(folder / "COMMAND-MAPPING.json")
    require(mapping["schema"] == "physical-action-to-environment-command/v1"
        and mapping["context"] == dict(case_id=terminal["case_id"], arm=terminal["arm"], out=str(out)),
        "original fullsuite command mapping source/context differs")
    require(commands.dtype == np.float32 and commands.shape == (terminal["actual_env_steps"], 7)
        and bool(np.isfinite(commands).all()) and mapping["complete_episode"] is True
        and mapping["env_steps"] == terminal["actual_env_steps"] and mapping["success"] == terminal["success"]
        and len(mapping["submitted"]) == len(mapping["environment"]) == len(commands), "full raw command coverage differs")
    chunks = {r["chunk"]: r for r in mapping["chunks"]}
    for i, (a, b) in enumerate(zip(mapping["submitted"], mapping["environment"])):
        require(a["env_index"] == b["env_index"] == i and a["chunk"] == b["chunk"] and a["command"] == b["command"],
            "submitted/environment raw command differs")
        received = np.asarray(chunks[a["chunk"]]["received_action"], dtype=np.float32)
        require(np.asarray(a["command"], dtype=np.float32).tobytes() == commands[i].tobytes()
            == received[:, a["frame"], a["subaction"]].tobytes(), "physical transport/submitted/raw command differs")
    return dict(actual_command_count=len(commands), transport_submitted_environment_exact=True)


def compare_two(job, spec, by, first, second):
    import numpy as np
    c, p = [by[(spec["case_id"], arm)] for arm in (first, second)]
    result = dict(case_id=spec["case_id"], task=spec["task"], init=spec["init"], data_exposure=spec.get("data_exposure", "UNKNOWN"),
        comparison=first + "_" + second, first_arm=first, second_arm=second, within_case_order=spec["within_case_order"], complete_pair=c["complete_episode"] and p["complete_episode"],
        first_success=c["success"], second_success=p["success"], rescued=False, lost=False,
        first_steps=c["actual_env_steps"], second_steps=p["actual_env_steps"], both_success_step_ratio=UNKNOWN,
        raw_initial_comparison=UNKNOWN, first2S_comparison=UNKNOWN, evidence_gaps=[],
        strict_comparison_qualification=UNKNOWN, not_a_quality_certificate=True)
    if not result["complete_pair"]:
        return result
    result.update(rescued=not c["success"] and p["success"], lost=c["success"] and not p["success"],
        both_success_step_ratio=p["actual_env_steps"] / c["actual_env_steps"] if c["success"] and p["success"] else UNKNOWN)
    try:
        roots = [episode_dir(job, spec["case_id"], arm) / "client" for arm in (first, second)]
        a, b = [raw_arrays(root / "initial_state.npz") for root in roots]
        initial = dict(sim_controller_arrays_exact=exact_arrays(a, b),
            raw_Python_NumPy_RNG_exact=read(roots[0] / "RNG-INITIAL.json") == read(roots[1] / "RNG-INITIAL.json"),
            model_XML_exact=(roots[0] / "model.xml").read_text() == (roots[1] / "model.xml").read_text())
        init_arrays = [np.load(root / "init_state.npy", allow_pickle=False) for root in roots]
        initial["raw_init_state_exact"] = (init_arrays[0].dtype == init_arrays[1].dtype
            and init_arrays[0].shape == init_arrays[1].shape and init_arrays[0].tobytes() == init_arrays[1].tobytes())
        images = [raw_arrays(root / "initial_observations.npz") for root in roots]
        require(all(set(x) == set(CAMERAS) for x in images), "both camera inventories required")
        camera = {}
        for name in CAMERAS:
            x, y = images[0][name], images[1][name]
            require(x.shape == y.shape == (128, 128, 3) and x.dtype == y.dtype == np.uint8, "actual initial camera geometry differs")
            d = np.abs(x.astype(np.int16) - y.astype(np.int16))
            camera[name] = dict(exact=x.tobytes() == y.tobytes(), changed_channels=int(np.count_nonzero(d)),
                changed_pixels=int(np.count_nonzero(np.any(d != 0, axis=2))), max_abs=int(d.max()))
        initial["camera_pixels"] = camera
        client_exact = all(v is True for k, v in initial.items() if k != "camera_pixels") and all(v["exact"] for v in camera.values())
        initial["client_saved_physics_RNG_pixels_exact"] = client_exact
        initial["saved_physics_RNG_pixels_exact"] = UNKNOWN
        result["raw_initial_comparison"] = initial
        rng_docs = [read(episode_dir(job, spec["case_id"], arm) / "SERVER-RESET-RNG.json") for arm in (first, second)]
        for rng in rng_docs:
            require(rng["schema"] == "tf-fullsuite-server-reset-rng/v1"
                and rng["actual_initialized_server_rng"] is True and rng["permissions"] == PERMISSIONS,
                "server actual reset RNG schema/provenance differs")
            for name in ("CPU", "CUDA"):
                item = rng[name]
                require(item["dtype"] == "torch.uint8" and type(item["shape"]) is list
                    and len(item["shape"]) == 1 and type(item["shape"][0]) is int and item["shape"][0] > 0
                    and len(item["values"]) == item["shape"][0]
                    and all(type(v) is int and 0 <= v <= 255 for v in item["values"]),
                    "actual CPU/CUDA RNG uint8 bytes/shape/dtype required")
        initial["actual_server_CPU_RNG_exact"] = rng_docs[0]["CPU"] == rng_docs[1]["CPU"]
        initial["actual_server_CUDA_RNG_exact"] = rng_docs[0]["CUDA"] == rng_docs[1]["CUDA"]
        initial["saved_physics_RNG_pixels_exact"] = (client_exact
            and initial["actual_server_CPU_RNG_exact"] and initial["actual_server_CUDA_RNG_exact"])
        initial["unsaved_simulator_internals"] = UNKNOWN
        result["raw_initial_comparison"] = initial
        for item in (c, p):
            if item.get("raw_command_component", {}).get("transport_submitted_environment_exact") is not True:
                result["evidence_gaps"].append(dict(scope="actual_raw_commands", arm=item["arm"],
                    error="per-episode original command audit did not establish exact full coverage"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result["evidence_gaps"].append(dict(scope="actual_raw_initial_or_commands", error=repr(exc)))
    statuses = [x["first2S"].get("status", UNKNOWN) if isinstance(x["first2S"], dict) else UNKNOWN for x in (c, p)]
    statuses = [x if isinstance(x, str) else UNKNOWN for x in statuses]
    result["first2S_comparison"] = dict(statuses_by_arm=dict(zip((first, second), statuses)),
        reference_and_comparison_observed=(sorted(statuses) == ["PASS_EXACT", "REFERENCE_STORED"]) if {first, second} == {"C", "P"} else "NOT_APPLICABLE_TARGET_ARM",
        raw_details_by_arm={x["arm"]: str(Path(x["source_dir"]) / "AUDIT.json") for x in (c, p)},
        reset_S0_before_lazy_scheduler_exception_must_be_read_separately=True,
        not_a_blanket_all_initial_fields_exact_claim=True)
    initial = result["raw_initial_comparison"]
    exact = initial.get("saved_physics_RNG_pixels_exact", UNKNOWN) if isinstance(initial, dict) else UNKNOWN
    component_failed = any(str(x["component_audit_status"]).startswith("FAILED")
        or str(x.get("finish_component_status", "")).startswith("FAILED") for x in (c, p))
    first2_ok = result["first2S_comparison"]["reference_and_comparison_observed"]
    known_client_difference = isinstance(initial, dict) and initial.get("client_saved_physics_RNG_pixels_exact") is False
    if exact is False or known_client_difference or component_failed or "FAILED" in statuses:
        result["strict_comparison_qualification"] = "INVALID_RECORDED_COMPONENT_DIFFERENCE"
    elif (exact is True and not result["evidence_gaps"] and all(not x["evidence_gaps"]
            and x["component_audit_status"] == "PASS_COMPLETE_EPISODE_COMPONENTS" for x in (c, p))
            and (first2_ok is True or first2_ok == "NOT_APPLICABLE_TARGET_ARM")):
        result["strict_comparison_qualification"] = "VALID_RECORDED_COMPONENTS_ONLY"
    return result


def costs(rows):
    started = [r for r in rows if r["started"]]
    covered = [r for r in started if r["all_attempt_cost_covered"]]
    all_covered = bool(started) and len(covered) == len(started)
    action = sum(r["action_total_sec"] for r in covered)
    g = sum(r["action_g"] for r in covered)
    full = sum(r["full_policy_elapsed_sec"] for r in covered)
    chunks = sum(r.get("counter_policy_chunks", 0) for r in covered)
    round_rows = [r for r in started if number(r["sum_T_round_sec"]) and type(r["sum_g"]) is int]
    rounds_complete = bool(started) and len(round_rows) == len(started)
    round_t, round_g = sum(r["sum_T_round_sec"] for r in round_rows), sum(r["sum_g"] for r in round_rows)
    known_forward = sum(r["target_forward_attempts"] for r in started if type(r["target_forward_attempts"]) is int)
    unknown_forward = [r for r in started if type(r["target_forward_attempts"]) is not int]
    return dict(started_attempts=len(started), complete_robot_episodes=sum(r["complete_episode"] for r in rows),
        complete_algorithm_failures_included=sum(r["complete_episode"] and r["success"] is False for r in rows),
        partial_attempts_retained=sum(r["started"] and not r["complete_episode"] for r in rows),
        all_started_attempt_costs_covered=all_covered, cost_missing_pair_arms=[[r["case_id"], r["arm"]] for r in started if not r["all_attempt_cost_covered"]],
        observed_covered_action_total_sec=action, observed_covered_action_g=g,
        observed_covered_full_policy_elapsed_sec=full, observed_covered_policy_chunks=chunks,
        C_update=action / g if all_covered and g > 0 else UNKNOWN,
        C_chunk=full / chunks if all_covered and chunks > 0 else UNKNOWN,
        action_per_chunk=action / chunks if all_covered and chunks > 0 else UNKNOWN,
        sum_T_round_sec=round_t if rounds_complete else UNKNOWN,
        sum_g=round_g if rounds_complete else UNKNOWN,
        cost_per_legal_round_commit=round_t / round_g if rounds_complete and round_g > 0 else UNKNOWN,
        known_forward_attempts=known_forward, all_forward_attempts=known_forward if not unknown_forward else UNKNOWN,
        unknown_attempt_conservative_debit=4000 * len(unknown_forward),
        conservative_forward_upper=known_forward + 4000 * len(unknown_forward),
        E2E_speedup=UNKNOWN, timing_scope="instrumented shared AutoDL exploratory; nested intervals are not additive")


def resource_summary(job):
    path = job / "GPU-RESOURCE-HISTORY.jsonl"
    try:
        require(path.stat().st_size <= 64 * 2**20, "oversized resource metadata history")
        samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        external = [{"snapshot_id": s.get("snapshot_id", UNKNOWN), "sampled_at_unix": s.get("sampled_at_unix", UNKNOWN),
            **r} for s in samples for r in s.get("compute_processes", []) if r.get("identity") == "EXTERNAL_OR_UNRESOLVED"]
        return dict(history=str(path), snapshots=len(samples), observed_snapshots=sum(s.get("status") == "OBSERVED" for s in samples),
            first_sample_id=samples[0].get("snapshot_id", UNKNOWN) if samples else UNKNOWN,
            last_sample_id=samples[-1].get("snapshot_id", UNKNOWN) if samples else UNKNOWN,
            external_or_unresolved_compute_observations=external, cost_comparability=UNKNOWN,
            reason="external/unresolved compute observed" if external else "sparse resource sampling does not establish continuous exclusivity",
            formal_exclusive_window_verified=False)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return dict(history=str(path), status=UNKNOWN, error=repr(exc), cost_comparability=UNKNOWN,
            formal_exclusive_window_verified=False)


def arm_summary(rows):
    full = [r for r in rows if r["complete_episode"]]
    return dict(planned=len(rows), started=sum(r["started"] for r in rows), complete=len(full),
        not_started=sum(not r["started"] for r in rows),
        partial=sum(r["started"] and not r["complete_episode"] for r in rows),
        successes=sum(r["success"] for r in full), failures=sum(not r["success"] for r in full),
        success_rate_observed_complete=sum(r["success"] for r in full) / len(full) if full else UNKNOWN,
        actual_env_steps=quantiles([r["actual_env_steps"] for r in full]),
        failure800_penalty=quantiles([r["failure800_penalty"] for r in full]),
        complete_episode_wall_sec=quantiles([r["client_episode_wall_sec"] for r in full]),
        client_infer_timing={key: quantiles([r["client_infer_timing"][key] for r in full
            if isinstance(r.get("client_infer_timing"), dict) and number(r["client_infer_timing"].get(key))])
            for key in ("reset_infer_wall_sec", "policy_infer_wall_sec", "cache_infer_wall_sec")},
        all_attempt_cost=costs(rows), note="complete denominator explicitly reported; PARTIAL is not an algorithm failure")


def comparison_summary(pairs):
    full = [r for r in pairs if r["complete_pair"]]
    return dict(planned=len(pairs), complete_pairs=len(full),
        first_wins_second_fails=sum(r["lost"] for r in full),
        first_fails_second_wins=sum(r["rescued"] for r in full),
        both_success=sum(r["first_success"] and r["second_success"] for r in full),
        both_failure=sum(not r["first_success"] and not r["second_success"] for r in full),
        success_rate_difference_second_minus_first=(sum(int(r["second_success"]) - int(r["first_success"])
            for r in full) / len(full)) if full else UNKNOWN,
        both_success_steps_second_over_first=quantiles([r["both_success_step_ratio"] for r in full
            if number(r["both_success_step_ratio"])]),
        saved_initial_pair_exact_count=sum(isinstance(r["raw_initial_comparison"], dict)
            and r["raw_initial_comparison"].get("saved_physics_RNG_pixels_exact") is True for r in full),
        initial_or_command_audit_gaps=sum(bool(r["evidence_gaps"]) for r in full),
        strict_comparison_valid_recorded_components=sum(r["strict_comparison_qualification"]
            == "VALID_RECORDED_COMPONENTS_ONLY" for r in full),
        strict_comparison_invalid=sum(r["strict_comparison_qualification"]
            == "INVALID_RECORDED_COMPONENT_DIFFERENCE" for r in full),
        strict_comparison_unknown=sum(r["strict_comparison_qualification"] == UNKNOWN for r in full),
        confidence_interval=UNKNOWN, noninferiority_established=False,
        denominator="only explicitly complete pairs; all missing and partial cases remain in row table")


def aggregate(plan, job):
    rows = [row(plan, job, spec, arm) for spec in plan["cases"] for arm in ARMS]
    by = {(r["case_id"], r["arm"]): r for r in rows}
    pairs = [compare_two(job, spec, by, first, second) for spec in plan["cases"]
        for first, second in (("T", "C"), ("T", "P"), ("C", "P"))]
    summaries = {arm: arm_summary([r for r in rows if r["arm"] == arm]) for arm in ARMS}
    pair_summaries = {name: comparison_summary([r for r in pairs if r["comparison"] == name])
        for name in ("T_C", "T_P", "C_P")}
    per_task = {str(task): dict(arms={arm: arm_summary([r for r in rows if r["task"] == task
        and r["arm"] == arm]) for arm in ARMS}, comparisons={name: comparison_summary([r for r in pairs
        if r["task"] == task and r["comparison"] == name]) for name in ("T_C", "T_P", "C_P")})
        for task in range(10)}
    complete = all(r["complete_episode"] for r in rows)
    cold = {}
    for index in (0, 1):
        ready, error = safe(job / f"server-worker{index}" / "READY.json")
        observed, observation_error = safe(job / f"server-worker{index}" / "MODEL-SETUP-OBSERVATION.json")
        cold[str(index)] = dict(source=str(job / f"server-worker{index}" / "READY.json"),
            setup_elapsed_sec=ready.get("setup_elapsed_sec", UNKNOWN) if ready else UNKNOWN,
            actual_service_start_to_ready_wall_sec=observed.get("persistent_server_start_to_ready_wall_sec", UNKNOWN) if observed else UNKNOWN,
            service_setup_observation_source=str(job / f"server-worker{index}" / "MODEL-SETUP-OBSERVATION.json"),
            service_setup_observation_error=observation_error,
            error=error, not_part_of_warm_episode_cost=True)
    return dict(schema="tf-fullsuite-long500-descriptive-result/v1", scientific_commit=plan["scientific_commit"],
        source_code_commit=plan.get("source_code_commit", UNKNOWN), suite=plan["suite"],
        status="COMPLETE_RECORDED1500_DESCRIPTIVE_COVERAGE" if complete else "PARTIAL_RECORDED_FIXED_LONG500_COVERAGE",
        rows=rows, comparisons=pairs, arms=summaries, paired=pair_summaries, per_task=per_task,
        panel_complete=complete, planned_episodes=1500, complete_episodes=sum(r["complete_episode"] for r in rows),
        planned_cases=500, complete_three_arm_cases=sum(all(by[(spec["case_id"], a)]["complete_episode"]
            for a in ARMS) for spec in plan["cases"]),
        partial_episodes=sum(r["started"] and not r["complete_episode"] for r in rows),
        not_started_episodes=sum(not r["started"] for r in rows),
        case_manifest=[dict(case_id=s["case_id"], task=s["task"], init=s["init"],
            data_exposure=s.get("data_exposure", UNKNOWN), within_case_order=s["within_case_order"],
            proposed_GPU_index=s["proposed_GPU_index"]) for s in plan["cases"]],
        data_exposure_sources=plan.get("data_exposure_sources", UNKNOWN),
        previous_reserved_TEST_not_independent_after_this_research_evaluation=True,
        engineering_quality_screen_observed=UNKNOWN, independent_test_complete_observed=UNKNOWN,
        noninferiority_observed=UNKNOWN, formal_performance_gate=UNKNOWN, permissions=PERMISSIONS,
        old_TF_ON_known_risk_NO_GO_unchanged=True, old_FAILED_22_cleanup_gaps_unchanged=True,
        no_quality_certificate=True, no_architecture_comparison=True, no_training=True, no_retry=True,
        no_old_quality_failure_early_stop=True, E2E_speedup=UNKNOWN, non_episode_target_forward_attempts=UNKNOWN,
        whole_model_GPU_active_time_sec=UNKNOWN, instance_shutdown_verified=UNKNOWN,
        valid_scope="user-authorized complete Long500 coverage, mixed prior FIT/DEV/reserved TEST exposure; research descriptive, not independent TEST",
        acceptance_threshold=plan["threshold"], production_schedule=dict(T="canonical original Target", C="continuous S", P="S,S,T"),
        resource_metadata=resource_summary(job), exploratory_cost_comparability=UNKNOWN,
        cold_start=cold, costs_definition=dict(C_update="SUM(action CUDA sec including terminal)/SUM(actual action g)",
            C_chunk="SUM(full policy host elapsed including synchronization)/Npolicy chunks",
            action_per_chunk="SUM(action CUDA sec)/Npolicy chunks",
            rounds="SUM(nonoverlapping complete T_round)/SUM(actual g), explicit full coverage only",
            failure_cost="every complete failure and partial attempt retained, missing coverage UNKNOWN",
            reset_cache="original client websocket reset/cache timing, distinct from policy and nested episode",
            nested_intervals="action, policy, client transport, episode are nested; do not add"),
        original_client_elapsed_boundary="after env construction before reset; includes reset/policy/cache/environment; excludes model load and env construction",
        absolute_work_stop=plan["absolute_work_stop"], absolute_cleanup_deadline=plan["absolute_cleanup_deadline"])


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), (dict, list))
            else r.get(k, UNKNOWN) for k in columns})
    return stream.getvalue()


def prose(r):
    common = "四权限始终false；旧NO_GO、FAILED及22清理缺证保留。此次全量覆盖经用户另行授权，为研究性描述，既非独立TEST，也不证明非劣或正式性能。"
    quality = ["# Long500三臂结果", "", common, "", "|arm|计划|启动|完整|成功|失败|PARTIAL|未开始|800罚分总|",
        "|---|---|---|---|---|---|---|---|---|"]
    for a, x in r["arms"].items():
        quality.append(f"|{a}|{x['planned']}|{x['started']}|{x['complete']}|{x['successes']}|{x['failures']}|{x['partial']}|{x['not_started']}|{x['failure800_penalty']['sum']}|")
    quality += ["", "|配对|完整N|前胜后败|前败后胜|双方成功|双方失败|步数比P50/P90/max|", "|---|---|---|---|---|---|---|"]
    for name, x in r["paired"].items():
        q = x["both_success_steps_second_over_first"]
        quality.append(f"|{name}|{x['complete_pairs']}|{x['first_wins_second_fails']}|{x['first_fails_second_wins']}|{x['both_success']}|{x['both_failure']}|{q['p50']}/{q['p90']}/{q['max']}|")
    quality += ["", "实际env steps与失败800罚分分开；双方成功长度仅条件描述，全部失败/缺项留在CSV和JSON。按task分层、旧FIT/DEV/原待留TEST来源见JSON，不能按query行重命名独立样本。", ""]
    system = ["# 探索性成本", "", "共享AutoDL带审计计时，仅描述成本，可比性UNKNOWN，不能作为正式bench或E2E加速。", "",
        "|arm|C_update|C_chunk(fullpolicy/chunk)|action/chunk|ΣT_round/Σg|未知成本attempt|", "|---|---|---|---|---|---|"]
    for a, x in r["arms"].items():
        c = x["all_attempt_cost"]
        system.append(f"|{a}|{c['C_update']}|{c['C_chunk']}|{c['action_per_chunk']}|{c['cost_per_legal_round_commit']}|{c['cost_missing_pair_arms']}|")
    system += ["", "包含完整失败与partial尝试；未知forward按4000上界计而不伪造精确。reset/cache/client壁钟与模型冷启动分列。嵌套时段不相加；调用数减少不等于加速。稀疏资源采样不能证明持续独占。", ""]
    return {"FACTS.md": "\n".join(["# 事实、未知与建议", "", f"{r['status']}，完整{r['complete_episodes']}/1500，完整三臂组{r['complete_three_arm_cases']}/500。", common,
        "UNKNOWN：独立测试、非劣、正式性能、全模型GPU活跃时间及实例关机。保留逐case原件、全部失败和审计差异，不自动改变权重/阈值。", ""]),
        "QUALITY-RESULT.md": "\n".join(quality), "SYSTEM-RESULT.md": "\n".join(system),
        "CLAIM-EVIDENCE-MATRIX.md": "# 主张与证据\n\n|主张|证据|边界|\n|---|---|---|\n|Long500三臂描述|新job完整终态及逐case表|不同暴露来源，不是独立TEST|\n|初态物理/RNG/双相机一致性|三种实际两两对账|保留不等和未保存内部状态UNKNOWN|\n|C/P首两S比较|同case持久server被动观察|短自然局不伪造2chunk，T不适用|\n|有效提交及policy成本|真实COUNTER/timing/g|仅探索性，缺覆盖UNKNOWN|\n|闭环非劣/正式E2E加速|无相应新证书|不能宣称|\n",
        "STOP-REPORT.md": "\n".join(["# 资源与终态", "", f"固定1500局，6,001,000 forward，10GiB新证据。停工{r['absolute_work_stop']}，清理{r['absolute_cleanup_deadline']}。",
        f"完整{r['complete_episodes']}，PARTIAL{r['partial_episodes']}，未开始{r['not_started_episodes']}；算法失败不沿旧六对门早停。资源/路由硬限仍停止，完整失败不重跑替换。",
        "等待/启动/失败/超时/清理均计预算。GPU进程清理由runner/guard证明，脚本退出不等于租卡关机。", common, ""]) }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    start, begin = time.time(), time.monotonic()
    job = args.job.resolve(strict=True)
    plan = load(job)
    require(len(os.sched_getaffinity(0)) <= plan["CPU_analysis_max"], "registered CPU analysis cap")
    out = args.out.resolve() if args.out is not None else job / "report-v1"
    require(out.parent == job and not out.exists(), "new report namespace only; no overwrite")
    deadline = min(start + 300, datetime.fromisoformat(plan["absolute_cleanup_deadline"]).timestamp())
    require(deadline > time.time(), "absolute cleanup deadline expired")
    out.mkdir(exist_ok=False)
    write(out / "PREREGISTRATION.json", dict(schema="tf-fullsuite-report-registration/v1", job=str(job),
        birth_epoch=start, deadline_epoch=deadline, CPU_max=plan["CPU_analysis_max"], storage_cap_bytes=REPORT_CAP,
        no_GPU=True, no_digest_computed=True, no_retry=True, permissions=PERMISSIONS))
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("bounded complete-coverage CPU report deadline")))
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    try:
        result = aggregate(plan, job)
        result.update(CPU_report_wall_sec=time.monotonic() - begin, created_at=datetime.now(timezone.utc).isoformat())
        write(out / "RESULT.json", result)
        write(out / "PANEL-STATUS.json", {k: result[k] for k in ("schema", "status", "panel_complete", "complete_episodes",
            "complete_three_arm_cases", "partial_episodes", "not_started_episodes", "permissions", "no_quality_certificate")})
        texts = prose(result)
        texts["episodes.csv"] = csv_text(result["rows"], ("case_id", "task", "init", "data_exposure", "arm", "within_case_order", "status",
            "complete_episode", "success", "actual_env_steps", "failure800_penalty", "policy_chunks", "S_chunks", "T_chunks",
            "target_forward_attempts", "action_total_sec", "action_g", "full_policy_elapsed_sec", "C_update", "C_chunk",
            "sum_T_round_sec", "sum_g", "client_episode_wall_sec", "client_infer_timing", "all_attempt_cost_covered", "evidence_gaps"))
        texts["paired.csv"] = csv_text(result["comparisons"], ("case_id", "task", "init", "data_exposure", "comparison", "first_arm", "second_arm",
            "complete_pair", "first_success", "second_success", "first_steps", "second_steps", "rescued", "lost", "both_success_step_ratio",
            "raw_initial_comparison", "first2S_comparison", "strict_comparison_qualification", "evidence_gaps"))
        require(sum(x.stat().st_size for x in out.iterdir() if x.is_file()) + sum(len(x.encode()) for x in texts.values())
            + 32768 < REPORT_CAP, "report texts and failure receipt reserve cap")
        for name, text in texts.items():
            with (out / name).open("x") as f:
                f.write(text)
        require(time.time() < deadline, "bounded report deadline")
        write(out / "CPU-FINAL.json", dict(status=result["status"], elapsed_sec=time.monotonic() - begin,
            GPU=False, no_digest_computed=True, permissions=PERMISSIONS, no_quality_certificate=True))
        print(json.dumps(dict(status=result["status"], complete_episodes=result["complete_episodes"], output=str(out))))
        return 0
    except BaseException as exc:
        write(out / "FAILED.json", dict(status="FAILED_CPU_FULLSUITE_REPORT", error=repr(exc),
            traceback=traceback.format_exc(), elapsed_sec=time.monotonic() - begin, permissions=PERMISSIONS))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == "__main__":
    raise SystemExit(main())



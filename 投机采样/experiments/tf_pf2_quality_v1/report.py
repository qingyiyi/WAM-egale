"""No-digest descriptive report for the fixed six-pair C/P DEV panel.

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
from tf_pf2_quality_v1.protocol import (ARMS, CAMERAS, PERMISSIONS, episode_dir, load, read, require)

UNKNOWN = "UNKNOWN"
REPORT_CAP = 4 * 2**20


def write(path, obj):
    path = Path(path)
    raw = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    used = sum(x.stat().st_size for x in path.parent.iterdir() if x.is_file())
    require(used + len(raw) <= REPORT_CAP, "4MiB report cap before write")
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
    out = episode_dir(job, spec["pair_id"], arm)
    names = ("EPISODE-TERMINAL.json", "COUNTER.json", "AUDIT.json", "FINISH_READY.json")
    docs = {name: safe(out / name) for name in names}
    terminal, counter, audit, finish = [docs[n][0] for n in names]
    gaps = [{"file": n, "error": error} for n, (_, error) in docs.items() if error is not None]
    started = (out / "CLIENT-STARTED.json").is_file() or any(docs[n][0] is not None for n in names)
    r = dict(pair_id=spec["pair_id"], stratum=spec["stratum"], task=spec["task"], init=spec["init"], arm=arm,
        within_pair_order=spec["within_pair_order"], source_dir=str(out), started=started,
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
            require(terminal["schema"] == "tf-pf2-full-episode-terminal/v1" and terminal["pair_id"] == spec["pair_id"]
                and terminal["arm"] == arm and terminal["task"] == spec["task"] and terminal["init"] == spec["init"]
                and terminal["context"] == dict(pair_id=spec["pair_id"], arm=arm, out=str(out))
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
                    client_total_wall_sec=terminal["client_total_wall_sec"])
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
                and finish["pair_id"] == spec["pair_id"] and finish["arm"] == arm
                and finish["full_episode_terminal"] is True and finish["async_drained"] is True
                and finish["permissions"] == PERMISSIONS, "original finish/drain component receipt differs")
            finish_valid = True
            r["finish_component_status"] = finish["status"]
        except (KeyError, ValueError, TypeError) as exc:
            r["evidence_gaps"].append(dict(file="FINISH_READY.json", error=repr(exc)))
    if counter is not None:
        try:
            require(counter["schema"] == "tf-pf2-episode-counter/v1" and counter["pair_id"] == spec["pair_id"]
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
            r["policy_resource_references"] = [{k: x.get(k, UNKNOWN) for k in
                ("successful_policy_ordinal", "role", "GPU_identity", "resource_before", "resource_after")} for x in policies]
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
            require(audit["schema"] == "tf-pf2-episode-audit/v1" and audit["pair_id"] == spec["pair_id"]
                and audit["arm"] == arm and audit["permissions"] == PERMISSIONS, "component audit identity differs")
            r.update(component_audit_status=audit["status"], first2S=audit["first2S"],
                numerical_isolation=audit["numerical_isolation"])
        except (KeyError, ValueError, TypeError) as exc:
            r["evidence_gaps"].append(dict(file="AUDIT.json", error=repr(exc)))
    if started and counter is None:
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


def compare_pair(job, spec, by):
    import numpy as np
    c, p = [by[(spec["pair_id"], arm)] for arm in ARMS]
    result = dict(pair_id=spec["pair_id"], task=spec["task"], init=spec["init"], stratum=spec["stratum"],
        within_pair_order=spec["within_pair_order"], complete_pair=c["complete_episode"] and p["complete_episode"],
        C_success=c["success"], P_success=p["success"], rescued=False, lost=False,
        C_steps=c["actual_env_steps"], P_steps=p["actual_env_steps"], both_success_step_ratio=UNKNOWN,
        raw_initial_comparison=UNKNOWN, first2S_comparison=UNKNOWN, evidence_gaps=[])
    if not result["complete_pair"]:
        return result
    result.update(rescued=not c["success"] and p["success"], lost=c["success"] and not p["success"],
        both_success_step_ratio=p["actual_env_steps"] / c["actual_env_steps"] if c["success"] and p["success"] else UNKNOWN)
    try:
        roots = [episode_dir(job, spec["pair_id"], arm) / "client" for arm in ARMS]
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
        initial["saved_physics_RNG_pixels_exact"] = all(v is True for k, v in initial.items() if k != "camera_pixels") and all(v["exact"] for v in camera.values())
        initial["unsaved_simulator_internals"] = UNKNOWN
        result["raw_initial_comparison"] = initial
        for item in (c, p):
            item["raw_command_component"] = client_commands(Path(item["source_dir"]), item)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result["evidence_gaps"].append(dict(scope="actual_raw_initial_or_commands", error=repr(exc)))
    statuses = [x["first2S"].get("status") if isinstance(x["first2S"], dict) else UNKNOWN for x in (c, p)]
    result["first2S_comparison"] = dict(statuses_by_arm=dict(zip(ARMS, statuses)),
        reference_and_comparison_observed=sorted(statuses) == ["PASS_EXACT", "REFERENCE_STORED"],
        raw_details_by_arm={x["arm"]: x["first2S"] for x in (c, p)},
        reset_S0_before_lazy_scheduler_exception_must_be_read_separately=True,
        not_a_blanket_all_initial_fields_exact_claim=True)
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
        all_started_attempt_costs_covered=all_covered, cost_missing_pair_arms=[[r["pair_id"], r["arm"]] for r in started if not r["all_attempt_cost_covered"]],
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


def aggregate(plan, job):
    rows = [row(plan, job, spec, arm) for spec in plan["pairs"] for arm in ARMS]
    by = {(r["pair_id"], r["arm"]): r for r in rows}
    pairs = [compare_pair(job, spec, by) for spec in plan["pairs"]]
    groups = {}
    for label in ("risk", "control", "all"):
        selected = [r for r in rows if label == "all" or r["stratum"] == label]
        paired = [r for r in pairs if (label == "all" or r["stratum"] == label) and r["complete_pair"]]
        arms = {}
        for arm in ARMS:
            arm_rows = [r for r in selected if r["arm"] == arm]
            full = [r for r in arm_rows if r["complete_episode"]]
            arms[arm] = dict(planned=len(arm_rows), started=sum(r["started"] for r in arm_rows), complete=len(full),
                successes=sum(r["success"] for r in full), failures=sum(not r["success"] for r in full),
                actual_env_steps=quantiles([r["actual_env_steps"] for r in full]),
                failure800_penalty=quantiles([r["failure800_penalty"] for r in full]),
                complete_episode_wall_sec=quantiles([r["client_episode_wall_sec"] for r in full]), all_attempt_cost=costs(arm_rows))
        groups[label] = dict(complete_pairs=len(paired), rescued=sum(r["rescued"] for r in paired),
            lost=sum(r["lost"] for r in paired), both_success=sum(r["C_success"] and r["P_success"] for r in paired),
            both_failure=sum(not r["C_success"] and not r["P_success"] for r in paired),
            saved_initial_pair_exact_count=sum(isinstance(r["raw_initial_comparison"], dict)
                and r["raw_initial_comparison"].get("saved_physics_RNG_pixels_exact") is True for r in paired),
            both_success_steps_P_over_C=quantiles([r["both_success_step_ratio"] for r in paired if number(r["both_success_step_ratio"])]),
            arms=arms, confidence_interval=UNKNOWN, independent_TEST=False)
    complete = all(r["complete_episode"] for r in rows)
    return dict(schema="tf-pf2-six-pair-panel-descriptive-result/v1", scientific_commit=plan["scientific_commit"],
        status="COMPLETE_RECORDED12_DESCRIPTIVE_PANEL" if complete else "PARTIAL_RECORDED_FIXED_PANEL",
        rows=rows, pairs=pairs, groups=groups, panel_complete=complete, planned_episodes=12,
        complete_episodes=sum(r["complete_episode"] for r in rows), planned_pairs=6,
        rescued=groups["all"]["rescued"], lost=groups["all"]["lost"],
        rescued_lost_scope="raw complete task/init outcome cells; inspect initial/first2S qualification separately, never causal quality certification",
        stop_new_waves=(groups["all"]["lost"] > 0 or any(str(r["component_audit_status"]).startswith("FAILED") for r in rows)
            or any(isinstance(r["raw_initial_comparison"], dict)
                and r["raw_initial_comparison"].get("saved_physics_RNG_pixels_exact") is False for r in pairs)
            or any(r["evidence_gaps"] for r in pairs if r["complete_pair"])),
        engineering_quality_screen_observed=UNKNOWN, independent_test_complete_observed=UNKNOWN,
        noninferiority_observed=UNKNOWN, formal_performance_gate=UNKNOWN, permissions=PERMISSIONS,
        old_TF_ON_known_risk_NO_GO_unchanged=True, old_FAILED_22_cleanup_gaps_unchanged=True,
        no_quality_certificate=True, no_architecture_comparison=True, no_training=True, no_retry=True,
        no_original_target_baseline_in_this_panel=True, E2E_speedup=UNKNOWN,
        non_episode_target_forward_attempts=UNKNOWN, whole_model_GPU_active_time_sec=UNKNOWN,
        instance_shutdown_verified=UNKNOWN, valid_scope="six fixed previously known DEV/risk pairs, not TEST",
        acceptance_threshold=plan["threshold"], production_schedule=dict(C="continuous S", P="S,S,T"),
        resource_metadata=resource_summary(job), exploratory_cost_comparability=UNKNOWN,
        costs_definition=dict(C_update="SUM(action total CUDA sec including terminal action advances)/SUM(actual action g)",
            C_chunk="SUM(full policy elapsed including return synchronization)/Npolicy chunks",
            action_per_chunk="SUM(action total CUDA sec)/Npolicy chunks",
            rounds="SUM(nonoverlapping complete T_round)/SUM(actual g), only explicit full coverage",
            failure_cost="all complete failures plus partial attempt costs; missing coverage remains UNKNOWN",
            nested_intervals="action, full policy and episode overlap; never add them"),
        original_client_elapsed_boundary="after environment construction, before reset; contains reset/policy/cache/environment; excludes initial model load and environment construction",
        absolute_work_stop=plan["absolute_work_stop"], absolute_cleanup_deadline=plan["absolute_cleanup_deadline"])


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (list, dict)) else row.get(k, UNKNOWN) for k in columns})
    return stream.getvalue()


def prose(result):
    table = ["|分层|完整pair|rescued C败P胜|lost C胜P败|双方成功|双方失败|", "|---|---|---|---|---|---|"]
    for label, g in result["groups"].items():
        table.append(f"|{label}|{g['complete_pairs']}|{g['rescued']}|{g['lost']}|{g['both_success']}|{g['both_failure']}|")
    caveat = "四项原权限全false，旧TF_ON NO_GO与22局部清理缺证不变；此panel不是独立TEST，不证明非劣。异常PARTIAL不作算法失败。"
    quality = ["# C/P完整任务描述", "", *table, "", caveat,
        "实际环境步与失败800罚分分开；双方成功长度比P50/P90/max仅条件统计，全部失败同时保留。线性分位数、逐局步骤、分层成本见JSON/CSV。", ""]
    system = ["# 探索性系统成本", "", "以下为共享AutoDL、带同步与记录的探索性成本，不能写成正式benchmark或E2E加速。", "",
        "|group|arm|C_update|C_chunk fullpolicy/chunk|action/chunk|Σround/Σg|成本缺失attempt|", "|---|---|---|---|---|---|---|"]
    for label, group in result["groups"].items():
        for arm, row in group["arms"].items():
            c = row["all_attempt_cost"]
            system.append(f"|{label}|{arm}|{c['C_update']}|{c['C_chunk']}|{c['action_per_chunk']}|{c['cost_per_legal_round_commit']}|{c['cost_missing_pair_arms']}|")
    system += ["", "C_update仅action/g；C_chunk仅完整policy/chunk。action、policy、episode为嵌套区间，不能相加。完整round覆盖未明确就UNKNOWN。所有完整失败与partial尝试均记成本，缺项不按0。E2E speedup UNKNOWN。", ""]
    system += ["资源采样仅是实际时刻的UUID/util/memory/compute PID观察；不能证明全时独占。非已证own生命周期标EXTERNAL_OR_UNRESOLVED，有外部或未解析负载时成本可比性UNKNOWN；未发现也不能证明正式独占。", ""]
    return {
        "FACTS.md": "\n".join(["# 已观察事实与未知", "", f"{result['status']}，完整{result['complete_episodes']}/12，rescued={result['rescued']}，lost={result['lost']}。", caveat,
            "原始初态、两相机/RNG逐pair差异及first2S/numerical isolation是组件证据；不是全局质量证书。", "建议按登记lost/隔离失败止损，不自动重训、改阈值或扩大panel。", ""]),
        "QUALITY-RESULT.md": "\n".join(quality),
        "SYSTEM-RESULT.md": "\n".join(system),
        "CLAIM-EVIDENCE-MATRIX.md": "# Claim与证据\n\n|claim|证据|允许边界|\n|---|---|---|\n|固定6pair C/P结果|完整终态/原始命令/组件audit|描述risk/control，不泛化TEST|\n|first2S精确|同pair持久server actual comparison|保留S0_before lazy scheduler例外，不能blanket称全部初态相同|\n|PF2恢复质量/非劣|此panel不足|UNKNOWN，旧NO_GO不变|\n|action与fullpolicy成本|COUNTER真实计时和g|仅探索性，覆盖缺失为UNKNOWN|\n|E2E加速/正式性能|没有相应许可或测量|禁止|\n",
        "STOP-REPORT.md": "\n".join(["# 终态、预算与止损", "", f"固定12局，无自动重跑；工作停止{result['absolute_work_stop']}，清理硬截止{result['absolute_cleanup_deadline']}。",
            f"lost={result['lost']}；stop_new_waves={result['stop_new_waves']}。未开始与PARTIAL分别列CSV/JSON，未强记失败。",
            "完整/异常尝试forward、未知4000保守上界、全部失败成本均见JSON。实际GPU活跃小时若无全局证据为UNKNOWN，不以wall相加冒充。",
            "report不停止服务器或实例；own进程清理由runner/watchdog归档，脚本退出不代表AutoDL已关机。", ""])}


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
    require(out.parent == job and not out.exists(), "new independent job/report directory; no overwrite")
    deadline = min(start + 120, datetime.fromisoformat(plan["absolute_cleanup_deadline"]).timestamp())
    require(deadline > time.time(), "absolute report/cleanup deadline expired")
    out.mkdir(exist_ok=False)
    write(out / "PREREGISTRATION.json", dict(schema="tf-pf2-report-registration/v1", goal=__doc__,
        job=str(job), birth_epoch=start, deadline_epoch=deadline, CPU_max=plan["CPU_analysis_max"],
        storage_cap_bytes=REPORT_CAP, no_digest_computed=True, no_GPU=True, no_retry=True,
        permissions=PERMISSIONS, cleanup="single CPU exit; no source writes or signals"))
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("bounded panel CPU report deadline")))
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    try:
        result = aggregate(plan, job)
        result["CPU_report_wall_sec"] = time.monotonic() - begin
        result["created_at"] = datetime.now(timezone.utc).isoformat()
        write(out / "RESULT.json", result)
        write(out / "PANEL-STATUS.json", {k: result[k] for k in
            ("schema", "status", "panel_complete", "complete_episodes", "rescued", "lost", "stop_new_waves", "permissions", "no_quality_certificate")})
        texts = prose(result)
        texts["panel.csv"] = csv_text(result["rows"], ("pair_id", "stratum", "task", "init", "arm", "within_pair_order", "status",
            "complete_episode", "success", "actual_env_steps", "failure800_penalty", "policy_chunks", "S_chunks", "T_chunks",
            "target_forward_attempts", "target_forward_returned", "target_forward_failed", "action_total_sec", "action_g",
            "full_policy_elapsed_sec", "C_update", "C_chunk", "action_per_chunk", "sum_T_round_sec", "sum_g",
            "cost_per_legal_round_commit", "client_episode_wall_sec", "all_attempt_cost_covered", "component_audit_status", "GPU_identity", "evidence_gaps"))
        texts["pairs.csv"] = csv_text(result["pairs"], ("pair_id", "stratum", "task", "init", "within_pair_order", "complete_pair",
            "C_success", "P_success", "rescued", "lost", "C_steps", "P_steps", "both_success_step_ratio", "raw_initial_comparison", "first2S_comparison", "evidence_gaps"))
        require(sum(x.stat().st_size for x in out.iterdir() if x.is_file())
            + sum(len(t.encode()) for t in texts.values()) + 16384 < REPORT_CAP, "report text and terminal reserve cap")
        for name, text in texts.items():
            with (out / name).open("x") as f:
                f.write(text)
        require(sum(x.stat().st_size for x in out.iterdir() if x.is_file()) <= REPORT_CAP, "4MiB report storage cap")
        require(time.time() < deadline, "bounded report deadline")
        write(out / "CPU-FINAL.json", dict(status=result["status"], elapsed_sec=time.monotonic() - begin,
            GPU=False, no_digest_computed=True, permissions=PERMISSIONS, no_quality_certificate=True))
        print(json.dumps(dict(status=result["status"], complete_episodes=result["complete_episodes"], output=str(out)), ensure_ascii=False))
        return 0
    except BaseException as exc:
        write(out / "FAILED.json", dict(status="FAILED_CPU_PANEL_REPORT", error=repr(exc), traceback=traceback.format_exc(),
            elapsed_sec=time.monotonic() - begin, permissions=PERMISSIONS, no_quality_certificate=True))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == "__main__":
    raise SystemExit(main())

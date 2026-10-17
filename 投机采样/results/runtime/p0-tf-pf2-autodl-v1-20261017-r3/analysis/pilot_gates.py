"""CPU1 analysis of the already registered r3 six-pair investment gates.

This does not run a policy, change gates, combine historical C, or grant a
quality/performance permission. No digest, GPU, SSH, training or subprocess.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

OWN = Path(__file__).resolve().parent
JOB = OWN.parent
R1 = JOB.parent / "p0-tf-pf2-autodl-v1-20261017-r1"
CODE = "9acc37cee9ae9b62b8311f1f8dbb01bffa37a86a"
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
UNKNOWN = "UNKNOWN"
CAP = 8 * 2**20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.stat().st_size <= 64 * 2**20, "oversized metadata input")
    return json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def write(out, name, data):
    raw = data.encode() if isinstance(data, str) else (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    used = sum(x.stat().st_size for x in OWN.rglob("*") if x.is_file())
    require(used + len(raw) <= CAP, "8MiB independent analysis cap")
    with (out / name).open("xb") as stream:
        stream.write(raw)


def numeric(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0


def distribution(values):
    values = sorted(values)
    def q(p):
        if not values:
            return UNKNOWN
        idx = (len(values) - 1) * p
        a, b = math.floor(idx), math.ceil(idx)
        return values[a] + (values[b] - values[a]) * (idx - a)
    return dict(n=len(values), p50=q(.5), p90=q(.9), max=max(values) if values else UNKNOWN,
        quantile="linear index=(n-1)*q")


def verdict(value):
    return "PASS" if value is True else "FAIL" if value is False else UNKNOWN


def component(row, paired):
    status = row.get("component_audit_status", UNKNOWN)
    if str(status).startswith("FAILED"):
        return False
    initial, first2 = paired.get("raw_initial_comparison"), paired.get("first2S_comparison")
    if isinstance(initial, dict) and initial.get("saved_physics_RNG_pixels_exact") is False:
        return False
    if isinstance(first2, dict) and "FAILED" in first2.get("statuses_by_arm", {}).values():
        return False
    isolation = row.get("numerical_isolation")
    expected = (status == "PASS_COMPLETE_EPISODE_COMPONENTS"
        and isinstance(initial, dict) and initial.get("saved_physics_RNG_pixels_exact") is True
        and isinstance(first2, dict) and first2.get("reference_and_comparison_observed") is True
        and isinstance(isolation, dict) and isolation.get("status") == "PASS_OBSERVED_ROUTE_STATE_AND_ROLE_CONTRACTS"
        and isolation.get("original_ObservedAPI_unchanged") is True and isolation.get("raw_S_g_unchanged") is True
        and row.get("evidence_gaps") == [] and paired.get("evidence_gaps") == [])
    return True if expected else UNKNOWN


def analyze(result_path):
    p, g, original = read(JOB / "plan.json"), read(JOB / "PILOT-GATES.json"), read(R1 / "PILOT-GATES.json")
    operating = read(JOB / "REMOTE-OPERATING-PLAN.json")
    require(g == original and g["schema"] == "tf-pf2-fixed-pilot-investment-gates/v1"
        and g["before_new_episode_start"] is True and g["permissions"] == PERMISSIONS
        and g["no_global_permission_change"] is True, "original registered investment gates differ")
    require(p["source_code_commit"] == CODE and p["permissions"] == PERMISSIONS
        and operating["permissions"] == PERMISSIONS and operating["pairs"] == p["pairs"]
        and Path(operating["job"]).name == JOB.name, "fixed r3 source/operating identity differs")
    require((g["P_success_required_risk"], g["P_success_required_control"], g["lost_success_max"],
        g["new_current_C_failure_P_success_in_risk_min"]) == (3, 3, 0, 1)
        and g["P_penalized_sum_not_above_C"] == ["risk", "control", "all"]
        and g["both_success_length_ratio_limits"] == dict(p50=1.10, p90=1.20, max=1.30, quantile="linear index=(n-1)*q")
        and g["P_over_C_cost_investment_limits"] == dict(C_update=1.25, common_success_total_policy_cost=1.25)
        and g["cost_unavailable_or_incomparable"] == "UNKNOWN_not_pass", "unrecognized gate values; no post hoc adjustment")
    report = read(result_path)
    require(report["schema"] == "tf-pf2-six-pair-panel-descriptive-result/v1"
        and report["scientific_commit"] == p["scientific_commit"] and report["permissions"] == PERMISSIONS,
        "only this new descriptive panel result is eligible")
    specs = {r["pair_id"]: r for r in p["pairs"]}
    require(set(specs) == set(range(1, 7)), "fixed six pair IDs required")
    rows = {}
    for row in report["rows"]:
        key = row["pair_id"], row["arm"]
        spec = specs.get(key[0])
        require(spec is not None and key[1] in ("C", "P") and key not in rows
            and (row["task"], row["init"], row["stratum"]) == (spec["task"], spec["init"], spec["stratum"])
            and row["source_dir"] == str(Path(operating["job"]) / "episodes" / f"pair-{key[0]:02d}" / key[1])
            and row["permissions"] == PERMISSIONS, "mixed/duplicate/historical C or unmatched r3 outcome")
        if row.get("complete_episode") is True:
            require(type(row["success"]) is bool and type(row["actual_env_steps"]) is int
                and 0 < row["actual_env_steps"] <= 800
                and row["failure800_penalty"] == (row["actual_env_steps"] if row["success"] else 800),
                "complete actual length and failure800 penalty differ")
        rows[key] = row
    observed_pairs = {r["pair_id"]: r for r in report["pairs"]}
    require(len(observed_pairs) == len(report["pairs"]), "duplicate descriptive pair")
    pairs = []
    for pid, spec in specs.items():
        c, pf = rows.get((pid, "C")), rows.get((pid, "P"))
        complete = c is not None and pf is not None and c.get("complete_episode") is True and pf.get("complete_episode") is True
        item = dict(pair_id=pid, task=spec["task"], init=spec["init"], stratum=spec["stratum"], complete=complete)
        if complete:
            item.update(C_success=c["success"], P_success=pf["success"], C_steps=c["actual_env_steps"], P_steps=pf["actual_env_steps"],
                C_penalty=c["failure800_penalty"], P_penalty=pf["failure800_penalty"],
                rescued=not c["success"] and pf["success"], lost=c["success"] and not pf["success"],
                both_success_ratio=pf["actual_env_steps"] / c["actual_env_steps"] if c["success"] and pf["success"] else UNKNOWN)
            audit = observed_pairs.get(pid, {})
            item["components"] = [component(x, audit) for x in (c, pf)]
        pairs.append(item)
    full = [r for r in pairs if r["complete"]]
    all_complete = len(full) == 6
    components = [x for r in full for x in r["components"]]
    checks = {}
    def gate(name, passed, observed, expected, note=None):
        checks[name] = dict(status=verdict(passed), observed=observed, registered=expected, note=note)
    gate("six_complete_identity_route", False if False in components else True if all_complete and all(x is True for x in components) else UNKNOWN,
        dict(complete_pairs=len(full), component_statuses=components), g["all_six_pairs_complete_and_identity_route_audit"])
    for stratum, required in (("risk", 3), ("control", 3)):
        actual = [rows[(pid, "P")] for pid, s in specs.items() if s["stratum"] == stratum
            and (pid, "P") in rows and rows[(pid, "P")].get("complete_episode") is True]
        successes = sum(r["success"] for r in actual)
        gate("P_all_" + stratum + "_success", False if any(not r["success"] for r in actual)
            else successes == required if len(actual) == required else UNKNOWN,
            dict(success=successes, complete=len(actual), planned=required), required)
    lost = sum(r["lost"] for r in full)
    rescued_risk = sum(r["rescued"] for r in full if r["stratum"] == "risk")
    gate("no_lost_success", False if lost > 0 else True if all_complete else UNKNOWN, lost, 0)
    risk_complete = sum(r["stratum"] == "risk" for r in full)
    gate("current_risk_rescued_min1", True if rescued_risk >= 1 else False if risk_complete == 3 else UNKNOWN,
        dict(rescued=rescued_risk, complete_risk_pairs=risk_complete), 1, "only this r3 current C/P; historical C never counted")
    for label in ("risk", "control", "all"):
        selected = [r for r in full if label == "all" or r["stratum"] == label]
        expected_n = 6 if label == "all" else 3
        c, pf = sum(r["C_penalty"] for r in selected), sum(r["P_penalty"] for r in selected)
        gate("penalty_" + label, pf <= c if len(selected) == expected_n else UNKNOWN,
            dict(C=c, P=pf, complete_pairs=len(selected), planned=expected_n), "P<=C using full failures=800")
    lengths = distribution([r["both_success_ratio"] for r in full if numeric(r["both_success_ratio"])])
    for name in ("p50", "p90", "max"):
        value, limit = lengths[name], g["both_success_length_ratio_limits"][name]
        passed = (False if name == "max" and numeric(value) and value > limit
            else value <= limit if all_complete and numeric(value) else UNKNOWN)
        gate("length_" + name, passed, dict(value=value, common_success_count=lengths["n"]), limit,
            "observed max counterexample is irreversible; later rescued pairs cannot erase it" if name == "max" else None)
    resource = report.get("resource_metadata", {})
    comparable = (report.get("exploratory_cost_comparability") is True
        or resource.get("cost_comparability") is True)
    all_cost = all_complete and all(r.get("all_attempt_cost_covered") is True for r in rows.values())
    cost_observations = {}
    ratios = {}
    for arm in ("C", "P"):
        selected = [r for (pid, a), r in rows.items() if a == arm]
        valid = len(selected) == 6 and all(numeric(r.get("action_total_sec")) and type(r.get("action_g")) is int and r["action_g"] > 0 for r in selected)
        t = sum(r["action_total_sec"] for r in selected) if valid else UNKNOWN
        n = sum(r["action_g"] for r in selected) if valid else UNKNOWN
        ratios[arm] = t / n if valid else UNKNOWN
        cost_observations[arm] = dict(action_total_sec=t, actual_action_g=n, C_update=ratios[arm])
    update_ratio = ratios["P"] / ratios["C"] if numeric(ratios["P"]) and numeric(ratios["C"]) and ratios["C"] > 0 else UNKNOWN
    gate("cost_C_update", update_ratio <= 1.25 if all_cost and comparable and numeric(update_ratio) else UNKNOWN,
        dict(P_over_C=update_ratio, arms=cost_observations, all_attempt_cost_covered=all_cost, comparable=comparable), 1.25)
    common_ids = [r["pair_id"] for r in full if r["C_success"] and r["P_success"]]
    common_rows = [rows[(pid, arm)] for pid in common_ids for arm in ("C", "P")]
    valid = bool(common_ids) and all(numeric(r.get("full_policy_elapsed_sec")) for r in common_rows)
    totals = {arm: sum(rows[(pid, arm)]["full_policy_elapsed_sec"] for pid in common_ids) if valid else UNKNOWN for arm in ("C", "P")}
    policy_ratio = totals["P"] / totals["C"] if valid and totals["C"] > 0 else UNKNOWN
    gate("cost_common_success_total_policy", policy_ratio <= 1.25 if all_complete and all_cost and comparable and numeric(policy_ratio) else UNKNOWN,
        dict(P_over_C=policy_ratio, common_success_pair_ids=common_ids, sum_full_policy_elapsed_sec=totals,
            comparable=comparable), 1.25, "SUM(full policy) on the same both-success pairs; not episode wall or mean per-pair ratios")
    states = [c["status"] for c in checks.values()]
    return dict(schema="tf-pf2-r3-registered-investment-analysis/v1", input_result=str(result_path),
        source_code_commit=CODE, registered_gates_sources=[str(R1 / "PILOT-GATES.json"), str(JOB / "PILOT-GATES.json")],
        pilot_investment_decision="NO_GO" if "FAIL" in states else "UNKNOWN_NOT_PASS" if UNKNOWN in states else "GO_FOR_DISCUSSION_ONLY",
        checks=checks, pairs=pairs, common_success_length_ratio=lengths,
        all_six_pairs_complete=all_complete, cost_comparability=UNKNOWN if not comparable else True,
        observations_do_not_establish_noninferiority=True, old_TF_ON_NO_GO_unchanged=True,
        no_global_permission_change=True, permissions=PERMISSIONS, no_new_execution_authorization=True,
        max_counterexamples=[r for r in full if numeric(r["both_success_ratio"]) and r["both_success_ratio"] > 1.30],
        no_historical_C_rescue=True, no_task_subset_hides_failure=True, no_new_training=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=OWN / "result-v1")
    args = ap.parse_args()
    source, out = args.result.resolve(strict=True), args.out.resolve()
    require(source.is_relative_to(JOB) and out.parent == OWN and not out.exists(), "r3 input/new analysis namespace only")
    require(len(os.sched_getaffinity(0)) == 1 and os.getenv("CUDA_VISIBLE_DEVICES") == "", "CPU1 and hidden CUDA required")
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    write(out, "PREREGISTRATION.json", dict(goal=__doc__, input_result=str(source), CPU1=True, GPU=False,
        wall_cap_sec=60, storage_cap_bytes=CAP, no_execution=True, permissions=PERMISSIONS))
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("bounded CPU investment analysis")))
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        result = analyze(source)
        result["CPU_wall_sec"] = time.monotonic() - started
        result["created_at"] = datetime.now(timezone.utc).isoformat()
        write(out, "INVESTMENT-RESULT.json", result)
        lines = ["# 登记的六对投入门", "", f"决策：{result['pilot_investment_decision']}。只针对当前r3固定panel。", "",
            "|门|判定|", "|---|---|", *[f"|{k}|{v['status']}|" for k, v in result["checks"].items()], "",
            "任一已观察共同成功长度比超过1.30即不可消除的FAIL；不能因后续rescued改门。缺失或计时不可比为UNKNOWN。",
            "四项原权限全false，旧TF_ON NO_GO不变；不借历史C算rescued，不授权扩规模/训练，不证明非劣。细节见JSON。", ""]
        write(out, "INVESTMENT-RESULT.md", "\n".join(lines))
        write(out, "CPU-FINAL.json", dict(status="COMPLETE_CPU_REGISTERED_GATE_CALCULATION", decision=result["pilot_investment_decision"],
            elapsed_sec=time.monotonic() - started, permissions=PERMISSIONS, GPU=False))
        print(json.dumps(dict(decision=result["pilot_investment_decision"], output=str(out)), ensure_ascii=False))
    except BaseException as exc:
        write(out, "FAILED.json", dict(status="FAILED_CPU_GATE_ANALYSIS_NOT_A_QUALITY_DECISION", error=repr(exc),
            elapsed_sec=time.monotonic() - started, permissions=PERMISSIONS))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()

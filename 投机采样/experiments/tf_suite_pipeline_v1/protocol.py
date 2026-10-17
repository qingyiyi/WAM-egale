"""Suite-specific, finite full-episode data/training/evaluation pipeline metadata."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
WORKTREE = ROOT / "投机采样/wam-p0-worktree"
SCHEMA = "tf-suite-pipeline/v1"
SUITES = ("libero_object", "libero_goal", "libero_spatial")
PHASES = ("T_POOL", "PARENT_DATA", "CURRENT_DATA", "FINAL_EVAL")
ARMS = ("T", "C", "P")
CAMERAS = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
READ_CAP = 64 * 2**20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.stat().st_size <= READ_CAP, "oversized suite metadata: " + str(path))
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= READ_CAP, "oversized suite receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)


def episode_dir(job, case_id, arm):
    require(type(case_id) is int and 0 <= case_id < 500 and arm in ARMS, "unregistered suite episode")
    return Path(job).resolve() / "episodes" / f"case-{case_id:04d}" / arm


def row(plan, row_id):
    require(type(row_id) is int and 0 <= row_id < len(plan["rows"]), "unregistered phase row")
    value = plan["rows"][row_id]
    require(value["row_id"] == row_id, "dense phase row identity differs")
    return value


def run_rows(plan):
    return [r for r in plan["rows"] if not r.get("reuse_source")]


def split_role(phase, init):
    if phase == "T_POOL":
        return "FIT" if init <= 17 else "CAL"
    if phase == "PARENT_DATA":
        return "FIT" if init <= 21 else "CAL"
    if phase == "CURRENT_DATA":
        return "FIT" if init <= 24 else "CAL"
    return "EVAL"


def load(job):
    job = Path(job).resolve(strict=True)
    p = read(job / "plan.json")
    require(p["schema"] == SCHEMA and p["phase"] in PHASES, "new suite phase/schema required")
    require(Path(p["job"]).resolve() == job and Path(p["root_path"]).resolve() == ROOT,
        "actual suite job/root source differs")
    info = p["suite_info"]
    require(info["suite"] in SUITES and p["suite"] == info["suite"], "suite matched target required")
    require(type(info["checkpoint_id"]) is str and info["checkpoint_id"].startswith("Wjjjh/")
        and bool(info["publisher_revision"]) and bool(info["model_path"]), "explicit community target/revision required")
    norm = info["norm_stat"]
    require(set(norm) == {"q01", "q99"} and all(len(norm[k]) == 30 and all(type(v) in (int, float)
        and math.isfinite(v) for v in norm[k]) for k in norm), "actual padded30 suite quantiles required")
    require(all(norm["q99"][i] > norm["q01"][i] for i in range(7))
        and all(norm[k][i] == 0 for k in norm for i in range(7, 30)), "suite seven-channel quantiles/padding differ")
    require(set(info["public_task_prompts"]) == {str(i) for i in range(10)}
        and all(type(v) is str and v for v in info["public_task_prompts"].values()), "ten actual suite prompts required")
    require(p["frozen_permissions"] == PERMISSIONS and p.get("permissions", PERMISSIONS) == PERMISSIONS,
        "historical permissions cannot change")
    require(p["mode"] == "verifier_k2_b2_bonus" and p["selector"] == "prefix_then_cost"
        and p["threshold"] == 0.007839680526450805 and p["evaluation_seed"] == 20260922
        and p["horizon"] == 800, "fixed scheduler/tree/acceptance/evaluation seed differs")
    require(p["max_GPU_workers"] == 1 and p["gpu_index"] in (0, 1)
        and p["server_threads_per_worker"] == p["client_threads_per_worker"] == 2
        and p["per_episode_wall_sec"] == 1200 and p["per_episode_forward_attempt_cap"] == 4000,
        "one isolated model worker and complete episode budget required")
    require(1 <= p["port"] <= 65535 and 1 <= p["master_port"] <= 65535 and p["port"] != p["master_port"],
        "distinct registered suite service/master ports required")
    affinity = p["CPU_affinity"]
    require(set(affinity) == {"server", "client", "control"}
        and all(isinstance(affinity[k], list) and len(affinity[k]) == n
            and all(type(i) is int and i >= 0 for i in affinity[k])
            for k, n in (("server", 2), ("client", 2), ("control", 1)))
        and len(set(affinity["server"] + affinity["client"])) == 4,
        "explicit two server/two client/one control CPU assignments required")
    require(type(p["remote_data_disk_floor_bytes"]) is int and p["remote_data_disk_floor_bytes"] >= 5 * 2**30
        and all(type(p[k]) is str and p[k] for k in
            ("source_code_commit", "va_python", "client_python", "LIBERO_CONFIG_PATH", "registry", "bank", "target_source_reference")),
        "finite disk floor and frozen suite runtime paths required")
    for key in ("no_smoke", "no_independent_tests", "no_SHA", "credential_storage"):
        require(p[key] is (False if key == "credential_storage" else True), "suite execution boundary differs: " + key)
    origin, work, hard = [datetime.fromisoformat(p[k]) for k in
        ("authorization_origin", "absolute_work_stop", "absolute_cleanup_deadline")]
    require(all(x.utcoffset() is not None for x in (origin, work, hard)) and origin < work < hard
        and (hard - work).total_seconds() >= 300, "finite timezone-aware suite budgets required")
    phase = p["phase"]
    case_ids = p.get("case_ids", list(range(500)))
    if phase == "FINAL_EVAL":
        require(isinstance(case_ids, list) and case_ids and case_ids == sorted(set(case_ids))
            and all(type(i) is int and 0 <= i < 500 for i in case_ids), "predeclared fixed evaluation case subset required")
    if phase == "T_POOL":
        expected = [(t, i, "T") for i in range(20) for t in range(10)]
    elif phase == "PARENT_DATA":
        expected = [(t, i, "C") for i in range(20, 23) for t in range(10)]
    elif phase == "CURRENT_DATA":
        expected = [(t, i, "C") for i in range(23, 26) for t in range(10)]
    else:
        expected = [(c % 10, c // 10, a) for c in case_ids
            for a in (("T", "C", "P"), ("C", "P", "T"), ("P", "T", "C"))[c % 3]]
    require(len(p["rows"]) == len(expected), "complete registered phase manifest required")
    reuse_plans = {}
    for j, (r, (task, init, arm)) in enumerate(zip(p["rows"], expected)):
        require((r["row_id"], r["case_id"], r["task"], r["init"], r["arm"], r["worker_index"])
            == (j, init * 10 + task, task, init, arm, 0)
            and Path(r["out"]).resolve() == episode_dir(job, init * 10 + task, arm), "dense suite row/order/out differs")
        must_reuse = phase == "FINAL_EVAL" and arm == "T" and init < 20
        require(bool(r.get("reuse_source")) == must_reuse, "FINAL_EVAL must reuse exactly T_POOL200 Target without rerun")
        if must_reuse:
            source = r["reuse_source"]
            prior_job = Path(source["job"]).resolve(strict=True)
            prior_out = Path(source["out"]).resolve(strict=True)
            require(prior_job != job and prior_out == episode_dir(prior_job, init * 10 + task, "T"), "reused Target source path differs")
            if str(prior_job) not in reuse_plans:
                old = read(prior_job / "plan.json")
                old_info = old["suite_info"]
                require(old["schema"] == SCHEMA and old["phase"] == "T_POOL"
                    and all(old_info[k] == info[k] for k in ("suite", "checkpoint_id", "publisher_revision", "norm_stat", "public_task_prompts"))
                    and Path(old_info["model_path"]).resolve(strict=True) == Path(info["model_path"]).resolve(strict=True)
                    and old["evaluation_seed"] == p["evaluation_seed"] and old["frozen_permissions"] == PERMISSIONS,
                    "reused Target must have identical suite target/norm/seed provenance")
                reuse_plans[str(prior_job)] = old
            terminal, counter, audit, finish = [read(prior_out / n) for n in
                ("EPISODE-TERMINAL.json", "COUNTER.json", "AUDIT.json", "FINISH_READY.json")]
            require(terminal["schema"] == "tf-suite-full-episode-terminal/v1" and terminal["complete_episode"] is True
                and terminal["horizon"] == 800 and terminal["case_id"] == init * 10 + task
                and terminal["task"] == task and terminal["init"] == init and terminal["arm"] == "T"
                and terminal["suite"] == info["suite"] and terminal["phase"] == "T_POOL"
                and (terminal["success"] is True or terminal["horizon_reached"] is True)
                and counter["schema"] == "tf-suite-episode-counter/v1" and counter["case_id"] == init * 10 + task
                and counter["arm"] == "T" and audit["case_id"] == init * 10 + task and audit["arm"] == "T"
                and finish["full_episode_terminal"] is True and finish["async_drained"] is True
                and all(x["permissions"] == PERMISSIONS for x in (terminal, counter, audit, finish)),
                "reused Target needs original full natural episode and closed component receipts, including failures")
    if phase != "FINAL_EVAL":
        data_root = Path(p["suite_storage_root"]).resolve()
        require(Path(p["collector_output_dir"]).resolve().is_relative_to(data_root), "phase columns must share suite data budget root")
    if phase != "T_POOL":
        require(p["selected"]["architecture"] == "transformer64" and p["selected"]["parameter_count"] == 53070,
            "suite-specific fixed TF64 checkpoint required")
        want = {"PARENT_DATA": "PARENT2000", "CURRENT_DATA": "CONT3000", "FINAL_EVAL": "ON3000"}[phase]
        require(p["selected"]["condition"] == want and p["selected"]["metadata"]["suite"] == info["suite"]
            and p["selected"]["metadata"]["suite_short"] == info["suite"].removeprefix("libero_"),
            "suite/phase draft provenance differs")
    require(type(p["job_forward_attempt_cap"]) is int and p["job_forward_attempt_cap"] > 0
        and type(p["new_evidence_cap_bytes"]) is int and p["new_evidence_cap_bytes"] > 0,
        "explicit finite suite job budgets required")
    return p

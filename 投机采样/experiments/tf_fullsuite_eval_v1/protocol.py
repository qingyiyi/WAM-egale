"""Authorized complete Long500 T/TF/TF+PF2 coverage; no digest or legacy loader."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
WORKTREE = ROOT / "投机采样/wam-p0-worktree"
SCHEMA = "tf-fullsuite-three-arm-eval/v1"
SCIENTIFIC_COMMIT = "4f04b35e44cbc00483dc6245ad57abec718385c9"
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
ARMS = ("T", "C", "P")
CAMERAS = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
ORDERS = ("T_C_P", "C_P_T", "P_T_C")
READ_CAP = 64 * 2**20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.stat().st_size <= READ_CAP, "oversized protocol/receipt: " + str(path))
    return json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def write(path, obj):
    path = Path(path)
    raw = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= READ_CAP, "oversized protocol receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def episode_dir(job, case_id, arm):
    require(type(case_id) is int and 0 <= case_id < 500 and arm in ARMS, "unregistered case/arm")
    return Path(job).resolve() / "episodes" / f"case-{case_id:04d}" / arm


def case(plan, case_id):
    require(type(case_id) is int and 0 <= case_id < 500, "unregistered case id")
    row = plan["cases"][case_id]
    require(row["case_id"] == case_id, "fixed init-major case manifest differs")
    return row


pair = case  # Source adapter compatibility only; all new evidence uses case_id.


def load(job):
    job = Path(job).resolve(strict=True)
    p = read(job / "plan.json")
    require(p["schema"] == SCHEMA and p["scientific_commit"] == SCIENTIFIC_COMMIT,
        "new fullsuite protocol identity differs")
    require(p["frozen_permissions"] == PERMISSIONS and p.get("permissions", PERMISSIONS) == PERMISSIONS
        and p["preserve_old_FAILED_22_gaps_and_NO_GO"] is True, "historical gates/failures must stay frozen")
    require(p.get("job", str(job)) == str(job) and p["suite"] == p["benchmark"] == "libero_10",
        "operating job/Long suite differs")
    require(len(p["cases"]) == 500, "exact 500 registered Long cases required")
    for i, row in enumerate(p["cases"]):
        init, task = divmod(i, 10)
        require((row["case_id"], row["task"], row["init"], row["proposed_GPU_index"], row["within_case_order"])
            == (i, task, init, (task + init) % 2, ORDERS[i % 3]) and row["execution_authorized"] is True,
            "fixed init-major task-balanced case/order/worker/authorization differs")
    require(p["horizon"] == 800 and p["evaluation_seed"] == 20260922 and p["mode"] == "verifier_k2_b2_bonus"
        and p["selector"] == "prefix_then_cost" and type(p["threshold"]) in (int, float)
        and math.isfinite(p["threshold"]) and p["threshold"] == 0.007839680526450805,
        "production seed/horizon/tree/acceptance differs")
    require(p["max_GPU_workers"] == 2 and p["max_complete_episodes"] == 1500
        and p["per_episode_wall_sec"] == 1200 and p["per_episode_forward_attempt_cap"] == 4000
        and p["job_forward_attempt_cap"] == 6001000 and p["non_episode_forward_attempt_cap"] == 1000
        and p["new_evidence_cap_bytes"] == 10 * 2**30, "registered Long coverage resource cap differs")
    for name in ("no_smoke", "no_independent_tests", "no_remote_SHA_including_hidden_loader_or_fingerprint",
            "no_training_or_new_FIT", "no_TEST_or_120_or_formal_benchmark"):
        require(p[name] is True, "coverage boundary changed: " + name)
    selected = p["selected"]
    require(selected["architecture"] == "transformer64" and selected["parameter_count"] == 53070
        and selected["metadata"]["condition"] == "TF_ON" and selected["metadata"]["total_updates"] == 3000
        and selected["metadata"]["permissions"] == PERMISSIONS, "fixed Long draft metadata differs")
    origin, work, hard = [datetime.fromisoformat(p[k]) for k in
        ("authorization_origin", "absolute_work_stop", "absolute_cleanup_deadline")]
    require(all(x.utcoffset() is not None for x in (origin, work, hard)) and origin < work < hard
        and (hard - origin).total_seconds() <= 48 * 3600
        and (hard - work).total_seconds() >= 900, "48h maximum authorization/deadline/cleanup reserve differs")
    require(p["credential_storage"] is False, "no credential evidence")
    return p

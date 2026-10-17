"""Authorized complete Long500 T/TF/TF+PF2 coverage; no digest or legacy loader."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
WORKTREE = ROOT / "投机采样/wam-p0-worktree"
SCHEMA = "tf-fullsuite-four-worker-eval/v2"
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
        require((row["case_id"], row["task"], row["init"], row["proposed_GPU_index"], row["within_case_order"], row["execution_worker_index"])
            == (i, task, init, (task + init) % 2, ORDERS[i % 3], (task + init) % 4) and row["execution_authorized"] is True,
            "fixed init-major task-balanced case/order/worker/authorization differs")
    require(p["horizon"] == 800 and p["evaluation_seed"] == 20260922 and p["mode"] == "verifier_k2_b2_bonus"
        and p["selector"] == "prefix_then_cost" and type(p["threshold"]) in (int, float)
        and math.isfinite(p["threshold"]) and p["threshold"] == 0.007839680526450805,
        "production seed/horizon/tree/acceptance differs")
    require(p["max_GPU_workers"] == 4 and p["physical_GPU_count"] == 2 and p["max_complete_episodes"] == 1500
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
    prior = Path(p["prior_job"]).resolve(strict=True)
    require(prior != job, "new four-worker namespace must differ from prior")
    old = read(prior / "plan.json")
    require(old["scientific_commit"] == p["scientific_commit"] and old["suite"] == p["suite"]
        and old["selected"]["path"] == p["selected"]["path"] and old["evaluation_seed"] == p["evaluation_seed"]
        and old["threshold"] == p["threshold"] and old["mode"] == p["mode"]
        and old["selector"] == p["selector"] and old["bank"] == p["bank"]
        and old["target_checkpoint_id"] == p["target_checkpoint_id"]
        and old["authorization_origin"] == p["authorization_origin"]
        and old["absolute_work_stop"] == p["absolute_work_stop"]
        and old["absolute_cleanup_deadline"] == p["absolute_cleanup_deadline"], "prior scientific identity or absolute series budget reset")
    completed = p["prior_completed_case_ids"]
    blocked = p.get("blocked_prior_partial_case_ids", [])
    require(completed == sorted(set(completed)) and blocked == sorted(set(blocked))
        and all(type(i) is int and 0 <= i < 500 for i in completed + blocked)
        and not set(completed).intersection(blocked), "explicit completed/blocked prior case lists differ")
    observed_started = set()
    episodes = prior / "episodes"
    if episodes.is_dir():
        for directory in episodes.iterdir():
            if directory.is_dir() and directory.name.startswith("case-"):
                i = int(directory.name[5:])
                require(0 <= i < 500, "unregistered predecessor case directory")
                if any((directory / arm).exists() for arm in ARMS):
                    observed_started.add(i)
    for spec in p["cases"]:
        i = spec["case_id"]
        if i in completed:
            for a in ARMS:
                terminal = read(episode_dir(prior, i, a) / "EPISODE-TERMINAL.json")
                finish = read(episode_dir(prior, i, a) / "FINISH_READY.json")
                counter = read(episode_dir(prior, i, a) / "COUNTER.json")
                audit = read(episode_dir(prior, i, a) / "AUDIT.json")
                require(terminal["complete_episode"] is True and terminal["quality_terminal"] is True
                    and terminal["status"] == "COMPLETE_RECORDED_FULL_EPISODE" and terminal["horizon"] == 800
                    and type(terminal["env_steps"]) is int and 0 < terminal["env_steps"] <= 800
                    and (terminal["success"] is True or terminal["horizon_reached"] is True)
                    and terminal["failure800_penalty"] == (terminal["env_steps"] if terminal["success"] else 800)
                    and terminal["case_id"] == i and terminal["arm"] == a
                    and terminal["task"] == spec["task"] and terminal["init"] == spec["init"]
                    and terminal["context"] == dict(case_id=i, arm=a, out=str(episode_dir(prior, i, a)))
                    and terminal["permissions"] == PERMISSIONS and type(terminal["success"]) is bool
                    and finish["full_episode_terminal"] is True and finish["async_drained"] is True
                    and finish["case_id"] == i and finish["arm"] == a
                    and finish["status"] in ("PASS", "FAILED_COMPONENT_AUDIT")
                    and finish["permissions"] == PERMISSIONS
                    and counter["schema"] == "tf-fullsuite-episode-counter/v1"
                    and counter["case_id"] == i and counter["arm"] == a
                    and counter["task"] == spec["task"] and counter["init"] == spec["init"]
                    and counter["policy_chunks"] == terminal["policy_chunks"]
                    and counter["permissions"] == PERMISSIONS
                    and audit["schema"] == "tf-fullsuite-episode-audit/v1"
                    and audit["case_id"] == i and audit["arm"] == a and audit["permissions"] == PERMISSIONS,
                    "prior reuse requires all three original complete/drained terminals, including failures")
    require(observed_started == set(completed) | set(blocked), "old partial/started cases cannot silently become new pending work")
    cleanup = read(prior / "GPU-OWNED-CLEANUP.json")
    require(cleanup["all_owned_work_exited"] is True and cleanup["all_owned_groups_empty"] is True,
        "prior GPU lifecycles must be fully cleaned before four-worker start")
    prior_known = 0
    for worker in range(old["max_GPU_workers"]):
        live = read(prior / f"server-worker{worker}" / "LIVE-COUNTER.json")
        require(type(live["target_forward_attempts"]) is int and live["target_forward_attempts"] >= 0
            and live["active_target_forwards"] == 0, "prior exact closed worker attempt accounting unavailable")
        prior_known += live["target_forward_attempts"]
    require(type(p["prior_target_forward_debit"]) is int and p["prior_target_forward_debit"] >= prior_known,
        "prior work cannot be reset or under-debited")
    pending = [s for s in p["cases"] if s["case_id"] not in set(completed) | set(blocked)]
    caps = p["worker_forward_caps"]
    for worker in range(4):
        n = sum(s["execution_worker_index"] == worker for s in pending)
        require(caps[str(worker)] == n * 12000 + 250, "remaining-case worker forward cap differs")
    require(sum(caps.values()) + p["prior_target_forward_debit"] <= p["job_forward_attempt_cap"],
        "prior work and new reservations exceed fixed series forward cap")
    return p


def source_job(plan, job, case_id):
    prior_ids = set(plan["prior_completed_case_ids"]) | set(plan.get("blocked_prior_partial_case_ids", []))
    return Path(plan["prior_job"]).resolve() if case_id in prior_ids else Path(job).resolve()


def pending_cases(plan):
    excluded = set(plan["prior_completed_case_ids"]) | set(plan.get("blocked_prior_partial_case_ids", []))
    return [s for s in plan["cases"] if s["case_id"] not in excluded]

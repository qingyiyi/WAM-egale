"""New, bounded six-pair TF-continuous/PF2 protocol. No digest or old loader."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
WORKTREE = ROOT / "投机采样/wam-p0-worktree"
SCHEMA = "tf-pf2-dual-autodl-campaign/v1"
SCIENTIFIC_COMMIT = "4f04b35e44cbc00483dc6245ad57abec718385c9"
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
ARMS = ("C", "P")
CAMERAS = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
FIXED_PAIRS = ((1, 1, 29, "risk", "C_P", 0), (2, 1, 36, "control", "P_C", 1),
    (3, 3, 26, "risk", "P_C", 0), (4, 3, 36, "control", "C_P", 1),
    (5, 4, 26, "risk", "C_P", 0), (6, 4, 36, "control", "P_C", 1))
READ_CAP = 64 * 2**20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.stat().st_size <= READ_CAP, "oversized protocol/receipt: " + str(path))
    return json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def write(path, obj):
    """Write one new receipt exclusively; never replace an old terminal."""
    path = Path(path)
    raw = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= READ_CAP, "oversized protocol receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def episode_dir(job, pair_id, arm):
    require(type(pair_id) is int and 1 <= pair_id <= 6 and arm in ARMS, "unregistered pair/arm")
    return Path(job).resolve() / "episodes" / f"pair-{pair_id:02d}" / arm


def pair(plan, pair_id):
    rows = [r for r in plan["pairs"] if r["pair_id"] == pair_id]
    require(len(rows) == 1, "one fixed pair required")
    return rows[0]


def load(job):
    """Validate new metadata only. Asset loading is the new deployment module's job."""
    job = Path(job).resolve(strict=True)
    p = read(job / "plan.json")
    require(p["schema"] == SCHEMA and p["scientific_commit"] == SCIENTIFIC_COMMIT,
        "new TF/PF2 scientific protocol identity differs")
    require(p["frozen_permissions"] == PERMISSIONS and p["preserve_old_FAILED_22_gaps_and_NO_GO"] is True,
        "old permissions and failures must remain frozen")
    require(p.get("permissions", PERMISSIONS) == PERMISSIONS and p.get("job", str(job)) == str(job),
        "active operating job/permission alias differs")
    rows = [(r["pair_id"], r["task"], r["init"], r["stratum"], r["within_pair_order"], r["proposed_GPU_index"])
        for r in p["pairs"]]
    require(rows == list(FIXED_PAIRS), "fixed balanced six-pair panel/order/worker assignment changed")
    require(p["horizon"] == 800 and p["evaluation_seed"] == 20260922 and p["mode"] == "verifier_k2_b2_bonus"
        and p["selector"] == "prefix_then_cost" and math.isfinite(p["threshold"])
        and p["threshold"] == 0.007839680526450805, "production horizon/seed/tree/acceptance changed")
    require(p["max_GPU_workers"] == 2 and p["max_complete_episodes"] == 12
        and p["per_episode_wall_sec"] == 1200 and p["per_episode_forward_attempt_cap"] == 4000
        and p["job_forward_attempt_cap"] == 49000 and p["non_episode_forward_attempt_cap"] == 1000
        and p["new_evidence_cap_bytes"] == 3 * 2**30, "registered finite panel budget changed")
    for name in ("no_smoke", "no_independent_tests", "no_remote_SHA_including_hidden_loader_or_fingerprint",
            "no_training_or_new_FIT", "no_TEST_or_120_or_formal_benchmark", "no_second_environment"):
        require(p[name] is True, "registered boundary changed: " + name)
    selected = p["selected"]
    require(selected["architecture"] == "transformer64" and selected["parameter_count"] == 53070
        and selected["metadata"]["condition"] == "TF_ON" and selected["metadata"]["total_updates"] == 3000
        and selected["metadata"]["permissions"] == PERMISSIONS, "fixed TF_ON checkpoint metadata changed")
    origin, work, hard = [datetime.fromisoformat(p[k]) for k in
        ("authorization_origin", "absolute_work_stop", "absolute_cleanup_deadline")]
    require(all(d.utcoffset() is not None for d in (origin, work, hard))
        and 0 < (hard - origin).total_seconds() <= 12 * 3600
        and 0 < (hard - work).total_seconds() >= 900 and origin < work,
        "12h absolute authorization/cleanup reserve changed")
    require(p["credential_storage"] is False, "credentials must not be written to evidence")
    return p

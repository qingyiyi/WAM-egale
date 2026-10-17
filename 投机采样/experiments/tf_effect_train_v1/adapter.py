"""Independent, strict 2500 -> 3000 loader; never installs an online default.

No forward, CUDA initialization, data collection, or training occurs at import.
The original ActionPredictor and parent loader remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

EXP = Path(__file__).resolve().parent.parent
if str(EXP) not in sys.path:
    sys.path.append(str(EXP))
from target_trace_draft_train_v1.adapter import (ActionPredictor, NORMALIZATION,
    model_source, sha)
from target_trace_draft_finetune_v1.adapter import load_checkpoint as load_parent

SCHEMA = "tf-effect-draft-3000/v1"
PLAN_SCHEMA = "tf-effect-training-plan/v1"
SELECTION_SCHEMA = "tf-effect-fixed-final-selection/v1"
FINAL_STATUS = "PASS_FIXED_FINAL_TF_EFFECT_TRAINING"
COMMIT = "b0dfac3a02f65de33a1217246b83003615166640"
SEED = 20260923
TAU = 0.007839680526450805
CONDITIONS = {"TF_CONT": "transformer64", "TF_ON": "transformer64", "MLP_ON": "mlp96"}
PARAMETERS = {"transformer64": 53070, "mlp96": 52928}
PARENT_SHAS = {
    "transformer64": "9cb92a4862579edc9cfb99210b0535e620daba56dd5abcc094a7d735dbadab07",
    "mlp96": "655e8647efa0fa44767b75b8e98a02ec6cbe84fc058b10aa15860f6cd609165e",
}
GENERATOR_SHA = "3657388fdce0763093d1b704c88a03b6836954160d43dddb9f75337a054d64ac"
PERMISSIONS = {
    "engineering_quality_screen": False,
    "independent_test_complete": False,
    "noninferiority_established": False,
    "formal_performance_permission": False,
}
KEYS = ("state", "previous", "target", "scalars", "delta", "task", "episode", "node")
FIT_IDS = tuple(t * 100 + i for t in range(10) for i in (30, 31))
CAL_IDS = tuple(t * 100 + 32 for t in range(10))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def stat(path):
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def validate_new_columns(path, *, expected_manifest_sha256=None):
    """CPU shape/split audit only. Reachability additionally needs its audit.

    This routine does not infer that a selected branch was actually reached.
    That property must come from the separately checked collector certificate.
    """
    path = Path(path).resolve(strict=True)
    require(path.stat().st_size <= 512 * 1024 * 1024, "compact columns exceed 512 MiB")
    pack = torch.load(path, map_location="cpu", weights_only=True)
    require(pack.get("schema") == "tf-effect-onpolicy-columns/v1", "new columns schema differs")
    manifest = pack.get("manifest", {})
    manifest_digest = digest_json(manifest)
    require(expected_manifest_sha256 is None or manifest_digest == expected_manifest_sha256,
        "new columns manifest changed")
    require(manifest.get("source_commit") == COMMIT
        and manifest.get("generator_condition") == "TF_CONT3000"
        and manifest.get("generator_total_updates") == 3000
        and manifest.get("generator_checkpoint_sha256") == GENERATOR_SHA,
        "actual CONT3000 generator identity differs from training child2500")
    generator = Path(manifest["generator_checkpoint"]).resolve(strict=True)
    require(sha(generator) == GENERATOR_SHA, "generator checkpoint bytes changed")
    require(manifest.get("fit_episode_ids") == list(FIT_IDS)
        and manifest.get("cal_episode_ids") == list(CAL_IDS)
        and manifest.get("complete_episodes") == 30
        and manifest.get("all_episodes_terminal") is True
        and manifest.get("extra_teacher_forwards") == 0
        and manifest.get("row_selection") == "root_and_actually_reached_parent"
        and manifest.get("counterfactual_nodes_in_columns") is False
        and manifest.get("actual_o3_preserved") is True
        and manifest.get("threshold") == TAU
        and manifest.get("production_mode") == "verifier_k2_b2_bonus",
        "fixed 20 FIT / 10 CAL terminal collection contract differs")
    splits = {}
    for name, episodes in (("FIT", FIT_IDS), ("CAL", CAL_IDS)):
        raw = pack.get(name, {})
        require(set(KEYS).issubset(raw), "missing compact columns: " + name)
        columns = {k: raw[k] for k in KEYS}
        n = len(columns["state"])
        require(0 < n <= 225000, "empty or excessive reached-node rows: " + name)
        shapes = {"state": (n, 112), "previous": (n, 112), "target": (n, 112),
            "scalars": (n, 3), "delta": (n,), "task": (n,), "episode": (n,), "node": (n,)}
        for key, shape in shapes.items():
            tensor = columns[key]
            require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape
                and tensor.device.type == "cpu", "column geometry differs: " + name + "/" + key)
            require(tensor.dtype == (torch.int64 if key in ("task", "episode", "node") else torch.float32),
                "column dtype differs: " + name + "/" + key)
            require(bool(torch.isfinite(tensor).all()), "nonfinite compact column: " + key)
        require(set(columns["episode"].tolist()) == set(episodes)
            and set(columns["task"].tolist()) == set(range(10))
            and torch.equal(columns["episode"] // 100, columns["task"]), "episode/task split differs")
        require(set(columns["node"].tolist()) <= {0, 1, 2}, "unknown root/parent node")
        for episode in episodes:
            require(bool(((columns["episode"] == episode) & (columns["node"] == 0)).any()),
                "episode lacks actual root rows")
        sigma, next_sigma, normalized_t = columns["scalars"].unbind(1)
        require(bool(((sigma >= 0) & (sigma <= 1) & (next_sigma >= 0)
            & (next_sigma <= sigma) & (normalized_t >= 0) & (normalized_t <= 1)).all()),
            "invalid sigma/timestep direction")
        require(bool((columns["delta"] > 0).all()) and torch.allclose(columns["delta"],
            (sigma - next_sigma).abs(), rtol=1e-5, atol=1e-7), "actual delta/sigma mismatch")
        if "state_reached" in raw:
            require(raw["state_reached"].shape == (n,) and raw["state_reached"].dtype == torch.bool
                and bool(raw["state_reached"].all()), "counterfactual row entered primary columns")
        splits[name] = {k: v.contiguous() for k, v in columns.items()}
    require(not set(splits["FIT"]["episode"].tolist()) & set(splits["CAL"]["episode"].tolist()),
        "new FIT/CAL overlap")
    require(sum(len(v["state"]) for v in splits.values()) <= 225000, "combined node-row cap exceeded")
    return splits["FIT"], splits["CAL"], manifest, manifest_digest


def verify_plan(plan_path):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("source_commit") == COMMIT
        and plan.get("seed") == SEED and plan.get("initial_total_updates") == 2500
        and plan.get("continuation_updates") == 500 and plan.get("total_updates") == 3000
        and plan.get("batch") == 512 and plan.get("target_half") == plan.get("same_query_half") == 256
        and plan.get("learning_rate") == 3e-5 and plan.get("weight_decay") == .01
        and plan.get("normalization_refit") is False and plan.get("optimizer_state_reset") is True
        and plan.get("permissions") == PERMISSIONS and plan.get("robot_quality_validated") is False
        and plan.get("deployed") is False, "strict independent training plan differs")
    require(plan.get("conditions") in (["TF_CONT", "TF_ON"], ["TF_CONT", "TF_ON", "MLP_ON"]),
        "only fixed TF pair and optional matched MLP-on are allowed")
    for path, expected in plan["source_sha256"].items():
        require(sha(path) == expected, "frozen source changed: " + path)
    for path, expected in plan["input_stat"].items():
        require(stat(path) == expected, "frozen data stat changed: " + path)
    for path, expected in plan["evidence_sha256"].items():
        require(sha(path) == expected, "frozen evidence changed: " + path)
    require(plan.get("new_manifest_sha256") == digest_json(plan["new_manifest"]), "manifest binding differs")
    return plan


def load_checkpoint(path, *, architecture, device, expected_channels, expected_sha256=None,
        expected_condition=None):
    channels = tuple(range(expected_channels)) if isinstance(expected_channels, int) else tuple(expected_channels)
    require(channels == tuple(range(7)), "original ordered seven action channels required")
    require(torch.device(device).type in ("cpu", "cuda"), "explicit CPU or CUDA predictor required")
    path = Path(path).resolve(strict=True)
    root = path.parent.parent
    require(not (root / "FAILED.json").exists() and not (root / "STOP_REQUESTED").exists()
        and not (path.parent / "FAILED.json").exists(), "failed/stopped training cannot load")
    digest = sha(path)
    require(expected_sha256 is None or digest == expected_sha256, "checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    condition = metadata.get("condition")
    require(condition in CONDITIONS and CONDITIONS[condition] == architecture
        and (expected_condition is None or expected_condition == condition), "condition/architecture mismatch")
    require(path == root / f"{condition}-seed{SEED}" / "checkpoint.pt", "noncanonical training cell")
    plan_path = root / "PLAN.json"
    plan = verify_plan(plan_path)
    require(condition in plan["conditions"] and metadata.get("plan") == str(plan_path)
        and metadata.get("plan_sha256") == sha(plan_path), "checkpoint plan binding differs")
    source = model_source()
    require(payload.get("schema") == SCHEMA and payload.get("architecture") == architecture
        and payload.get("specification") == source.SPECS[architecture], "new independent schema differs")
    for key in ("seed", "initial_total_updates", "continuation_updates", "total_updates", "batch",
            "target_half", "same_query_half", "learning_rate", "weight_decay", "normalization_refit",
            "optimizer_state_reset", "permissions", "robot_quality_validated", "deployed",
            "new_manifest_sha256"):
        require(metadata.get(key) == plan[key], "checkpoint recipe binding differs: " + key)
    cell = plan["cells"][condition]
    require(metadata.get("same_query_source") == cell["same_query_source"]
        and metadata.get("data_manifest_sha256") == digest_json(cell["data_manifest"]),
        "training half/source manifest differs")
    parent_path = Path(metadata["parent_checkpoint"]).resolve(strict=True)
    parent_sha = PARENT_SHAS[architecture]
    require(str(parent_path) == cell["parent_checkpoint"]
        and metadata.get("parent_checkpoint_sha256") == parent_sha
        and cell["parent_checkpoint_sha256"] == parent_sha and sha(parent_path) == parent_sha,
        "actual seed23 child2500 parent differs")
    parent = load_parent(parent_path, architecture=architecture, device="cpu",
        expected_channels=channels, expected_sha256=parent_sha)
    norm = payload.get("normalization", {})
    require(set(norm) == set(NORMALIZATION), "normalizer inventory differs")
    for key, shape in NORMALIZATION.items():
        value = norm[key]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
            and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
            and torch.equal(value, getattr(parent, key))
            and (not key.endswith("std") or bool((value > 0).all())), "parent normalization changed: " + key)
    model = parent.model
    state = payload.get("model_state_dict", {})
    require(set(state) == set(model.state_dict()), "model inventory differs")
    for key, ref in model.state_dict().items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.shape == ref.shape
            and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "invalid model tensor: " + key)
    model.load_state_dict(state, strict=True)
    require(sum(v.numel() for v in model.parameters()) == PARAMETERS[architecture], "fixed model size changed")
    return ActionPredictor(model, norm, architecture=architecture, checkpoint_path=path,
        checkpoint_sha256=digest, device=device)


def load_selected(training_dir, condition, *, device, expected_channels):
    root = Path(training_dir).resolve(strict=True)
    plan = verify_plan(root / "PLAN.json")
    require(condition in plan["conditions"], "condition not preregistered")
    final, selected = read(root / "FINAL-RESULT.json"), read(root / "selected.json")
    require(final.get("status") == FINAL_STATUS and final.get("complete_runs") == len(plan["conditions"])
        and final.get("permissions") == PERMISSIONS and final.get("robot_quality_validated") is False,
        "complete fixed-final training required")
    require(selected.get("schema") == SELECTION_SCHEMA
        and selected.get("selection_role") == "fixed_final_update500_needs_complete_DEV"
        and selected.get("permissions") == PERMISSIONS and selected.get("robot_quality_validated") is False,
        "selection must not claim robot quality")
    row = selected["conditions"][condition]
    record = read(Path(row["checkpoint"]).parent / "RESULT.json")
    require(record.get("status") == "PASS_FIXED_FINAL_CELL"
        and record.get("checkpoint_sha256") == row["checkpoint_sha256"]
        and record.get("full_old_and_new_CAL_CPU_loader_parity") is True
        and record.get("normalization_exact_parent") is True, "training cell incomplete")
    return load_checkpoint(row["checkpoint"], architecture=CONDITIONS[condition],
        expected_condition=condition, device=device, expected_channels=expected_channels,
        expected_sha256=row["checkpoint_sha256"])

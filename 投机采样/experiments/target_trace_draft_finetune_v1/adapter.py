"""Strict opt-in adapter for 2000+500-update two-stage real-data drafts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

_EXP = str(Path(__file__).resolve().parent.parent)
if _EXP not in sys.path:
    sys.path.append(_EXP)
from target_trace_draft_train_v1.adapter import (ActionPredictor, ARCHITECTURES,
    NORMALIZATION, model_source, sha)

SCHEMA = "target-trace-draft-two-stage/v1"
FINAL_STATUS = "PASS_SIX_TWO_STAGE_FINETUNING_RUNS"


def load_checkpoint(path, *, architecture, device, expected_channels, expected_sha256=None):
    channels = tuple(range(expected_channels)) if isinstance(expected_channels, int) else tuple(expected_channels)
    if channels != tuple(range(7)) or architecture not in ARCHITECTURES:
        raise ValueError("original compact action geometry and known architecture required")
    path = Path(path).resolve(strict=True)
    digest = sha(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("approved two-stage checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    source = model_source()
    if (payload.get("schema") != SCHEMA or payload.get("architecture") != architecture
            or payload.get("specification") != source.SPECS[architecture]
            or metadata.get("training_source") != "target_pretrain_then_mixed_historical_same_query_finetune"
            or metadata.get("seed") not in (20260922, 20260923, 20260924)
            or metadata.get("pretraining_updates") != 2000 or metadata.get("finetune_updates") != 500
            or metadata.get("total_updates") != 2500 or metadata.get("batch") != 512
            or metadata.get("target_half") != 256 or metadata.get("same_query_half") != 256
            or metadata.get("learning_rate") != 3e-5 or metadata.get("weight_decay") != .01
            or metadata.get("normalization_refit") is not False or metadata.get("deployed") is not False):
        raise ValueError("strict two-stage checkpoint protocol differs")
    parent = Path(metadata["parent_checkpoint"]).resolve(strict=True)
    parent_record = json.loads((parent.parent / "RESULT.json").read_text())
    parent_digest = sha(parent)
    if (parent_digest != metadata.get("parent_checkpoint_sha256")
            or parent_digest != parent_record.get("checkpoint_sha256")
            or parent_record.get("architecture") != architecture
            or parent_record.get("seed") != metadata["seed"]):
        raise ValueError("stage1 checkpoint lineage differs")
    parent_payload = torch.load(parent, map_location="cpu", weights_only=True)
    if (parent_payload.get("schema") != "target-trace-draft-pretraining/v1"
            or parent_payload.get("metadata", {}).get("updates") != 2000
            or parent_payload.get("architecture") != architecture):
        raise ValueError("stage1 source is not approved target-only pretraining")
    normalization = payload.get("normalization", {})
    if set(normalization) != set(NORMALIZATION):
        raise ValueError("two-stage normalization inventory differs")
    for key, shape in NORMALIZATION.items():
        value = normalization[key]
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())
                or key.endswith("std") and not bool((value > 0).all())
                or not torch.equal(value, parent_payload["normalization"][key])):
            raise ValueError("stage1 normalization changed: " + key)
    with torch.random.fork_rng(devices=[]):
        model = source.make_model(architecture).float()
    state = payload.get("model_state_dict", {})
    if set(state) != set(model.state_dict()):
        raise ValueError("model state inventory differs")
    for key, reference in model.state_dict().items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError("two-stage model tensor invalid: " + key)
    model.load_state_dict(state, strict=True)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("CPU or CUDA required")
    return ActionPredictor(model, normalization, architecture=architecture,
        checkpoint_path=path, checkpoint_sha256=digest, device=device)


def load_selected(training_dir, architecture, *, device, expected_channels):
    root = Path(training_dir).resolve(strict=True)
    if (root / "FAILED.json").exists():
        raise ValueError("failed two-stage job cannot supply a selected checkpoint")
    final = json.loads((root / "FINAL-RESULT.json").read_text())
    selected = json.loads((root / "selected.json").read_text())
    if (final.get("status") != FINAL_STATUS or final.get("complete_runs") != 6
            or selected.get("selection_role") != "complete_closed_loop_DEV_candidate_only"):
        raise ValueError("six complete two-stage training runs required")
    for arch in ARCHITECTURES:
        for seed in (20260922, 20260923, 20260924):
            cell = root / f"{arch}-seed{seed}"
            if (cell / "FAILED.json").exists() or not (cell / "RESULT.json").is_file():
                raise ValueError("two-stage cell failed or missing result")
            record = json.loads((cell / "RESULT.json").read_text())
            if (record.get("architecture") != arch or record.get("seed") != seed
                    or record.get("pretraining_updates") != 2000
                    or record.get("finetune_updates") != 500 or record.get("total_updates") != 2500
                    or record.get("full_CAL_new_adapter_bitwise_exact") is not True
                    or record.get("normalization_exact_parent") is not True
                    or not (cell / "checkpoint.pt").is_file()):
                raise ValueError("two-stage complete cell provenance differs")
    row = selected["architectures"][architecture]
    checkpoint = Path(row["checkpoint"]).resolve(strict=True)
    if checkpoint != root / f"{architecture}-seed{row['seed']}" / "checkpoint.pt":
        raise ValueError("selected two-stage checkpoint path differs")
    predictor = load_checkpoint(checkpoint, architecture=architecture, device=device,
        expected_channels=expected_channels, expected_sha256=row["checkpoint_sha256"])
    predictor.selection_role = selected["selection_role"]
    return predictor

"""Opt-in predictor for real target-trace pretraining, with no online install."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

OWN = Path(__file__).resolve().parent
MODELS = OWN.parent / "draft_architecture_v1/models.py"
SCHEMA = "target-trace-draft-pretraining/v1"
ARCHITECTURES = {"transformer64": 53070, "mlp96": 52928}
NORMALIZATION = {"feature_mean": (227,), "feature_std": (227,),
                 "residual_mean": (112,), "residual_std": (112,)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_source():
    spec = importlib.util.spec_from_file_location("_target_trace_draft_architecture", MODELS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActionPredictor:
    def __init__(self, model, normalization, *, architecture, checkpoint_path,
                 checkpoint_sha256, device):
        self.model = model.eval().to(device=device, dtype=torch.float32)
        self.device = next(self.model.parameters()).device
        for key in NORMALIZATION:
            setattr(self, key, normalization[key].detach().clone().to(self.device))
        self.architecture = architecture
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_sha256 = checkpoint_sha256
        self.output_dim, self.heads = 112, 2

    @torch.no_grad()
    def predict(self, current_state, previous_direction, sigma, next_sigma, timestep):
        if (not isinstance(current_state, torch.Tensor)
                or not isinstance(previous_direction, torch.Tensor)
                or current_state.shape != previous_direction.shape or current_state.numel() != 112):
            raise ValueError("matching compact 112-element actions required")
        if current_state.device != self.device or previous_direction.device != self.device:
            raise ValueError("predictor and action devices differ")
        if (not torch.is_floating_point(current_state) or not torch.is_floating_point(previous_direction)
                or not all(math.isfinite(float(x)) for x in (sigma, next_sigma, timestep))
                or not bool(torch.isfinite(current_state).all())
                or not bool(torch.isfinite(previous_direction).all())):
            raise ValueError("floating, finite actions and times required")
        state = current_state.detach().reshape(1, 112).float()
        previous = previous_direction.detach().reshape(1, 112).float()
        times = torch.tensor([[sigma, next_sigma, timestep / 1000.]],
                             device=self.device, dtype=torch.float32)
        features = torch.cat((state, previous, times), 1)
        raw = self.model((features - self.feature_mean) / self.feature_std)
        value = previous[:, None] + raw * self.residual_std[None, None] + self.residual_mean[None, None]
        value = value.reshape(2, *current_state.shape).to(current_state.dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite draft prediction")
        return value


def load_checkpoint(path, *, architecture, device, expected_channels,
                    expected_sha256=None):
    channels = tuple(range(expected_channels)) if isinstance(expected_channels, int) else tuple(expected_channels)
    if channels != tuple(range(7)) or architecture not in ARCHITECTURES:
        raise ValueError("original seven ordered action channels and known architecture required")
    path = Path(path).resolve(strict=True)
    actual_sha256 = sha(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("approved checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source = model_source()
    metadata = payload.get("metadata", {})
    if (payload.get("schema") != SCHEMA or payload.get("architecture") != architecture
            or payload.get("specification") != source.SPECS[architecture]
            or metadata.get("training_source") != "historical_target_only_actual_queries"
            or metadata.get("seed") not in (20260922, 20260923, 20260924)
            or metadata.get("updates") != 2000 or metadata.get("batch") != 512
            or metadata.get("fit_episodes") != 174 or metadata.get("cal_episodes") != 20
            or metadata.get("deployed") is not False):
        raise ValueError("pretraining checkpoint schema or fixed protocol differs")
    normalization = payload.get("normalization", {})
    if set(normalization) != set(NORMALIZATION):
        raise ValueError("normalization inventory differs")
    for key, shape in NORMALIZATION.items():
        value = normalization[key]
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                or key.endswith("std") and not bool((value > 0).all())):
            raise ValueError("normalization tensor differs: " + key)
    with torch.random.fork_rng(devices=[]):
        model = source.make_model(architecture).float()
    state = payload.get("model_state_dict", {})
    if set(state) != set(model.state_dict()):
        raise ValueError("model tensor inventory differs")
    for key, reference in model.state_dict().items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError("model tensor differs: " + key)
    model.load_state_dict(state, strict=True)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("CPU or CUDA predictor required")
    return ActionPredictor(model, normalization, architecture=architecture,
        checkpoint_path=path, checkpoint_sha256=actual_sha256, device=device)


def load_selected(training_dir, architecture, *, device, expected_channels):
    root = Path(training_dir).resolve(strict=True)
    result = json.loads((root / "FINAL-RESULT.json").read_text())
    selected = json.loads((root / "selected.json").read_text())
    if (result.get("status") != "PASS_SIX_TARGET_TRACE_PRETRAINING_RUNS"
            or result.get("complete_runs") != 6
            or selected.get("selection_role") != "complete_closed_loop_DEV_candidate_only"):
        raise ValueError("all six complete training runs required")
    row = selected["architectures"][architecture]
    path = Path(row["checkpoint"]).resolve(strict=True)
    expected = root / f"{architecture}-seed{row['seed']}" / "checkpoint.pt"
    if path != expected:
        raise ValueError("selected checkpoint escaped canonical seed folder")
    predictor = load_checkpoint(path, architecture=architecture, device=device,
        expected_channels=expected_channels, expected_sha256=row["checkpoint_sha256"])
    predictor.selection_role = selected["selection_role"]
    return predictor


def cpu_gpu_numeric_parity(training_dir, architecture, *, device, expected_channels):
    if torch.device(device).type != "cuda":
        raise ValueError("online parity requires the designated CUDA device")
    cpu = load_selected(training_dir, architecture, device="cpu", expected_channels=expected_channels)
    gpu = load_selected(training_dir, architecture, device=device, expected_channels=expected_channels)
    records = []
    for name, state, previous, sigma, next_sigma, timestep in (
            ("ramp", torch.linspace(-.25, .25, 112).reshape(7, 4, 4),
             torch.linspace(.125, -.125, 112).reshape(7, 4, 4), .65, .62, 650.),
            ("zero", torch.zeros(7, 4, 4), torch.zeros(7, 4, 4), .2, .18, 200.)):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.bfloat16, .015625)):
            x, p = state.to(dtype), previous.to(dtype)
            expected = cpu.predict(x, p, sigma, next_sigma, timestep)
            actual = gpu.predict(x.to(device), p.to(device), sigma, next_sigma, timestep).cpu()
            error = float((expected.float() - actual.float()).abs().max())
            if actual.shape != expected.shape or actual.dtype != dtype or error > tolerance:
                raise ValueError(f"CPU/GPU draft parity differs: {name}/{dtype}/{error}")
            records.append(dict(case=name, dtype=str(dtype), shape=list(actual.shape),
                                max_abs=error, tolerance=tolerance, target_forwards=0))
    return gpu, records

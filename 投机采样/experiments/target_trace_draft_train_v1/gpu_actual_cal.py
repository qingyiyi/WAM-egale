"""Full real-CAL predictor parity; import is inert, GPU execution explicit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

try:
    from .adapter import load_selected, model_source
except ImportError:
    from adapter import load_selected, model_source


def actual_cal_gpu_parity(training_dir, architecture, *, device,
                          expected_channels=tuple(range(7)), out_path=None):
    """Check every real CAL query against original decode_batch on one GPU.

    Strict gate: adapter vs original mathematical interface on the *same* GPU.
    CPU/GPU differences and local acceptance flips are descriptive diagnostics.
    Neither this gate nor its data are a robot success-rate evaluation.
    No target model, simulator, synthetic query, or target forward is used.
    """
    start = time.monotonic()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("full actual CAL GPU parity requires CUDA explicitly")
    root = Path(training_dir).resolve(strict=True)
    plan = json.loads((root / "PLAN.json").read_text())
    selected = json.loads((root / "selected.json").read_text())["architectures"][architecture]
    pack = torch.load(plan["data"], map_location="cpu", weights_only=False)
    indices = torch.as_tensor(pack["views"]["cal20"], dtype=torch.long)
    columns = pack["columns"]
    cal = {name: columns[name][indices].contiguous()
           for name in ("state", "previous", "target", "scalars", "delta", "task", "episode")}
    if (len(indices) != plan["cal_rows"] or len(torch.unique(cal["episode"])) != 20
            or set(cal["task"].tolist()) != set(range(10))):
        raise ValueError("full real CAL20 identity/count differs")
    # A running server already owns this context. Initializing it in a standalone
    # explicit invocation is also deliberate; there is no GPU call at import.
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    predictor = load_selected(root, architecture, device=device, expected_channels=expected_channels)
    payload = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
    source = model_source()
    with torch.random.fork_rng(devices=[]):
        reference = source.Predictor(architecture, payload["normalization"])
    reference.model.load_state_dict(payload["model_state_dict"], strict=True)
    reference.model.to(device=device, dtype=torch.float32).eval()
    for name in ("feature_mean", "feature_std", "residual_mean", "residual_std"):
        setattr(reference, name, getattr(reference, name).to(device=device))
    n = len(indices)
    outputs = torch.empty((n, 2, 112), device=device, dtype=torch.bfloat16)
    references = torch.empty_like(outputs)
    # Preserve both interfaces' real batch-one path and BF16 query rounding.
    # Store the compact outputs and compare once, avoiding an extra sync gate
    # and extra CPU transfer at each of the 20,237 queries.
    with torch.inference_mode():
        for i in range(n):
            state = cal["state"][i].reshape(7, 4, 4).bfloat16().to(device)
            previous = cal["previous"][i].reshape(7, 4, 4).bfloat16().to(device)
            sigma, next_sigma, timestep = cal["scalars"][i].tolist()
            timestep *= 1000.
            outputs[i] = predictor.predict(state, previous, sigma, next_sigma, timestep).reshape(2, 112)
            data = dict(state=state.reshape(1, 112).float(),
                previous=previous.reshape(1, 112).float(),
                scalars=torch.tensor([[sigma, next_sigma, timestep / 1000.]], device=device, dtype=torch.float32))
            references[i] = reference.decode_batch(data).reshape(2, 112).bfloat16()
    output = outputs.cpu().float()
    reference_output = references.cpu().float()
    cpu_saved = torch.load(Path(selected["checkpoint"]).parent / "cal_predictions.pt",
                           map_location="cpu", weights_only=True)
    if cpu_saved.shape != output.shape:
        raise ValueError("complete saved CPU CAL predictions differ in shape")
    same_device_diff = (output - reference_output).abs()
    cpu_gpu_diff = (output - cpu_saved).abs()
    delta = cal["delta"][:, None]
    target = cal["target"][:, None]
    gpu_error = ((output - target).square().sum(2).sqrt() * delta).min(1).values
    cpu_error = ((cpu_saved - target).square().sum(2).sqrt() * delta).min(1).values
    tau = plan["threshold"]
    gpu_accept, cpu_accept = gpu_error <= tau, cpu_error <= tau
    per_task = {}
    for task in range(10):
        mask = cal["task"] == task
        per_task[str(task)] = dict(rows=int(mask.sum()),
            cpu_acceptance=float(cpu_accept[mask].float().mean()),
            gpu_acceptance=float(gpu_accept[mask].float().mean()),
            cpu_accept_gpu_reject=int((cpu_accept[mask] & ~gpu_accept[mask]).sum()),
            cpu_reject_gpu_accept=int((~cpu_accept[mask] & gpu_accept[mask]).sum()))
    finite = bool(torch.isfinite(output).all() and torch.isfinite(reference_output).all())
    same_device_exact = torch.equal(output, reference_output)
    cpu_rng_same = torch.equal(cpu_rng, torch.get_rng_state())
    cuda_rng_same = torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    passed = finite and same_device_exact and cpu_rng_same and cuda_rng_same
    report = dict(status="PASS_FULL_REAL_CAL_SAME_DEVICE_ADAPTER_PARITY" if passed
                  else "FAIL_FULL_REAL_CAL_SAME_DEVICE_ADAPTER_PARITY",
        architecture=architecture, checkpoint=selected["checkpoint"],
        checkpoint_sha256=selected["checkpoint_sha256"], CAL_episodes=20, rows=n,
        complete_CAL=True, synthetic_queries=0, target_forwards=0, LIBERO_episodes=0,
        same_device_reference="original models.Predictor.decode_batch(batch1), same weights/norm/BF16 query",
        same_device_bitwise_exact=same_device_exact, same_device_max_abs=float(same_device_diff.max()),
        finite=finite, CPU_rng_unchanged=cpu_rng_same, CUDA_rng_unchanged=cuda_rng_same,
        cpu_gpu_diagnostic=dict(bitwise_exact=torch.equal(output, cpu_saved),
            max_abs=float(cpu_gpu_diff.max()),
            mean_abs=float(cpu_gpu_diff.mean()),
            equal_query_count=int((output == cpu_saved).reshape(n, -1).all(1).sum()),
            cpu_accept_gpu_reject=int((cpu_accept & ~gpu_accept).sum()),
            cpu_reject_gpu_accept=int((~cpu_accept & gpu_accept).sum()),
            not_a_robot_quality_gate=True, tasks=per_task),
        elapsed_seconds=time.monotonic() - start,
        robot_success_validated=False, recursive_tree_quality_validated=False)
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    del outputs, references, reference
    if not passed:
        raise ValueError("full real CAL adapter parity failed; see recorded report")
    return predictor, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=("transformer64", "mlp96"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _, result = actual_cal_gpu_parity(args.training_dir, args.architecture,
                                     device=args.device, out_path=args.out)
    print(json.dumps({k: result[k] for k in ("status", "rows", "elapsed_seconds")}), flush=True)

"""Explicit same-GPU, batch-one, BF16 adapter parity on both real CAL sets.

Importing this module does not allocate a CUDA context. CPU/GPU agreement is a
descriptive diagnostic; same-GPU production adapter/original decode agreement
is the actual interface gate. Neither is a robot quality certificate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch

OWN = Path(__file__).resolve().parent
if str(OWN.parent) not in sys.path:
    sys.path.append(str(OWN.parent))
from tf_effect_train_v1.adapter import (COMMIT, CONDITIONS, PERMISSIONS, SEED, TAU,
    load_selected, load_checkpoint, model_source, sha, read, stat, require,
    verify_plan, validate_new_columns)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def prepare(job, training_dir, condition, *, wall_cap_seconds=900):
    """Register a later GPU call, with no model forward and no CUDA call here."""
    job, training_dir = Path(job).resolve(), Path(training_dir).resolve(strict=True)
    require(not job.exists() and 60 <= wall_cap_seconds <= 1800, "new GPU CAL job and finite cap required")
    job.mkdir(parents=True, exist_ok=False)
    registration = dict(schema="tf-effect-real-CAL-registration/v1", source_commit=COMMIT,
        goal="old CAL20237 + new CAL10 same-device adapter/decode batch1 parity",
        training_dir=str(training_dir), condition=condition, cpu_threads=2, GPU="one explicitly designated CUDA device",
        birth_epoch=time.time(), deadline_epoch=time.time()+wall_cap_seconds,
        wall_cap_seconds=wall_cap_seconds, storage_cap_bytes=32*1024*1024,
        terminal_status="REGISTERED_NOT_PREPARED", cleanup_plan="single caller-owned process; release predictor tensors; no spawned service",
        evidence_path=str(job), permissions=PERMISSIONS)
    write(job / "PREREGISTRATION.json", registration)
    try:
        require(not torch.cuda.is_initialized(), "standalone CPU preparation must not own CUDA")
        require(condition in CONDITIONS, "unknown fixed condition")
        training_plan = verify_plan(training_dir / "PLAN.json")
        predictor = load_selected(training_dir, condition, device="cpu", expected_channels=range(7))
        plan = dict(schema="tf-effect-real-CAL-plan/v1", source_commit=COMMIT,
            training_dir=str(training_dir), condition=condition, architecture=CONDITIONS[condition],
            checkpoint=predictor.checkpoint_path, checkpoint_sha256=predictor.checkpoint_sha256,
            training_plan_sha256=sha(training_dir / "PLAN.json"), data=training_plan["data"],
            new_columns=training_plan["new_columns"], new_manifest_sha256=training_plan["new_manifest_sha256"],
            old_CAL_rows=20237, new_CAL_rows=training_plan["new_CAL_rows"],
            input_stat={str(Path(training_plan[key])): stat(training_plan[key]) for key in ("data", "new_columns")},
            source_sha256={str(OWN / name): sha(OWN / name) for name in ("adapter.py", "gpu_actual_cal.py")},
            birth_epoch=registration["birth_epoch"], deadline_epoch=registration["deadline_epoch"],
            wall_cap_seconds=wall_cap_seconds, storage_cap_bytes=32*1024*1024,
            permissions=PERMISSIONS, robot_quality_validated=False)
        write(job / "PLAN.json", plan)
        write(job / "PREPARED.json", dict(status="PREPARED_NO_FORWARD_NO_CUDA", plan_sha256=sha(job / "PLAN.json"),
            CPU_interface_only=True, CUDA_initialized=False, permissions=PERMISSIONS))
        return plan
    except BaseException as exc:
        write(job / "FAILED.json", dict(status="FAILED_CPU_CAL_PREPARATION", error=repr(exc), permissions=PERMISSIONS))
        raise


def evaluate(predictor, reference, columns, cpu_saved, device, *, deadline):
    n = len(columns["state"])
    require(cpu_saved.shape == (n, 2, 112), "complete CPU CAL predictions differ in shape")
    outputs = torch.empty((n, 2, 112), device=device, dtype=torch.bfloat16)
    references = torch.empty_like(outputs)
    with torch.inference_mode():
        for index in range(n):
            if index % 128 == 0:
                require(time.monotonic() < deadline, "GPU CAL wall cap expired")
            state = columns["state"][index].reshape(7, 4, 4).bfloat16().to(device)
            previous = columns["previous"][index].reshape(7, 4, 4).bfloat16().to(device)
            sigma, next_sigma, normalized_time = columns["scalars"][index].tolist()
            timestep = normalized_time * 1000.
            outputs[index] = predictor.predict(state, previous, sigma, next_sigma, timestep).reshape(2, 112)
            batch = dict(state=state.reshape(1, 112).float(), previous=previous.reshape(1, 112).float(),
                scalars=torch.tensor([[sigma, next_sigma, timestep / 1000.]], device=device, dtype=torch.float32))
            references[index] = reference.decode_batch(batch).reshape(2, 112).bfloat16()
    output, expected = outputs.cpu().float(), references.cpu().float()
    finite = bool(torch.isfinite(output).all() and torch.isfinite(expected).all() and torch.isfinite(cpu_saved).all())
    same_device_exact = torch.equal(output, expected)
    difference = (output - expected).abs()
    cpu_difference = (output - cpu_saved).abs()
    target = columns["target"][:, None].float()
    delta = columns["delta"][:, None].float()
    # Convert both operands before subtraction; never subtract BF16 directions.
    gpu_errors = ((output - target).square().sum(2).sqrt() * delta).min(1).values
    cpu_errors = ((cpu_saved.float() - target).square().sum(2).sqrt() * delta).min(1).values
    gpu_accept, cpu_accept = gpu_errors <= TAU, cpu_errors <= TAU
    groups = {}
    masks = {"all": torch.ones(n, dtype=torch.bool)}
    masks.update({f"task{task}": columns["task"] == task for task in range(10)})
    sigma = columns["scalars"][:, 0]
    masks.update({"sigma/high": sigma >= 2/3, "sigma/mid": (sigma >= 1/3) & (sigma < 2/3), "sigma/low": sigma < 1/3})
    if "node" in columns:
        masks.update(root=columns["node"] == 0, reached_parent=columns["node"] != 0)
    for name, mask in masks.items():
        if bool(mask.any()):
            groups[name] = dict(rows=int(mask.sum()), gpu_acceptance=float(gpu_accept[mask].float().mean()),
                cpu_acceptance=float(cpu_accept[mask].float().mean()),
                cpu_accept_gpu_reject=int((cpu_accept[mask] & ~gpu_accept[mask]).sum()),
                cpu_reject_gpu_accept=int((~cpu_accept[mask] & gpu_accept[mask]).sum()))
    return dict(rows=n, complete_CAL=True, finite=finite, same_device_bitwise_exact=same_device_exact,
        same_device_max_abs=float(difference.max()),
        cpu_gpu_diagnostic=dict(bitwise_exact=torch.equal(output, cpu_saved), max_abs=float(cpu_difference.max()),
            equal_query_count=int((output == cpu_saved).reshape(n, -1).all(1).sum()), groups=groups,
            not_a_robot_quality_gate=True))


def actual_cal_gpu_parity(job, *, device, out_path=None):
    """Run a separately prepared job, also usable inside an owned VA server.

    The caller must supervise/clean its own process. The finite deadline includes
    CPU preparation, waiting, loading and setup; no target-model call is made.
    """
    start = time.monotonic()
    job = Path(job).resolve(strict=True)
    require(not (job / "STARTED.json").exists() and not (job / "FAILED.json").exists()
        and not (job / "STOP_REQUESTED").exists(), "fresh prepared GPU CAL job required")
    plan = read(job / "PLAN.json")
    prepared = read(job / "PREPARED.json")
    require(prepared.get("plan_sha256") == sha(job / "PLAN.json")
        and plan.get("schema") == "tf-effect-real-CAL-plan/v1" and plan.get("source_commit") == COMMIT
        and plan.get("permissions") == PERMISSIONS, "prepared GPU CAL contract changed")
    condition, architecture = plan["condition"], plan["architecture"]
    require(CONDITIONS.get(condition) == architecture, "condition/architecture mismatch")
    device = torch.device(device)
    require(device.type == "cuda", "same-device CAL requires explicit CUDA")
    registration = read(job / "PREREGISTRATION.json")
    require(plan["birth_epoch"] == registration["birth_epoch"]
        and plan["deadline_epoch"] == registration["deadline_epoch"], "CAL cumulative deadline changed")
    remaining = plan["deadline_epoch"] - time.time()
    if remaining <= 0:
        expired = dict(status="TIMEOUT_BEFORE_GPU_CAL", error="registered wall cap expired during preparation/wait",
            elapsed_since_registration=time.time()-registration["birth_epoch"], target_forwards=0, LIBERO_episodes=0,
            permissions=PERMISSIONS)
        write(job / "FAILED.json", expired); write(job / "status.json", expired)
        raise TimeoutError("GPU CAL registered wall cap expired; preparation/wait are charged")
    deadline = start + remaining
    write(job / "STARTED.json", dict(status="RUNNING_REAL_GPU_CAL", pid=os.getpid(), condition=condition,
        device=str(device), deadline_monotonic=deadline, target_forwards=0, LIBERO_episodes=0))
    try:
        for path, digest in plan["source_sha256"].items():
            require(sha(path) == digest, "GPU CAL source changed")
        for path, expected in plan["input_stat"].items():
            require(stat(path) == expected, "GPU CAL input stat changed")
        training_root = Path(plan["training_dir"])
        training_plan = verify_plan(training_root / "PLAN.json")
        require(sha(training_root / "PLAN.json") == plan["training_plan_sha256"], "frozen training plan changed")
        pack = torch.load(plan["data"], map_location="cpu", weights_only=False)
        indices = torch.as_tensor(pack["views"]["cal20"], dtype=torch.long)
        old_cal = {key: pack["columns"][key][indices].contiguous()
            for key in ("state", "previous", "target", "scalars", "delta", "task", "episode")}
        require(len(indices) == 20237 and len(torch.unique(old_cal["episode"])) == 20
            and set(old_cal["task"].tolist()) == set(range(10)), "full historical CAL identity differs")
        _, new_cal, _, _ = validate_new_columns(plan["new_columns"], expected_manifest_sha256=plan["new_manifest_sha256"])
        require(len(new_cal["state"]) == plan["new_CAL_rows"] == training_plan["new_CAL_rows"], "new CAL row identity differs")
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        predictor = load_selected(training_root, condition, device=device, expected_channels=range(7))
        require(predictor.checkpoint_sha256 == plan["checkpoint_sha256"], "GPU CAL checkpoint changed")
        payload = torch.load(plan["checkpoint"], map_location="cpu", weights_only=True)
        source = model_source()
        with torch.random.fork_rng(devices=[]):
            reference = source.Predictor(architecture, payload["normalization"])
        reference.model.load_state_dict(payload["model_state_dict"], strict=True)
        reference.model.to(device=device, dtype=torch.float32).eval()
        for name in ("feature_mean", "feature_std", "residual_mean", "residual_std"):
            setattr(reference, name, getattr(reference, name).to(device=device))
        folder = Path(plan["checkpoint"]).parent
        sets = {}
        for name, columns, saved_name in (("old_CAL", old_cal, "cal_predictions.pt"),
                ("new_CAL", new_cal, "new_cal_predictions.pt")):
            saved = torch.load(folder / saved_name, map_location="cpu", weights_only=True)
            sets[name] = evaluate(predictor, reference, columns, saved, device, deadline=deadline)
        cpu_rng_same = torch.equal(cpu_rng, torch.get_rng_state())
        cuda_rng_same = torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
        passed = all(row["finite"] and row["same_device_bitwise_exact"] for row in sets.values()) and cpu_rng_same and cuda_rng_same
        result = dict(status="PASS_FULL_OLD_AND_NEW_REAL_CAL_SAME_DEVICE_PARITY" if passed
            else "FAIL_FULL_OLD_AND_NEW_REAL_CAL_SAME_DEVICE_PARITY", source_commit=COMMIT,
            condition=condition, architecture=architecture, seed=SEED, checkpoint=plan["checkpoint"],
            checkpoint_sha256=plan["checkpoint_sha256"], device=str(device), sets=sets,
            full_old_and_new_CAL=True, synthetic_queries=0, target_forwards=0, LIBERO_episodes=0,
            same_device_reference="unchanged models.Predictor.decode_batch; real BF16 rounded queries, batch1, identical weights/norm",
            CPU_rng_unchanged=cpu_rng_same, CUDA_rng_unchanged=cuda_rng_same,
            elapsed_seconds=time.monotonic()-start, robot_quality_validated=False, permissions=PERMISSIONS)
        require(time.monotonic() < deadline, "GPU CAL deadline exceeded after final synchronization")
        write(job / "RESULT.json", result)
        if out_path is not None:
            out_path = Path(out_path)
            require(out_path.resolve().parent == job, "GPU CAL report must remain inside registered job")
            require(not out_path.exists(), "do not overwrite an existing GPU CAL report")
            write(out_path, result)
        del reference
        if not passed:
            raise ValueError("same-device real-CAL adapter parity failed; original RESULT preserved")
        require(sum(path.stat().st_size for path in job.rglob("*") if path.is_file()) <= plan["storage_cap_bytes"], "GPU CAL storage cap")
        write(job / "status.json", result)
        return predictor, result
    except BaseException as exc:
        failed = dict(status="FAILED_REAL_GPU_CAL", error=repr(exc), elapsed_seconds=time.monotonic()-start,
            target_forwards=0, LIBERO_episodes=0, permissions=PERMISSIONS)
        write(job / "FAILED.json", failed); write(job / "status.json", failed)
        raise


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true"); mode.add_argument("--run", action="store_true")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--wall-cap-seconds", type=int, default=900)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        if args.training_dir is None or args.condition is None:
            parser.error("--prepare requires --training-dir and --condition")
        result = prepare(args.job, args.training_dir, args.condition, wall_cap_seconds=args.wall_cap_seconds)
        print(json.dumps(dict(status="PREPARED_NO_FORWARD_NO_CUDA", condition=result["condition"])), flush=True)
    else:
        if args.training_dir or args.condition:
            parser.error("--run uses the frozen GPU CAL PLAN")
        # Standalone invocation has an OS deadline as well as the loop deadline.
        plan = read(args.job / "PLAN.json")
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("GPU CAL process wall cap")))
        remaining = plan["deadline_epoch"] - time.time()
        if remaining <= 0:
            # Let the core write its immutable timeout receipt, with no CUDA call.
            actual_cal_gpu_parity(args.job, device=args.device)
            raise AssertionError("expired CAL unexpectedly returned")
        signal.alarm(math.ceil(remaining))
        try:
            _, result = actual_cal_gpu_parity(args.job, device=args.device)
            print(json.dumps(dict(status=result["status"], sets={name: row["rows"] for name, row in result["sets"].items()},
                elapsed_seconds=result["elapsed_seconds"])), flush=True)
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    main()

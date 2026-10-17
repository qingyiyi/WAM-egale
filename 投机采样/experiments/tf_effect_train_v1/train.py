"""Explicit prepare/run for the one finite TF-cont / TF-on comparison.

No execution at import. Run only after an evidence-backed trigger and a complete
30-episode collection. This module never starts a GPU or simulator process.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
if str(EXP) not in sys.path:
    sys.path.append(str(EXP))
from tf_effect_train_v1.adapter import (SCHEMA, PLAN_SCHEMA, SELECTION_SCHEMA,
    FINAL_STATUS, COMMIT, SEED, TAU, CONDITIONS, PARENT_SHAS, GENERATOR_SHA, PERMISSIONS,
    load_parent, load_checkpoint, model_source, sha, read, require, stat,
    digest_json, validate_new_columns)

PARENT_ROOT = ROOT / "投机采样/results/runtime/p0-target-trace-draft-finetune-v1-20261015-r1"
OLD_CONT = ROOT / "投机采样/results/runtime/p0-tf-usable-train-v1-20261016-r1/TF_CONT-seed20260923/checkpoint.pt"


def write(path, value):
    import json
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def frozen_functions():
    # Strictly reuse the reviewed extractor and its original source, not a copy
    # of the sampler, objective, decode, errors, or metric implementation.
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only process must hide CUDA")
    from tf_usable_train_v1.train import math_functions
    return math_functions()


def remaining(registration):
    value = registration["deadline_epoch"] - time.time()
    require(value > 0, "entire registered job wall cap expired; retries do not reset it")
    return value


def arm_deadline(registration):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("registered CPU job wall cap")))
    signal.alarm(max(1, math.ceil(remaining(registration))))


def check_runtime():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
        "CPU-only training requires no CUDA context")
    require(len(os.sched_getaffinity(0)) <= 2, "pin this process to at most two allowed CPU cores")
    torch.set_num_threads(2)


def check_storage(job, cap):
    size = sum(path.stat().st_size for path in job.rglob("*") if path.is_file())
    require(size <= cap, "registered training storage cap exceeded")
    return size


def trigger_certificate(path, conditions):
    certificate = read(path)
    require(certificate.get("schema") == "tf-effect-training-trigger/v1"
        and certificate.get("source_commit") == COMMIT and certificate.get("training_trigger") is True
        and isinstance(certificate.get("mechanism"), str) and certificate["mechanism"].strip()
        and certificate.get("observed_support") and certificate.get("inference_limits")
        and set(conditions) <= set(certificate.get("allowed_conditions", [])),
        "positive, bounded mechanism trigger required before training preparation")
    evidence = certificate.get("evidence", [])
    require(bool(evidence), "trigger has no traceable evidence")
    for entry in evidence:
        require(sha(entry["path"]) == entry["sha256"], "trigger evidence changed")
    return certificate


def collection_certificate(path, columns, manifest_sha256):
    certificate = read(path)
    require(certificate.get("schema") == "tf-effect-data-audit/v1"
        and certificate.get("status") == "PASS_FULL_30_EPISODE_REACHED_NODE_AUDIT"
        and certificate.get("source_commit") == COMMIT
        and Path(certificate["input_pack"]).resolve(strict=True) == columns
        and certificate.get("input_pack_stat") == stat(columns)
        and certificate.get("manifest_sha256") == manifest_sha256
        and certificate.get("all_rows_reached") is True
        and certificate.get("all_episodes_terminal") is True
        and certificate.get("fit_cal_disjoint") is True
        and certificate.get("no_development_or_test_rows") is True
        and certificate.get("collector_equivalence_passed") is True
        and certificate.get("extra_teacher_forwards") == 0,
        "full collector/equivalence/reachability certificate required")
    require(bool(certificate.get("evidence")), "collector certificate has no raw audit/equivalence evidence")
    for entry in certificate["evidence"]:
        require(sha(entry["path"]) == entry["sha256"], "collector evidence changed")
    return certificate


def historical(plan, funcs):
    require(read(plan["data_audit"])["status"] == "PASS_COMPLETE_NUMERIC_AUDIT", "target source audit incomplete")
    require(read(plan["same_query_audit"])["status"] == "PASS_COMPLETE_SAME_QUERY_AUX_NUMERIC_AUDIT", "old mixed audit incomplete")
    pack = torch.load(plan["data"], map_location="cpu", weights_only=False)
    keys = ("state", "previous", "target", "scalars", "delta", "task", "episode")
    fit = {k: pack["columns"][k][pack["views"]["fit174"]].contiguous() for k in keys}
    cal = {k: pack["columns"][k][pack["views"]["cal20"]].contiguous() for k in keys}
    same_query = funcs["same_query"](torch.load(plan["same_query_data"], map_location="cpu", weights_only=False))
    require(len(fit["state"]) == 159691 and len(cal["state"]) == 20237
        and len(same_query["state"]) == 14997, "frozen historical recipe cardinality differs")
    require(not set(fit["episode"].tolist()) & set(cal["episode"].tolist())
        and not set(same_query["episode"].tolist()) & set(cal["episode"].tolist()), "historical FIT/CAL overlap")
    return fit, cal, same_query


def prepare(args):
    job = args.job.resolve()
    require(not job.exists(), "preparation needs a new namespace; failed jobs are immutable")
    require(300 <= args.wall_cap_seconds <= 3600 and 128 <= args.storage_cap_mib <= 512,
        "CPU preparation+training cap must be 300..3600 seconds and 128..512 MiB")
    conditions = ["TF_CONT", "TF_ON"] + (["MLP_ON"] if args.include_mlp_on else [])
    job.mkdir(parents=True, exist_ok=False)
    registration = dict(schema="tf-effect-training-registration/v1", goal="one finite mechanism-triggered TF training comparison",
        source_commit=COMMIT, inputs=dict(onpolicy=str(args.onpolicy.resolve()), data_audit=str(args.data_audit.resolve()),
            trigger=str(args.trigger.resolve())), cpu_threads=2, GPU=False,
        birth_epoch=time.time(), deadline_epoch=time.time() + args.wall_cap_seconds,
        wall_cap_seconds=args.wall_cap_seconds, storage_cap_bytes=args.storage_cap_mib * 1024 * 1024,
        terminal_status="REGISTERED_NOT_PREPARED", cleanup_plan="single CPU process; alarm and external timeout; no workers/services",
        evidence_path=str(job), conditions=conditions, permissions=PERMISSIONS)
    write(job / "PREREGISTRATION.json", registration)
    write(job / "status.json", dict(status="PREPARING_CPU_ONLY", pid=os.getpid()))
    try:
        check_runtime(); arm_deadline(registration)
        funcs = frozen_functions()
        trigger = trigger_certificate(args.trigger, conditions)
        columns = args.onpolicy.resolve(strict=True)
        new_fit, new_cal, manifest, manifest_sha = validate_new_columns(columns)
        audit = collection_certificate(args.data_audit, columns, manifest_sha)
        base = read(PARENT_ROOT / "PLAN.json")
        final = read(PARENT_ROOT / "FINAL-RESULT.json")
        require(final.get("status") == "PASS_SIX_TWO_STAGE_FINETUNING_RUNS"
            and final.get("complete_runs") == 6 and not (PARENT_ROOT / "FAILED.json").exists(), "frozen parent job incomplete")
        fit, cal, same_query = historical(base, funcs)
        new_ids = set(new_fit["episode"].tolist()) | set(new_cal["episode"].tolist())
        require(not new_ids & (set(fit["episode"].tolist()) | set(cal["episode"].tolist())
            | set(same_query["episode"].tolist())), "new episodes overlap historical fit/cal")
        cells = {}
        for condition in conditions:
            architecture = CONDITIONS[condition]
            parent_path = PARENT_ROOT / f"{architecture}-seed{SEED}" / "checkpoint.pt"
            parent = load_parent(parent_path, architecture=architecture, device="cpu", expected_channels=range(7),
                expected_sha256=PARENT_SHAS[architecture])
            require(not torch.cuda.is_initialized(), "prepare initialized CUDA")
            source_name = "original_same_query_14997" if condition == "TF_CONT" else "current_TF_CONT3000_reached_FIT20"
            dataset = dict(historical_target_fit=dict(path=base["data"], rows=159691, episodes=174),
                same_query_source=source_name, same_query_rows=len(same_query["state"]) if condition == "TF_CONT" else len(new_fit["state"]),
                new_data_manifest_sha256=manifest_sha if condition != "TF_CONT" else None,
                generator_is_training_parent=False, new_CAL_used_for_optimization=False)
            cells[condition] = dict(architecture=architecture, seed=SEED, parent_checkpoint=str(parent_path),
                parent_checkpoint_sha256=PARENT_SHAS[architecture], same_query_source=source_name, data_manifest=dataset)
            del parent
        files = [OWN / x for x in ("train.py", "adapter.py", "gpu_actual_cal.py")]
        files += [EXP / x for x in ("tf_usable_train_v1/train.py", "tf_usable_train_v1/adapter.py",
            "target_trace_draft_finetune_v1/train.py", "target_trace_draft_finetune_v1/adapter.py",
            "target_trace_draft_train_v1/adapter.py", "draft_architecture_v1/models.py")]
        evidences = [args.trigger.resolve(strict=True), args.data_audit.resolve(strict=True),
            PARENT_ROOT / "PLAN.json", PARENT_ROOT / "FINAL-RESULT.json", PARENT_ROOT / "DATA-SOURCE-PROOF.json",
            Path(base["data_audit"]), Path(base["same_query_audit"])]
        evidences += [Path(entry["path"]).resolve(strict=True) for entry in trigger["evidence"]]
        evidences += [Path(entry["path"]).resolve(strict=True) for entry in audit["evidence"]]
        require(sha(OLD_CONT) == GENERATOR_SHA, "known prior CONT3000 reference changed")
        plan = dict(schema=PLAN_SCHEMA, source_commit=COMMIT, seed=SEED, conditions=conditions, cells=cells,
            data=base["data"], data_audit=base["data_audit"], same_query_data=base["same_query_data"],
            same_query_audit=base["same_query_audit"], new_columns=str(columns), new_data_audit=str(args.data_audit.resolve()),
            training_trigger=str(args.trigger.resolve()), new_manifest=manifest, new_manifest_sha256=manifest_sha,
            initial_total_updates=2500, continuation_updates=500, total_updates=3000, batch=512,
            target_half=256, same_query_half=256, learning_rate=3e-5, weight_decay=.01,
            normalization_refit=False, optimizer_state_reset=True, draw_rule="unchanged task->episode->row; fixed seed offsets +700000/+800000",
            loss="unchanged frozen min-head MSE + .1 mean-head MSE + .01 relu(.05-normalized gap)",
            checkpoint_selection="final update500 only; no CAL or seed selection", generator=manifest["generator_checkpoint"],
            generator_checkpoint_sha256=manifest["generator_checkpoint_sha256"],
            old_CAL_rows=20237, new_FIT_rows=len(new_fit["state"]), new_CAL_rows=len(new_cal["state"]),
            old_CONT_reference=str(OLD_CONT), old_CONT_reference_sha256=sha(OLD_CONT),
            source_sha256={str(p): sha(p) for p in files},
            input_stat={str(p): stat(p) for p in (Path(base["data"]), Path(base["same_query_data"]), columns, PARENT_ROOT / "draws.pt")},
            evidence_sha256={str(p): sha(p) for p in evidences},
            permissions=PERMISSIONS, robot_quality_validated=False, deployed=False)
        write(job / "PLAN.json", plan)
        write(job / "DATA-SOURCE-PROOF.json", dict(status="PASS_CPU_DATA_AND_LINEAGE_ONLY", new_manifest_sha256=manifest_sha,
            new_fit_episode_ids=manifest["fit_episode_ids"], new_cal_episode_ids=manifest["cal_episode_ids"],
            old_target_fit_episode_ids=sorted(set(fit["episode"].tolist())), old_CAL_episode_ids=sorted(set(cal["episode"].tolist())),
            fit_rows=len(fit["state"]), old_CAL_rows=len(cal["state"]), old_same_query_rows=len(same_query["state"]),
            new_FIT_rows=len(new_fit["state"]), new_CAL_rows=len(new_cal["state"]),
            collector_certificate_status=audit["status"], no_new_forward=True, CUDA_initialized=False))
        record = dict(status="PREPARED_NOT_TRAINED", elapsed_since_registration=time.time()-registration["birth_epoch"],
            plan_sha256=sha(job / "PLAN.json"), source_files=len(files), no_model_forward=True, CUDA_initialized=False,
            output_bytes=check_storage(job, registration["storage_cap_bytes"]), permissions=PERMISSIONS)
        write(job / "PREPARED.json", record); write(job / "status.json", record)
        print(record, flush=True)
    except BaseException as exc:
        write(job / "FAILED.json", dict(status="FAILED_CPU_PREPARATION", error=repr(exc),
            elapsed_since_registration=time.time()-registration["birth_epoch"], permissions=PERMISSIONS))
        raise
    finally:
        signal.alarm(0)


def reference_cpu(predictor, data, outputs):
    source = model_source()
    with torch.random.fork_rng(devices=[]):
        reference = source.Predictor(predictor.architecture, {key: getattr(predictor, key) for key in
            ("feature_mean", "feature_std", "residual_mean", "residual_std")})
    reference.model.load_state_dict(predictor.model.state_dict(), strict=True)
    reference.model.eval()
    with torch.inference_mode():
        for index in range(len(outputs)):
            state = data["state"][index].reshape(1, 112).bfloat16().float()
            previous = data["previous"][index].reshape(1, 112).bfloat16().float()
            sigma, next_sigma, normalized_time = data["scalars"][index].tolist()
            # Reproduce original predict's scalar conversion boundary exactly.
            batch = dict(state=state, previous=previous, scalars=torch.tensor(
                [[sigma, next_sigma, (normalized_time * 1000.) / 1000.]], dtype=torch.float32))
            actual = reference.decode_batch(batch).reshape(2, 112).bfloat16().float()
            require(torch.equal(actual, outputs[index]), "full same-CPU original decode parity differs")


def run(job):
    from tf_effect_train_v1.adapter import verify_plan
    job = job.resolve(strict=True)
    require(not (job / "STARTED.json").exists() and not (job / "FAILED.json").exists()
        and not (job / "STOP_REQUESTED").exists() and (job / "PREPARED.json").is_file(), "fresh prepared job required")
    registration = read(job / "PREREGISTRATION.json")
    complete, active_condition, active_updates = [], None, 0
    try:
        check_runtime(); arm_deadline(registration)
        funcs = frozen_functions(); plan = verify_plan(job / "PLAN.json")
        require(read(job / "PREPARED.json")["plan_sha256"] == sha(job / "PLAN.json"), "prepared plan changed")
        write(job / "STARTED.json", dict(status="RUNNING_CPU_ONLY", pid=os.getpid(),
            cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_threads=torch.get_num_threads(),
            CUDA_initialized=False, plan_sha256=sha(job / "PLAN.json"), registration_deadline=registration["deadline_epoch"]))
        fit, old_cal, old_same = historical(plan, funcs)
        new_fit, new_cal, _, _ = validate_new_columns(plan["new_columns"], expected_manifest_sha256=plan["new_manifest_sha256"])
        draws = dict(target=funcs["draw"](fit, SEED+700000), old_same=funcs["draw"](old_same, SEED+800000),
            new_same=funcs["draw"](new_fit, SEED+800000))
        original = torch.load(PARENT_ROOT / "draws.pt", map_location="cpu", weights_only=True)[str(SEED)]
        require(torch.equal(draws["target"], original["target"])
            and torch.equal(draws["old_same"], original["same_query"]), "original TF_CONT draw schedule differs")
        torch.save(draws, job / "draws.pt")
        cpu_rng = torch.get_rng_state().clone()
        for condition in plan["conditions"]:
            active_condition, active_updates = condition, 0
            start = time.monotonic(); cell = plan["cells"][condition]; architecture = cell["architecture"]
            predictor = load_parent(cell["parent_checkpoint"], architecture=architecture, device="cpu",
                expected_channels=range(7), expected_sha256=cell["parent_checkpoint_sha256"])
            norm = {key: getattr(predictor, key).clone() for key in ("feature_mean", "feature_std", "residual_mean", "residual_std")}
            same = old_same if condition == "TF_CONT" else new_fit
            same_draws = draws["old_same"] if condition == "TF_CONT" else draws["new_same"]
            # Before diagnostics are not used to stop, select a seed, or select a checkpoint.
            before_old = torch.load(Path(cell["parent_checkpoint"]).parent / "cal_predictions.pt", map_location="cpu", weights_only=True)
            before_new = funcs["predict_all"](predictor, new_cal)
            predictor.model.train()
            optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=3e-5, weight_decay=.01)
            logs = []
            for update, (target_ids, same_ids) in enumerate(zip(draws["target"], same_draws), 1):
                require(not (job / "STOP_REQUESTED").exists(), "STOP_REQUESTED")
                remaining(registration)
                batch = {key: torch.cat((fit[key][target_ids], same[key][same_ids]), 0)
                    for key in ("state", "previous", "target", "scalars")}
                optimizer.zero_grad(set_to_none=True)
                loss = funcs["loss_value"](funcs["decode"](predictor, batch), batch, predictor.residual_std)
                require(bool(torch.isfinite(loss)), "nonfinite training loss")
                loss.backward()
                require(all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in predictor.model.parameters()),
                    "nonfinite training gradient")
                optimizer.step(); active_updates = update
                if update == 1 or update % 50 == 0:
                    logs.append(dict(update=update, loss=float(loss.detach()), seconds=time.monotonic()-start))
                    write(job / "status.json", dict(status="TRAINING_FIXED_TF_EFFECT", condition=condition,
                        update=update, complete_runs=len(complete), seconds_since_registration=time.time()-registration["birth_epoch"]))
            predictor.model.eval()
            require(all(torch.equal(norm[key], getattr(predictor, key)) for key in norm), "frozen normalizer changed")
            folder = job / f"{condition}-seed{SEED}"; folder.mkdir(exist_ok=False)
            checkpoint = folder / "checkpoint.pt"
            metadata = {key: plan[key] for key in ("seed", "initial_total_updates", "continuation_updates", "total_updates", "batch",
                "target_half", "same_query_half", "learning_rate", "weight_decay", "normalization_refit", "optimizer_state_reset",
                "new_manifest_sha256", "permissions", "robot_quality_validated", "deployed")}
            metadata.update(condition=condition, parent_checkpoint=cell["parent_checkpoint"],
                parent_checkpoint_sha256=cell["parent_checkpoint_sha256"], same_query_source=cell["same_query_source"],
                data_manifest_sha256=digest_json(cell["data_manifest"]), plan=str(job / "PLAN.json"), plan_sha256=sha(job / "PLAN.json"))
            torch.save(dict(schema=SCHEMA, architecture=architecture, specification=model_source().SPECS[architecture],
                model_state_dict=predictor.model.state_dict(), normalization=norm, metadata=metadata), checkpoint)
            loaded = load_checkpoint(checkpoint, architecture=architecture, device="cpu", expected_channels=range(7), expected_condition=condition)
            predictions, metrics = {}, {}
            for name, data, before in (("old_CAL", old_cal, before_old), ("new_CAL", new_cal, before_new)):
                values = funcs["predict_all"](loaded, data)
                require(torch.equal(values, funcs["predict_all"](predictor, data)), "full CPU loader/in-memory parity failed")
                reference_cpu(loaded, data, values)
                predictions[name] = values
                metrics[name] = dict(before=funcs["metrics"](funcs["errors"](before, data), data),
                    after=funcs["metrics"](funcs["errors"](values, data), data, funcs["errors"](before, data)))
            old_cont_parity = None
            if condition == "TF_CONT":
                require(sha(OLD_CONT) == plan["old_CONT_reference_sha256"], "old CONT reference changed")
                old_payload = torch.load(OLD_CONT, map_location="cpu", weights_only=True)
                old_cont_parity = all(torch.equal(value, old_payload["model_state_dict"][key])
                    for key, value in loaded.model.state_dict().items())
                require(old_cont_parity, "fair TF_CONT must reproduce prior fixed500 weights, not alter recipe")
            require(torch.equal(cpu_rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "CPU RNG/CUDA changed")
            torch.save(predictions["old_CAL"], folder / "cal_predictions.pt")
            torch.save(predictions["new_CAL"], folder / "new_cal_predictions.pt")
            record = dict(status="PASS_FIXED_FINAL_CELL", condition=condition, architecture=architecture, seed=SEED,
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), initial_total_updates=2500,
                continuation_updates=500, total_updates=3000, parent_checkpoint=cell["parent_checkpoint"],
                parent_checkpoint_sha256=cell["parent_checkpoint_sha256"], full_old_and_new_CAL_CPU_loader_parity=True,
                same_CPU_original_decode_batch_exact=True, normalization_exact_parent=True, CPU_rng_unchanged=True,
                CUDA_initialized=False, old_TF_CONT_all_model_tensors_bitwise_exact=old_cont_parity,
                CAL_metrics=metrics, loss_log=logs, elapsed_seconds=time.monotonic()-start,
                new_GPU_CAL_required=True, robot_quality_validated=False, permissions=PERMISSIONS)
            write(folder / "RESULT.json", record); complete.append(record)
            check_storage(job, registration["storage_cap_bytes"])
        selected = {row["condition"]: {key: row[key] for key in ("checkpoint", "checkpoint_sha256", "seed", "architecture")}
            for row in complete}
        write(job / "selected.json", dict(schema=SELECTION_SCHEMA,
            selection_role="fixed_final_update500_needs_complete_DEV", conditions=selected,
            robot_quality_validated=False, deployed=False, permissions=PERMISSIONS))
        final = dict(status=FINAL_STATUS, complete_runs=len(complete), conditions=plan["conditions"],
            elapsed_since_registration=time.time()-registration["birth_epoch"], target_forwards=0, LIBERO_episodes=0,
            CUDA_initialized=False, new_GPU_CAL_required=True, robot_quality_validated=False, permissions=PERMISSIONS,
            output_bytes=check_storage(job, registration["storage_cap_bytes"]))
        write(job / "FINAL-RESULT.json", final); write(job / "status.json", final)
        print(final, flush=True)
    except BaseException as exc:
        failed = dict(status="FAILED_FIXED_TF_EFFECT_TRAINING", condition=active_condition, updates_completed=active_updates,
            complete_runs=len(complete), error=repr(exc), elapsed_since_registration=time.time()-registration["birth_epoch"],
            CUDA_initialized=torch.cuda.is_initialized(), permissions=PERMISSIONS)
        write(job / "FAILED.json", failed); write(job / "status.json", failed)
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true"); choice.add_argument("--run", action="store_true")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--onpolicy", type=Path); parser.add_argument("--data-audit", type=Path)
    parser.add_argument("--trigger", type=Path); parser.add_argument("--include-mlp-on", action="store_true")
    parser.add_argument("--wall-cap-seconds", type=int, default=1800)
    parser.add_argument("--storage-cap-mib", type=int, default=256)
    args = parser.parse_args()
    if args.prepare:
        if any(value is None for value in (args.onpolicy, args.data_audit, args.trigger)):
            parser.error("--prepare requires --onpolicy, --data-audit, --trigger")
        prepare(args)
    else:
        if args.onpolicy or args.data_audit or args.trigger or args.include_mlp_on:
            parser.error("--run uses only the frozen prepared PLAN")
        run(args.job)


if __name__ == "__main__":
    main()

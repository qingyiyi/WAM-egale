"""Frozen 500-update mixed-FIT follow-up; no GPU, new teacher, or simulator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CPU-only two-stage training requires CUDA_VISIBLE_DEVICES empty")
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
sys.path.insert(0, str(OWN.parent))
sys.path.insert(0, str(OWN))
from adapter import ARCHITECTURES, FINAL_STATUS, SCHEMA, load_checkpoint, model_source, sha
from target_trace_draft_train_v1.adapter import load_checkpoint as load_stage1

SEEDS = (20260922, 20260923, 20260924)
TAU = 0.007839680526450805


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def require(value, message):
    if not value:
        raise ValueError(message)


def draw(data, seed):
    generator = torch.Generator().manual_seed(seed)
    tasks = torch.randint(10, (500 * 256,), generator=generator)
    result = torch.empty_like(tasks)
    for task in range(10):
        positions = (tasks == task).nonzero().flatten()
        episodes = torch.unique(data["episode"][data["task"] == task]).tolist()
        choices = torch.randint(len(episodes), (len(positions),), generator=generator)
        for i, episode in enumerate(episodes):
            slots = positions[choices == i]
            pool = ((data["task"] == task) & (data["episode"] == episode)).nonzero().flatten()
            result[slots] = pool[torch.randint(len(pool), (len(slots),), generator=generator)]
    return result.reshape(500, 256)


def decode(predictor, data):
    features = torch.cat((data["state"], data["previous"], data["scalars"]), 1)
    raw = predictor.model((features - predictor.feature_mean) / predictor.feature_std)
    return data["previous"][:, None] + raw * predictor.residual_std[None, None] + predictor.residual_mean[None, None]


def loss_value(prediction, data, std):
    mse = (prediction - data["target"][:, None]).square().mean(2)
    gap = ((prediction[:, 0] - prediction[:, 1]) / std).square().mean(1)
    return mse.min(1).values.mean() + .1 * mse.mean() + .01 * torch.relu(.05 - gap).mean()


@torch.no_grad()
def predict_all(predictor, data):
    values = torch.empty((len(data["state"]), 2, 112), dtype=torch.float32)
    for i in range(len(values)):
        sigma, next_sigma, normalized_time = data["scalars"][i].tolist()
        values[i] = predictor.predict(data["state"][i].reshape(7, 4, 4).bfloat16(),
            data["previous"][i].reshape(7, 4, 4).bfloat16(),
            sigma, next_sigma, normalized_time * 1000.).reshape(2, 112).float()
    require(bool(torch.isfinite(values).all()), "nonfinite full real-data prediction")
    return values


def errors(prediction, data):
    return (prediction - data["target"][:, None]).square().sum(2).sqrt() * data["delta"][:, None]


def metrics(err, data, previous_err=None):
    best = err.min(1).values
    accepted = best <= TAU
    masks = {"all": torch.ones(len(best), dtype=torch.bool)}
    masks.update({f"task{t}": data["task"] == t for t in range(10)})
    for name, mask in (("high", data["scalars"][:, 0] >= 2/3),
                       ("mid", (data["scalars"][:, 0] >= 1/3) & (data["scalars"][:, 0] < 2/3)),
                       ("low", data["scalars"][:, 0] < 1/3)):
        masks["sigma/" + name] = mask
    if "node" in data:
        masks["root"], masks["nonroot"] = data["node"] == 0, data["node"] != 0
        for name in ("high", "mid", "low"):
            for node in ("root", "nonroot"):
                masks[node + "/" + name] = masks[node] & masks["sigma/" + name]
    groups = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            continue
        value = best[mask]
        row = dict(rows=int(mask.sum()), pass_count=int(accepted[mask].sum()),
            pass_rate=float(accepted[mask].float().mean()), scaled_l2_mean=float(value.mean()),
            scaled_l2_p50=float(torch.quantile(value, .5)), scaled_l2_p90=float(torch.quantile(value, .9)))
        if previous_err is not None:
            previous_ok = previous_err.min(1).values[mask] <= TAU
            old_count = int(previous_ok.sum())
            lost = int((previous_ok & ~accepted[mask]).sum())
            row.update(pretrain_pass_count=old_count, pretrain_pass_to_fail_count=lost,
                pretrain_pass_retention=1 - lost / old_count if old_count else None)
        groups[name] = row
    return dict(groups=groups,
        macro_task_acceptance=statistics.mean(groups[f"task{t}"]["pass_rate"] for t in range(10)),
        macro_task_scaled_l2=statistics.mean(groups[f"task{t}"]["scaled_l2_mean"] for t in range(10)))


def same_query(pack):
    rows = pack["current"]["train_rows"] + pack["aux_rows"]
    require(len(rows) == 14997 and len(pack["current"]["train_rows"]) == 3906
            and len(pack["aux_rows"]) == 11091, "same-query full FIT cardinality differs")
    require(all(r["split"] == "train" and r["episode_index"] < 18 for r in rows), "same-query CAL leakage")
    return dict(state=torch.stack([r["state"] for r in rows]),
        previous=torch.stack([r["previous"] for r in rows]),
        target=torch.stack([r["target"] for r in rows]),
        scalars=torch.tensor([[r["sigma"], r["next_sigma"], r["timestep"] / 1000.] for r in rows]),
        delta=torch.tensor([abs(r["sigma"] - r["next_sigma"]) for r in rows]),
        task=torch.tensor([r["task_index"] for r in rows]),
        episode=torch.tensor([r["task_index"] * 100 + r["episode_index"] for r in rows]),
        node=torch.tensor([r["node"] for r in rows]))


def main(job):
    start = time.monotonic()
    job = job.resolve(strict=True)
    plan = read(job / "PLAN.json")
    require(plan["schema"] == "target-trace-draft-two-stage-plan/v1" and plan["finetune_updates"] == 500
            and plan["batch"] == 512 and plan["learning_rate"] == 3e-5
            and plan["target_half"] == plan["same_query_half"] == 256
            and plan["seeds"] == list(SEEDS), "fixed plan differs")
    require(not (job / "STARTED.json").exists(), "training job already started; do not overwrite")
    write(job / "STARTED.json", dict(pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_threads=2, CUDA_initialized=torch.cuda.is_initialized(), plan_sha256=sha(job / "PLAN.json")))
    require(not torch.cuda.is_initialized(), "CUDA initialized")
    complete = []
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("two-stage complete-run CPU limit")))
    signal.alarm(1200)
    try:
        require(read(plan["data_audit"])["status"] == "PASS_COMPLETE_NUMERIC_AUDIT", "target audit incomplete")
        aux_audit = read(plan["same_query_audit"])
        require(aux_audit["status"] == "PASS_COMPLETE_SAME_QUERY_AUX_NUMERIC_AUDIT", "same-query audit incomplete")
        require(read(Path(plan["pretraining_dir"]) / "FINAL-RESULT.json")["complete_runs"] == 6,
                "all original pretraining runs required")
        target_pack = torch.load(plan["data"], map_location="cpu", weights_only=False)
        columns = target_pack["columns"]
        fit_ids, cal_ids = target_pack["views"]["fit174"], target_pack["views"]["cal20"]
        keys = ("state", "previous", "target", "scalars", "delta", "task", "episode")
        fit = {k: columns[k][fit_ids].contiguous() for k in keys}
        cal = {k: columns[k][cal_ids].contiguous() for k in keys}
        require(len(fit["state"]) == 159691 and len(cal["state"]) == 20237,
                "target FIT/CAL row counts differ")
        require(not set(fit["episode"].tolist()) & set(cal["episode"].tolist()), "target episode leakage")
        sq_pack = torch.load(plan["same_query_data"], map_location="cpu", weights_only=False)
        sq = same_query(sq_pack)
        require(not set(sq["episode"].tolist()) & set(cal["episode"].tolist()), "same-query episode leaks into CAL")
        del target_pack, columns, sq_pack
        draws = {str(seed): dict(target=draw(fit, seed + 700000), same_query=draw(sq, seed + 800000)) for seed in SEEDS}
        torch.save(draws, job / "draws.pt")
        write(job / "DATA-SOURCE-PROOF.json", dict(target_fit_rows=len(fit["state"]), cal_rows=len(cal["state"]),
            same_query_rows=len(sq["state"]), target_fit_episode_ids=sorted(set(fit["episode"].tolist())),
            same_query_episode_ids=sorted(set(sq["episode"].tolist())), cal_episode_ids=sorted(set(cal["episode"].tolist())),
            same_query_provenance=plan["same_query_provenance"], generator_inventory=aux_audit["generator_draft_inventory"],
            normalization_refit=False, quality25_to29_tensors_read=False, new_teacher_or_environment=False,
            same_draws_for_architectures=True, target_samples_per_update=256, same_query_samples_per_update=256))
        for seed in SEEDS:
            for architecture in ARCHITECTURES:
                cell_start = time.monotonic()
                folder = job / f"{architecture}-seed{seed}"
                folder.mkdir()
                parent = Path(plan["pretraining_dir"]) / f"{architecture}-seed{seed}" / "checkpoint.pt"
                parent_record = read(parent.parent / "RESULT.json")
                predictor = load_stage1(parent, architecture=architecture, device="cpu",
                    expected_channels=tuple(range(7)), expected_sha256=parent_record["checkpoint_sha256"])
                norm = {k: getattr(predictor, k).clone() for k in ("feature_mean", "feature_std", "residual_mean", "residual_std")}
                before_rng = torch.get_rng_state().clone()
                before_cal = torch.load(parent.parent / "cal_predictions.pt", map_location="cpu", weights_only=True)
                before_cal_errors = errors(before_cal, cal)
                before_sq_errors = errors(predict_all(predictor, sq), sq)
                predictor.model.train()
                optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=3e-5, weight_decay=.01)
                logs = []
                for update, (target_ids, same_ids) in enumerate(zip(draws[str(seed)]["target"], draws[str(seed)]["same_query"]), 1):
                    batch = {k: torch.cat((fit[k][target_ids], sq[k][same_ids]), 0)
                             for k in ("state", "previous", "target", "scalars")}
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_value(decode(predictor, batch), batch, predictor.residual_std)
                    require(bool(torch.isfinite(loss)), "nonfinite mixed-FIT loss")
                    loss.backward()
                    require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                for p in predictor.model.parameters()), "nonfinite mixed-FIT gradient")
                    optimizer.step()
                    if update == 1 or update % 50 == 0:
                        logs.append(dict(update=update, loss=float(loss.detach()), seconds=time.monotonic() - cell_start))
                        write(job / "status.json", dict(status="FINETUNING", architecture=architecture,
                            seed=seed, update=update, complete_runs=len(complete), elapsed_seconds=time.monotonic() - start))
                predictor.model.eval()
                require(all(torch.equal(norm[k], getattr(predictor, k)) for k in norm), "stage1 normalizer mutated")
                checkpoint = folder / "checkpoint.pt"
                metadata = dict(training_source="target_pretrain_then_mixed_historical_same_query_finetune",
                    seed=seed, pretraining_updates=2000, finetune_updates=500, total_updates=2500,
                    batch=512, target_half=256, same_query_half=256, learning_rate=3e-5, weight_decay=.01,
                    parent_checkpoint=str(parent), parent_checkpoint_sha256=parent_record["checkpoint_sha256"],
                    normalization_refit=False, same_query_provenance=plan["same_query_provenance"],
                    plan=str(job / "PLAN.json"), plan_sha256=sha(job / "PLAN.json"), deployed=False)
                source = model_source()
                torch.save(dict(schema=SCHEMA, architecture=architecture, specification=source.SPECS[architecture],
                    model_state_dict=predictor.model.state_dict(), normalization=norm, metadata=metadata), checkpoint)
                production = load_checkpoint(checkpoint, architecture=architecture, device="cpu", expected_channels=tuple(range(7)))
                after_cal = predict_all(production, cal)
                # Every actual CAL query checks the training state through the new opt-in schema loader.
                require(torch.equal(after_cal, predict_all(predictor, cal)), "complete CAL new-loader parity failed")
                after_sq_errors = errors(predict_all(production, sq), sq)
                require(torch.equal(before_rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "RNG/CUDA changed")
                after_cal_errors = errors(after_cal, cal)
                result = dict(architecture=architecture, seed=seed, checkpoint=str(checkpoint),
                    checkpoint_sha256=production.checkpoint_sha256, parent_checkpoint=str(parent),
                    parent_checkpoint_sha256=parent_record["checkpoint_sha256"],
                    pretraining_updates=2000, finetune_updates=500, total_updates=2500,
                    before_cal=metrics(before_cal_errors, cal), after_cal=metrics(after_cal_errors, cal, before_cal_errors),
                    before_same_query_FIT=metrics(before_sq_errors, sq),
                    after_same_query_FIT=metrics(after_sq_errors, sq, before_sq_errors),
                    full_CAL_new_adapter_bitwise_exact=True, normalization_exact_parent=True,
                    CPU_rng_unchanged=True, CUDA_initialized=False, loss_log=logs,
                    elapsed_seconds=time.monotonic() - cell_start, robot_quality_validated=False)
                torch.save(after_cal, folder / "cal_predictions.pt")
                torch.save(dict(before_errors=before_sq_errors, after_errors=after_sq_errors), folder / "same_query_FIT_errors.pt")
                write(folder / "RESULT.json", result)
                complete.append(result)
                print(json.dumps(dict(event="COMPLETE_FINETUNE_CELL", architecture=architecture, seed=seed,
                    after_cal=result["after_cal"]["macro_task_acceptance"],
                    seconds=result["elapsed_seconds"])), flush=True)
        selections, summary = {}, []
        for architecture in ARCHITECTURES:
            runs = [r for r in complete if r["architecture"] == architecture]
            ordered = sorted(runs, key=lambda r: (-r["after_cal"]["macro_task_acceptance"],
                                                  r["after_cal"]["macro_task_scaled_l2"], r["seed"]))
            middle = ordered[1]
            selections[architecture] = dict(seed=middle["seed"], checkpoint=middle["checkpoint"],
                checkpoint_sha256=middle["checkpoint_sha256"],
                cal_macro_task_acceptance=middle["after_cal"]["macro_task_acceptance"],
                cal_macro_task_scaled_l2=middle["after_cal"]["macro_task_scaled_l2"])
            summary.append(dict(architecture=architecture, seeds=3, selected=selections[architecture],
                median_before_CAL=statistics.median(r["before_cal"]["macro_task_acceptance"] for r in runs),
                median_after_CAL=statistics.median(r["after_cal"]["macro_task_acceptance"] for r in runs),
                median_before_same_query_FIT=statistics.median(r["before_same_query_FIT"]["macro_task_acceptance"] for r in runs),
                median_after_same_query_FIT=statistics.median(r["after_same_query_FIT"]["macro_task_acceptance"] for r in runs)))
        write(job / "selected.json", dict(schema="target-trace-draft-two-stage-selection/v1",
            selection_role="complete_closed_loop_DEV_candidate_only", architectures=selections,
            selection_rule=plan["selection"], robot_quality_validated=False, deployed=False))
        final = dict(status=FINAL_STATUS, complete_runs=6, summary=summary,
            elapsed_seconds=time.monotonic() - start, cpu_threads=2, cpu_affinity=sorted(os.sched_getaffinity(0)),
            CUDA_initialized=False, target_forwards=0, new_LIBERO_episodes=0, quality25_to29_tensors_read=False,
            deployed=False, same_query_provenance=plan["same_query_provenance"],
            limitation="Fixed developmental follow-up after observing init24 failure; FIT gains and CAL local errors do not establish closed-loop quality.")
        write(job / "FINAL-RESULT.json", final)
        write(job / "status.json", dict(status=FINAL_STATUS, complete_runs=6, elapsed_seconds=final["elapsed_seconds"]))
        signal.alarm(0)
        print(json.dumps({k: final[k] for k in ("status", "complete_runs", "elapsed_seconds")}), flush=True)
    except BaseException as error:
        write(job / "FAILED.json", dict(status="FAILED", error=repr(error), complete_runs=len(complete),
            elapsed_seconds=time.monotonic() - start, CUDA_initialized=torch.cuda.is_initialized()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    main(parser.parse_args().job)

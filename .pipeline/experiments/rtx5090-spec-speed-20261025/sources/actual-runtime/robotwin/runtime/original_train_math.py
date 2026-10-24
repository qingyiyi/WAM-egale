"""Six fixed real-data pretraining runs; CPU only, no LIBERO or target call."""
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
    raise RuntimeError("target-trace CPU training requires CUDA_VISIBLE_DEVICES empty")
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
sys.path.insert(0, str(OWN))
from adapter import ARCHITECTURES, SCHEMA, load_checkpoint, model_source, sha

SEEDS = (20260922, 20260923, 20260924)
UPDATES, BATCH, LR, WEIGHT_DECAY = 2000, 512, 1e-4, .01
THRESHOLD = 0.007839680526450805
HISTORICAL = Path("/var/tmp/cxz-wam-quality-20261014/p0-morning-long10-train-six-v1-20261015-r1/dataset.pt")
SELECTION = "Per architecture: median seed in descending CAL macro-task acceptance, ascending macro-task scaled L2, ascending seed; final update only; complete closed-loop DEV candidate, no deployment"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def require(value, message):
    if not value:
        raise ValueError(message)


def normalize(data):
    features = torch.cat((data["state"], data["previous"], data["scalars"]), 1)
    residual = data["target"] - data["previous"]
    return dict(feature_mean=features.mean(0), feature_std=features.std(0, correction=0).clamp_min(1e-5),
                residual_mean=residual.mean(0), residual_std=residual.std(0, correction=0).clamp_min(1e-5))


def draw_indices(fit, seed):
    generator = torch.Generator().manual_seed(seed)
    tasks = torch.randint(10, (UPDATES * BATCH,), generator=generator)
    result = torch.empty_like(tasks)
    for task in range(10):
        task_positions = (tasks == task).nonzero().flatten()
        episodes = torch.unique(fit["episode"][fit["task"] == task]).tolist()
        episode_choices = torch.randint(len(episodes), (len(task_positions),), generator=generator)
        for i, episode in enumerate(episodes):
            positions = task_positions[episode_choices == i]
            pool = ((fit["task"] == task) & (fit["episode"] == episode)).nonzero().flatten()
            result[positions] = pool[torch.randint(len(pool), (len(positions),), generator=generator)]
    return result.reshape(UPDATES, BATCH)


def data_loss(prediction, batch, residual_std):
    mse = (prediction - batch["target"][:, None]).square().mean(2)
    gap = ((prediction[:, 0] - prediction[:, 1]) / residual_std).square().mean(1)
    return mse.min(1).values.mean() + .1 * mse.mean() + .01 * torch.relu(.05 - gap).mean()


@torch.no_grad()
def predict_rows(predictor, data):
    values = torch.empty((len(data["state"]), 2, 112), dtype=torch.float32)
    for i in range(len(values)):
        x = data["state"][i].reshape(7, 4, 4).bfloat16()
        p = data["previous"][i].reshape(7, 4, 4).bfloat16()
        sigma, next_sigma, normalized_timestep = data["scalars"][i].tolist()
        values[i] = predictor.predict(x, p, sigma, next_sigma, normalized_timestep * 1000.).reshape(2, 112).float()
    require(bool(torch.isfinite(values).all()), "nonfinite production-interface predictions")
    return values


def metrics(prediction, data):
    error = (prediction - data["target"][:, None]).square().sum(2).sqrt() * data["delta"][:, None]
    error = error.min(1).values
    groups = {}
    for task in range(10):
        mask = data["task"] == task
        require(bool(mask.any()), "CAL lacks task " + str(task))
        groups[str(task)] = dict(rows=int(mask.sum()), acceptance=float((error[mask] <= THRESHOLD).float().mean()),
            scaled_l2_mean=float(error[mask].mean()), scaled_l2_p95=float(torch.quantile(error[mask], .95)))
    return dict(rows=len(error), tasks=groups,
        macro_task_acceptance=statistics.mean(x["acceptance"] for x in groups.values()),
        macro_task_scaled_l2=statistics.mean(x["scaled_l2_mean"] for x in groups.values()),
        micro_acceptance=float((error <= THRESHOLD).float().mean()),
        micro_scaled_l2=float(error.mean()),
        meaning="target-only next actual query, step1..49; not a speculative root or new tree acceptance")


def historical_data(pack):
    rows = pack["current"]["validation_rows"]
    return dict(state=torch.stack([r["state"] for r in rows]),
        previous=torch.stack([r["previous"] for r in rows]),
        target=torch.stack([r["target"] for r in rows]),
        scalars=torch.tensor([[r["sigma"], r["next_sigma"], r["timestep"] / 1000.] for r in rows]),
        delta=torch.tensor([abs(r["sigma"] - r["next_sigma"]) for r in rows]),
        node=torch.tensor([r["node"] for r in rows]),
        task=torch.tensor([r["task_index"] for r in rows]))


def historical_six_diagnostic(prediction, old_prediction, data):
    error = ((prediction - data["target"][:, None]).square().sum(2).sqrt() * data["delta"][:, None]).min(1).values
    old_error = ((old_prediction - data["target"][:, None]).square().sum(2).sqrt() * data["delta"][:, None]).min(1).values
    accepted, old_accepted = error <= THRESHOLD, old_error <= THRESHOLD
    root = data["node"] == 0
    all_rows = torch.ones(len(root), dtype=torch.bool)
    groups = {}
    for name, mask in (("all", all_rows), ("root", root),
                       ("task0/root", root & (data["task"] == 0)),
                       ("task1/root", root & (data["task"] == 1))):
        n = int(old_accepted[mask].sum())
        lost = int((old_accepted[mask] & ~accepted[mask]).sum())
        groups[name] = dict(rows=int(mask.sum()), pass_count=int(accepted[mask].sum()), old_pass_count=n,
            old_pass_to_fail_count=lost, old_pass_to_fail_rate=lost / n if n else 0.,
            successor_l2_mean=float(error[mask].mean()), old_successor_l2_mean=float(old_error[mask].mean()))
    g = groups
    values = dict(root_pass_count=g["root"]["pass_count"] - g["root"]["old_pass_count"],
        task0_root_pass_count=g["task0/root"]["pass_count"] - g["task0/root"]["old_pass_count"],
        task1_root_pass_count=g["task1/root"]["pass_count"] - g["task1/root"]["old_pass_count"],
        all_l2_improvement=g["all"]["old_successor_l2_mean"] - g["all"]["successor_l2_mean"],
        root_l2_improvement=g["root"]["old_successor_l2_mean"] - g["root"]["successor_l2_mean"],
        root_retention_margin=.01 - g["root"]["old_pass_to_fail_rate"])
    return dict(groups=groups, values=values, passed=all(v >= 0 for v in values.values()),
                diagnostic_only=True, not_robot_success_gate=True, not_used_for_seed_selection=True)


def prepare(data_path, audit_path, out):
    audit = read(audit_path)
    require(audit.get("status") == "PASS_COMPLETE_NUMERIC_AUDIT", "complete independent data audit required")
    history = audit["history"]
    require(Path(history["pack"]).resolve() == data_path and history["rows"] == 179928,
            "audited data path/row count differs")
    require(data_path.stat().st_size == history["pack_bytes"], "audited data size differs")
    pack = torch.load(data_path, map_location="cpu", weights_only=False)
    columns = pack["columns"]
    required = {"state", "previous", "target", "scalars", "delta", "task", "init", "episode", "step"}
    require(required <= set(columns), "compact dataset column missing")
    require(len(columns["state"]) == 179928, "full TRAIN194 row count differs")
    for name, shape in (("state", (179928, 112)), ("previous", (179928, 112)),
                        ("target", (179928, 112)), ("scalars", (179928, 3))):
        require(columns[name].shape == shape and columns[name].dtype == torch.float32
                and bool(torch.isfinite(columns[name]).all()), "data tensor shape/dtype/finite differs: " + name)
    require(bool(((columns["step"] >= 1) & (columns["step"] <= 49)).all()), "only next actual query step1..49 allowed")
    fit_ids = torch.as_tensor(pack["views"]["fit174"], dtype=torch.long)
    cal_ids = torch.as_tensor(pack["views"]["cal20"], dtype=torch.long)
    require(len(fit_ids) + len(cal_ids) == len(columns["state"]), "FIT/CAL do not cover TRAIN194")
    require(len(torch.unique(torch.cat((fit_ids, cal_ids)))) == len(columns["state"]), "FIT/CAL row overlap")
    fit = {k: columns[k][fit_ids].contiguous() for k in required}
    cal = {k: columns[k][cal_ids].contiguous() for k in required}
    fit_episodes, cal_episodes = set(fit["episode"].tolist()), set(cal["episode"].tolist())
    expected_fit = {t * 100 + i for t in range(10) for i in range(18)
                    if not (t in (0, 1) and i in (6, 7, 8))}
    expected_cal = {t * 100 + i for t in range(10) for i in (18, 19)}
    require(fit_episodes == expected_fit and cal_episodes == expected_cal
            and not fit_episodes & cal_episodes, "FIT174/CAL20 complete episode split differs")
    normalization = normalize(fit)
    models = model_source()
    parameters = {}
    for architecture in ARCHITECTURES:
        with torch.random.fork_rng(devices=[]):
            parameters[architecture] = sum(p.numel() for p in models.make_model(architecture).parameters())
    require(parameters == ARCHITECTURES, "architecture capacity match differs")
    draws = {str(seed): draw_indices(fit, seed) for seed in SEEDS}
    torch.save(draws, out / "draws.pt")
    torch.save(normalization, out / "normalization.pt")
    plan = dict(schema="target-trace-draft-training-plan/v1", data=str(data_path), audit=str(audit_path),
        audit_status=audit["status"], source="historical_target_only_actual_queries", fit_episodes=174,
        cal_episodes=20, fit_rows=len(fit_ids), cal_rows=len(cal_ids),
        fit_episode_ids=sorted(fit_episodes), cal_episode_ids=sorted(cal_episodes),
        architectures=list(ARCHITECTURES), parameters=parameters, seeds=list(SEEDS),
        updates=UPDATES, batch=BATCH, learning_rate=LR, weight_decay=WEIGHT_DECAY,
        optimizer="AdamW", loss="min-head MSE + .1 mean-head MSE + .01 ReLU(.05-normalized head gap)",
        retention_KD=False, normalization="FIT174 only, population std, clamp_min1e-5",
        sampler="task uniform; episode uniform within task; actual query uniform within episode; common draws per seed",
        threshold=THRESHOLD, checkpoint="fixed final update2000; no early stopping", selection=SELECTION,
        historical_six_gates="diagnostic only; not seed selection or robot quality admission",
        cpu_threads=2, cpu_affinity=sorted(os.sched_getaffinity(0)), CUDA_visible_devices="",
        closed_loop_plan="separate full Long10 init24 T/oldAoff/newMLP/newTF, all full horizon800",
        deployed=False, extra_target_forwards=0)
    write(out / "PLAN.json", plan)
    return models, fit, cal, normalization, draws, plan


def main(args):
    start = time.monotonic()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    complete = []
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("1800 second complete-run limit")))
    signal.alarm(1800)
    try:
        require(not torch.cuda.is_initialized(), "CUDA initialized")
        models, fit, cal, normalization, draws, plan = prepare(args.data.resolve(), args.audit.resolve(), out)
        # The baseline is diagnostic and never enters the new students' loss.
        sys.path.insert(0, str(ROOT / "投机采样/scripts"))
        from p0_batched_verify import TwoHeadResidualActionPredictor
        current = read(ROOT / "投机采样/configs/current_speculative_candidate.json")
        with torch.random.fork_rng(devices=[]):
            baseline = TwoHeadResidualActionPredictor.from_checkpoint(current["draft_checkpoint"])
        old_cal = predict_rows(baseline, cal)
        write(out / "BASELINE-CAL.json", metrics(old_cal, cal))
        del baseline, old_cal
        historical_pack = torch.load(HISTORICAL, map_location="cpu", weights_only=False)
        historical = historical_data(historical_pack)
        historical_old = historical_pack["current"]["old_val"]
        del historical_pack
        for seed in SEEDS:
            for architecture in ARCHITECTURES:
                cell_start = time.monotonic()
                folder = out / f"{architecture}-seed{seed}"
                folder.mkdir()
                torch.manual_seed(seed)
                predictor = models.Predictor(architecture, normalization)
                predictor.model.train()
                optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                logs = []
                for update, ids in enumerate(draws[str(seed)], 1):
                    batch = {key: fit[key][ids] for key in ("state", "previous", "scalars", "target")}
                    optimizer.zero_grad(set_to_none=True)
                    loss = data_loss(predictor.decode_batch(batch), batch, predictor.residual_std)
                    require(bool(torch.isfinite(loss)), "training loss became nonfinite")
                    loss.backward()
                    require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                for p in predictor.model.parameters()), "training gradient became nonfinite")
                    optimizer.step()
                    if update == 1 or update % 250 == 0:
                        logs.append(dict(update=update, loss=float(loss.detach()), elapsed_seconds=time.monotonic() - cell_start))
                        write(out / "status.json", dict(status="TRAINING", architecture=architecture,
                            seed=seed, update=update, complete_runs=len(complete), elapsed_seconds=time.monotonic() - start))
                predictor.model.eval()
                checkpoint = folder / "checkpoint.pt"
                torch.save(dict(schema=SCHEMA, architecture=architecture, specification=models.SPECS[architecture],
                    model_state_dict=predictor.model.state_dict(), normalization=normalization,
                    metadata=dict(training_source="historical_target_only_actual_queries", seed=seed,
                        updates=UPDATES, batch=BATCH, fit_episodes=174, cal_episodes=20,
                        plan=str(out / "PLAN.json"), deployed=False)), checkpoint)
                before_rng = torch.get_rng_state().clone()
                production = load_checkpoint(checkpoint, architecture=architecture,
                    device="cpu", expected_channels=tuple(range(7)))
                cal_prediction = predict_rows(production, cal)
                # Real checkpoint roundtrip and production API check on every CAL query.
                reference_prediction = predict_rows(predictor, cal)
                require(torch.equal(cal_prediction, reference_prediction), "full CAL production roundtrip changed predictions")
                require(torch.equal(before_rng, torch.get_rng_state()), "loading/prediction changed host RNG")
                diagnostic_prediction = predict_rows(production, historical)
                require(not torch.cuda.is_initialized(), "CUDA initialized during training/evaluation")
                cal_metrics = metrics(cal_prediction, cal)
                diagnostic = historical_six_diagnostic(diagnostic_prediction, historical_old, historical)
                torch.save(cal_prediction, folder / "cal_predictions.pt")
                result = dict(architecture=architecture, seed=seed, updates=UPDATES,
                    checkpoint=str(checkpoint), checkpoint_sha256=production.checkpoint_sha256,
                    parameter_count=ARCHITECTURES[architecture], cal=cal_metrics,
                    historical_six_diagnostic=diagnostic, production_API_all_CAL_bitwise_exact=True,
                    CPU_rng_unchanged=True, CUDA_initialized=False, loss_log=logs,
                    elapsed_seconds=time.monotonic() - cell_start, selection_role="complete_closed_loop_DEV_candidate_only")
                write(folder / "RESULT.json", result)
                complete.append(result)
                write(out / "PARTIAL-RESULT.json", dict(status="PARTIAL", complete_runs=len(complete),
                    completed=[dict(architecture=x["architecture"], seed=x["seed"]) for x in complete]))
                print(json.dumps(dict(event="COMPLETE_CELL", architecture=architecture, seed=seed,
                    seconds=result["elapsed_seconds"], cal_macro_acceptance=cal_metrics["macro_task_acceptance"])), flush=True)
        selections = {}
        summary = []
        for architecture in ARCHITECTURES:
            runs = [x for x in complete if x["architecture"] == architecture]
            require(len(runs) == 3, "architecture missing seed")
            ranked = sorted(runs, key=lambda x: (-x["cal"]["macro_task_acceptance"],
                                                 x["cal"]["macro_task_scaled_l2"], x["seed"]))
            middle = ranked[1]
            selections[architecture] = dict(seed=middle["seed"], checkpoint=middle["checkpoint"],
                checkpoint_sha256=middle["checkpoint_sha256"],
                cal_macro_task_acceptance=middle["cal"]["macro_task_acceptance"],
                cal_macro_task_scaled_l2=middle["cal"]["macro_task_scaled_l2"])
            summary.append(dict(architecture=architecture, seeds=3,
                median_macro_task_acceptance=statistics.median(x["cal"]["macro_task_acceptance"] for x in runs),
                median_macro_task_scaled_l2=statistics.median(x["cal"]["macro_task_scaled_l2"] for x in runs),
                diagnostic_six_gate_passes=sum(x["historical_six_diagnostic"]["passed"] for x in runs),
                selected=selections[architecture]))
        write(out / "selected.json", dict(schema="target-trace-draft-selection/v1", selection_rule=SELECTION,
            selection_role="complete_closed_loop_DEV_candidate_only", architectures=selections,
            robot_success_validated=False, recursive_tree_quality_validated=False, deployed=False))
        final = dict(status="PASS_SIX_TARGET_TRACE_PRETRAINING_RUNS", complete_runs=len(complete),
            summary=summary, CUDA_initialized=False, cpu_threads=2, target_forwards=0, LIBERO_episodes=0,
            source="historical_target_only_actual_queries", elapsed_seconds=time.monotonic() - start,
            selection_role="complete_closed_loop_DEV_candidate_only", deployed=False,
            limitations=["Target-only trajectories differ from each draft's recursive verifier distribution.",
                "CAL local acceptance and old six gates do not establish task success or trace preservation.",
                "Full Long10 closed-loop comparison and local timing remain separate work."])
        write(out / "FINAL-RESULT.json", final)
        write(out / "status.json", dict(status=final["status"], complete_runs=6, elapsed_seconds=final["elapsed_seconds"]))
        signal.alarm(0)
        print(json.dumps({k: final[k] for k in ("status", "complete_runs", "elapsed_seconds")}), flush=True)
    except BaseException as error:
        write(out / "FAILED.json", dict(status="FAILED", error=repr(error), complete_runs=len(complete),
            CUDA_initialized=torch.cuda.is_initialized(), elapsed_seconds=time.monotonic() - start))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())

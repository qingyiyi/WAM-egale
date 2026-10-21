"""One registered 500-update branch, fixed final checkpoint, no CAL selection.

Preparation only until the complete fixed observer FIT/CAL audit is available.
Usage: train_branch.py PLAN.json OUTPUT_DIR
"""
from pathlib import Path
import ast
import datetime
import json
import os
import sys
import time
import traceback

os.sched_setaffinity(0, {18, 19})
import torch
from draft import Transformer, Predictor, SPEC

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
base = Path(__file__).resolve().parent
plan = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
out.mkdir(exist_ok=False)
started = time.time()
deadline = min(started + 1800, datetime.datetime.fromisoformat(plan["deadline"]).timestamp())
report = dict(status="PREPARING_FIXED_BRANCH", updates_completed=0, log=[],
    robot_quality_validated=False, formal_performance_permission=False,
    owned=dict(pid=os.getpid(), pgid=os.getpgid(0),
        birth_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])))


def save():
    (out / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")


def draws(data, seed):
    """Same task→episode→query draws as original mixed training, six tasks."""
    rng = torch.Generator().manual_seed(seed)
    tasks = torch.randint(6, (500 * 256,), generator=rng)
    result = torch.empty_like(tasks)
    for task in range(6):
        positions = (tasks == task).nonzero().flatten()
        episodes = data["episode"][data["task"] == task].unique()
        assert len(episodes) > 0
        choices = torch.randint(len(episodes), (len(positions),), generator=rng)
        for index, episode in enumerate(episodes):
            slots = positions[choices == index]
            pool = ((data["task"] == task) & (data["episode"] == episode)).nonzero().flatten()
            result[slots] = pool[torch.randint(len(pool), (len(slots),), generator=rng)]
    return result.reshape(500, 256)


try:
    condition = plan["condition"]
    parents = dict(CHILD2500="PARENT2000", CONT3000="CHILD2500", ON3000="CHILD2500")
    assert condition in parents
    assert json.loads((base / "DATA-AUDIT.json").read_text())["status"] == "PASS_TARGET48_COMPLETE_SUPERVISION"
    audit = Path(plan["observer_audit"])
    assert json.loads((audit / "RESULT.json").read_text())["status"] == "PASS_COMPLETE_FIXED18_REACHED_SUPERVISION"
    parent_path = Path(plan["parent_checkpoint"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    assert parent["schema"] == "robotwin-tf64/v1" and parent["specification"] == SPEC
    assert parent["metadata"]["condition"] == parents[condition]
    target = torch.load(base / "target48.pt", map_location="cpu", weights_only=True)
    observed = torch.load(audit / "observed18.pt", map_location="cpu", weights_only=True)
    assert observed["schema"] == "robotwin-reached18/v1"
    assert observed["stage"] == ("CURRENT" if condition == "ON3000" else "PARENT")
    assert observed["generator"] == ("CONT3000" if condition == "ON3000" else "PARENT2000")
    identity = observed["generator_identity"]
    if condition == "CHILD2500":
        assert identity["checkpoint"] == str(parent_path)
    elif condition == "CONT3000":
        assert identity["checkpoint"] == parent["metadata"]["parent_checkpoint"]
    else:
        assert identity["metadata"]["parent_checkpoint"] == str(parent_path)
    fit, visits = target["data"]["FIT"], observed["data"]["FIT"]
    assert len(fit["episode"].unique()) == 36 and len(visits["episode"].unique()) == 12
    assert not set(fit["episode"].tolist()) & set(visits["episode"].tolist())
    for data in (fit, visits):
        assert set(data["task"].tolist()) == set(range(6))
        assert all(torch.isfinite(data[k]).all() for k in ("state", "previous", "target", "scalars"))
    # Pure math from the real original functions, not a renamed surrogate loss.
    namespace = dict(torch=torch)
    source = ast.parse((base / "original_train_math.py").read_text())
    nodes = [x for x in source.body if isinstance(x, ast.FunctionDef)
             and x.name in ("normalize", "data_loss")]
    assert {x.name for x in nodes} == {"normalize", "data_loss"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_train_math.py", "exec"), namespace)
    norm = parent["normalization"]
    fitted = namespace["normalize"](fit)
    assert set(norm) == set(fitted) and all(torch.equal(norm[k], fitted[k]) for k in norm)
    assert parent["channels"] == target["channels"]
    seed = 20260923
    target_ids, observer_ids = draws(fit, seed + 700000), draws(visits, seed + 800000)
    torch.save(dict(target=target_ids, observer=observer_ids), out / "draws.pt")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Transformer().float().cuda()
    model.load_state_dict(parent["model_state_dict"])
    predictor = Predictor(model, norm)
    gpu = [{k: d[k].cuda() for k in ("state", "previous", "target", "scalars")}
           for d in (fit, visits)]
    target_ids, observer_ids = target_ids.cuda(), observer_ids.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    model.train()
    report["status"] = "RUNNING_FIXED_500_UPDATES"
    for step in range(500):
        assert time.time() < deadline
        batch = {k: torch.cat((gpu[0][k][target_ids[step]], gpu[1][k][observer_ids[step]]))
                 for k in gpu[0]}
        optimizer.zero_grad(set_to_none=True)
        loss = namespace["data_loss"](predictor.decode_batch(batch), batch, predictor.residual_std)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        report["updates_completed"] = step + 1
        if step == 0 or (step + 1) % 50 == 0:
            report["log"].append(dict(update=step + 1, loss=float(loss.detach())))
            save()
    model.eval()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.equal(getattr(predictor, k).cpu(), norm[k]) for k in norm)
    total = 2500 if condition == "CHILD2500" else 3000
    metadata = dict(condition=condition, seed=seed, updates=total, additional_updates=500,
        batch=512, target_half=256, observer_half=256, lr=3e-5, weight_decay=.01,
        optimizer_state_reset=True, normalization="Target FIT36 fixed", selection="fixed final update",
        parent_checkpoint=str(parent_path), observer_audit=str(audit),
        observer_generator=observed["generator"], parameters=sum(p.numel() for p in model.parameters()),
        target_revision=parent["metadata"]["target_revision"], quality_passed=False)
    torch.save(dict(schema="robotwin-tf64/v1", specification=SPEC,
        model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        normalization=norm, metadata=metadata, channels=parent["channels"]), out / "checkpoint.pt")
    report.update(status="COMPLETE_FIXED_BRANCH_NOT_DEPLOYMENT_CAL_OR_ROBOT_CERTIFICATE",
                  metadata=metadata, CAL_used_for_training_or_selection=False,
                  additional_teacher_forwards=0)
except BaseException as exc:
    report.update(status="FAILED_FIXED_BRANCH", error=repr(exc), traceback=traceback.format_exc())
finally:
    report["elapsed_seconds"] = time.time() - started
    save()
if report["status"] != "COMPLETE_FIXED_BRANCH_NOT_DEPLOYMENT_CAL_OR_ROBOT_CERTIFICATE":
    raise SystemExit(1)

"""Audit complete reached-query collection; never train from a partial panel.

Usage: audit_observed.py PLAN.json OUTPUT_DIR
The plan supplies the fixed 18 cases and explicit original attempt directories.
Completed algorithmic failures remain data; infrastructure partials are listed
but never converted into completed episodes or silently retried.
"""
from pathlib import Path
import datetime
import json
import os
import sys
import time
import traceback

os.sched_setaffinity(0, {18, 19})
import torch

torch.set_num_threads(2)
plan = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
out.mkdir(exist_ok=False)
started = time.time()
deadline = min(started + 1800, datetime.datetime.fromisoformat(plan["deadline"]).timestamp())
report = dict(status="RUNNING_CPU_AUDIT", episodes=[], excluded_attempts=[],
              missing=[], extra_teacher_forwards=0, full_conditions_for_teacher_requery=False)
report["owned"] = dict(pid=os.getpid(), pgid=os.getpgid(0),
    birth_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]))
tasks = ["place_object_basket", "stack_blocks_two", "handover_block",
         "lift_pot", "press_stapler", "open_microwave"]


def save():
    (out / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")


try:
    assert not torch.cuda.is_initialized()
    expected = json.loads(Path(plan["fixed_manifest"]).read_text())
    assert len(expected) == 18
    stage = plan["stage"]
    assert stage in ("PARENT", "CURRENT")
    generator = "PARENT2000" if stage == "PARENT" else "CONT3000"
    valid_ids = (8, 9, 10) if stage == "PARENT" else (11, 12, 13)
    wanted = {f"{x['task']}-valid{x['valid_index']:02d}": x for x in expected}
    assert set(wanted) == {f"{task}-valid{v:02d}" for task in tasks for v in valid_ids}
    records = {}
    for directory in plan["attempts"]:
        root = Path(directory)
        status = json.loads((root / "STATUS.json").read_text())
        for item in status["episodes"]:
            key = item["case"]
            assert key in wanted
            if item["status"] not in ("COMPLETE_SUCCESS", "COMPLETE_ALGORITHMIC_FAILURE"):
                report["excluded_attempts"].append(dict(root=str(root), record=item))
                continue
            assert key not in records, "Do not merge repeated complete episodes"
            records[key] = (root, item)
    cols = {s: {k: [] for k in ("state", "previous", "target", "scalars",
                                "task", "episode", "chunk", "step")}
            for s in ("FIT", "CAL")}
    references = []
    generator_identity = None
    for key, case in wanted.items():
        assert time.time() < deadline
        if key not in records:
            report["missing"].append(key)
            continue
        root, record = records[key]
        ep = json.loads((root / key / "STATUS.json").read_text())
        assert ep["status"] == record["status"] and all(ep["initial_checks"].values())
        task_id = tasks.index(case["task"])
        valid = case["valid_index"]
        split = "CAL" if valid == valid_ids[-1] else "FIT"
        assert case["role"] == dict(stage=stage, split=split)
        requests = [json.loads(line) for line in (root / "server/requests.jsonl").read_text().splitlines()]
        requests = [x for x in requests if x["case"] == key and x["route"] == "policy"]
        chunks = sorted((root / "server" / key).glob("chunk-*.pt"))
        assert len(chunks) == len(requests) == ep["policy_chunks"] > 0
        seen = 0
        for index, (path, request) in enumerate(zip(chunks, requests)):
            assert time.time() < deadline
            data = torch.load(path, map_location="cpu", weights_only=False)
            assert data["schema"] == "robotwin-reached-spec-query/v1"
            assert data["case"] == key and data["chunk"] == index
            assert request["chunk"] == index + 1
            assert data["case_manifest"] == case
            assert data["generator"]["condition"] == generator
            identity = dict(metadata=data["generator"], checkpoint=data["checkpoint"])
            if generator_identity is None:
                generator_identity = identity
            assert identity == generator_identity
            assert data["source_kind"] == "selected reached root/parent only"
            assert data["full_conditions_for_teacher_requery"] is False
            assert data["tau"] == plan["tau"]
            assert request["counts"]["action_B2_update1_returned"] == 1
            audit = request["audit"]
            assert audit["counters"]["committed_scheduler_update_count"] == 50
            # Reconstruct expected reached nodes from the recorded actual route.
            identities = []
            for tree in audit["trees"]:
                identities.append((tree["step"], 0, tree["depth"]))
                if tree["depth"] >= 1:
                    identities.append((tree["step"] + 1, 1 + tree["parent"], tree["depth"]))
            rows = data["rows"]
            assert [(x["step"], x["node"], x["depth"]) for x in rows] == identities
            assert len({x["step"] for x in rows}) == len(rows)
            for row in rows:
                assert row["source"] == "actual_reached_root_or_selected_parent"
                assert 0 <= row["step"] < 50
                for field in ("state", "previous", "target"):
                    value = row[field]
                    assert value.numel() == 512 and value.dtype == torch.bfloat16
                    assert torch.isfinite(value).all()
                    cols[split][field].append(value.reshape(512).float())
                sigma, nxt, t = map(float, (row["sigma"], row["next_sigma"], row["timestep"]))
                assert 0 <= nxt <= sigma <= 1 and abs(t - 1000 * sigma) < .002
                cols[split]["scalars"].append(torch.tensor([sigma, nxt, t / 1000]))
                for field, value in (("task", task_id), ("episode", task_id * 100 + valid),
                                     ("chunk", index), ("step", row["step"])):
                    cols[split][field].append(value)
                seen += 1
            references.append(dict(path=str(path), bytes=path.stat().st_size,
                                   mtime_ns=path.stat().st_mtime_ns, case=key, rows=len(rows)))
        report["episodes"].append(dict(case=key, split=split, rows=seen, success=ep["success"],
            actual_commands=ep["environment_commands"], failure_penalty=ep["failure_penalty"],
            horizon=ep["task_step_limit"], policy_chunks=ep["policy_chunks"]))
        save()
    report["status"] = "PARTIAL_FIXED_PANEL_NO_TRAINING_PACK"
    if not report["missing"]:
        data = {s: {k: torch.stack(v) if k in ("state", "previous", "target", "scalars")
                    else torch.tensor(v, dtype=torch.int64) for k, v in c.items()}
                for s, c in cols.items()}
        assert data["FIT"]["episode"].unique().numel() == 12
        assert data["CAL"]["episode"].unique().numel() == 6
        assert not set(data["FIT"]["episode"].tolist()) & set(data["CAL"]["episode"].tolist())
        torch.save(dict(schema="robotwin-reached18/v1", stage=stage, generator=generator,
                        generator_identity=generator_identity, data=data,
                        episodes=report["episodes"]), out / "observed18.pt")
        report["status"] = "PASS_COMPLETE_FIXED18_REACHED_SUPERVISION"
    report.update(FIT_rows=len(cols["FIT"]["state"]), CAL_rows=len(cols["CAL"]["state"]),
                  independent_test_complete=False, all_complete_failures_retained=True)
    (out / "SHARD-REFERENCES.json").write_text(json.dumps(references, indent=2) + "\n")
except BaseException as exc:
    report.update(status="FAILED_REACHED_DATA_AUDIT", error=repr(exc),
                  traceback=traceback.format_exc())
finally:
    report["elapsed_seconds"] = time.time() - started
    save()
if report["status"] == "FAILED_REACHED_DATA_AUDIT":
    raise SystemExit(1)

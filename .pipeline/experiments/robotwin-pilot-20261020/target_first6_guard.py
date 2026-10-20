"""Bounded queue: complete frozen manifest -> original Target first six FIT.

This queue stops here. Speculative collection/training is not yet implemented.
"""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import os
import signal
import subprocess
import time
import traceback

R = Path("/root/cxz/robotwin")
J = R / "runs/target-first6-3"
J.mkdir(exist_ok=False)
os.sched_setaffinity(0, {18})
auth = json.loads((R / "runs/AUTHORIZATION.json").read_text())
start = time.time()
previous = json.loads((R / "runs/target-first6-1/RESERVATION.json").read_text())
deadline = previous["absolute_deadline"]
cleanup_reserve = 900
state = {"status": "WAITING_FIXED_MANIFEST", "start": start, "deadline": deadline,
         "GPU_lifecycle_seconds": 0, "episodes": [], "owned": [],
         "next_stage_implemented": False}
workers = []
seen = {}
gpu_start = None


def save():
    p = J / "STATUS.json"
    p.with_suffix(".tmp").write_text(json.dumps(state, indent=2) + "\n")
    p.with_suffix(".tmp").replace(p)


def identity(pid):
    return {"pid": pid, "birth_ticks": int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]),
            "pgid": os.getpgid(pid)}


def same(row):
    try:
        return identity(row["pid"]) == row
    except (FileNotFoundError, ProcessLookupError):
        return False


def discover():
    groups = {r["pgid"] for r in seen.values()}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            row = identity(int(p.name))
            if row["pgid"] in groups:
                seen[row["pid"]] = row
        except (FileNotFoundError, ProcessLookupError):
            pass


def cleanup():
    discover()
    for row in seen.values():
        if same(row):
            os.kill(row["pid"], signal.SIGTERM)
    end = time.time() + 15
    for worker in workers:
        if worker.poll() is None:
            try:
                worker.wait(timeout=max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                pass
    for row in seen.values():
        if same(row):
            os.kill(row["pid"], signal.SIGKILL)
    for worker in workers:
        worker.wait(timeout=15)


def launch(command, env, log):
    with log.open("x") as output:
        process = subprocess.Popen(command, cwd=R, env=env, stdout=output,
                                   stderr=output, start_new_session=True)
    workers.append(process)
    row = identity(process.pid)
    seen[row["pid"]] = row
    state["owned"] = list(seen.values())
    save()
    return process


def check_storage():
    size = sum(p.stat().st_size for p in J.rglob("*") if p.is_file())
    if size > 2 * 2**30:
        raise RuntimeError("First-six evidence storage cap")
    state["storage_bytes"] = size


try:
    retry = json.loads((R / "runs/RETRY-FIRSTCASE-AUTHORIZATION.json").read_text())
    if retry.get("allowed") is not True or retry.get("case") != "place_object_basket-valid00":
        raise RuntimeError("Explicit first-case retry authorization missing")
    failed = json.loads((R / "runs/target-first6-2/place_object_basket-valid00/STATUS.json").read_text())
    if failed["status"] != "PARTIAL_OR_INVALID" or failed["environment_commands"] != 0:
        raise RuntimeError("Authorized failed-attempt identity differs")
    (J / "RESERVATION.json").write_text(json.dumps({
        "goal": "Original Target six complete FIT episodes, one per fixed task, with teacher supervision",
        "sources": auth["versions"], "input": "/root/cxz/robotwin/runs/prefilter-4/STATUS.json",
        "continuation_of": "target-first6-2; user-authorized first-case retry, other five unattempted",
        "retry_authorization": retry,
        "case_selection": "valid_index=0 for each task; fixed before seeing policy results",
        "wall_clock_cap_seconds": 7200, "absolute_deadline": deadline,
        "per_episode_cap_seconds": 1800, "per_episode_teacher_attempts": 8000,
        "episodes_cap": 6, "storage_cap_bytes": 2 * 2**30,
        "GPU": auth["GPU_uuid"], "server_cpus": [20, 21], "client_cpus": [16, 17],
        "original_T0": auth["T0"], "original_preparation_deadline": auth["CPU_preparation_deadline"],
        "evaluation_seed": 0, "reset": "original GPU BF16 full512 bank, then T5 CPU offload",
        "source_changes": "independent observer wrappers; original source not overwritten",
        "cleanup_reserve_seconds": cleanup_reserve,
        "cleanup": "PID+birth checked; no automatic retry; do not start next chain after first6"
    }, indent=2) + "\n")
    save()
    while True:
        if time.time() + 1800 + cleanup_reserve >= deadline:
            raise TimeoutError("Insufficient lifetime after manifest wait")
        control = R / "runs/prefilter-control-7/STATUS.json"
        if control.exists():
            d = json.loads(control.read_text())
            if d["status"] == "COMPLETE_EXPERT_MANIFEST":
                break
            if d["status"].startswith(("FAILED", "STOPPED")):
                raise RuntimeError("Expert manifest dependency failed; no automatic retry")
        time.sleep(5)
    if time.time() >= datetime.fromisoformat(auth["CPU_preparation_deadline"]).timestamp():
        raise TimeoutError("Original preparation deadline expired")
    lock = (R / "runs/GPU1-LEASE.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = json.loads((R / "runs/GPU-LEDGER.json").read_text())
    if ledger["consumed_seconds"] + deadline - time.time() > auth["GPU_lifecycle_cap_seconds"]:
        raise RuntimeError("Original GPU lifecycle budget insufficient")
    contexts = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    if any(line.startswith(auth["GPU_uuid"]) for line in contexts.splitlines()):
        raise RuntimeError("GPU1 occupied after expert cleanup")
    manifest = json.loads((R / "runs/prefilter-4/STATUS.json").read_text())
    if [t["task"] for t in manifest["tasks"]] != auth["tasks"]:
        raise RuntimeError("Fixed task coverage differs")
    cases = [t["valid"][0] for t in manifest["tasks"]]
    if any(len(t["valid"]) != 19 for t in manifest["tasks"]):
        raise RuntimeError("Manifest incomplete")
    if any(c["valid_index"] != 0 or c["role"] != {"stage": "TARGET", "split": "FIT"} for c in cases):
        raise RuntimeError("First-six split differs")
    (J / "INPUT-MANIFEST.json").write_text(json.dumps(cases, indent=2) + "\n")
    for directory in ("tmp", "cache/inductor", "cache/matplotlib", "cache/torch_extensions"):
        (R / directory).mkdir(parents=True, exist_ok=True)
    common = {**os.environ, "CUDA_VISIBLE_DEVICES": auth["GPU_uuid"], "OMP_NUM_THREADS": "2",
              "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
              "XDG_CACHE_HOME": str(R / "cache"),
              "HF_HOME": str(R / "cache/hf"), "HF_HUB_OFFLINE": "1",
              "TMPDIR": str(R / "tmp"), "CUDA_CACHE_PATH": str(R / "cache/cuda"),
              "TORCHINDUCTOR_CACHE_DIR": str(R / "cache/inductor"),
              "TORCH_EXTENSIONS_DIR": str(R / "cache/torch_extensions"),
              "MPLCONFIGDIR": str(R / "cache/matplotlib")}
    client_env = {**common, **json.loads((R / "runs/CLIENT-ENVIRONMENT.json").read_text()),
                  "CUDA_VISIBLE_DEVICES": auth["GPU_uuid"], "PYTHONPATH": str(R / "runs"),
                  "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    if any("/compat/" in p for p in client_env["LD_LIBRARY_PATH"].split(":")):
        raise RuntimeError("Client driver repair not retained")
    model_env = {**common, "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
                 "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "34961",
                 "PYTHONPATH": str(R / "source/lingbot-va-pinned")}
    gpu_start = time.time()
    state.update(status="LOADING_TARGET_AND_BANK", GPU_start=gpu_start,
                 CPU_wait_seconds=gpu_start - start)
    server = launch(["/root/cxz/envs/va/bin/python", "-B", str(R / "runs/target_server.py"),
                     str(J / "server")], model_env, J / "server.log")
    ready_deadline = min(time.time() + 900, deadline - 1800 - cleanup_reserve)
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"Target server exited before ready: {server.returncode}")
        p = J / "server/STATUS.json"
        if p.exists():
            d = json.loads(p.read_text())
            state["server_stage"] = d["status"]
            if d["status"] == "READY":
                break
            if d["status"] == "FAILED":
                raise RuntimeError("Target initialization failed: " + d.get("error", ""))
        if time.time() >= ready_deadline:
            raise TimeoutError("Target setup/bank cap")
        save()
        time.sleep(5)
    for case in cases:
        if time.time() + 1800 + cleanup_reserve >= deadline:
            state.update(status="PARTIAL_BUDGET_NO_NEXT_EPISODE",
                         stop_reason="Whole episode and cleanup reserve unavailable")
            break
        name = case["task"] + "-valid00"
        case_path = J / f"{name}-CASE.json"
        case_path.write_text(json.dumps(case, indent=2))
        out = J / name
        state.update(status="RUNNING_TARGET_COMPLETE_EPISODE", active_case=name)
        client = launch([str(R / "envs/client/bin/python"), "-B",
                         str(R / "runs/target_client.py"), str(case_path), str(out)],
                        client_env, J / f"{name}.log")
        case_deadline = time.time() + 1800
        while client.poll() is None:
            discover()
            if server.poll() is not None:
                raise RuntimeError("Server terminated during complete episode")
            if time.time() >= case_deadline:
                raise TimeoutError("Complete episode lifecycle cap; preserve PARTIAL")
            check_storage()
            state.update(owned=list(seen.values()), GPU_lifecycle_seconds=time.time() - gpu_start)
            save()
            time.sleep(5)
        terminal = json.loads((out / "STATUS.json").read_text())
        state["episodes"].append({"case": name, "client_returncode": client.returncode,
                                   "status": terminal["status"], "success": terminal.get("success"),
                                   "environment_commands": terminal["environment_commands"],
                                   "policy_chunks": terminal["policy_chunks"],
                                   "lifecycle_seconds": terminal["lifecycle_seconds"]})
        save()
        if client.returncode != 0 or not terminal["status"].startswith("COMPLETE_"):
            raise RuntimeError("Complete episode infrastructure/identity failure; no automatic retry")
    else:
        state["status"] = "COMPLETE_FIRST6_TARGET_ONLY"
except BaseException as error:
    state.update(status="FAILED_OR_PARTIAL", error=repr(error), traceback=traceback.format_exc())
finally:
    cleanup()
    state.update(end=time.time(), lifecycle_seconds=time.time() - start,
                 remaining_owned=[row for row in seen.values() if same(row)],
                 returncodes=[p.returncode for p in workers])
    if gpu_start is not None:
        state["GPU_lifecycle_seconds"] = time.time() - gpu_start
        with (R / "runs/GPU-LEDGER.lock").open("a") as ledger_lock:
            fcntl.flock(ledger_lock, fcntl.LOCK_EX)
            ledger = json.loads((R / "runs/GPU-LEDGER.json").read_text())
            ledger["consumed_seconds"] += state["GPU_lifecycle_seconds"]
            ledger["jobs"].append({"namespace": str(J), **state})
            (R / "runs/GPU-LEDGER.tmp").write_text(json.dumps(ledger, indent=2) + "\n")
            (R / "runs/GPU-LEDGER.tmp").replace(R / "runs/GPU-LEDGER.json")
    save()

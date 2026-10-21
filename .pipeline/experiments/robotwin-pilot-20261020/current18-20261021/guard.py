"""Newly authorized CURRENT18 queue; fixed CONT3000, no robot retry."""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import os
import signal
import subprocess
import time
import traceback
import sys

R = Path("/root/cxz/robotwin")
B = R / "runs/tf-migration-20261021"
J = Path(sys.argv[1])
if (J/"STATUS.json").exists():
    raise RuntimeError("Attempt already exists; automatic restart forbidden")
reg=json.loads((J/"REGISTRATION.json").read_text())
os.sched_setaffinity(0, {18})
auth = json.loads((R / "runs/AUTHORIZATION.json").read_text())
wait_start=time.time()
while True:
    dep=json.loads((B/'INTERFACE-GUARD-r3.json').read_text())
    if dep['status']=='COMPLETE_INTERFACE_GATE':break
    if dep['status'].startswith('FAILED'):
        (J/'STATUS.json').write_text(json.dumps(dict(status='BLOCKED_INTERFACE_PREREQUISITE',wait_seconds=time.time()-wait_start,error=dep.get('error'))));raise SystemExit(1)
    if time.time()+2700>=datetime.fromisoformat(reg['must_finish_before']).timestamp():
        (J/'STATUS.json').write_text(json.dumps(dict(status='BLOCKED_INSUFFICIENT_FULL_EPISODE_RESERVE',wait_seconds=time.time()-wait_start)));raise SystemExit(1)
    time.sleep(30)
start = time.time()
deadline = min(start+reg["execution_lifecycle_cap_seconds"],datetime.fromisoformat(reg["must_finish_before"]).timestamp())
cleanup_reserve = 900
state = {"status": "PREPARING_CURRENT18", "start": start, "deadline": deadline,
         "GPU_lifecycle_seconds": 0, "episodes": [], "owned": [],
         "next_stage_implemented": False, "waiting_seconds": time.time()-wait_start}
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
    if size > reg["storage_cap_bytes"]:
        raise RuntimeError("CURRENT18 evidence storage cap")
    state["storage_bytes"] = size


try:
    assert json.loads((R/"runs/branch-cal-20261021/CONT3000/RESULT.json").read_text())["status"]=="PASS_FINITE_GPU_ADAPTER_DECODE_NOT_CLOSED_LOOP"
    assert json.loads((J/"PLAN.json").read_text())["checkpoint"] == json.loads((R/"runs/branch-cal-20261021/CONT3000/RESULT.json").read_text())["checkpoint"]
    if time.time()>=datetime.fromisoformat(reg["prepare_deadline"]).timestamp():raise TimeoutError("New preparation deadline expired")
    (J/"RESERVATION.json").write_text(json.dumps({**reg,"actual_start":start,"absolute_deadline":deadline,"original_T0":auth["T0"],"original_preparation_deadline":auth["CPU_preparation_deadline"],"prior_first6":str(R/"runs/target-first6-3"),"reset_bank":"reuse exact full512 BF16 bank; no new prompt encoding"},indent=2)+"\n")
    save()
    lock = (R / "runs/GPU1-LEASE.lock").open("a")
    second_lock = (R / "runs/GPU1.lease").open("a")
    fcntl.flock(second_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    all_cases = [t["valid"][i] for i in range(11,14) for t in manifest["tasks"]]
    cases = all_cases # All 18 fixed CURRENT cases have never started.
    if any(len(t["valid"]) != 19 for t in manifest["tasks"]):
        raise RuntimeError("Manifest incomplete")
    if cases!=json.loads((J/"INPUT-MANIFEST.json").read_text()):raise RuntimeError("Frozen remaining cases differ")
    if len(cases)!=18 or sum(c["role"]=={"stage":"CURRENT","split":"FIT"} for c in cases)!=12:raise RuntimeError("Fixed CURRENT FIT12/CAL6 coverage differs")
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
                  "CUDA_VISIBLE_DEVICES": auth["GPU_uuid"], "PYTHONPATH": str(B)+os.pathsep+str(R / "runs"),
                  "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    if any("/compat/" in p for p in client_env["LD_LIBRARY_PATH"].split(":")):
        raise RuntimeError("Client driver repair not retained")
    model_env = {**common, "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
                 "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "34971",
                 "PYTHONPATH": str(R / "source/lingbot-va-pinned")}
    gpu_start = time.time()
    state.update(status="LOADING_TARGET_AND_BANK", GPU_start=gpu_start,
                 CPU_wait_seconds=gpu_start - start)
    server = launch(["/root/cxz/envs/va/bin/python", "-B", str(B / "collection_server.py"),
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
        name = case["task"] + f"-valid{case['valid_index']:02d}"
        case_path = J / f"{name}-CASE.json"
        case_path.write_text(json.dumps(case, indent=2))
        out = J / name
        state.update(status="RUNNING_CONT3000_COMPLETE_EPISODE", active_case=name)
        client = launch([str(R / "envs/client/bin/python"), "-B",
                         str(B / "collection_client.py"), str(case_path), str(out)],
                        client_env, J / f"{name}.log")
        case_deadline = time.time() + 1800
        while client.poll() is None:
            discover()
            if server.poll() is not None:
                raise RuntimeError("Server terminated during complete episode")
            if time.time() >= case_deadline:
                raise TimeoutError("Complete episode lifecycle cap; preserve PARTIAL")
            check_storage()
            server_state=json.loads((J/"server/STATUS.json").read_text())
            if server_state["teacher_attempted"]>reg["new_teacher_attempts_cap"]:raise RuntimeError("Teacher budget exceeded")
            state.update(owned=list(seen.values()),teacher_attempted=server_state["teacher_attempted"],teacher_returned=server_state["teacher_returned"],GPU_lifecycle_seconds=time.time() - gpu_start)
            save()
            time.sleep(5)
        terminal = json.loads((out / "STATUS.json").read_text())
        state["episodes"].append({"case": name,"role":case["role"],"initial":case["initial"], "client_returncode": client.returncode,
                                   "status": terminal["status"], "success": terminal.get("success"),
                                   "environment_commands": terminal["environment_commands"],
                                   "policy_chunks": terminal["policy_chunks"],
                                   "lifecycle_seconds": terminal["lifecycle_seconds"]})
        save()
        if client.returncode != 0 or not terminal["status"].startswith("COMPLETE_"):
            raise RuntimeError("Complete episode infrastructure/identity failure; no automatic retry")
    else:
        state["status"] = "COMPLETE_CURRENT18_VISITED_COLLECTION"
except BaseException as error:
    state.update(status="FAILED_OR_PARTIAL", error=repr(error), traceback=traceback.format_exc())
finally:
    cleanup()
    p=J/"server/STATUS.json"
    if p.exists():
        sd=json.loads(p.read_text());state.update(teacher_attempted=sd["teacher_attempted"],teacher_returned=sd["teacher_returned"],policy_chunks=sd["policy_chunks"])
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

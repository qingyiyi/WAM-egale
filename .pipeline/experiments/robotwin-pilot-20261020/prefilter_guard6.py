"""Resume the authorized real expert manifest after the library-path repair."""
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
J = R / "runs/prefilter-control-6"
W = R / "runs/prefilter-3"
J.mkdir(exist_ok=False)
os.sched_setaffinity(0, {18})
auth = json.loads((R / "runs/AUTHORIZATION.json").read_text())
state = {"status": "PREPARING", "start": time.time(), "policy_episodes": 0,
         "teacher_forwards": 0, "owned": [], "GPU_lifecycle_seconds": 0}
proc = None
gpu_start = None
seen = {}


def save():
    p = J / "STATUS.json"
    p.with_suffix(".tmp").write_text(json.dumps(state, indent=2) + "\n")
    p.with_suffix(".tmp").replace(p)


def identity(pid):
    return {"pid": pid, "pgid": os.getpgid(pid),
            "birth_ticks": int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])}


def same(rec):
    try:
        return identity(rec["pid"]) == rec
    except (FileNotFoundError, ProcessLookupError):
        return False


def discover():
    if not seen:
        return
    group = next(iter(seen.values()))["pgid"]
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                rec = identity(int(p.name))
                if rec["pgid"] == group:
                    seen[rec["pid"]] = rec
            except (FileNotFoundError, ProcessLookupError):
                pass


def stop_owned():
    discover()
    for rec in seen.values():
        if same(rec):
            os.kill(rec["pid"], signal.SIGTERM)
    if proc is not None and proc.poll() is None:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    for rec in seen.values():
        if same(rec):
            os.kill(rec["pid"], signal.SIGKILL)
    if proc is not None:
        proc.wait(timeout=15)


try:
    lock = (R / "runs/GPU1-LEASE.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repair = json.loads((R / "runs/driver-interop-base-2/APPLIED-REPAIR.json").read_text())
    ledger = json.loads((R / "runs/GPU-LEDGER.json").read_text())
    deadline = min(time.time() + 3600,
                   datetime.fromisoformat(auth["CPU_preparation_deadline"]).timestamp() - 300)
    if deadline - time.time() < 3300:
        raise RuntimeError("Insufficient original preparation window for fixed manifest")
    if ledger["consumed_seconds"] + 3900 + 900 > auth["GPU_lifecycle_cap_seconds"]:
        raise RuntimeError("Original GPU budget cannot reserve preparation and cleanup")
    contexts = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    if any(line.startswith(auth["GPU_uuid"]) for line in contexts.splitlines()):
        raise RuntimeError("GPU1 occupied; no launch over existing context")
    env = {**os.environ, **json.loads((R / "runs/CLIENT-ENVIRONMENT.json").read_text()),
           "CUDA_VISIBLE_DEVICES": auth["GPU_uuid"], "PYTHONPATH": str(R / "runs"),
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "HF_HUB_OFFLINE": "1", "CUDA_CACHE_PATH": str(R / "cache/cuda"),
           "WARP_CACHE_PATH": str(R / "cache/warp")}
    if any("/compat/" in p for p in env["LD_LIBRARY_PATH"].split(":")):
        raise RuntimeError("Repaired client must not inherit forward-compat libcuda")
    (J / "RESERVATION.json").write_text(json.dumps({
        "goal": "Freeze six tasks x 19 expert-valid seeds before any policy selection",
        "authorization": "User resumed bounded RoboTwin migration after repair",
        "source_versions": auth["versions"], "input_manifest": auth["tasks"],
        "GPU": auth["GPU_uuid"], "wall_clock_cap_seconds": 3600,
        "deadline": deadline, "cleanup_reserve_seconds": 300,
        "storage_cap_bytes": 2 * 2**30, "expert_attempt_cap": 1200,
        "policy_episodes": 0, "teacher_forwards": 0,
        "evidence_path": str(W), "cleanup": "PID and birth identity checked; no automatic retry",
        "original_T0": auth["T0"], "original_CPU_deadline": auth["CPU_preparation_deadline"]
    }, indent=2) + "\n")
    gpu_start = time.time()
    with (J / "worker.log").open("x") as output:
        proc = subprocess.Popen(
            [str(R / "envs/client/bin/python"), "-B", str(R / "runs/prefilter.py"), str(W)],
            cwd=R, env=env, stdout=output, stderr=output, start_new_session=True)
    owned = identity(proc.pid)
    seen[proc.pid] = owned
    state.update(status="RUNNING_EXPERT_MANIFEST", root_owned=owned, GPU_start=gpu_start)
    save()
    while proc.poll() is None:
        discover()
        state.update(owned=list(seen.values()), GPU_lifecycle_seconds=time.time() - gpu_start)
        save()
        if time.time() >= deadline:
            state["stop_reason"] = "WALL_CLOCK_CAP"
            break
        size = sum(p.stat().st_size for p in W.rglob("*") if p.is_file()) if W.exists() else 0
        if size > 2 * 2**30:
            state["stop_reason"] = "STORAGE_CAP"
            break
        tail = (J / "worker.log").read_text(errors="replace")[-12000:]
        if "OIDN Error:" in tail or "illegal memory access" in tail:
            state["stop_reason"] = "RENDER_INFRASTRUCTURE_ERROR"
            break
        time.sleep(5)
    stop_owned()
    terminal = json.loads((W / "STATUS.json").read_text()) if (W / "STATUS.json").exists() else {}
    state.update(returncode=proc.returncode, root_reaped=True,
                 worker_terminal=terminal.get("status", "MISSING"),
                 expert_attempts=terminal.get("expert_attempts"),
                 valid_cases=sum(len(t["valid"]) for t in terminal.get("tasks", [])))
    state["status"] = (
        "COMPLETE_EXPERT_MANIFEST"
        if proc.returncode == 0 and state["worker_terminal"] == "COMPLETE_FIXED_SCENES_NO_POLICY_QUALITY_RESULT"
        else "FAILED_OR_PARTIAL")
except BaseException as error:
    state.update(status="FAILED_OR_BLOCKED", error=repr(error), traceback=traceback.format_exc())
    if proc is not None:
        stop_owned()
finally:
    state.update(end=time.time(), lifecycle_seconds=time.time() - state["start"],
                 remaining_owned=[r for r in seen.values() if same(r)])
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

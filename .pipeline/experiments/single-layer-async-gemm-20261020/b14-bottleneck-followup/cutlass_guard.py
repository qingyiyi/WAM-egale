"""CPU-only build then bounded GPU0 replay, with owned-process cleanup."""
from pathlib import Path
import datetime
import json
import os
import signal
import subprocess
import time
import traceback

B = Path(__file__).resolve().parent
R = Path("/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized")
UUID = "GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726"
PY = "/root/cxz/envs/va/bin/python"
os.sched_setaffinity(0, {4, 5})
start = time.time()
deadline = min(start + 1800, datetime.datetime.fromisoformat(
    "2026-10-20T20:20:28+08:00").timestamp())
state = dict(status="RUNNING", start=start, deadline=deadline, jobs=[],
             prior_full_block_calls=1392, new_full_block_cap=105, primitive_cap=0)
for name in ("tmp", "cache", "build", "generated"):
    (B / name).mkdir(exist_ok=True)
env = dict(os.environ, CUDA_HOME="/root/cxz/ops/persistent-tile-20261019/cuda-12.4",
           WAM_FA2_BUILD_ROOT=str(B / "build"), TORCH_CUDA_ARCH_LIST="8.0",
           TMPDIR=str(B / "tmp"), XDG_CACHE_HOME=str(B / "cache"),
           CUDA_CACHE_PATH=str(B / "cache/cuda"), MAX_JOBS="2",
           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=str(R.parents[1]), GPU_PRELAUNCH_EMPTY="1")


def save():
    (B / "CUTLASS-STATUS.json").write_text(json.dumps(state, indent=2) + "\n")


def identity(pid):
    return dict(pid=pid, birth_ticks=int(Path(f"/proc/{pid}/stat").read_text()
                .rsplit(")", 1)[1].split()[19]), pgid=os.getpgid(pid))


def same(row):
    try:
        return identity(row["pid"]) == row
    except (FileNotFoundError, ProcessLookupError):
        return False


try:
    for name, cmd, cap, gpu in [
        ("cutlass-build", [PY, str(R / "build_whole_layer.py"), "build_ext", "--build-lib", str(R), "--build-temp", str(B / "cutlass-build")], 900, False),
        ("cutlass-gate", [PY, "-B", str(B / "cutlass_gate.py")], 300, True),
        ("cutlass-timing-1", [PY, "-B", str(B / "cutlass_timing.py"), "1"], 300, True),
        ("cutlass-timing-2", [PY, "-B", str(B / "cutlass_timing.py"), "2"], 300, True)]:
        if name.startswith("cutlass-timing"):
            gate = json.loads((B / "CUTLASS-GATE-RESULT.json").read_text())
            if not gate.get("all_pass", False):
                raise RuntimeError("Numerical gate failed; no timing")
        cap = min(cap, int(deadline - time.time() - 60))
        if cap <= 0:
            raise TimeoutError("No lifecycle budget")
        if gpu:
            contexts = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                 "--format=csv,noheader"], text=True)
            if any(x.startswith(UUID) for x in contexts.splitlines()):
                raise RuntimeError("GPU0 occupied; no worker launched")
        with (B / (name + ".log")).open("x") as log:
            worker = subprocess.Popen(cmd, cwd=R, stdout=log, stderr=log,
                env=dict(env, CUDA_VISIBLE_DEVICES=UUID if gpu else ""), start_new_session=True)
        row = dict(name=name, owned=identity(worker.pid), start=time.time(), gpu=gpu)
        state["jobs"].append(row)
        save()
        try:
            rc = worker.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            row["timeout"] = True
            if same(row["owned"]):
                os.killpg(row["owned"]["pgid"], signal.SIGTERM)
            try:
                rc = worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if same(row["owned"]):
                    os.killpg(row["owned"]["pgid"], signal.SIGKILL)
                rc = worker.wait(timeout=15)
        row.update(returncode=rc, end=time.time(), lifecycle_seconds=time.time()-row["start"],
                   root_reaped=True, root_identity_alive=same(row["owned"]))
        save()
        if rc != 0 or row.get("timeout"):
            raise RuntimeError(name + " failed; no automatic retry")
    results = [json.loads((B / n).read_text()) for n in ["CUTLASS-GATE-RESULT.json", "cutlass-timing-1-RESULT.json", "cutlass-timing-2-RESULT.json"]]
    state["full_block_calls"] = sum(x["full_block_calls"] for x in results)
    if any(x["status"].startswith("FAILED") for x in results) or state["full_block_calls"] > 105:
        raise RuntimeError("Worker failure or budget exceeded")
    state["status"] = "COMPLETE"
except BaseException as error:
    state.update(status="FAILED_OR_PARTIAL", error=repr(error), traceback=traceback.format_exc())
finally:
    state.update(end=time.time(), lifecycle_seconds=time.time()-start)
    save()

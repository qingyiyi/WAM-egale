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
reg=json.loads((B/'REGISTRATION.json').read_text())
deadline=datetime.datetime.fromisoformat(reg['deadline']).timestamp()
state = dict(status='RUNNING',start=start,deadline=deadline,jobs=[],new_full_block_cap=24,primitive_cap=150)
for name in ("tmp", "cache", "build", "generated"):
    (B / name).mkdir(exist_ok=True)
env = dict(os.environ, CUDA_HOME="/root/cxz/ops/persistent-tile-20261019/cuda-12.4",
           WAM_FA2_BUILD_ROOT=str(B / "build"), TORCH_CUDA_ARCH_LIST="8.0",
           TMPDIR=str(B / "tmp"), XDG_CACHE_HOME=str(B / "cache"),
           CUDA_CACHE_PATH=str(B / "cache/cuda"), MAX_JOBS="2",
           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=str(R.parents[1]),WAM_CONTEXT_TILE_M="128")


def save():
    (B / "STATUS.json").write_text(json.dumps(state, indent=2) + "\n")


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
        ("prepare", [PY, "-B", str(B / "prepare_profile.py")], 60, False),
        ("build", [PY, str(B / "build_profile.py"), "build_ext",
                   "--build-lib", str(B), "--build-temp", str(B / "build")], 900, False),
        ("profile", [PY, "-B", str(B / "profile_worker.py")], 900, True)]:
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
    result = json.loads((B / "RESULT.json").read_text())
    if result["status"] != "COMPLETE_CURRENT_B14_PROFILE":
        raise RuntimeError("Worker reported incomplete result")
    state["status"] = "COMPLETE"
except BaseException as error:
    state.update(status="FAILED_OR_PARTIAL", error=repr(error), traceback=traceback.format_exc())
finally:
    state.update(end=time.time(), lifecycle_seconds=time.time()-start)
    save()

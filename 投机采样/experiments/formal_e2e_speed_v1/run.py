"""Serial, AB/BA, local-only A_off/T natural episode timing campaign.

Only this explicit entrypoint can allocate a GPU. Prepare/analyze/tests are CPU.
Every episode gets a fresh server, so task timing and cold model startup remain
separate. Existing project locks prevent overlap with current GPU campaigns.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

from speed_contract import CLIENT, EXP, OWN, ROOT, VA, WORKTREE, read, sha, validate, write

for source in (EXP / "single_worker_chc_gpu_v4", EXP / "speculative_e2e_v3",
               EXP / "morning_onpolicy_long8_v4"):
    sys.path.append(str(source))
from resource_guard import NVML
from launch_identity import pin as pin_launch, bootstrap_key
from cleanup import roots as cleanup_roots
import ownership


class Campaign:
    def __init__(self, job: Path, plan: dict):
        self.job, self.plan = job, plan
        self.nv = NVML()
        self.pins = {}
        self.roots = []
        self.current = None
        self.started = time.monotonic()
        self.deadline = self.started + plan["max_run_wall_sec"]
        self.next_sample = 0.
        self.completed = []
        self.stage = "preflight"

    def environment(self, row: dict, out: Path):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("WAM_P0_", "P0_", "LINGBOT_", "FORMAL_SPEED_", "PREFIX_ACCEPT_"))}
        env.update({
            "FORMAL_SPEED_JOB": str(self.job), "FORMAL_SPEED_KEY": row["key"],
            "PREFIX_ACCEPT_ENABLED": "1", "PREFIX_ACCEPT_GPU_APPROVED": "1",
            "CUDA_VISIBLE_DEVICES": "0", "LINGBOT_SEED": str(self.plan["seed"]),
            "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
            "LINGBOT_DISABLE_FSDP": "1", "LINGBOT_CACHE_MODE": "original",
            "LINGBOT_ATTN_MODE": "torch", "LINGBOT_NSYS_PROFILE": "off",
            "LINGBOT_VA_LIBERO_SUITE": "libero_10",
            "LINGBOT_VA_LIBERO_CHECKPOINT": self.plan["target_checkpoint_path"],
            "LINGBOT_VA_LIBERO_CHECKPOINT_ID": self.plan["target_checkpoint_id"],
            "LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY": self.plan["registry"],
            "WAM_VAE_DEVICE": "cuda", "WAM_P0_PHASE_PROFILE": "0",
            "WAM_P0_ACTION_DRAFT_MODE": "off" if row["arm"] == "T" else self.plan["mode"],
            "WAM_P0_ACTION_DRAFT_SELECTOR": "global",
            "WAM_P0_ACTION_DRAFT_THRESHOLD": str(self.plan["threshold"]),
            "WAM_P0_ACTION_DRAFT_B2_CHECKPOINT": self.plan["draft_checkpoint"],
            "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0", "OMP_NUM_THREADS": "7",
            "MKL_NUM_THREADS": "7", "OPENBLAS_NUM_THREADS": "7",
            "NUMEXPR_NUM_THREADS": "7", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        })
        env.pop("P0_NATIVE_DIAGNOSTIC_DIR", None)
        env.pop("WAM_P0_STATE_TEACHER_DIR", None)
        return env

    def telemetry(self, stage):
        sample = self.nv.sample()
        gpu = subprocess.run([
            "nvidia-smi", "-i", "0",
            "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.gr,clocks.mem",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        apps = subprocess.run([
            "nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        if "A100-SXM4-40GB" not in gpu.stdout:
            raise RuntimeError("formal speed must use local A100-SXM4-40GB; never AutoDL")
        foreign = []
        for entry in sample["compute"] + sample["graphics"]:
            pid = entry["pid"]
            try:
                proc = ownership.read_process(pid)
                owned = ownership.is_owned(proc, self.pins)
                uid = proc.uid
                user = pwd.getpwuid(uid).pw_name
            except (FileNotFoundError, ProcessLookupError):
                owned, uid, user = False, None, None
            if not owned:
                foreign.append(dict(pid=pid, uid=uid, user=user,
                                    memory_bytes=entry["memory_bytes"]))
        row = dict(monotonic=time.monotonic(), stage=stage, nvml=sample,
                   nvidia_smi_gpu_csv=gpu.stdout.strip(),
                   nvidia_smi_compute_apps_csv=apps.stdout.strip(),
                   foreign=foreign, own_root_pids=sorted(self.pins))
        with (self.job / "GPU-TELEMETRY.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if foreign:
            raise RuntimeError("foreign GPU process during formal timing: " + str(foreign))
        return row

    def idle(self):
        count = 0
        limit = time.monotonic() + 180
        while count < self.plan["idle_admission_samples"]:
            row = self.telemetry("idle_admission")
            nv = row["nvml"]
            quiet = (nv["memory_used_bytes"] == 0 and not nv["compute"]
                     and not nv["graphics"] and nv["utilization_percent"] <= 2)
            count = count + 1 if quiet else 0
            if time.monotonic() > limit:
                raise TimeoutError("exclusive idle GPU window unavailable")
            time.sleep(1)

    def guard(self, stage):
        if time.monotonic() > self.deadline:
            raise TimeoutError("bounded formal timing campaign wall deadline")
        if (self.job / "STOP_REQUESTED").exists():
            raise RuntimeError("STOP_REQUESTED")
        if time.monotonic() >= self.next_sample:
            self.telemetry(stage)
            self.next_sample = time.monotonic() + self.plan["sample_interval_sec"]

    def spawn(self, cmd, log: Path, env):
        with log.open("x") as stream:
            proc = subprocess.Popen(cmd, cwd=WORKTREE, env=env,
                                    stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        self.roots.append(proc)
        boot = {}
        try:
            self.pins[proc.pid] = pin_launch(proc, cmd, on_boot=lambda info: boot.update(info=info))
        except BaseException:
            # An early exec failure may occur before exact argv stabilizes.
            # Reap only the spawned PID if the bootstrap identity is unchanged;
            # never signal an unpinned process group or another user's job.
            if proc.poll() is None and "info" in boot:
                now = ownership.read_process(proc.pid)
                if bootstrap_key(now) != bootstrap_key(boot["info"]):
                    raise RuntimeError("launcher bootstrap PID changed; signal refused")
                os.kill(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    now = ownership.read_process(proc.pid)
                    if bootstrap_key(now) != bootstrap_key(boot["info"]):
                        raise RuntimeError("bootstrap PID changed before SIGKILL")
                    os.kill(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
            raise
        return proc

    def wait_ready(self, server, out: Path):
        limit = min(time.monotonic() + 600, self.deadline)
        while True:
            self.guard("server_loading")
            if server.poll() is not None:
                raise RuntimeError("model server exited before READY; see server.log")
            if (out / "READY.json").exists():
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{self.plan['port']}/healthz", timeout=1) as response:
                        if response.status == 200:
                            return read(out / "READY.json")
                except (OSError, TimeoutError):
                    pass
            if time.monotonic() > limit:
                raise TimeoutError("server readiness exceeded 600 seconds")
            time.sleep(1)

    def one(self, row):
        out = self.job / "episodes" / row["key"]
        out.mkdir(parents=True, exist_ok=False)
        env = self.environment(row, out)
        server = client = None
        started = time.monotonic()
        self.stage = "episode_" + row["key"]
        try:
            self.idle()
            server_cmd = [str(VA), "-m", "torch.distributed.run", "--nproc_per_node=1",
                          "--master_port=" + str(self.plan["master_port"]),
                          "--local-ranks-filter=0", "--tee=3",
                          str(OWN / "server_entry.py"), "--config-name", "libero",
                          "--port", str(self.plan["port"]), "--save_root", str(out / "server_output")]
            launch = time.monotonic()
            server = self.spawn(server_cmd, out / "server.log", env)
            ready = self.wait_ready(server, out)
            ready_observed = time.monotonic()
            if ready["key"] != row["key"] or ready["arm"] != row["arm"]:
                raise ValueError("server READY identity mismatch")
            client_cmd = [str(CLIENT), str(OWN / "client_entry.py"),
                          "--port", str(self.plan["port"]),
                          "--master-port", str(self.plan["master_port"]),
                          "--benchmarks", "libero_10", "--task-indices", str(row["task"]),
                          "--episodes-per-task", "1", "--episode-start", str(row["init"]),
                          "--condition", row["arm"], "--campaign-id", self.job.name,
                          "--worker-id", row["key"], "--trace-mode", "off",
                          "--latency-benchmark", "--out-dir", str(out / "client"),
                          "--fail-fast", "--no-resume", "--worktree", str(WORKTREE)]
            client = self.spawn(client_cmd, out / "client.log", env)
            client_start = time.monotonic()
            limit = min(client_start + self.plan["per_episode_wall_timeout_sec"], self.deadline)
            while client.poll() is None:
                self.guard("natural_episode")
                if server.poll() is not None:
                    raise RuntimeError("server exited during episode")
                if time.monotonic() > limit:
                    raise TimeoutError("natural episode exceeded 1500 seconds")
                time.sleep(1)
            if client.returncode != 0:
                raise RuntimeError("LIBERO client exited nonzero; see client.log")
            client_done = time.monotonic()
            (out / "FINISH_REQUESTED").write_text("client complete; async drain outside task timer\n")
            close_limit = min(time.monotonic() + 90, self.deadline)
            while not (out / "FINISH_READY.json").exists():
                self.guard("post_episode_drain")
                if (out / "FINISH_FAILED.json").exists():
                    raise RuntimeError("server accounting close failed")
                if server.poll() is not None or time.monotonic() > close_limit:
                    raise RuntimeError("server failed to drain after episode")
                time.sleep(.5)
            result = dict(row=row, model_launch_to_READY_observed_sec=ready_observed-launch,
                          model_launch_to_READY_worker_sec=ready["ready_monotonic"]-launch,
                          server_ready_to_client_spawn_sec=client_start-ready_observed,
                          client_process_wall_sec=client_done-client_start,
                          cold_server_launch_to_client_exit_sec=client_done-launch,
                          model_init_sec=ready["target_server_init_sec"],
                          private_setup_sec=ready["private_setup_sec"],
                          client_status="completed", concurrent_model_workers=1,
                          concurrent_EGL_clients=1, model_reload_per_episode=True)
            write(out / "RUN-TIMING.json", result)
            self.completed.append(row["key"])
        finally:
            processes = [p for p in (client, server) if p is not None]
            if processes:
                closed = cleanup_roots(processes, self.pins,
                                       deadline=time.monotonic() + 25, wait_seconds=.5)
                write(out / "OWNED-CLEANUP.json", closed)
            self.next_sample = 0

    def run(self):
        try:
            for row in self.plan["schedule"]:
                self.one(row)
            write(self.job / "GPU-COLLECTION.json", dict(
                status="PASS_LOCAL_FORMAL_E2E_COLLECTION", completed=self.completed,
                expected=len(self.plan["schedule"]),
                wall_sec=time.monotonic()-self.started,
                no_foreign_GPU_processes=True))
        except BaseException as exc:
            write(self.job / "FAILED.json", dict(
                status="FAILED", stage=self.stage, completed=self.completed,
                error=repr(exc), traceback=traceback.format_exc(),
                wall_sec=time.monotonic()-self.started))
            raise
        finally:
            cleanup_roots(self.roots, self.pins,
                          deadline=time.monotonic() + 25, wait_seconds=.5)
            self.nv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    args = ap.parse_args()
    job = args.job.resolve()
    plan = validate(job)
    if any((job / file).exists() for file in ("GPU-COLLECTION.json", "FAILED.json")):
        raise ValueError("one-shot campaign already terminal; never resume or overwrite")
    # Full seven-file SHA verification is deliberately before GPU admission.
    for weight in plan["target_weights"]:
        if sha(Path(weight["path"])) != weight["sha256"]:
            raise ValueError("target model weight SHA changed: " + weight["path"])
    for port in (plan["port"], plan["master_port"]):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    locks = [ROOT / ".pipeline/.chunk-template-gpu.lock",
             ROOT / "投机采样/results/runtime/.speculative_minimal_gpu.lock",
             ROOT / "投机采样/results/runtime/.shared-prefix-GPU.lock"]
    opened = []
    try:
        for path in locks:
            file = path.open("a")
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            opened.append(file)
        Campaign(job, plan).run()
    finally:
        for file in reversed(opened):
            file.close()


if __name__ == "__main__":
    main()

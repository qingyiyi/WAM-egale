"""Bounded single-worker suite phase full-episode runner; no probes, tests or digests.

One persistent target per supplied GPU. All registered algorithm failures and passive audit differences are retained.
Only routing/resource/absolute limits interrupt work; no old quality early-stop.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from tf_suite_pipeline_v1.protocol import (ROOT, EXP, WORKTREE, PERMISSIONS, episode_dir,
    row as phase_row, run_rows, read, write, load, require)


class HardStop(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def process(pid):
    try:
        base = Path(f"/proc/{pid}")
        fields = (base / "stat").read_text().rpartition(") ")[2].split()
        return dict(pid=pid, birth_ticks=int(fields[19]), pgid=os.getpgid(pid), state=fields[0],
            uid=base.stat().st_uid)
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(pin):
    actual = process(pin["pid"])
    return (actual is not None and actual["birth_ticks"] == pin["birth_ticks"]
        and actual["pgid"] == pin["pgid"] == pin["pid"] and actual["uid"] == pin.get("uid", os.getuid()))


def forensic_groups(pins):
    """Observe membership only; never signal a group whose root proof is lost."""
    wanted = {p["pgid"]: p for p in pins}
    groups, errors = {str(g): [] for g in wanted}, []
    if not wanted:
        return dict(groups=[], scan_errors=[], scan_complete=True, observed_at=now())
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rpartition(") ")[2].split()
            pgid = int(fields[2])
            if pgid not in wanted:
                continue
            groups[str(pgid)].append(dict(pid=int(path.name), parent_pid=int(fields[1]), pgid=pgid,
                birth_ticks=int(fields[19]), uid=path.stat().st_uid, state=fields[0]))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as exc:
            errors.append(dict(pid=int(path.name), error=repr(exc)))
    return dict(groups=[dict(original_root_pin=wanted[int(pgid)], members=members,
        original_root_proof_currently_matches=matches(wanted[int(pgid)]),
        live_members=[m for m in members if m["state"] not in ("Z", "X")],
        classification="EMPTY" if not members else "MEMBERS_PRESENT_IDENTITY_NOT_INFERRED_FROM_PGID_ALONE",
        no_post_root_exit_group_signal_without_birth_proof=True) for pgid, members in groups.items()],
        scan_errors=errors, scan_complete=not errors, observed_at=now())


class Runner:
    def __init__(self, job, plan):
        self.job, self.p = job, plan
        self.work = datetime.fromisoformat(plan["absolute_work_stop"]).timestamp()
        self.hard = datetime.fromisoformat(plan["absolute_cleanup_deadline"]).timestamp()
        self.started, self.started_mono = time.time(), time.monotonic()
        self.allowed = sorted(os.sched_getaffinity(0))
        self.affinity = {0: dict(server=plan["CPU_affinity"]["server"], client=plan["CPU_affinity"]["client"])}
        require(len(self.affinity[0]["server"]) == len(self.affinity[0]["client"]) == 2
            and len(set(self.affinity[0]["server"] + self.affinity[0]["client"])) == 4,
            "four distinct registered suite model/client CPU cores required")
        self.lock, self.spawn_lock = threading.RLock(), threading.RLock()
        self.resource_lock = threading.Lock()
        self.last_resource_sample, self.resource_sequence = 0, 0
        self.hard_event, self.soft_event = threading.Event(), threading.Event()
        self.hard_reason, self.soft_reasons = None, []
        self.entries, self.servers, self.current, self.results = [], {}, {}, {}
        self.cleanup_signals, self.cleanup_exits, self.cleanup_refused = [], [], []
        self.last_storage_check, self.storage_bytes = 0, 0
        self.health = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.report_result = None
        self.observed_isolation = set()
        self.last_status_write = 0

    def atomic(self, path, data):
        with self.lock:
            temp = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}")
            with temp.open("x") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write("\n")
            os.replace(temp, path)

    def atomic_new(self, path, data):
        """Publish one complete immutable receipt, refusing every old name."""
        with self.lock:
            temp = path.with_name(path.name + f".new-{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}")
            try:
                with temp.open("x") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.link(temp, path)  # atomic exclusive publication; never replace
            finally:
                if temp.exists():
                    temp.unlink()

    def event(self, kind, **fields):
        with self.lock:
            with (self.job / "RUNNER-EVENTS.jsonl").open("a") as f:
                f.write(json.dumps(dict(at=now(), kind=kind, **fields), ensure_ascii=False, allow_nan=False) + "\n")

    def passive_observation(self, reason, **details):
        # New user-authorized full coverage explicitly supersedes old six-pair
        # quality early-stop, not the original scientific failures/permissions.
        with self.lock:
            record = dict(reason=reason, at=now(), **details)
            self.soft_reasons.append(record)
            self.event("PASSIVE_OBSERVATION_NO_QUALITY_EARLY_STOP", reason=reason, **details)

    def stop_hard(self, reason, **details):
        with self.lock:
            if not self.hard_event.is_set():
                self.hard_reason = dict(reason=reason, at=now(), **details)
                self.hard_event.set()
                if not (self.job / "STOP_REQUESTED").exists():
                    with (self.job / "STOP_REQUESTED").open("x") as f:
                        f.write(reason + "\n")
                self.atomic(self.job / "HARD-STOP.json", dict(status="HARD_RESOURCE_OR_PROTOCOL_STOP",
                    detail=self.hard_reason, permissions=PERMISSIONS))

    def status(self, stage):
        with self.lock:
            if time.monotonic() - self.last_status_write < 1:
                return
            self.last_status_write = time.monotonic()
            self.atomic(self.job / "status.json", dict(status="RUNNING_REGISTERED_SUITE_PHASE", stage=stage,
                at=now(), suite=self.p["suite"], phase=self.p["phase"], current=dict(self.current),
                closed_row_ids=sorted(self.results),
                complete_rows=sum(r.get("complete_episode") is True for r in self.results.values()),
                actual_new_complete_episodes=sum(r.get("complete_episode") is True and r.get("no_new_episode") is not True
                    for r in self.results.values()), hard_stop=self.hard_event.is_set(),
                storage_bytes=self.storage_bytes, wall_sec=time.monotonic() - self.started_mono,
                performance_claim=False, permissions=PERMISSIONS))

    def guard(self, stage):
        if self.hard_event.is_set():
            raise HardStop(str(self.hard_reason))
        if time.time() >= self.work:
            self.stop_hard("absolute_work_stop", stage=stage)
        if (self.job / "STOP_REQUESTED").exists():
            self.stop_hard("STOP_REQUESTED", stage=stage)
        for index in (0,):
            worker = self.job / f"server-worker{index}"
            if (worker / "ISOLATION-FAILED.json").is_file():
                if index not in self.observed_isolation:
                    self.observed_isolation.add(index)
                    self.passive_observation("PASSIVE_NUMERICAL_ISOLATION_OBSERVATION", worker=index,
                        evidence=str(worker / "ISOLATION-FAILED.json"))
            if (worker / "FAILED.json").is_file() or (self.job / f"WORKER-{index}-STOP.json").is_file():
                self.stop_hard("OWNED_SERVER_HARD_FAILURE", worker=index, stage=stage)
            live = worker / "LIVE-COUNTER.json"
            if live.is_file():
                state = read(live)
                require(type(state["target_forward_attempts"]) is int
                    and 0 <= state["target_forward_attempts"] <= self.p["job_forward_attempt_cap"], "worker forward hard cap exceeded")
        if time.monotonic() - self.last_storage_check >= 10:
            with self.lock:
                total = 0
                for path in self.job.rglob("*"):
                    if path.is_file():
                        try:
                            total += path.stat().st_size
                        except FileNotFoundError:
                            continue
                self.storage_bytes, self.last_storage_check = total, time.monotonic()
                if total > self.p["new_evidence_cap_bytes"]:
                    self.stop_hard("REGISTERED_SUITE_EVIDENCE_CAP", observed_bytes=total)
                if shutil.disk_usage(self.job).free < self.p["remote_data_disk_floor_bytes"]:
                    self.stop_hard("REGISTERED_DISK_FREE_FLOOR")
                if self.p["phase"] != "FINAL_EVAL":
                    compact = Path(self.p["suite_storage_root"])
                    compact_bytes = 0
                    for x in compact.rglob("*") if compact.exists() else ():
                        try:
                            if x.is_file():
                                compact_bytes += x.stat().st_size
                        except FileNotFoundError:
                            continue
                    if compact_bytes > 2**30:
                        self.stop_hard("SHARED_SUITE_COMPACT_ONE_GiB_CAP", observed_bytes=compact_bytes)
        if self.hard_event.is_set():
            raise HardStop(str(self.hard_reason))
        self.resource_sample_if_due()
        self.status(stage)

    def resource_sample_if_due(self):
        """Actual resource metadata, never a model forward or performance test."""
        if time.monotonic() - self.last_resource_sample < 60 or not self.resource_lock.acquire(blocking=False):
            return
        try:
            if time.monotonic() - self.last_resource_sample < 60:
                return
            self.last_resource_sample = time.monotonic()
            self.resource_sequence += 1
            snapshot_id = f"gpu-resource-{self.resource_sequence:06d}"
            sampled_start = time.time()
            queries = {}
            executable = shutil.which("nvidia-smi")
            specs = dict(gpu="uuid,index,utilization.gpu,memory.used,memory.total",
                compute="gpu_uuid,pid,process_name,used_gpu_memory")
            for label, fields in specs.items():
                query = None
                try:
                    require(executable is not None, "nvidia-smi unavailable")
                    option = "--query-gpu=" if label == "gpu" else "--query-compute-apps="
                    log = self.job / f"GPU-QUERY-{self.resource_sequence:06d}-{label}.csv"
                    query = self.spawn([executable, option + fields, "--format=csv,noheader,nounits"], log,
                        self.environment(0, report=True), self.allowed[:1], f"resource-query-{snapshot_id}-{label}")
                    until = min(time.time() + 4, self.hard - 5)
                    while not self.wait_reaped(query) and time.time() < until:
                        time.sleep(.05)
                    if not self.wait_reaped(query):
                        self.signal_owned(query, signal.SIGTERM)
                        if not self.wait_reaped(query, timeout=1):
                            self.signal_owned(query, signal.SIGKILL)
                            self.wait_reaped(query, timeout=1)
                    text = log.read_text(errors="replace")
                    require(len(text.encode()) <= 65536, "resource metadata unexpectedly large")
                    rows = list(csv.reader(text.splitlines())) if query["proc"].returncode == 0 else []
                    queries[label] = dict(status="OBSERVED" if query["proc"].returncode == 0 else "UNKNOWN",
                        pid=query["proc"].pid, birth_ticks=query["pin"]["birth_ticks"],
                        returncode=query["proc"].returncode, rows=rows, source=str(log))
                except Exception as exc:
                    if query is None:
                        with self.lock:
                            query = next((r for r in reversed(self.entries)
                                if r["role"] == f"resource-query-{snapshot_id}-{label}"), None)
                    if query is not None and not self.wait_reaped(query):
                        self.signal_owned(query, signal.SIGTERM)
                        if not self.wait_reaped(query, timeout=1):
                            self.signal_owned(query, signal.SIGKILL)
                            self.wait_reaped(query, timeout=1)
                    queries[label] = dict(status="UNKNOWN", error=repr(exc))
            devices, computes, parse_errors = [], [], []
            def numeric(value):
                try:
                    return float(value.strip())
                except ValueError:
                    return "UNKNOWN"
            for values in queries.get("gpu", {}).get("rows", []):
                if len(values) != 5:
                    parse_errors.append(dict(query="gpu", fields=values))
                    continue
                devices.append(dict(uuid=values[0].strip(), physical_index=values[1].strip(),
                    utilization_gpu_percent=numeric(values[2]), memory_used_MiB=numeric(values[3]),
                    memory_total_MiB=numeric(values[4])))
            with self.lock:
                pins = {r["proc"].pid: r["pin"] for r in self.entries if r["pin"] is not None}
            for values in queries.get("compute", {}).get("rows", []):
                if len(values) != 4:
                    parse_errors.append(dict(query="compute", fields=values))
                    continue
                try:
                    pid = int(values[1].strip())
                except ValueError:
                    parse_errors.append(dict(query="compute", fields=values))
                    continue
                pin, observed = pins.get(pid), process(pid)
                verified = pin is not None and pin["owner_job"] == str(self.job) and matches(pin)
                computes.append(dict(gpu_uuid=values[0].strip(), pid=pid, process_name=values[2].strip(),
                    used_gpu_memory_MiB=numeric(values[3]),
                    identity="OWNED_VERIFIED_CURRENT_LIFETIME" if verified else "EXTERNAL_OR_UNRESOLVED",
                    observed_birth_ticks=observed["birth_ticks"] if observed is not None else "UNKNOWN",
                    verified_owned_birth_ticks=pin["birth_ticks"] if verified else "UNKNOWN"))
            snapshot = dict(schema="tf-suite-real-resource-metadata/v1", snapshot_id=snapshot_id,
                sampled_at_unix=time.time(), sampling_started_unix=sampled_start, sampled_at=now(),
                status="OBSERVED" if all(q["status"] == "OBSERVED" for q in queries.values()) and devices and not parse_errors
                    else "PARTIAL_OR_UNKNOWN", query_processes=queries, GPUs=devices, compute_processes=computes,
                parse_errors=parse_errors, no_model_forward=True, not_an_independent_test=True,
                sampling_does_not_prove_continuous_exclusivity=True, permissions=PERMISSIONS)
            self.atomic(self.job / "GPU-RESOURCE-CURRENT.json", snapshot)
            with self.lock:
                with (self.job / "GPU-RESOURCE-HISTORY.jsonl").open("a") as f:
                    f.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception as exc:
            # Metadata must never become an intervention in the robot loop.
            try:
                self.atomic(self.job / "GPU-RESOURCE-CURRENT.json", dict(schema="tf-suite-real-resource-metadata/v1",
                    snapshot_id=f"gpu-resource-{self.resource_sequence:06d}", sampled_at_unix=time.time(),
                    status="UNKNOWN", error=repr(exc), no_model_forward=True,
                    sampling_does_not_prove_continuous_exclusivity=True, permissions=PERMISSIONS))
            except Exception:
                # The separate storage guard still enforces real disk limits.
                # A metadata-only failure does not change an admitted episode.
                pass
        finally:
            self.resource_lock.release()

    def environment(self, index, *, report=False):
        prefixes = ("WAM_P0_", "P0_", "LINGBOT_", "FORMAL_SPEED_", "PREFIX_ACCEPT_", "RESET_ONLINE_",
            "RESET_TEXT_BANK_", "TARGET_TRACE_", "TF_EFFECT_", "TORCHELASTIC_")
        env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes) and k not in
            ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "MASTER_PORT", "MASTER_ADDR")}
        env.update(CUDA_VISIBLE_DEVICES="" if report else str(self.p["gpu_index"]), LINGBOT_SEED=str(self.p["evaluation_seed"]),
            PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", LINGBOT_DISABLE_FSDP="1",
            LINGBOT_CACHE_MODE="original", LINGBOT_ATTN_MODE="torch", LINGBOT_NSYS_PROFILE="off",
            PREFIX_ACCEPT_ENABLED="1", PREFIX_ACCEPT_GPU_APPROVED="1", RESET_TEXT_BANK_ENABLED="1",
            LINGBOT_VA_LIBERO_SUITE=self.p["suite"], LINGBOT_VA_LIBERO_CHECKPOINT=self.p["suite_info"]["model_path"],
            LINGBOT_VA_LIBERO_CHECKPOINT_ID=self.p["suite_info"]["checkpoint_id"],
            LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=self.p["registry"],
            WAM_VAE_DEVICE="cuda", WAM_P0_PHASE_PROFILE="0", WAM_P0_ACTION_DRAFT_MODE=self.p["mode"],
            WAM_P0_ACTION_DRAFT_SELECTOR="global", WAM_P0_ACTION_DRAFT_THRESHOLD=str(self.p["threshold"]),
            WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=(self.p.get("selected") or {}).get("path", str(self.job / "UNUSED_T_ONLY_NO_SUITE_DRAFT")),
            MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(self.p["gpu_index"]),
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false", LIBERO_CONFIG_PATH=self.p["LIBERO_CONFIG_PATH"],
            RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
            MASTER_ADDR="127.0.0.1", MASTER_PORT=str(self.p["master_port"]))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1" if report else "2"
        app = Path(self.p.get("root_path", str(ROOT)))
        paths = [app / "userlib", Path(self.p["va_python"]).resolve().parents[1] / "lib",
            Path(self.p["client_python"]).resolve().parents[1] / "lib"]
        env["LD_LIBRARY_PATH"] = ":".join([str(x) for x in paths] + [os.getenv("LD_LIBRARY_PATH", "")])
        pythonpaths = [ROOT / "third_party/LIBERO-official-20260922",
            WORKTREE, EXP, ROOT / "投机采样/scripts"]
        env["PYTHONPATH"] = ":".join([str(x) for x in pythonpaths] + [os.getenv("PYTHONPATH", "")])
        return env

    def spawn(self, argv, log, env, affinity, role):
        with self.spawn_lock:
            require(time.time() < self.hard, "no owned process after hard cleanup deadline")
            if role.startswith(("server-", "client-")):
                require(not self.hard_event.is_set() and not (self.job / "STOP_REQUESTED").exists()
                    and time.time() < self.work, "no new GPU/client process after a hard stop")
            log.parent.mkdir(parents=True, exist_ok=True)
            previous = os.sched_getaffinity(0)
            with log.open("xb") as stream:
                try:
                    os.sched_setaffinity(0, set(affinity))
                    child = subprocess.Popen(list(map(str, argv)), stdin=subprocess.DEVNULL, stdout=stream,
                        stderr=subprocess.STDOUT, env=env, cwd=WORKTREE, close_fds=True, start_new_session=True)
                finally:
                    os.sched_setaffinity(0, previous)
            record = dict(proc=child, pin=None, role=role, argv=list(map(str, argv)), reaped=False)
            with self.lock:
                self.entries.append(record)
            actual = process(child.pid)
            if actual is None:
                child.wait(timeout=5)
                record["reaped"] = True
                raise HardStop(f"owned {role} exited before lifetime registration; returncode={child.returncode}")
            require(actual["pgid"] == child.pid and actual["uid"] == os.getuid(), "owned fresh session identity differs")
            pin = dict(pid=child.pid, birth_ticks=actual["birth_ticks"], pgid=actual["pgid"], uid=actual["uid"],
                owner_job=str(self.job), role=role, argv=record["argv"], CPU_affinity=list(affinity), registered_at=now())
            record["pin"] = pin
            try:
                self.atomic_new(self.job / f"OWNED-{child.pid}.json", pin)
            except BaseException:
                # A child already exists. Keep its exact in-memory pin so the
                # finally cleanup can stop it even if registration I/O failed.
                raise
            self.event("OWNED_PROCESS_STARTED", **pin)
            return record

    def wait_reaped(self, record, timeout=0):
        try:
            rc = record["proc"].wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        with self.lock:
            if not record["reaped"]:
                self.cleanup_exits.append(dict(pid=record["proc"].pid, role=record["role"], returncode=rc, at=now()))
                record["reaped"] = True
        return True

    def signal_owned(self, record, sig):
        child, pin = record["proc"], record["pin"]
        if self.wait_reaped(record):
            return
        if pin is None or not matches(pin):
            with self.lock:
                self.cleanup_refused.append(dict(pid=child.pid, role=record["role"],
                    reason="birth/uid/PGID proof no longer matches; no signal sent", at=now()))
            return
        try:
            os.killpg(pin["pgid"], sig)
            with self.lock:
                self.cleanup_signals.append(dict(pid=pin["pid"], birth_ticks=pin["birth_ticks"], pgid=pin["pgid"],
                    signal=signal.Signals(sig).name, at=now(), owner_job=str(self.job)))
        except ProcessLookupError:
            pass

    def cleanup(self, *, gpu_only=False):
        with self.lock:
            records = [r for r in self.entries if not gpu_only or r["role"].startswith(("server-", "client-"))]
        for record in records:
            self.signal_owned(record, signal.SIGTERM)
        until = min(time.time() + 15, self.hard - 10)
        while time.time() < until and any(not self.wait_reaped(r) for r in records):
            time.sleep(.1)
        for record in records:
            if not self.wait_reaped(record):
                self.signal_owned(record, signal.SIGKILL)
        for record in records:
            if not self.wait_reaped(record, timeout=max(0, min(5, self.hard - time.time()))):
                self.cleanup_refused.append(dict(pid=record["proc"].pid, reason="owned process not reaped by cleanup cap"))
        forensic = forensic_groups([r["pin"] for r in records if r["pin"] is not None])
        groups_empty = forensic["scan_complete"] and not any(g["members"] for g in forensic["groups"])
        no_live = forensic["scan_complete"] and not any(g["live_members"] for g in forensic["groups"])
        receipt = dict(status="OWNED_GROUPS_EMPTY" if all(r["reaped"] for r in records) and groups_empty else "OWNED_GROUPS_REMAIN_OR_UNKNOWN",
            all_roots_reaped=all(r["reaped"] for r in records), signals=list(self.cleanup_signals),
            exits=list(self.cleanup_exits), refused=list(self.cleanup_refused), at=now(),
            all_owned_groups_empty=groups_empty, no_live_owned_group_members_observed=no_live,
            all_owned_work_exited=all(r["reaped"] for r in records) and no_live,
            FORENSIC_group_members=forensic,
            foreign_processes_signalled=False, permissions=PERMISSIONS, instance_shutdown="NOT_REQUESTED_NOT_EXECUTED")
        return receipt

    def start_servers(self):
        for port in (self.p["port"], self.p["master_port"]):
            with socket.socket() as endpoint:
                endpoint.bind(("127.0.0.1", port))
        self.guard("single_suite_model_admission")
        require(time.time() + 900 + 1320 < self.work, "model setup plus one complete episode reserve unavailable")
        server = self.spawn([self.p["va_python"], "-B", OWN / "server_entry.py", "--job", self.job,
            "--worker-index", 0, "--port", self.p["port"], "--master-port", self.p["master_port"],
            "--forward-cap", self.p["job_forward_attempt_cap"]], self.job / "server-worker0/server.log",
            self.environment(0), self.affinity[0]["server"], "server-0")
        self.servers[0] = server
        until = min(server["pin"]["birth_ticks"] / os.sysconf("SC_CLK_TCK") + 900,
            time.monotonic() + self.work - time.time())
        while True:
            self.guard("actual_suite_server_readiness")
            require(server["proc"].poll() is None and time.monotonic() < until, "suite model startup failed or900s cap")
            path = self.job / "server-worker0/READY.json"
            if path.is_file():
                ready = read(path)
                require(ready["schema"] == "tf-suite-server-ready/v1" and ready["status"] == "READY"
                    and ready["pid"] == server["proc"].pid and ready["worker_index"] == 0
                    and ready["suite"] == self.p["suite"] and ready["phase"] == self.p["phase"]
                    and ready["port"] == self.p["port"] and ready["master_port"] == self.p["master_port"]
                    and ready["physical_GPU_index"] == self.p["gpu_index"] and ready["permissions"] == PERMISSIONS
                    and ready["target_forward_cap"] == self.p["job_forward_attempt_cap"]
                    and ready["setup_RNG_restored"] is True and ready["target_RNG_captured_before_draft"] is True
                    and ready["O3_installed"] is True and ready["complete_text_bank"] is True, "actual suite READY identity differs")
                try:
                    with self.health.open(f"http://127.0.0.1:{self.p['port']}/healthz", timeout=1) as response:
                        require(response.status == 200, "actual model service not healthy yet")
                        response.read(1024)
                except (OSError, TimeoutError, ValueError):
                    time.sleep(.2)
                    continue
                write(self.job / "server-worker0/MODEL-SETUP-OBSERVATION.json", dict(
                    schema="tf-suite-actual-service-setup/v1", pid=server["proc"].pid,
                    birth_ticks=server["pin"]["birth_ticks"],
                    persistent_server_start_to_ready_wall_sec=time.monotonic() - server["pin"]["birth_ticks"] / os.sysconf("SC_CLK_TCK"),
                    scope="actual import/model/bank/setup and service readiness; not isolated weight load", permissions=PERMISSIONS))
                return
            time.sleep(.25)

    def finish_receipts(self, out, spec):
        path = out / "FINISH_REQUESTED"
        if not path.exists():
            path.write_text("actual suite client exited; drain its original complete episode\n")
        names = ["FINISH_READY.json", "COUNTER.json", "AUDIT.json", "SERVER-TERMINAL.json"]
        if self.p["phase"] != "FINAL_EVAL":
            names.append("COLLECTION-AUDIT.json")
        until = min(time.time() + 120, self.work)
        while not all((out / name).is_file() for name in names):
            self.guard("actual_suite_finish_drain")
            require(self.servers[0]["proc"].poll() is None and time.time() < until, "suite finish120s cap/server exit")
            time.sleep(.2)
        docs = {name: read(out / name) for name in names}
        for name, value in docs.items():
            require(value["row_id"] == spec["row_id"] and value["case_id"] == spec["case_id"]
                and value["arm"] == spec["arm"] and value["permissions"] == PERMISSIONS,
                "suite close receipt identity differs: " + name)
        ready, counter = docs["FINISH_READY.json"], docs["COUNTER.json"]
        require(ready["status"] in ("PASS", "FAILED_COMPONENT_AUDIT") and ready["full_episode_terminal"] is True
            and ready["async_drained"] is True and counter["schema"] == "tf-suite-episode-counter/v1"
            and type(counter["target_forward_attempts"]) is int and 0 <= counter["target_forward_attempts"] <= 4000,
            "suite complete/drained or forward accounting differs")
        return docs

    def run_row(self, spec):
        rid, cid, arm = spec["row_id"], spec["case_id"], spec["arm"]
        result = dict(schema="tf-suite-row-result/v1", row_id=rid, case_id=cid, task=spec["task"], init=spec["init"],
            arm=arm, suite=self.p["suite"], phase=self.p["phase"], complete_episode=False, permissions=PERMISSIONS)
        out = episode_dir(self.job, cid, arm)
        try:
            if spec.get("reuse_source"):
                origin = Path(spec["reuse_source"]["out"]).resolve(strict=True)
                terminal = read(origin / "EPISODE-TERMINAL.json")
                result.update(status="REUSED_READONLY_COMPLETE_TARGET", complete_episode=True,
                    source_job=spec["reuse_source"]["job"], source_out=str(origin), source_row_id=terminal["row_id"],
                    success=terminal["success"], env_steps=terminal["env_steps"], failure800_penalty=terminal["failure800_penalty"],
                    new_target_forwards=0, no_new_episode=True, old_terminal_unchanged=True)
                return result
            self.guard("complete_suite_episode_admission")
            require(time.time() + 1320 < self.work, "whole suite episode1200s plus finish120s unavailable")
            out.mkdir(parents=True, exist_ok=False)
            self.current[0] = dict(row_id=rid, case_id=cid, arm=arm, stage="complete_episode")
            child = self.spawn([self.p["client_python"], "-B", OWN / "client_entry.py", "--job", self.job,
                "--row-id", rid, "--case-id", cid, "--arm", arm, "--out", out, "--port", self.p["port"],
                "--worker-index", 0, "--master-port", self.p["master_port"]], out / "client.log",
                self.environment(0), self.affinity[0]["client"], f"client-row{rid}-{arm}")
            until = child["pin"]["birth_ticks"] / os.sysconf("SC_CLK_TCK") + 1200
            while not self.wait_reaped(child):
                self.guard("running_original_complete_suite_episode")
                require(self.servers[0]["proc"].poll() is None and time.monotonic() < until,
                    "suite server exited or actual complete episode1200s cap")
                time.sleep(.25)
            terminal = read(out / "EPISODE-TERMINAL.json") if (out / "EPISODE-TERMINAL.json").is_file() else None
            if child["proc"].returncode != 0 or terminal is None or terminal.get("complete_episode") is not True:
                if not (out / "FINISH_REQUESTED").exists():
                    (out / "FINISH_REQUESTED").write_text("partial suite client exited; retain known attempted work\n")
                raise HardStop("partial original suite client; no automatic retry, not an algorithm failure")
            require(terminal["schema"] == "tf-suite-full-episode-terminal/v1" and terminal["row_id"] == rid
                and terminal["case_id"] == cid and terminal["arm"] == arm and terminal["suite"] == self.p["suite"]
                and terminal["phase"] == self.p["phase"] and terminal["context"]
                    == dict(row_id=rid, case_id=cid, arm=arm, out=str(out)) and terminal["permissions"] == PERMISSIONS,
                "suite actual client terminal routing differs")
            docs = self.finish_receipts(out, spec)
            result.update(status="COMPLETE_RECORDED_FULL_EPISODE", complete_episode=True, source_out=str(out),
                success=terminal["success"], env_steps=terminal["env_steps"], failure800_penalty=terminal["failure800_penalty"],
                policy_chunks=terminal["policy_chunks"], target_forward_attempts=docs["COUNTER.json"]["target_forward_attempts"],
                component_status=docs["AUDIT.json"]["status"], finish_status=docs["FINISH_READY.json"]["status"],
                collector_invalid=docs["COUNTER.json"].get("collector_invalid", "NOT_COLLECTED_EVAL"),
                collection_status=docs.get("COLLECTION-AUDIT.json", {}).get("status", "NOT_COLLECTED_EVAL"))
            return result
        except BaseException as exc:
            result.update(status="PARTIAL_HARD_FAILURE_NO_RETRY", error=repr(exc), traceback=traceback.format_exc())
            self.stop_hard("SUITE_ROW_PROTOCOL_OR_RESOURCE_FAILURE", row_id=rid, error=repr(exc))
            return result
        finally:
            result["finished_at"] = now()
            self.results[rid] = result
            self.current.pop(0, None)
            write(self.job / f"ROW-{rid:04d}-RESULT.json", result)

    def shutdown_idle_servers(self):
        if self.hard_event.is_set() or 0 not in self.servers:
            return
        server = self.servers[0]
        if server["proc"].poll() is None:
            path = self.job / "server-worker0/SHUTDOWN_REQUESTED"
            if not path.exists():
                path.write_text("all admitted suite episodes closed; idle model may exit\n")
        until = min(time.time() + 90, self.work)
        while not self.wait_reaped(server) and time.time() < until:
            time.sleep(.2)

    def execute(self):
        require((self.job / "DEPLOYMENT-COMPLETE.json").is_file(), "actual suite source/assets readiness required")
        guard = read(self.job / "GUARD-STARTED.json")
        observed = process(guard["pid"])
        require(observed is not None and observed["birth_ticks"] == guard["birth_ticks"]
            and observed["uid"] == os.getuid() and guard["hard_deadline"] == self.hard and guard["work_stop"] == self.work,
            "independent suite guard identity/deadline differs")
        write(self.job / "RUN-STARTED.json", dict(schema="tf-suite-run-start/v1", pid=os.getpid(),
            birth_ticks=process(os.getpid())["birth_ticks"], at=now(), suite=self.p["suite"], phase=self.p["phase"],
            CPU_affinity=self.p["CPU_affinity"], absolute_work_stop=self.p["absolute_work_stop"],
            absolute_cleanup_deadline=self.p["absolute_cleanup_deadline"], permissions=PERMISSIONS))
        old_signals, failure, paused = {}, None, False
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_signals[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda received, _: self.stop_hard("OWNED_SUITE_RUNNER_SIGNAL", signal=received))
        try:
            if not (self.job / "STOP_AFTER_EPISODE").exists():
                self.start_servers()
            for spec in self.p["rows"]:
                if (self.job / "STOP_AFTER_EPISODE").exists():
                    paused = True
                    break
                self.guard("suite_row_boundary")
                if not spec.get("reuse_source") and time.time() + 1320 >= self.work:
                    self.event("ROW_NOT_STARTED_INSUFFICIENT_REGISTERED_RESERVE", row_id=spec["row_id"])
                    break
                result = self.run_row(spec)
                if not result["complete_episode"]:
                    break
                self.status("complete_suite_row_preserved")
                if self.hard_event.is_set():
                    break
        except BaseException as exc:
            failure = dict(error=repr(exc), traceback=traceback.format_exc())
            self.stop_hard("SUITE_RUNNER_HARD_FAILURE", error=repr(exc))
        finally:
            try:
                self.shutdown_idle_servers()
            except BaseException as exc:
                failure = dict(previous=failure, shutdown_error=repr(exc))
            final = self.cleanup()
            write(self.job / "GPU-OWNED-CLEANUP.json", final)
            write(self.job / "FINAL-OWNED-CLEANUP.json", final)
            complete = sum(r["complete_episode"] for r in self.results.values())
            full = complete == len(self.p["rows"])
            status = ("COMPLETE_RECORDED_SUITE_PHASE" if full and final["all_owned_groups_empty"] and failure is None
                else "PAUSED_AFTER_COMPLETE_EPISODE" if paused and not self.hard_event.is_set()
                else "PARTIAL_OR_FAILED_SUITE_PHASE")
            write(self.job / "SUPERVISOR-RESULT.json", dict(schema="tf-suite-supervisor/v1", status=status,
                suite=self.p["suite"], phase=self.p["phase"], planned_rows=len(self.p["rows"]), complete_rows=complete,
                actual_new_complete_episodes=sum(r["complete_episode"] and r.get("no_new_episode") is not True for r in self.results.values()),
                reused_complete_target_rows=sum(r.get("no_new_episode") is True for r in self.results.values()),
                missing_row_ids=[r["row_id"] for r in self.p["rows"] if r["row_id"] not in self.results],
                row_results=[self.results[i] for i in sorted(self.results)], failure=failure, hard_reason=self.hard_reason,
                cleanup=final, elapsed_wall_sec=time.monotonic() - self.started_mono, no_retry=True,
                no_quality_certificate=True, formal_performance=False, permissions=PERMISSIONS,
                instance_shutdown="NOT_REQUESTED_NOT_EXECUTED"))
            write(self.job / "ALL-OWNED-CLEANUP.json", dict(status="OWNED_GROUPS_EMPTY_RUNNER_EXITS_NEXT"
                if final["all_owned_groups_empty"] else "OWNED_GROUPS_REMAIN_OR_UNKNOWN",
                all_roots_reaped=final["all_roots_reaped"], all_owned_groups_empty=final["all_owned_groups_empty"],
                FORENSIC_group_members=final["FORENSIC_group_members"], current_CPU_runner_exits_next=True,
                independent_guard_observes_final_exit=True, permissions=PERMISSIONS))
            for sig, old in old_signals.items():
                signal.signal(sig, old)
        return 0 if full and final["all_owned_groups_empty"] and failure is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    try:
        p = load(job)
        return Runner(job, p).execute()
    except BaseException as exc:
        if not (job / "RUNNER-ENTRY-FAILED.json").exists():
            write(job / "RUNNER-ENTRY-FAILED.json", dict(status="FAILED_SUITE_RUNNER_ENTRY", error=repr(exc),
                traceback=traceback.format_exc(), permissions=PERMISSIONS))
        raise


if __name__ == "__main__":
    raise SystemExit(main())


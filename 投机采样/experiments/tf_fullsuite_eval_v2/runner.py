"""Bounded dual-worker Long500 three-arm full-episode runner; no probes, tests or digests.

One persistent target per supplied GPU. All registered algorithm failures and passive audit differences are retained.
Only routing/resource/absolute limits interrupt work; no old quality early-stop.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from tf_fullsuite_eval_v2.protocol import (ROOT, EXP, WORKTREE, PERMISSIONS, episode_dir,
    case, read, write, load, require, pending_cases)


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
        require(len(self.allowed) >= 16, "sixteen allowed CPU cores required for four server/client workers")
        self.affinity = {i: dict(server=self.allowed[4*i:4*i+2], client=self.allowed[4*i+2:4*i+4]) for i in range(4)}
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
            self.atomic(self.job / "status.json", dict(status="RUNNING_FIXED_FULL_EPISODE_PANEL", stage=stage,
                at=now(), current=dict(self.current), completed_cases=sorted(self.results),
                complete_episodes=sum(len(r.get("episodes", [])) for r in self.results.values()),
                no_new_wave=self.soft_event.is_set(), hard_stop=self.hard_event.is_set(),
                storage_bytes=self.storage_bytes, wall_sec=time.monotonic() - self.started_mono,
                performance_claim=False, permissions=PERMISSIONS))

    def guard(self, stage):
        if self.hard_event.is_set():
            raise HardStop(str(self.hard_reason))
        if time.time() >= self.work:
            self.stop_hard("absolute_work_stop", stage=stage)
        if (self.job / "STOP_REQUESTED").exists():
            self.stop_hard("STOP_REQUESTED", stage=stage)
        for index in range(4):
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
                    and 0 <= state["target_forward_attempts"] <= self.p["worker_forward_caps"][str(index)], "worker forward hard cap exceeded")
        if time.monotonic() - self.last_storage_check >= 10:
            with self.lock:
                total = 0
                for path in (x for folder in (self.job, Path(self.p["prior_job"])) for x in folder.rglob("*")):
                    if path.is_file():
                        try:
                            total += path.stat().st_size
                        except FileNotFoundError:
                            continue
                self.storage_bytes, self.last_storage_check = total, time.monotonic()
                if total > self.p["new_evidence_cap_bytes"]:
                    self.stop_hard("NEW_EVIDENCE_10GiB_CAP", observed_bytes=total)
                if shutil.disk_usage(self.job).free < self.p["remote_data_disk_floor_bytes"]:
                    self.stop_hard("REGISTERED_DISK_FREE_FLOOR")
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
            snapshot = dict(schema="tf-fullsuite-real-resource-metadata/v1", snapshot_id=snapshot_id,
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
                self.atomic(self.job / "GPU-RESOURCE-CURRENT.json", dict(schema="tf-fullsuite-real-resource-metadata/v1",
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
        env.update(CUDA_VISIBLE_DEVICES="" if report else str(index % 2), LINGBOT_SEED=str(self.p["evaluation_seed"]),
            PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", LINGBOT_DISABLE_FSDP="1",
            LINGBOT_CACHE_MODE="original", LINGBOT_ATTN_MODE="torch", LINGBOT_NSYS_PROFILE="off",
            PREFIX_ACCEPT_ENABLED="1", PREFIX_ACCEPT_GPU_APPROVED="1", RESET_TEXT_BANK_ENABLED="1",
            LINGBOT_VA_LIBERO_SUITE="libero_10", LINGBOT_VA_LIBERO_CHECKPOINT=self.p["target_checkpoint_path"],
            LINGBOT_VA_LIBERO_CHECKPOINT_ID=self.p["target_checkpoint_id"],
            LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=self.p["registry"],
            WAM_VAE_DEVICE="cuda", WAM_P0_PHASE_PROFILE="0", WAM_P0_ACTION_DRAFT_MODE=self.p["mode"],
            WAM_P0_ACTION_DRAFT_SELECTOR="global", WAM_P0_ACTION_DRAFT_THRESHOLD=str(self.p["threshold"]),
            WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=self.p["selected"]["path"],
            MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(index % 2),
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false", LIBERO_CONFIG_PATH=self.p["LIBERO_CONFIG_PATH"],
            RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
            MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29781 + index))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1" if report else "2"
        app = Path(self.p.get("root", str(ROOT)))
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
        for index in range(4):
            for port in (19781 + index, 29781 + index):
                with socket.socket() as local:
                    local.bind(("127.0.0.1", port))
            self.guard("persistent_server_admission")
            self.servers[index] = self.spawn([self.p["va_python"], "-B", OWN / "server_entry.py", "--job", self.job,
                "--worker-index", index, "--port", 19781 + index, "--master-port", 29781 + index, "--forward-cap", self.p["worker_forward_caps"][str(index)]],
                self.job / f"server-worker{index}" / "server.log", self.environment(index), self.affinity[index]["server"], f"server-{index}")
            self.servers[index]["ready_deadline_monotonic"] = min(
                self.servers[index]["pin"]["birth_ticks"] / os.sysconf("SC_CLK_TCK") + 900,
                time.monotonic() + self.work - time.time())
        pending = set(range(4))
        while pending:
            self.guard("actual_persistent_server_readiness")
            for index in list(pending):
                server = self.servers[index]
                require(server["proc"].poll() is None, f"server{index} exited during actual model setup")
                require(time.monotonic() < server["ready_deadline_monotonic"], f"server{index} model900s startup cap")
                path = self.job / f"server-worker{index}" / "READY.json"
                if not path.is_file():
                    continue
                ready = read(path)
                require(ready["status"] == "READY" and ready["pid"] == server["proc"].pid
                    and ready["worker_index"] == index and ready["port"] == 19781 + index
                    and ready["master_port"] == 29781 + index and ready["permissions"] == PERMISSIONS
                    and ready["model_persistent"] is True and ready["setup_RNG_restored"] is True
                    and ready["target_RNG_captured_before_draft"] is True and ready["complete_text_bank"] is True
                    and ready["O3_installed"] is True and ready["target_forward_cap"] == self.p["worker_forward_caps"][str(index)]
                    and ready["TF_ON_checkpoint"] == self.p["selected"]["path"], "persistent READY identity/math differs")
                try:
                    with self.health.open(f"http://127.0.0.1:{19781 + index}/healthz", timeout=1) as response:
                        if response.status != 200:
                            continue
                        response.read(1024)
                except (OSError, TimeoutError):
                    continue
                pending.remove(index)
                self.atomic_new(self.job / f"server-worker{index}" / "MODEL-SETUP-OBSERVATION.json",
                    dict(schema="tf-fullsuite-actual-service-setup-observation/v1", worker_index=index,
                        pid=server["proc"].pid, birth_ticks=server["pin"]["birth_ticks"],
                        spawn_birth_monotonic=server["pin"]["birth_ticks"] / os.sysconf("SC_CLK_TCK"),
                        observed_service_ready_monotonic=time.monotonic(),
                        persistent_server_start_to_ready_wall_sec=time.monotonic()
                            - server["pin"]["birth_ticks"] / os.sysconf("SC_CLK_TCK"),
                        scope="actual server import/model/bank/setup until READY and service health; not isolated model-load timing",
                        at=now(), permissions=PERMISSIONS))
                self.event("ACTUAL_SERVICE_READY", worker=index, pid=server["proc"].pid,
                    health_check_is_model_service_admission_only=True)
            if pending:
                time.sleep(.3)

    def finish_receipts(self, out, index):
        path = out / "FINISH_REQUESTED"
        if not path.exists():
            with path.open("x") as f:
                f.write("client process exited; close actual recorded episode\n")
        until = min(time.time() + 120, self.work)
        required = ("FINISH_READY.json", "COUNTER.json", "AUDIT.json", "SERVER-TERMINAL.json")
        while not all((out / n).is_file() for n in required):
            self.guard("actual_episode_finish_drain")
            require(self.servers[index]["proc"].poll() is None, "persistent server exited before finish receipts")
            require(time.time() < until, "episode finish/drain120s cap")
            time.sleep(.2)
        ready, counter, audit, terminal = [read(out / n) for n in required]
        require(ready["status"] in ("PASS", "FAILED_COMPONENT_AUDIT") and ready["full_episode_terminal"] is True
            and ready["async_drained"] is True and ready["permissions"] == PERMISSIONS
            and ready["case_id"] == terminal["case_id"] == audit["case_id"] == counter["case_id"]
            and ready["arm"] == terminal["arm"] == audit["arm"] == counter["arm"], "closed recorded episode receipts disagree")
        require(0 <= counter["target_forward_attempts"] <= 4000, "complete episode forward cap exceeded")
        return ready, counter, audit, terminal

    def run_case(self, case_id):
        spec = case(self.p, case_id)
        index = spec["execution_worker_index"]
        result = dict(case_id=case_id, worker=index, episodes=[], status="ADMITTED_NOT_COMPLETE", at=now())
        with self.lock:
            self.current[index] = dict(case_id=case_id, stage="case_admission")
        try:
            # Do not consult soft_event inside a case: finish all three admitted arms.
            self.guard("whole_case_admission")
            require(time.time() + 3 * (1200 + 120) <= self.work
                and time.time() + 3 * (1200 + 120) + 900 <= self.hard,
                "whole three-episode case plus finish/cleanup reserve unavailable")
            require(spec["execution_authorized"] is True, "case is not execution-authorized")
            for arm in spec["within_case_order"].split("_"):
                self.guard("whole_episode_admission")
                require(time.time() + 1200 + 120 <= self.work, "whole1200s episode plus finish reserve unavailable")
                out = episode_dir(self.job, case_id, arm)
                out.mkdir(parents=True, exist_ok=False)
                with self.lock:
                    self.current[index] = dict(case_id=case_id, arm=arm, stage="complete_episode")
                server = self.servers[index]
                require(server["proc"].poll() is None, "one persistent server lifecycle required")
                client = self.spawn([self.p["client_python"], "-B", OWN / "client_entry.py", "--job", self.job,
                    "--case-id", case_id, "--arm", arm, "--out", out, "--port", 19781 + index,
                    "--worker-index", index, "--master-port", 29781 + index], out / "client.log",
                    self.environment(index), self.affinity[index]["client"], f"client-case{case_id}-{arm}")
                deadline = client["pin"]["birth_ticks"] / os.sysconf("SC_CLK_TCK") + 1200
                while not self.wait_reaped(client):
                    self.guard("running_original_complete_episode")
                    require(server["proc"].poll() is None, "persistent server exited during complete episode")
                    if time.monotonic() >= deadline:
                        self.stop_hard("COMPLETE_EPISODE1200s_CAP", case_id=case_id, arm=arm)
                        raise HardStop("client whole-episode hard timeout")
                    time.sleep(.25)
                ep = read(out / "EPISODE-TERMINAL.json") if (out / "EPISODE-TERMINAL.json").is_file() else None
                # FINISH is a file on the actual idle server, never an infer test.
                if client["proc"].returncode != 0 or ep is None or ep.get("complete_episode") is not True:
                    if not (out / "FINISH_REQUESTED").exists():
                        (out / "FINISH_REQUESTED").write_text("partial client exited; preserve incomplete server accounting\n")
                    self.stop_hard("PARTIAL_CLIENT_EXCEPTION_NOT_ALGORITHM_FAILURE", case_id=case_id, arm=arm,
                        returncode=client["proc"].returncode)
                    raise HardStop("partial original client; no retry")
                require(ep["case_id"] == case_id and ep["arm"] == arm and ep["permissions"] == PERMISSIONS,
                    "client episode routed outside registered case")
                finish, counter, audit, server_terminal = self.finish_receipts(out, index)
                result["episodes"].append(dict(case_id=case_id, arm=arm, success=ep["success"],
                    env_steps=ep["env_steps"], failure800_penalty=ep["failure800_penalty"],
                    target_forward_attempts=counter["target_forward_attempts"],
                    finish_status=finish["status"], audit_status=audit["status"], complete_episode=True))
                if finish["status"] == "FAILED_COMPONENT_AUDIT" or str(audit["status"]).startswith("FAILED"):
                    self.passive_observation("PASSIVE_COMPLETE_EPISODE_COMPONENT_OBSERVATION", case_id=case_id, arm=arm,
                        evidence=str(out / "AUDIT.json"))
            from tf_fullsuite_eval_v2.report import row as report_row, compare_two
            by = {(case_id, arm): report_row(self.p, self.job, spec, arm) for arm in ("T", "C", "P")}
            observations = [compare_two(self.job, spec, by, a, b) for a, b in (("T", "C"), ("T", "P"), ("C", "P"))]
            result.update(status="COMPLETE_RECORDED_CASE", observations=observations,
                case_qualified="UNKNOWN_NOT_A_QUALITY_CERTIFICATE", permissions=PERMISSIONS)
            for observed in observations:
                if observed["lost"]:
                    self.passive_observation("RECORDED_LOST_SUCCESS_NO_EARLY_STOP", case_id=case_id,
                        comparison=observed["comparison"])
                initial = observed["raw_initial_comparison"]
                if not isinstance(initial, dict) or initial.get("saved_physics_RNG_pixels_exact") is not True or observed["evidence_gaps"]:
                    result["case_qualified"] = False
                    self.passive_observation("RECORDED_INITIAL_OR_COMMAND_AUDIT_GAP_NO_EARLY_STOP", case_id=case_id,
                        comparison=observed["comparison"], evidence=str(self.job / f"CASE-{case_id:04d}-RESULT.json"))
        except BaseException as exc:
            result.update(status="PARTIAL_HARD_FAILURE_NO_RETRY", error=repr(exc), traceback=traceback.format_exc())
            self.stop_hard("CASE_HARD_PROTOCOL_OR_RESOURCE_FAILURE", case_id=case_id, error=repr(exc))
        finally:
            result["finished_at"] = now()
            with self.lock:
                self.results[case_id] = result
                self.current.pop(index, None)
                write(self.job / f"CASE-{case_id:04d}-RESULT.json", result)
            self.status("case_terminal_preserved")
        return result

    def worker_loop(self, index):
        for spec in pending_cases(self.p):
            if spec["execution_worker_index"] != index:
                continue
            if self.hard_event.is_set():
                break
            self.guard("case_barrier_admission")
            if time.time() + 3 * (1200 + 120) > self.work or time.time() + 3 * (1200 + 120) + 900 > self.hard:
                self.event("CASE_NOT_STARTED_INSUFFICIENT_REGISTERED_RESERVE", worker=index, case_id=spec["case_id"])
                break
            result = self.run_case(spec["case_id"])
            if result["status"] != "COMPLETE_RECORDED_CASE":
                break

    def waves(self):
        # Each physical GPU owns its fixed parity list. Entire T/C/P cases stay
        # together and use one persistent target; there is no DDP or task retry.
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="tf-fullsuite") as pool:
            futures = [pool.submit(self.worker_loop, index) for index in range(4)]
            while not all(f.done() for f in futures):
                if self.hard_event.is_set():
                    self.cleanup(gpu_only=True)
                time.sleep(.2)
            for future in as_completed(futures):
                future.result()

    def shutdown_idle_servers(self):
        if self.hard_event.is_set():
            return
        for index, server in self.servers.items():
            if server["proc"].poll() is None:
                path = self.job / f"server-worker{index}" / "SHUTDOWN_REQUESTED"
                if not path.exists():
                    with path.open("x") as f:
                        f.write("all admitted three-arm cases archived; idle persistent server may exit\n")
        until = min(time.time() + 90, self.work)
        while time.time() < until and any(not self.wait_reaped(r) for r in self.servers.values()):
            time.sleep(.2)

    def run_report(self):
        if time.time() + 305 >= self.work or time.time() + 315 >= self.hard:
            self.report_result = dict(status="REPORT_NOT_ADMITTED_BEFORE_INDEPENDENT_WORK_STOP", GPU=False,
                reason="guard stops every birth-pinned child at original work stop; no deadline extension")
            write(self.job / "REPORT-TERMINAL.json", self.report_result)
            return
        child = self.spawn([self.p["va_python"], "-B", OWN / "report.py", "--job", self.job],
            self.job / "report.log", self.environment(0, report=True), self.allowed[:1], "cpu-report")
        until = min(time.time() + 305, self.work)
        while not self.wait_reaped(child) and time.time() < until:
            time.sleep(.1)
        if not self.wait_reaped(child):
            self.signal_owned(child, signal.SIGTERM)
            if not self.wait_reaped(child, timeout=5):
                self.signal_owned(child, signal.SIGKILL)
                self.wait_reaped(child, timeout=5)
        self.report_result = dict(status="REPORT_PROCESS_EXITED" if child["reaped"] else "REPORT_OWNED_REMAIN",
            returncode=child["proc"].returncode, result_present=(self.job / "report-v1/RESULT.json").is_file(),
            GPU=False, no_retry=True)
        write(self.job / "REPORT-TERMINAL.json", self.report_result)

    def execute(self):
        require((self.job / "DEPLOYMENT-COMPLETE.json").is_file(), "actual deployment receipt required before GPU start")
        require((self.job / "GUARD-STARTED.json").is_file(), "independent absolute guard must already run")
        guard = read(self.job / "GUARD-STARTED.json")
        actual_guard = process(guard["pid"])
        require(actual_guard is not None and actual_guard["birth_ticks"] == guard["birth_ticks"]
            and guard["hard_deadline"] == self.hard and guard["work_stop"] == self.work,
            "independent guard lifetime/deadlines differ")
        require(not (self.job / "RUN-STARTED.json").exists(), "one runner lifecycle; no retry")
        write(self.job / "RUN-STARTED.json", dict(schema="tf-fullsuite-dual-run-start/v1", pid=os.getpid(),
            birth_ticks=process(os.getpid())["birth_ticks"], at=now(), CPU_affinity=self.affinity,
            independent_guard_pid=guard["pid"], absolute_work_stop=self.p["absolute_work_stop"],
            absolute_cleanup_deadline=self.p["absolute_cleanup_deadline"], permissions=PERMISSIONS))
        failure, gpu_cleanup = None, None
        old_signals = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_signals[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda received, _frame: self.stop_hard("OWNED_RUNNER_SIGNAL", signal=received))
        try:
            self.start_servers()
            self.waves()
        except BaseException as exc:
            failure = dict(error=repr(exc), traceback=traceback.format_exc())
            self.stop_hard("RUNNER_HARD_FAILURE", error=repr(exc))
        finally:
            try:
                self.shutdown_idle_servers()
                gpu_cleanup = self.cleanup(gpu_only=True)
                write(self.job / "GPU-OWNED-CLEANUP.json", gpu_cleanup)
                if gpu_cleanup["all_owned_work_exited"]:
                    self.run_report()
                else:
                    self.report_result = dict(status="REPORT_NOT_ADMITTED_GPU_GROUP_MEMBERS_REMAIN_OR_UNKNOWN",
                        GPU=False, no_retry=True)
                    write(self.job / "REPORT-TERMINAL.json", self.report_result)
            except BaseException as exc:
                failure = dict(previous=failure, cleanup_or_report_error=repr(exc), traceback=traceback.format_exc())
            final_cleanup = self.cleanup()
            write(self.job / "FINAL-OWNED-CLEANUP.json", final_cleanup)
            raw_terminals = []
            for spec in self.p["cases"]:
                for arm in ("T", "C", "P"):
                    path = episode_dir(self.job, spec["case_id"], arm) / "EPISODE-TERMINAL.json"
                    if not path.is_file():
                        continue
                    try:
                        terminal = read(path)
                        raw_terminals.append(dict(case_id=spec["case_id"], arm=arm,
                            complete_episode=terminal.get("complete_episode") is True,
                            success=terminal.get("success"), env_steps=terminal.get("env_steps"),
                            status=terminal.get("status"), source=str(path)))
                    except (OSError, ValueError, TypeError) as exc:
                        raw_terminals.append(dict(case_id=spec["case_id"], arm=arm,
                            complete_episode=False, metadata_error=repr(exc), source=str(path)))
            complete_cases = [k for k, value in self.results.items() if value["status"] == "COMPLETE_RECORDED_CASE"]
            report_ok = self.report_result is not None and self.report_result.get("returncode") == 0
            status = ("COMPLETE_RECORDED_LONG500_THREE_ARM_COVERAGE" if len(complete_cases) + len(self.p["prior_completed_case_ids"]) == 500 and failure is None
                and final_cleanup["all_owned_groups_empty"] and report_ok
                else "RECORDED_LONG500_WITH_CLOSURE_GAP" if len(complete_cases) + len(self.p["prior_completed_case_ids"]) == 500 and failure is None
                else "PARTIAL_LONG500_PASSIVE_OBSERVATIONS" if self.soft_event.is_set() and failure is None
                else "PARTIAL_OR_FAILED_LONG500_COVERAGE")
            write(self.job / "SUPERVISOR-RESULT.json", dict(schema="tf-fullsuite-dual-supervisor/v1", status=status,
                failure=failure, hard_reason=self.hard_reason, passive_observations=self.soft_reasons,
                case_results=[self.results[k] for k in sorted(self.results)], complete_cases=complete_cases,
                complete_episodes_new=sum(r["complete_episode"] for r in raw_terminals),
                prior_completed_case_ids=self.p["prior_completed_case_ids"],
                logical_complete_episodes=sum(r["complete_episode"] for r in raw_terminals) + 3*len(self.p["prior_completed_case_ids"]), raw_episode_terminals=raw_terminals,
                closed_episode_receipts_count=sum(len(v["episodes"]) for v in self.results.values()),
                all_own_children_reaped=final_cleanup["all_roots_reaped"], all_owned_groups_empty=final_cleanup["all_owned_groups_empty"],
                cleanup=final_cleanup, report=self.report_result,
                elapsed_wall_sec=time.monotonic() - self.started_mono, permissions=PERMISSIONS,
                wall_sec_since_original_authorization=time.time() - datetime.fromisoformat(self.p["authorization_origin"]).timestamp(),
                GPU_UUID="UNKNOWN", external_GPU_process_load="UNKNOWN_PARENT_METADATA_NOT_ATTACHED",
                exclusive_GPU_window_verified=False,
                no_quality_certificate=True, performance_claim=False, instance_shutdown="NOT_REQUESTED_NOT_EXECUTED"))
            registered_live = []
            for path in self.job.glob("OWNED-*.json"):
                pin = read(path)
                if pin.get("owner_job") == str(self.job) and matches(pin):
                    registered_live.append(pin)
            write(self.job / "ALL-OWNED-CLEANUP.json", dict(status="OWNED_GROUPS_EMPTY_RUNNER_EXITS_NEXT"
                if final_cleanup["all_owned_groups_empty"] else "OWNED_GROUPS_REMAIN_OR_UNKNOWN",
                all_roots_reaped=final_cleanup["all_roots_reaped"], remaining_registered_live=registered_live,
                all_owned_groups_empty=final_cleanup["all_owned_groups_empty"],
                FORENSIC_group_members=final_cleanup["FORENSIC_group_members"],
                all_registered_other_than_current_runner_exited=not any(p["pid"] != os.getpid() for p in registered_live),
                current_CPU_runner_exits_next=True, independent_guard_observes_final_exit=True,
                instance_shutdown="NOT_REQUESTED_NOT_EXECUTED", at=now(), permissions=PERMISSIONS))
            for sig, old in old_signals.items():
                signal.signal(sig, old)
        return 0 if len(complete_cases) + len(self.p["prior_completed_case_ids"]) == 500 and final_cleanup["all_owned_groups_empty"] and report_ok else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    try:
        p = load(job)
        return Runner(job, p).execute()
    except BaseException as exc:
        path = job / "RUNNER-ENTRY-FAILED.json"
        if not path.exists():
            write(path, dict(status="FAILED_RUNNER_ENTRY_OR_FINALIZATION", error=repr(exc),
                traceback=traceback.format_exc(), at=now(), permissions=PERMISSIONS,
                no_retry=True, instance_shutdown="NOT_REQUESTED_NOT_EXECUTED"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())

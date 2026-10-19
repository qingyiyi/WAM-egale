"""Resource-only adoption of the existing bounded Object queue; no model patches."""
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    args = parser.parse_args()
    ops = args.registration.resolve().parent
    cfg = json.loads(args.registration.read_text())
    base = (Path.home() / "cxz").resolve(strict=True)
    source = Path(cfg["source_root"]).resolve(strict=True)
    job = Path(cfg["parent_job"]).resolve(strict=True)
    assert ops.is_relative_to(base) and source.is_relative_to(base) and job.is_relative_to(base)
    os.chdir(source)
    os.sched_setaffinity(0, {42})
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "投机采样/experiments"))
    from object_capture_recovery_v3.controller import Controller
    from object_final_placement_v1.common import atomic, read, write, require, make_plan, stamp
    from object_final_placement_v2.remote import gpu_metadata, released_primary_slot
    from tf_suite_pipeline_v1.runner import process, matches
    from tf_suite_pipeline_v1.protocol import PERMISSIONS

    class Parallel(Controller):
        def __init__(self):
            super().__init__(job)
            self.pid_map = {}
            self.adopted = False
            self.old_stopped = False
            self.old = cfg["old_controller"]
            self.blocks = {}

        def status(self, state, **extra):
            super().status(state, resource_dispatcher=str(ops),
                max_model_workers_per_GPU=2, admission_blocks=self.blocks, **extra)

        def server(self, item):
            folder = item["job"]
            ready_file = folder / "server-worker0/READY.json"
            if not ready_file.is_file():
                return None
            ready = read(ready_file)
            pid = ready["pid"]
            pin = next((read(p) for p in folder.glob("OWNED-*.json")
                        if read(p).get("pid") == pid), None)
            if pin is None or not matches(pin):
                return None
            actual = process(pid)
            proc = Path("/proc") / str(pid)
            if not actual or (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] in ("Z", "T"):
                return None
            expected = self.r["GPU_UUIDs"][str(item["gpu"])]
            require("GPU-" + ready["GPU_identity"]["uuid"].removeprefix("GPU-") == expected,
                    "actual initialized model GPU UUID differs")
            devices = []
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if re.fullmatch(r"/dev/nvidia\d+", target):
                        devices.append(target)
                except FileNotFoundError:
                    pass
            counter = folder / "server-worker0/LIVE-COUNTER.json"
            if not devices or not counter.is_file() or time.time() - counter.stat().st_mtime > 120:
                return None
            counts = read(counter)
            if counts["target_forward_returned"] <= 0:
                return None
            require(counts["target_forward_failed"] == 0,
                    "live actual CUDA teacher execution failed")
            return dict(pid=pid, birth_ticks=pin["birth_ticks"], uid=pin["uid"],
                        phase=folder.name, GPU=expected, devices=sorted(set(devices)),
                        target_forward_returned=counts["target_forward_returned"],
                        counter_age_seconds=time.time() - counter.stat().st_mtime)

        def free_slots(self):
            result = []
            self.blocks = {}
            for gpu in (0, 1):
                items = [x for x in self.active.values() if x["gpu"] == gpu]
                if len(items) >= 2:
                    continue
                servers = [self.server(x) for x in items]
                if any(x is None for x in servers):
                    self.blocks[str(gpu)] = "WAITING_LIVE_OWNED_READY_AND_CUDA_COUNTER"
                    continue
                uuid, free, observed = gpu_metadata(gpu)
                require(uuid == self.r["GPU_UUIDs"][str(gpu)], "physical UUID differs")
                # With one known live CUDA server, one reported CUDA context is a
                # unique set assignment. Later additions are mapped by set delta.
                # No NSpid conversion, process-name guess, or unknown PID bypass.
                if len(observed) != len(servers):
                    self.blocks[str(gpu)] = "NVML_CONTEXT_COUNT_DIFFERS_FROM_KNOWN_LIVE_SERVERS"
                    continue
                keys = {(gpu, x["pid"], x["birth_ticks"]) for x in servers}
                current = {key: value for key, value in self.pid_map.items()
                           if key in keys and value in observed}
                unassigned = [x for x in servers if (gpu, x["pid"], x["birth_ticks"]) not in current]
                spare = observed - set(current.values())
                if len(unassigned) == len(spare) == 1:
                    x = unassigned[0]
                    key = (gpu, x["pid"], x["birth_ticks"])
                    current[key] = next(iter(spare))
                    with (ops / "CONTEXT-MAPPING.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(at=stamp(), container=x,
                            NVML_host_pid=current[key], observed_NVML=sorted(observed),
                            method="UNIQUE_LIVE_CUDA_CONTEXT_SET_AND_SUBSEQUENT_SET_DELTA")) + "\n")
                elif unassigned or spare:
                    self.blocks[str(gpu)] = "CONTEXT_MAPPING_NOT_UNIQUE"
                    continue
                self.pid_map = {key: value for key, value in self.pid_map.items() if key[0] != gpu}
                self.pid_map.update(current)
                if free < 22000:
                    self.blocks[str(gpu)] = "LESS_THAN_ORIGINAL_22000_MIB_ADMISSION_RESERVE"
                    continue
                for slot in (0, 1):
                    if any(x["slot"] == slot for x in items):
                        continue
                    primary = released_primary_slot(Path(self.r["primary_campaigns"][str(gpu)]), slot)
                    if primary == set():
                        result.append((gpu, slot))
            return result

        def adopt(self):
            require(matches(self.old), "old dispatcher birth/UID/PGID differs")
            command = (Path("/proc") / str(self.old["pid"]) / "cmdline").read_bytes().decode()
            require("object_capture_recovery_v3/controller.py" in command and str(job) in command,
                    "only the exact project CPU dispatcher may be handed over")
            until = time.time() + 60
            while time.time() < until:
                self.guard()
                wchan = (Path("/proc") / str(self.old["pid"]) / "wchan").read_text().strip()
                if wchan in ("hrtimer_nanosleep", "do_nanosleep"):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("no safe sleeping dispatcher checkpoint")
            require(matches(self.old), "old identity changed before pause")
            os.kill(self.old["pid"], signal.SIGSTOP)
            self.old_stopped = True
            until = time.time() + 3
            while process(self.old["pid"])["state"] != "T":
                require(time.time() < until, "old dispatcher did not stop at checkpoint")
                time.sleep(.05)
            snapshot = read(job / "status.json")
            atomic(ops / "OLD-STATUS-AT-HANDOVER.json", snapshot)
            require(snapshot["first200_queued"] is True, "matched baseline and fixed pending queue required")
            for name in snapshot["active"]:
                folder = job / name
                plan = read(folder / "plan.json")
                gpu = plan["gpu_index"]
                slot = plan["port"] - 19900 - gpu * 10
                require(slot in (0, 1), "original fixed port/slot identity")
                self.active[name] = dict(job=folder, gpu=gpu, slot=slot)
            self.finished = snapshot["completed"]
            pending = []
            baseline_gpu = read(job / "MATCHED_TARGET200/plan.json")["gpu_index"]
            for name in snapshot["pending"]:
                match = re.fullmatch(r"FINAL_(\d+)_(\d+)", name)
                require(match is not None, "unexpected pending phase")
                first, last = map(int, match.groups())
                require((first, last) in ((0, 99), (100, 199), (200, 274),
                        (275, 349), (350, 424), (425, 499)), "only original case partitions")
                gpu = baseline_gpu if first < 200 else (0 if first < 350 else 1)
                slot = None if first < 200 else (first - 200 - gpu * 150) // 75
                pending.append(dict(name=name, gpu=gpu, slot=slot,
                    cases=list(range(first, last + 1)), cap=800000 if first < 200 else 900000))
            require(self.free_slots(), "no proven extra slot; retain original dispatcher")
            self.guard()
            old_pointer = read(job / "CONTROLLER-LAUNCH.json")
            atomic(ops / "ORIGINAL-CONTROLLER-LAUNCH.json", old_pointer)
            own = process(os.getpid())
            pin = {key: own[key] for key in ("pid", "birth_ticks", "pgid", "uid")}
            atomic(ops / "OWNED-CONTROLLER.json", pin)
            atomic(job / "CONTROLLER-LAUNCH.json", dict(**pin, resource_only_handover=str(ops),
                superseded_controller_receipt=str(ops / "ORIGINAL-CONTROLLER-LAUNCH.json")))
            self.adopted = True
            require(matches(self.old), "old identity differs before CPU-only cleanup")
            os.kill(self.old["pid"], signal.SIGKILL)
            atomic(ops / "HANDOVER.json", dict(at=stamp(), old=self.old, new=pin,
                old_CPU_dispatcher_signals=["SIGSTOP", "SIGKILL"],
                model_client_guard_signals=[], old_natural_exit=False,
                original_absolute_work_stop=self.r["absolute_work_stop"],
                original_absolute_cleanup_deadline=self.r["absolute_cleanup_deadline"],
                original_budget_reset=False, pending=pending, active=list(self.active)))
            return pending

        def run(self):
            pending = self.adopt()
            selected = read(job / "SELECTED-ON3000.json")
            template = read(job / "FINAL-TEMPLATE.json")
            while pending or self.active:
                self.guard()
                require(sum(p.stat().st_size for p in ops.rglob("*") if p.is_file()) <= 16 * 2**20,
                        "bounded operations evidence storage")
                self.refresh()
                for gpu, slot in self.free_slots():
                    if (gpu, slot) not in self.free_slots():
                        continue
                    item = next((x for x in pending if x["gpu"] == gpu and x["slot"] in (None, slot)), None)
                    if item is None:
                        continue
                    folder = job / item["name"]
                    first = 16 + gpu * 8 + slot * 4
                    config = dict(self.r, T_POOL_source=str(job / "MATCHED_TARGET200"))
                    make_plan(config, folder, template, selected, item["cases"], gpu,
                        dict(server=[first, first + 1], client=[first + 2, first + 3], control=[40 + gpu]),
                        19900 + gpu * 10 + slot, 29700 + gpu * 10 + slot, item["cap"])
                    self.launch(folder, gpu, slot, item["name"])
                    pending.remove(item)
                self.status("RUNNING_OR_QUEUED_ALL_REMOTE_OBJECT500",
                    pending=[x["name"] for x in pending], first200_queued=True,
                    new_matched_Target200_is_additional_resource_debit=True)
                time.sleep(20 if pending or self.active else 0)
            terminal = dict(status="COMPLETE_DESCRIPTIVE_REMOTE_OBJECT500", at=stamp(),
                completed=self.finished, permissions=PERMISSIONS, independent_TEST=False,
                formal_performance=False, resource_dispatcher=str(ops))
            atomic(job / "TERMINAL.json", terminal)
            atomic(ops / "TERMINAL.json", terminal)

    controller = Parallel()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("owned dispatcher stop")))
    try:
        controller.run()
    except BaseException as exc:
        error = dict(status="PARTIAL_RESOURCE_DISPATCH_NO_AUTOMATIC_RETRY", at=stamp(),
                     error=repr(exc), traceback=traceback.format_exc(), adopted=controller.adopted)
        atomic(ops / "TERMINAL.json", error)
        if not controller.adopted and controller.old_stopped and matches(controller.old):
            os.kill(controller.old["pid"], signal.SIGCONT)
            atomic(ops / "OLD-CONTROLLER-RESUMED.json", dict(at=stamp(), pin=controller.old))
        elif controller.adopted:
            for item in controller.active.values():
                flag = item["job"] / "STOP_AFTER_EPISODE"
                if not flag.exists():
                    flag.write_text("resource dispatcher failed; preserve current complete episode; no retry\n")
            atomic(job / "TERMINAL.json", dict(**error, permissions=PERMISSIONS))
        raise


if __name__ == "__main__":
    main()

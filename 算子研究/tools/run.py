"""等待无争用窗口，顺序运行两个独立验证/计时进程，核验自有PID并清理。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .replay import resource, write


def identity(pid):
    path = Path(f"/proc/{pid}")
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, birth_ticks=int(fields[19]), uid=path.stat().st_uid)
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    out = args.registration.parent
    deadline = datetime.fromisoformat(reg["absolute_cleanup_deadline"]).timestamp()
    os.sched_setaffinity(0, {4, 5})
    write(out / "SUPERVISOR-OWNED.json", identity(os.getpid()))
    prior = reg.get("prior_consumed", {})
    result = dict(status="WAITING_IDLE_GPU", jobs=[], GPU_lifecycle_seconds=prior.get("GPU_lifecycle_seconds", 0),
                  full_model_forwards=prior.get("full_model_forwards", 0),
                  block_forwards=prior.get("block_forwards", 0), prior_consumed=prior)
    child = owned = None
    try:
        for index in range(1, reg.get("process_count", 2) + 1):
            clean = 0
            while time.time() < deadline - 300:
                observation = resource()
                fields = [x.strip() for x in observation["gpu"].split(",")]
                idle = not observation["pids"] and fields[0] == reg["physical_gpu_uuid"] and float(fields[1]) <= 256
                clean = clean + 1 if idle else 0
                write(out / "HEARTBEAT.json", dict(status="WAITING_IDLE_GPU", process_index=index,
                    at=datetime.now().astimezone().isoformat(), consecutive_idle=clean, **observation))
                if (out / "STOP").exists():
                    raise InterruptedError("用户/任务停止标记")
                if clean >= 2:
                    break
                time.sleep(15)
            else:
                raise TimeoutError("未获得完整作业/清理窗口")
            remaining = reg["GPU_lifecycle_seconds_cap"] - result["GPU_lifecycle_seconds"]
            if remaining < 300:
                raise TimeoutError("累计GPU生命周期余量不足")
            engines = reg.get("timing_engines_by_process", [["original", "baseline", "optimized"]] * 2)[index - 1]
            expected_model = 6 * (3 + (reg.get("timing_rounds", 3)
                + reg.get("warmup_calls_per_engine", 0)) * len(engines))
            if result["full_model_forwards"] + expected_model > reg["full_model_forward_cap"]:
                raise RuntimeError("没有下一完整worker的forward预算")
            if result["block_forwards"] + expected_model + 6 > reg["block_forward_cap"]:
                raise RuntimeError("没有下一完整worker的block预算")
            directory = out / f"process-{index}"
            command = [sys.executable, "-B", "-m", "tools.replay", "--registration", str(args.registration.resolve()),
                       "--output", str(directory)]
            command += ["--timing-engines", *engines]
            started = time.monotonic()
            receipt = dict(command=command, started_at=datetime.now().astimezone().isoformat())
            with (out / f"process-{index}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                owned = identity(child.pid)
                receipt["owned"] = owned
                write(out / f"process-{index}-OWNED.json", receipt)
                try:
                    child_deadline = time.monotonic() + min(600, remaining - 30, deadline - time.time() - 30)
                    receipt["resource_samples"] = []
                    while child.poll() is None:
                        if time.monotonic() >= child_deadline:
                            raise subprocess.TimeoutExpired(command, 600)
                        try:
                            child.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            observation = resource()
                            receipt["resource_samples"].append(dict(at=datetime.now().astimezone().isoformat(), **observation))
                            if set(observation["pids"].splitlines()) - {str(child.pid)}:
                                receipt["foreign_GPU_process_detected"] = True
                    code = child.returncode
                except subprocess.TimeoutExpired:
                    receipt["timeout"] = True
                    if identity(child.pid) == owned:
                        os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if identity(child.pid) == owned:
                            os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
                    code = child.returncode
            elapsed = time.monotonic() - started
            receipt.update(exit_code=code, natural_exit=not receipt.get("timeout", False), lifecycle_seconds=elapsed,
                           finished_at=datetime.now().astimezone().isoformat(), exited=child.poll() is not None)
            result["GPU_lifecycle_seconds"] += elapsed
            result["jobs"].append(receipt)
            if (directory / "RESULT.json").exists():
                worker = json.loads((directory / "RESULT.json").read_text())
                for field in ("full_model_forwards", "block_forwards"):
                    result[field] += worker[field]
            write(out / "RESOURCE-AND-STOP-REPORT.json", result)
            if code:
                raise RuntimeError("首个不完整worker后不自动续跑第二进程")
            if receipt.get("foreign_GPU_process_detected"):
                raise RuntimeError("监督采样发现GPU争用，不启动第二进程")
            if result["full_model_forwards"] > reg["full_model_forward_cap"] or result["block_forwards"] > reg["block_forward_cap"]:
                raise RuntimeError("累计调用上限")
        result["status"] = "COMPLETE_REGISTERED_PROCESSES"
    except BaseException as error:
        result.update(status="PARTIAL_OR_FAILED", error=repr(error))
    finally:
        if child is not None and child.poll() is None and identity(child.pid) == owned:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if identity(child.pid) == owned:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        result.update(finished_at=datetime.now().astimezone().isoformat(),
                      owned_children_exited=child is None or child.poll() is not None)
        write(out / "RESOURCE-AND-STOP-REPORT.json", result)


if __name__ == "__main__":
    main()

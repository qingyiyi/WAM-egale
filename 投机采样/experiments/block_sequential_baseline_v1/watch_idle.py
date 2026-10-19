"""Bounded CPU-only wait for a changed, clean local GPU timing window.

It launches the registered whole-block matrix at most once. A previous failed
A/A is retained. NVML idle utilization must become <=5 percent twice, unlike
the known 24--25 percent anomaly. No model, CUDA, retries, or foreign signals
are used by this watcher itself.
"""
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    out = args.registration.parent
    os.sched_setaffinity(0, {4})
    born = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    write(out / "WATCH-OWNED.json", dict(pid=os.getpid(), birth_ticks=int(born), uid=os.getuid()))
    deadline = datetime.fromisoformat(reg["cleanup_deadline"]).timestamp()
    started = time.monotonic()
    polls, clean = 0, 0
    state = dict(status="WAITING_CHANGED_CLEAN_LOCAL_TIMING_WINDOW", new_GPU_launches=0)
    try:
        while time.time() < deadline - 300:
            if (out / "STOP_WATCH").exists():
                state["status"] = "STOP_REQUESTED_NO_GPU_LAUNCH"
                break
            processes = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                timeout=15, text=True).strip()
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                timeout=15, text=True).strip()
            fields = [x.strip() for x in gpu.split(",")]
            eligible = (not processes and len(fields) == 4 and fields[0] == reg["expected_gpu_uuid"]
                        and float(fields[1]) == 0 and float(fields[2]) >= 8000 and float(fields[3]) <= 5)
            clean = clean + 1 if eligible else 0
            polls += 1
            state.update(at=datetime.now().astimezone().isoformat(), polls=polls,
                         consecutive_clean=clean, GPU=gpu, process_pids=processes)
            write(out / "WATCH-HEARTBEAT.json", state)
            if clean >= 2:
                state.update(status="ONE_REGISTERED_GPU_ATTEMPT_LAUNCHED", new_GPU_launches=1)
                write(out / "WATCH-HEARTBEAT.json", state)
                with (out / "WATCH-child.stdout").open("x") as log:
                    result = subprocess.run(reg["command"], stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, env=dict(os.environ))
                state.update(status="ONE_ATTEMPT_FINISHED_NO_RETRY", returncode=result.returncode)
                break
            time.sleep(min(60, max(1, deadline - 300 - time.time())))
        else:
            state["status"] = "EXPIRED_NO_ELIGIBLE_GPU_WINDOW"
    except Exception as error:
        state.update(status="WATCH_FAILED_NO_RETRY", error=repr(error))
    finally:
        state.update(finished_at=datetime.now().astimezone().isoformat(),
                     CPU_watch_lifecycle_wall_seconds=time.monotonic() - started)
        write(out / "WATCH-RESULT.json", state)


if __name__ == "__main__":
    main()

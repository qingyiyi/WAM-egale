"""Small CPU attempt supervisor: cap, real PID identity, output and exit receipt."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return dict(pid=pid, starttime_ticks=int(stat.rsplit(")", 1)[1].split()[19]),
                uid=Path(f"/proc/{pid}").stat().st_uid)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    registration = json.loads((args.attempt / "REGISTRATION.json").read_text())
    if registration.get("GPU_count") != 0 or not 0 < registration["wall_clock_cap_seconds"] <= 7200:
        raise PermissionError("CPU supervisor cannot authorize GPU or unbounded work")
    os.sched_setaffinity(0, {4, 5})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    start = time.monotonic()
    receipt = dict(started_at=datetime.now().astimezone().isoformat(),
                   terminal_status="RUNNING_CPU_ONLY", command=command,
                   GPU_count=0, CPU_affinity=[4, 5],
                   original_registration=str((args.attempt / "REGISTRATION.json").resolve()))
    write(args.attempt / "RECEIPT.json", receipt)
    process = None
    owned = None
    with (args.attempt / "stdout.log").open("w") as stdout, (args.attempt / "stderr.log").open("w") as stderr:
        try:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            owned = identity(process.pid)
            write(args.attempt / "OWNED-PROCESS.json", owned)
            code = process.wait(timeout=registration["wall_clock_cap_seconds"])
            receipt.update(exit_code=code,
                           terminal_status="COMPLETE_EXIT0" if code == 0 else "FAILED_EXIT_NONZERO",
                           exit_method="NATURAL")
        except subprocess.TimeoutExpired:
            if owned == identity(process.pid):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=10)
                    exit_method = "OWNED_SIGTERM_AFTER_CAP"
                except subprocess.TimeoutExpired:
                    if owned == identity(process.pid):
                        os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait(timeout=10)
                    exit_method = "OWNED_SIGKILL_AFTER_TERM"
                receipt.update(exit_code=code, terminal_status="TIMEOUT", exit_method=exit_method)
            else:
                raise RuntimeError("PID identity changed; no cleanup signal issued")
        finally:
            receipt.update(finished_at=datetime.now().astimezone().isoformat(),
                           lifecycle_wall_seconds=time.monotonic() - start,
                           owned_process=owned,
                           owned_process_exited=(process is not None and process.poll() is not None))
            write(args.attempt / "RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    raise SystemExit(0 if receipt["terminal_status"] == "COMPLETE_EXIT0" else 1)


if __name__ == "__main__":
    main()

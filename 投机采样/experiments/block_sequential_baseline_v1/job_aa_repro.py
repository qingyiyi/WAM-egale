"""Future dedicated-worker supervisor; no execution without new authorization.

This file is prepared and CPU negative-tested only. It does not create an
authorization, infer an unused old budget, rent a GPU, or start an episode.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .cpu_job import identity, write


def main():
    started = time.monotonic()
    from .runtime import require_gpu_authorization
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    auth = require_gpu_authorization(args.authorization, "accuracy")
    if not 0 < auth.get("storage_cap_bytes", 0) <= 2**30:
        raise PermissionError("An explicit storage cap no larger than 1 GiB is required")
    os.sched_setaffinity(0, {4, 5})
    args.attempt.mkdir(parents=True, exist_ok=False)
    write(args.attempt / "REGISTRATION.json", auth)
    deadline = datetime.fromisoformat(auth["cleanup_deadline"])
    cap = min(auth["gpu_lifecycle_seconds_cap"] - (time.monotonic() - started),
              (deadline - datetime.now().astimezone()).total_seconds())
    if cap < 300:
        raise PermissionError("No full initialization/cleanup reserve remains")
    command = [sys.executable, "-B", "-m", "experiments.block_sequential_baseline_v1.gpu_aa_repro",
               "--authorization", str(args.authorization.resolve()),
               "--repository", str(args.repository.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
               "--fixtures", str(args.fixtures.resolve()), "--output", str((args.attempt / "matrix").resolve())]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    receipt = dict(started_at=datetime.now().astimezone().isoformat(), terminal_status="RUNNING_AUTHORIZED_GPU_WORKER",
                   command=command, original_budget_origin_unchanged=True)
    process = None
    owned = None
    write(args.attempt / "RECEIPT.json", receipt)
    with (args.attempt / "stdout.log").open("w") as stdout, (args.attempt / "stderr.log").open("w") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            owned = identity(process.pid)
            write(args.attempt / "OWNED-PROCESS.json", owned)
            code = process.wait(timeout=max(1, cap - 15))
            receipt.update(exit_code=code, terminal_status="COMPLETE_EXIT0" if code == 0 else "FAILED_EXIT_NONZERO", exit_method="NATURAL")
        except subprocess.TimeoutExpired:
            if owned != identity(process.pid):
                raise RuntimeError("Owned PID identity changed; no signal sent")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
                method = "OWNED_SIGTERM_AFTER_CAP"
            except subprocess.TimeoutExpired:
                if owned == identity(process.pid):
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
                method = "OWNED_SIGKILL_AFTER_TERM"
            receipt.update(exit_code=code, terminal_status="TIMEOUT", exit_method=method)
        except Exception as error:
            receipt.update(terminal_status="FAILED_SUPERVISOR_OR_SPAWN",
                           reason=dict(type=type(error).__name__, message=str(error)))
            raise
        finally:
            if process is not None and process.poll() is None and owned is not None:
                if owned == identity(process.pid):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                        receipt["exit_method"] = "OWNED_SIGTERM_DURING_EXCEPTION_CLEANUP"
                    except subprocess.TimeoutExpired:
                        if owned == identity(process.pid):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                        receipt["exit_method"] = "OWNED_SIGKILL_DURING_EXCEPTION_CLEANUP"
            receipt.update(finished_at=datetime.now().astimezone().isoformat(),
                           lifecycle_wall_seconds=time.monotonic() - started,
                           owned_process=owned, owned_process_exited=(process is not None and process.poll() is not None))
            write(args.attempt / "RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    raise SystemExit(0 if receipt["terminal_status"] == "COMPLETE_EXIT0" else 1)


if __name__ == "__main__":
    main()

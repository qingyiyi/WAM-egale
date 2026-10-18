"""CPU-only bounded admission wait, then exactly one predeclared matrix job."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
import launch


def main():
    job=Path(sys.argv[1]).resolve(strict=True)
    os.sched_setaffinity(0,{7})
    plan=json.loads((job/'REGISTRATION.json').read_text())
    wait_cap=plan.get('CPU_admission_wait_cap_seconds',1800)
    idle_required=plan.get('continuous_idle_seconds',120)
    started=time.monotonic();until=started+wait_cap;idle_since=None
    launch.write(job/'ADMISSION-WATCH-STARTED.json',dict(**launch.pin(os.getpid()),
        at=datetime.now(timezone.utc).isoformat(),CPU=1,CPU_wait_cap_seconds=wait_cap,GPU_jobs_started=0))
    while time.monotonic()<until and not (job/'CANCEL-ADMISSION').exists():
        sample=launch.gpu()
        launch.write(job/'ADMISSION-STATUS.json',dict(at=datetime.now(timezone.utc).isoformat(),
            state='WAITING_IDLE_GPU_CPU_ONLY',sample=sample,wait_wall_seconds=time.monotonic()-started,GPU_jobs_started=0))
        idle=not sample['compute_pids'] and not sample['graphics_pids'] and sample['utilization']<=5 and sample['used_MiB']<=256
        if idle:
            if idle_since is None:idle_since=time.monotonic()
        else:idle_since=None
        if idle_since is not None and time.monotonic()-idle_since>=idle_required:
            launch.write(job/'ADMISSION-RELEASED.json',dict(at=datetime.now(timezone.utc).isoformat(),
                CPU_wait_wall_seconds=time.monotonic()-started,resource_only_not_numeric_retry=True))
            code=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('launch.py')),str(job)]).returncode
            launch.write(job/'ADMISSION-WATCH-EXIT.json',dict(returncode=code,actual_job_launched_once=True,
                total_wall_seconds=time.monotonic()-started))
            return code
        time.sleep(30)
    launch.write(job/'ADMISSION-WATCH-EXIT.json',dict(status='BLOCKED_NO_IDLE_WINDOW_OR_CANCELLED',
        GPU_jobs_started=0,total_wall_seconds=time.monotonic()-started))
    return 2


if __name__=='__main__':raise SystemExit(main())

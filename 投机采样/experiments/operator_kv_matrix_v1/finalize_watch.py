"""Bounded CPU-only result return; never starts or retries a GPU worker."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'.pipeline/experiments/operator-kv-matrix-20261018'


def write(path,value):
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    pending.replace(path)


def main():
    job=Path(sys.argv[1]).resolve(strict=True)
    os.sched_setaffinity(0,{7})
    started=time.monotonic();cap=3600
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    write(job/'CPU-RETURN-STARTED.json',dict(pid=os.getpid(),birth_ticks=int(stat[19]),uid=os.getuid(),
        at=datetime.now(timezone.utc).isoformat(),CPU=1,wall_cap_seconds=cap,
        no_GPU=True,no_SHA=True,no_retry=True,output=str(OUT),extra_storage_cap_bytes=32*2**20))
    outcome=dict(status='CPU_RESULT_RETURN_PENDING',GPU_jobs_started=0)
    try:
        while time.monotonic()-started<cap:
            terminal=job/'TERMINAL.json';admission=job/'ADMISSION-WATCH-EXIT.json'
            if terminal.exists() or admission.exists():break
            time.sleep(15)
        else:raise TimeoutError('CPU result return cap; no GPU or controller was signalled')
        with (job/'CPU-AGGREGATION.log').open('xb') as log:
            completed=subprocess.run([sys.executable,'-B',str(OUT/'analyze.py'),str(job)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        outcome.update(status='CPU_AGGREGATION_COMPLETE' if completed.returncode==0 else 'CPU_AGGREGATION_FAILED',
            aggregation_returncode=completed.returncode,child_exit_observed=True)
        if (OUT/'SUMMARY.json').exists():
            summary=json.loads((OUT/'SUMMARY.json').read_text())
            outcome.update(matrix_audit_complete=summary['audit_complete'],points=summary['points'],issues=summary['issues'])
        if terminal.exists():outcome['GPU_job_terminal']=json.loads(terminal.read_text())
        if admission.exists():outcome['admission_terminal']=json.loads(admission.read_text())
    except BaseException as exc:
        outcome.update(status='CPU_RESULT_RETURN_FAILED',error=repr(exc),traceback=traceback.format_exc())
    finally:
        outcome.update(at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
            no_GPU_workers_started_by_returner=True,no_foreign_processes_signalled=True)
        write(job/'CPU-RETURN-RESULT.json',outcome)
        write(OUT/'CURRENT-STATUS.json',dict(runtime=str(job),**outcome))
    return 0 if outcome['status']=='CPU_AGGREGATION_COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())

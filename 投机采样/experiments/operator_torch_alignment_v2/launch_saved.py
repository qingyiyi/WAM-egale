"""Bounded shared-GPU correctness child with birth-checked cleanup, no exclusive lock."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'operator_kv_matrix_v1'))
import launch


def main():
    job=Path(sys.argv[1]).resolve(strict=True);plan=json.loads((job/'REGISTRATION.json').read_text())
    os.sched_setaffinity(0,{7});started=time.monotonic();child=None;identity=None;failure=None
    launch.write(job/'STARTED.json',dict(at=datetime.now(timezone.utc).isoformat(),**launch.pin(os.getpid())))
    def stop(*_):raise InterruptedError('owned saved-input launcher stop')
    signal.signal(signal.SIGTERM,stop)
    try:
        sample=launch.gpu();launch.write(job/'RESOURCE-BEFORE.json',sample)
        assert sample['free_MiB']>=8192,'at least8GiB free for operator-only shared verification'
        env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with (job/'worker.log').open('xb') as log:
            child=subprocess.Popen([plan['python'],'-B',str(Path(__file__).with_name('saved_input_validate.py')),str(job)],
                env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            identity=launch.pin(child.pid);assert identity and identity['uid']==os.getuid()
            launch.write(job/'OWNED-PROCESS.json',identity)
            while child.poll() is None:
                assert time.monotonic()-started<plan['GPU_lifecycle_cap_seconds']-30,'saved input lifecycle cleanup reserve'
                time.sleep(.25)
            assert child.wait()==0,'saved input verification child failed'
    except BaseException as exc:failure=repr(exc);raise
    finally:
        if child is not None and child.poll() is None and identity is not None and launch.matches(identity):
            os.killpg(identity['pgid'],signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if launch.matches(identity):os.killpg(identity['pgid'],signal.SIGKILL)
                child.wait(timeout=10)
        launch.write(job/'TERMINAL.json',dict(status='COMPLETE_OWNED_CLEANUP' if failure is None else 'PARTIAL',error=failure,
            at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
            returncode=child.returncode if child else None,owned_child_exited=child is None or child.poll() is not None,
            no_foreign_processes_signalled=True,timing_performed=False))


if __name__=='__main__':main()

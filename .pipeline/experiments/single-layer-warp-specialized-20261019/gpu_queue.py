"""Wait for GPU0's timing queue to close, then run finite operator jobs.

No target policy or speculative source is edited. GPU1 timing may continue.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

BASE=Path('/root/cxz/ops/whole-layer-warp-20261020')
WORK=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究')
TIMING=Path('/root/cxz/performance/libero-four-suite-20261019/isolated')
UUID='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'


def write(name,data):
    p=BASE/name;q=p.with_suffix('.pending');q.write_text(json.dumps(data,indent=2)+'\n');q.replace(p)


def pin(pid):
    return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),
                pgid=os.getpgid(pid),uid=Path(f'/proc/{pid}').stat().st_uid)


def alive(identity):
    try:return pin(identity['pid'])==identity
    except (FileNotFoundError,ProcessLookupError):return False


def contexts():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    return [x.strip().split(',')[1].strip() for x in raw.splitlines() if x.split(',')[0].strip()==UUID]


def main():
    os.sched_setaffinity(0,{4,5})
    state=dict(status='CPU_WAITING_FOR_GPU0_TIMING_RELEASE',jobs=[],GPU=UUID,
        GPU_context_in_queue=False,owned_queue=pin(os.getpid()),
        CPU_wait_start=datetime.now(timezone.utc).isoformat(),
        absolute_wait_deadline='2026-10-20T00:15:00+00:00',
        GPU_lifecycle_cap_seconds=3600,block_attempts_cap=600,full30_calls=0,robot_episodes=0)
    write('GPU-QUEUE-STATUS.json',state)
    try:
        while True:
            if time.time()>datetime.fromisoformat(state['absolute_wait_deadline']).timestamp():
                raise TimeoutError('morning CPU wait deadline; no automatic new lease')
            p=TIMING/'GPU0-QUEUE-STATUS.json';q=json.loads(p.read_text())
            # Require all six queued phases to be terminal and no active phase.
            finished=q.get('active') is None and len(q.get('completed',[]))==6
            # A stopped CPU admission attempt is also a released card, provided
            # its independent guard is CLEAN and it never started a GPU server.
            # Failed timing phase is preserved and is not re-run here.
            released_partial=False
            if q.get('status')=='STOPPED_OR_PARTIAL_ISOLATED_QUEUE' and q.get('active'):
                job=Path(q['active']['job']);f=job/'GUARD-FINAL.json'
                released_partial=(f.exists() and json.loads(f.read_text()).get('status')=='CLEAN'
                    and not (job/'server-worker0').exists()
                    and not list(job.glob('episodes/*/*/EPISODE-TERMINAL.json')))
            finished=finished or released_partial
            builds=sorted(BASE.glob('BUILD-*-RESULT.json'),key=lambda p:p.stat().st_mtime)
            built=builds and json.loads(builds[-1].read_text())['status']=='COMPILED'
            if finished and built and not contexts():break
            state.update(timing_status=q.get('status'),timing_completed=len(q.get('completed',[])),
                compiler_complete=bool(built),checked_at=datetime.now(timezone.utc).isoformat())
            write('GPU-QUEUE-STATUS.json',state);time.sleep(30)

        t0=datetime.now(timezone.utc)
        work_stop=min(t0+timedelta(minutes=50),datetime.fromisoformat('2026-10-20T00:30:00+00:00'))
        cleanup=work_stop+timedelta(minutes=10)
        reg=dict(goal='one complete layer in one cooperative persistent warp-specialized kernel',
            user_authorization='latest user: implement now; tomorrow speed/precision/issues',
            physical_gpu_uuid=UUID,actual_GPU_T0=t0.isoformat(),
            absolute_work_stop=work_stop.isoformat(),absolute_cleanup_deadline=cleanup.isoformat(),
            GPU_lifecycle_cap_seconds=3600,full_block_forward_cap=600,new_storage_cap_bytes=3*1024**3,
            max_CPU_threads=2,full30=0,episodes=0,baseline_frozen=True,
            production_admitted=False,formal_performance_permission=False,
            input_manifest='M32 BLOCK-INPUT-MANIFEST.json; nine unchanged actual block fixtures',
            source_files={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (WORK/'wam_ops/optimized/csrc/whole_layer.cu',WORK/'wam_ops/optimized/whole_layer.py',
                          WORK/'wam_ops/optimized/build_whole_layer.py')})
        write('GPU-REGISTRATION.json',reg)
        state.update(status='GPU_RUNNING_BOUNDED_OPERATOR_RESEARCH',actual_GPU_T0=t0.isoformat())
        write('GPU-QUEUE-STATUS.json',state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(BASE/'tmp'),
            XDG_CACHE_HOME=str(BASE/'cache'),CUDA_CACHE_PATH=str(BASE/'cache/cuda'),
            TORCH_HOME=str(BASE/'cache/torch'),HF_HOME=str(BASE/'cache/huggingface'))
        for name in ('cuda','torch','huggingface'):(BASE/'cache'/name).mkdir(exist_ok=True)
        for name,timing in [('validate-1',False),('timing-1',True),('timing-2',True)]:
            if time.time()+700>work_stop.timestamp():
                raise TimeoutError('no complete next worker and cleanup reserve')
            if contexts():raise RuntimeError('GPU busy before next operator worker')
            cmd=['/root/cxz/envs/va/bin/python','-B',str(BASE/'validate_whole_layer.py'),
                '--registration',str(BASE/'GPU-REGISTRATION.json'),'--output',str(BASE/name)]
            if timing:cmd.append('--timing')
            env['PYTHONPATH']=str(WORK)
            begin=time.monotonic()
            with (BASE/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=log,stderr=log,start_new_session=True)
                owned=pin(child.pid);write(name+'-OWNED.json',dict(**owned,argv=cmd))
                timed_out=False
                try:rc=child.wait(timeout=660)
                except subprocess.TimeoutExpired:
                    timed_out=True
                    if alive(owned):os.killpg(owned['pgid'],signal.SIGTERM)
                    try:rc=child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        if alive(owned):os.killpg(owned['pgid'],signal.SIGKILL)
                        rc=child.wait(timeout=30)
            record=dict(name=name,owned=owned,returncode=rc,timed_out=timed_out,
                lifecycle_seconds=time.monotonic()-begin,root_reaped=True,remaining_NVML=contexts())
            state['jobs'].append(record);write(name+'-CLEANUP.json',record);write('GPU-QUEUE-STATUS.json',state)
            result_path=BASE/name/'RESULT.json'
            if rc or not result_path.exists():raise RuntimeError(name+' did not complete; no dependent job')
            result=json.loads(result_path.read_text())
            if result['status']=='FAILED_OR_PARTIAL':raise RuntimeError(name+' failed; no dependent job')
            if any(not row['original_vs_whole']['finite_pass'] or not row['tile_exact_once']
                   for row in result['rows']):
                raise RuntimeError('nonfinite/structural failure; no timing')
        state['status']='FINITE_OPERATOR_RESEARCH_COMPLETE_NOT_PRODUCTION_ADMITTED'
    except BaseException as exc:
        state.update(status='STOPPED_FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        state.update(finished_at=datetime.now(timezone.utc).isoformat(),remaining_NVML=contexts(),
                     foreign_processes_signalled=False)
        write('GPU-QUEUE-STATUS.json',state)


if __name__=='__main__':main()

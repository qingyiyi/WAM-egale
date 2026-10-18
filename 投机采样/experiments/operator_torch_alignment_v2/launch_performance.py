"""Two predeclared independent real-input runs; bounded owned-process cleanup."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3]
OWN=Path(__file__).resolve().parent
RUNTIME=ROOT/'投机采样/results/runtime'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def pin(pid):
    p=Path('/proc')/str(pid)
    try:
        f=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,birth_ticks=int(f[19]),uid=p.stat().st_uid,pgid=int(f[2]),state=f[0])
    except FileNotFoundError:return None


def matches(record):
    actual=pin(record['pid'])
    return actual is not None and all(actual[k]==record[k] for k in ('pid','birth_ticks','uid','pgid'))


def gpu():
    a=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.free',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    b=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=10)
    c=subprocess.run(['nvidia-smi','pmon','-c','1','-s','m'],capture_output=True,text=True,timeout=10)
    if a.returncode or b.returncode or c.returncode:raise RuntimeError('resource visibility unavailable')
    fields=[x.strip() for x in a.stdout.strip().split(',')]
    pids=[int(x.strip()) for x in b.stdout.splitlines() if x.strip().isdigit()]
    graphics=[]
    for line in c.stdout.splitlines():
        f=line.split()
        if len(f)>2 and f[0].isdigit() and f[1].isdigit():
            if 'G' in f[2]:graphics.append(int(f[1]))
            if 'C' in f[2]:pids.append(int(f[1]))
    return dict(uuid=fields[0],utilization=int(fields[1]),used_MiB=int(fields[2]),free_MiB=int(fields[3]),
                compute_pids=pids,graphics_pids=graphics)


def main():
    job=Path(sys.argv[1]).resolve(strict=True)
    assert job.parent==RUNTIME and not (job/'LAUNCH-STARTED.json').exists()
    plan=json.loads((job/'REGISTRATION.json').read_text());started=time.monotonic()
    gpu_deadline=min(started+plan['GPU_lifecycle_cap_seconds'],started+plan['absolute_release_unix']-time.time())
    active_cap=min(600,int(gpu_deadline-started)-90)
    assert active_cap>=180,'not enough complete performance capture window'
    write(job/'LAUNCH-STARTED.json',dict(at=datetime.now(timezone.utc).isoformat(),**pin(os.getpid())))
    locks=[];active=None;records=[];status='PARTIAL';error=None
    def stop(*_):raise InterruptedError('owned operator launcher stopped')
    signal.signal(signal.SIGTERM,stop)
    try:
        for name in ('.speculative_minimal_gpu.lock','.shared-prefix-GPU.lock'):
            f=(RUNTIME/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        existing=Path('/tmp/atten-mlir-gpu-exclusive.lock')
        if existing.exists():
            f=existing.open('r');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        for index in (1,):
            idle_since=None;observations=[]
            while True:
                assert time.monotonic()-started<plan['overall_wall_cap_seconds']-900
                assert time.monotonic()+active_cap+60<gpu_deadline,'remaining cumulative GPU lifecycle reserve'
                sample=gpu();observations.append(sample)
                empty_context=not sample['compute_pids'] and not sample['graphics_pids'] and sample['used_MiB']<=256
                idle=empty_context and (sample['utilization']<=5 or plan.get('allow_empty_context_capture',False))
                if idle:
                    if idle_since is None:idle_since=time.monotonic()
                    if time.monotonic()-idle_since>=10:break
                else:idle_since=None
                if len(observations)>=60:raise RuntimeError('confirmed idle GPU not available within admission cap')
                time.sleep(1)
            worker=job/f'process{index}';worker.mkdir()
            source=RUNTIME/'p0-wam-static-prompt-baseline-v2-20261016-r1/SOURCE-PLAN.json'
            shutil.copy2(source,worker/'SOURCE-PLAN.json')
            write(worker/'PLAN.json',dict(fixture_root=str(RUNTIME/'p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4'),
                CPU_worker=[4,5],worker_active_cap_seconds=active_cap,cleanup_cap_seconds=120,
                no_SHA=True,canonical_policy_replays=3,new_robot_episodes=0,
                attention_rounds=3,attention_iterations=10,block_iterations=5,graph_calls=8,
                physical_host_exclusivity_not_proven=True))
            prior=json.loads(source.read_text());ref=prior['frozen_reference']
            env={k:v for k,v in os.environ.items() if not k.startswith(('WAM_P0_','P0_','LINGBOT_'))}
            env.update(CUDA_VISIBLE_DEVICES='0',OPERATOR_MATRIX_JOB=str(worker),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',LINGBOT_ATTN_MODE='torch',LINGBOT_NSYS_PROFILE='off',
                LINGBOT_SEED='20260922',LINGBOT_VA_LIBERO_SUITE='libero_10',
                LINGBOT_VA_LIBERO_CHECKPOINT=ref['target_checkpoint_path'],LINGBOT_VA_LIBERO_CHECKPOINT_ID=ref['target_checkpoint_id'],
                LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=ref['registry'],WAM_VAE_DEVICE='cuda',WAM_P0_PHASE_PROFILE='0',
                WAM_P0_ACTION_DRAFT_MODE='off',WAM_P0_ACTION_DRAFT_SELECTOR='prefix_then_cost',
                WAM_P0_ACTION_DRAFT_THRESHOLD=str(ref['threshold']),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(29870+index),
                RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',PYTHONHASHSEED='0')
            argv=[plan['python'],'-B',str(OWN/'worker_performance.py'),'--config-name','libero','--port',str(19870+index),
                  '--save_root',str(worker/'server')]
            write(worker/'RESOURCE-ADMISSION.json',dict(observations=observations,confirmed_empty_context_seconds=10,
                GPU_utilization_counter_qualified=all(x['utilization']<=5 for x in observations),
                visible_processes_empty=True,latencies_exploratory_due_to_initial_counter=True,
                user_window_confirmation=plan['user_window_confirmation']))
            with (worker/'worker.log').open('xb') as log:
                active=subprocess.Popen(argv,cwd=ROOT/'投机采样/wam-p0-worktree',env=env,
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                identity=pin(active.pid);assert identity is not None and identity['uid']==os.getuid()
                write(worker/'OWNED-PROCESS.json',dict(**identity,argv=argv));clock=time.monotonic()
                while active.poll() is None:
                    assert time.monotonic()-clock<active_cap,'worker lifecycle activity cap'
                    assert time.monotonic()+60<gpu_deadline,'cumulative original GPU lifecycle cap'
                    assert sum(p.stat().st_size for p in job.rglob('*') if p.is_file())<512*2**20,'512MiB new evidence cap'
                    time.sleep(.5)
                rc=active.wait();record=dict(process=index,returncode=rc,wall_seconds=time.monotonic()-clock,
                    actual_exit_observed=True,owned_pid=identity['pid'])
                records.append(record);write(worker/'PROCESS-EXIT.json',record);active=None
                assert rc==0,'complete matrix worker failed; preserve outputs and stop, no pass selection'
                outcome=json.loads((worker/'RESULT.json').read_text())
                assert outcome['status']=='COMPLETE_PERFORMANCE_CAPTURE_VALIDATION_DEFERRED' and len(outcome['points'])==54
            after=gpu();write(worker/'RESOURCE-AFTER.json',after)
        status='COMPLETE_ONE_PROCESS_PERFORMANCE_CAPTURE_VALIDATION_DEFERRED'
    except BaseException as exc:error=repr(exc);raise
    finally:
        if active is not None:
            identity=pin(active.pid)
            if active.poll() is None and identity is not None and identity['uid']==os.getuid() and matches(identity):
                os.killpg(identity['pgid'],signal.SIGTERM)
                try:active.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if matches(identity):os.killpg(identity['pgid'],signal.SIGKILL)
                    active.wait(timeout=10)
            records.append(dict(returncode=active.returncode,owned_pid=active.pid,cleanup_exit_observed=active.poll() is not None))
        for lock in reversed(locks):fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        write(job/'TERMINAL.json',dict(status=status,error=error,processes=records,wall_seconds=time.monotonic()-started,
            owned_children_exited=active is None or active.poll() is not None,no_foreign_processes_signalled=True,
            new_robot_episodes=0,new_hashes_computed=False))


if __name__=='__main__':main()

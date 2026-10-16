"""Independent CPU outer supervisor; only explicit execution starts one GPU worker.

NVML runs in separately pinned short-lived children with bounded waits. The
supervisor itself never calls NVML or CUDA, so it can enforce the deadline and
clean its exact owned worker even if driver introspection blocks.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
import traceback

OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN.parent))
from wam_quality_diagnostic_v1.replay_protocol import ROOT,RUNTIME,load,prepare,read,write,require
VA=Path('/rjs/cxz/miniconda3/envs/va/bin/python')

def execute(job,p):
    from target_trace_draft_fresh_dev_v1.legacy import launcher
    Base,cleanup=launcher()
    class Driver:
        def __init__(self):self.roots=[];self.pins={}
    d=Driver();begin=time.monotonic();deadline=begin+2700;active_deadline=begin+2400
    write(job/'RUN-STARTED.json',dict(status='EXPLICIT_REVIEWED_Q02',pid=os.getpid(),monotonic=begin,
        deadline_monotonic=deadline,active_deadline_monotonic=active_deadline,
        no_NVML_or_CUDA_in_supervisor=True,max_GPU_workers=1,performance_claim=False))
    env={k:v for k,v in os.environ.items() if not k.startswith(('WAM_P0_','P0_','LINGBOT_','FORMAL_SPEED_',
        'PREFIX_ACCEPT_','RESET_TEXT_BANK_','TARGET_TRACE_','Q02_'))}
    env.update(CUDA_VISIBLE_DEVICES='0',LINGBOT_SEED=str(p['seed']),PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
        LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',LINGBOT_ATTN_MODE='torch',LINGBOT_NSYS_PROFILE='off',
        LINGBOT_VA_LIBERO_SUITE='libero_10',LINGBOT_VA_LIBERO_CHECKPOINT=p['reference_plan']['target_checkpoint_path'],
        LINGBOT_VA_LIBERO_CHECKPOINT_ID=p['reference_plan']['target_checkpoint_id'],
        LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=p['reference_plan']['registry'],WAM_VAE_DEVICE='cuda',
        WAM_P0_PHASE_PROFILE='0',WAM_P0_ACTION_DRAFT_MODE='verifier_k2_b2_bonus',WAM_P0_ACTION_DRAFT_SELECTOR='global',
        WAM_P0_ACTION_DRAFT_THRESHOLD=str(p['reference_plan']['threshold']),
        WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=p['reference_plan']['draft_checkpoint'],
        RESET_TEXT_BANK_ENABLED='1',PREFIX_ACCEPT_ENABLED='1',PREFIX_ACCEPT_GPU_APPROVED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
        RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(p['master_port']),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    # No EGL client or simulator is imported or spawned.
    env.pop('MUJOCO_GL',None);env.pop('PYOPENGL_PLATFORM',None)
    sequence=0
    def spawn(argv,log,environment):return Base.spawn(d,argv,log,environment)
    def probe():
        nonlocal sequence
        sequence+=1;out=job/f'GPU-PROBE-{sequence:04d}.json';e=dict(env);e['CUDA_VISIBLE_DEVICES']=''
        proc=spawn([str(VA),'-B',str(OWN/'resource_probe.py'),'--out',str(out)],job/f'GPU-PROBE-{sequence:04d}.log',e)
        until=min(time.monotonic()+15,deadline-20)
        while proc.poll() is None and time.monotonic()<until:time.sleep(.05)
        if proc.poll() is None:
            cleanup([proc],d.pins,deadline=min(time.monotonic()+20,deadline),wait_seconds=0.)
            raise RuntimeError('bounded NVML child timed out; outer supervisor remains independent')
        require(proc.returncode==0 and out.is_file(),'bounded actual NVML probe failed')
        return read(out)
    worker=None;failure=None;closed=None;final_nv=None
    try:
        first=probe();require(first['memory_free_bytes']>=22*2**30,'22GiB admission headroom unavailable; no automatic wait/retry')
        require(shutil.disk_usage(job).free>=5*2**30,'disk floor5GiB required')
        worker=spawn([str(VA),'-B',str(OWN/'replay_worker.py'),'--job',str(job),'--execute-reviewed'],job/'worker.log',env)
        next_probe=0.
        while worker.poll() is None:
            require(time.monotonic()<active_deadline,'40-minute active deadline; five-minute owned cleanup reserve')
            require(not (job/'STOP_REQUESTED').exists(),'STOP_REQUESTED')
            require(shutil.disk_usage(job).free>=5*2**30,'disk floor reached')
            require(sum(x.stat().st_size for x in job.rglob('*') if x.is_file())<=2*2**30,'2GiB replay output cap reached')
            if time.monotonic()>=next_probe:
                row=probe();require(row['memory_free_bytes']>=.5*2**30,'runtime GPU headroom below half GiB')
                next_probe=time.monotonic()+15
            time.sleep(.2)
        require(worker.returncode==0,'worker exited nonzero; original reproduction failure remains a STOP')
        result=read(job/'WORKER-RESULT.json')
        require(result['status']=='COMPLETE_Q02_CONTROLLED_NUMERIC_REPLAY_PENDING_OWNED_CLEANUP'
            and result['same_query_qualified'] is True and result['complete_policy_cache_chunks']==20
            and result['packed_tree_queries']==32 and result['real_candidate_queries']==224
            and result['extra_target_forwards']==256 and result['total_target_forwards']<=4000
            and result['all_per_query_KV_and_RNG_unchanged'] is True,'complete bounded real query result required')
        require(not list(job.rglob('*FAILED*.json')),'failure markers take precedence')
    except BaseException as e:
        failure=dict(status='STOP_Q02_REPLAY_OR_ADMISSION',error=repr(e),traceback=traceback.format_exc(),
            same_query_qualified=False,D01_admitted=False,root_cause='UNKNOWN')
    finally:
        try:
            closed=cleanup(d.roots,d.pins,deadline=deadline,wait_seconds=.5)
            final_nv=probe()
            require(all(x['pid'] not in d.pins for x in final_nv['compute']+final_nv['graphics']),
                    'own process remains in actual final NVML process inventory')
        except BaseException as e:
            if failure is None:failure=dict(status='STOP_Q02_CLEANUP_OR_FINAL_NVML',error=repr(e),same_query_qualified=False,D01_admitted=False)
        finally:
            try:
                closed=cleanup(d.roots,d.pins,deadline=deadline,wait_seconds=0.)
                write(job/'FINAL-OWNED-CLEANUP.json',closed)
            except BaseException as e:
                failure=dict(status='STOP_Q02_FINAL_OWNED_CLEANUP',error=repr(e),same_query_qualified=False,D01_admitted=False)
    if failure is not None:
        if not (job/'FAILED.json').exists():write(job/'FAILED.json',failure)
        write(job/'SUPERVISOR-RESULT.json',dict(status='STOP',failure=failure,cleanup=closed,elapsed_sec=time.monotonic()-begin,
            NVML_independent_outer_supervisor=True,no_automatic_retry=True))
        raise RuntimeError(failure['error'])
    write(job/'RESULT.json',dict(status='COMPLETE_Q02_REAL_INPUT_NUMERIC_DIAGNOSTIC',
        analysis=read(job/'WORKER-RESULT.json'),all_owned_processes_reaped=closed['all_roots_reaped'],
        final_NVML_own_GPU_processes_empty=True,elapsed_sec=time.monotonic()-begin,
        NVML_independent_outer_supervisor=True,root_cause='UNKNOWN',D01_admitted=False,performance_claim=False))

def main():
    q=argparse.ArgumentParser();q.add_argument('--job',type=Path,required=True)
    q.add_argument('--prepare',action='store_true');q.add_argument('--execute-reviewed',action='store_true');a=q.parse_args()
    if a.prepare:
        require(not a.execute_reviewed,'CPU preparation cannot also execute GPU')
        p=prepare(a.job);print(json.dumps(dict(status='PREPARED_CPU_ONLY',job=p['job'],GPU_used=False)));return
    require(a.execute_reviewed,'only --prepare or explicit --execute-reviewed allowed')
    job=a.job.resolve();p=load(job)
    require(not any((job/n).exists() for n in ('RUN-STARTED.json','RESULT.json','FAILED.json','WORKER-RESULT.json')),'one attempt only')
    with socket.socket() as sock:sock.bind(('127.0.0.1',p['master_port']))
    locks=[]
    try:
        for path in (RUNTIME/'.speculative_minimal_gpu.lock',RUNTIME/'.shared-prefix-GPU.lock'):
            f=path.open('a+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:f.close();raise RuntimeError('another local project GPU job owns '+str(path))
            locks.append(f)
        def stop(*_):raise SystemExit('signal requested Q02 owned cleanup')
        signal.signal(signal.SIGTERM,stop)
        execute(job,p)
    finally:
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()

if __name__=='__main__':main()

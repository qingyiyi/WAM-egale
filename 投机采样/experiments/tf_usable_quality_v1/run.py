"""Explicit one-model/one-EGL, bounded four-arm 40-episode DEV per site."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone

EXP=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(EXP))
from tf_usable_quality_v1.protocol import OWN,ROOT,RUNTIME,WORKTREE,load,prepare,read,write
from tf_usable_quality_v1.legacy import launcher

ProvenLauncher,cleanup_roots=launcher()


class Campaign(ProvenLauncher):
    def __init__(self,job,plan):
        super().__init__(job,plan)
        self.model_active=False
        self.last_sample=None
        self.total_target_forwards=0
        self.final_deadline=min(self.deadline,time.monotonic()+max(0,
            (datetime.fromisoformat(plan['absolute_cleanup_deadline'])-
             datetime.now(timezone.utc)).total_seconds()))
        self.task_deadline=min(self.final_deadline-plan['cleanup_reserve_sec'],
            time.monotonic()+max(0,(datetime.fromisoformat(plan['absolute_work_stop'])-
                datetime.now(timezone.utc)).total_seconds()))
        if self.task_deadline<=time.monotonic():
            self.nv.close()
            raise TimeoutError('common two-site work deadline already passed')
        result=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=15,check=True)
        if result.stdout.strip() not in (self.plan['required_gpu_name'],
                                          self.plan['required_gpu_name'].removeprefix('NVIDIA ')):
            self.nv.close()
            raise ValueError('complete DEV GPU model differs from fixed site profile')

    def environment(self,row,out,arm_budget=None):
        prefixes=('WAM_P0_','P0_','LINGBOT_','FORMAL_SPEED_','PREFIX_ACCEPT_','RESET_ONLINE_',
                  'TARGET_TRACE_DEV_','TARGET_TRACE_FOLLOWUP_','TF_USABLE_','RESET_TEXT_BANK_')
        env={k:v for k,v in os.environ.items() if not k.startswith(prefixes)}
        env.update(TF_USABLE_JOB=str(self.job),TF_USABLE_ARM=row['arm'],TF_USABLE_CASE=row['key'],
            TF_USABLE_ARM_BUDGET=str(arm_budget if arm_budget is not None else self.plan['job_target_forward_cap']-self.total_target_forwards),
            PREFIX_ACCEPT_ENABLED='1',PREFIX_ACCEPT_GPU_APPROVED='1',RESET_TEXT_BANK_ENABLED='1',
            CUDA_VISIBLE_DEVICES='0',LINGBOT_SEED=str(self.plan['seed']),PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
            LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',LINGBOT_ATTN_MODE='torch',LINGBOT_NSYS_PROFILE='off',
            LINGBOT_VA_LIBERO_SUITE='libero_10',LINGBOT_VA_LIBERO_CHECKPOINT=self.plan['target_checkpoint_path'],
            LINGBOT_VA_LIBERO_CHECKPOINT_ID=self.plan['target_checkpoint_id'],
            LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=self.plan['registry'],WAM_VAE_DEVICE='cuda',WAM_P0_PHASE_PROFILE='0',
            WAM_P0_ACTION_DRAFT_MODE='off' if row['arm']=='T' else self.plan['mode'],
            WAM_P0_ACTION_DRAFT_SELECTOR='global',WAM_P0_ACTION_DRAFT_THRESHOLD=str(self.plan['threshold']),
            WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=(self.plan['selected'][row['arm']]['checkpoint']
                                               if row['arm']!='T' else self.plan['selected']['TF_CHILD']['checkpoint']),
            MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        env.update({key:str(self.plan['OMP_NUM_THREADS']) for key in
                    ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        return env

    def telemetry(self,stage):
        sample=self.nv.sample()
        row=dict(monotonic=time.monotonic(),stage=stage,nvml=sample,
                 shared_GPU_allowed=True,performance_claim=False,own_root_pids=sorted(self.pins))
        with (self.job/'GPU-TELEMETRY.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        self.last_sample=sample
        return row

    def guard(self,stage):
        self.stage=stage
        if time.monotonic()>self.task_deadline:
            raise TimeoutError('fixed per-site/global active deadline; 300 seconds reserved for cleanup')
        if (self.job/'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED')
        if time.monotonic()>=self.next_sample:
            if shutil.disk_usage(self.job).free<self.plan['disk_floor_GiB']*2**30:
                raise RuntimeError('project disk headroom below five GiB')
            output=sum(f.stat().st_size for f in self.job.rglob('*') if f.is_file())
            if output>self.plan['output_budget_GiB']*2**30:
                raise RuntimeError('complete DEV output exceeds three GiB')
            sample=self.telemetry(stage)['nvml']
            if self.model_active and sample['memory_free_bytes']<self.plan['runtime_free_GiB']*2**30:
                raise RuntimeError('runtime free GPU memory below half GiB')
            self.next_sample=time.monotonic()+15
            write(self.job/'status.json',dict(status='RUNNING',stage=stage,completed=len(self.completed),
                planned=len(self.plan['cases']),active=self.current,elapsed_sec=time.monotonic()-self.started,
                total_target_forwards=self.total_target_forwards,
                performance_claim=False,output_bytes=output),replace=True)

    def admission(self):
        self.next_sample=0
        while True:
            self.guard('waiting_22GiB_headroom')
            if self.last_sample['memory_free_bytes']>=self.plan['initial_free_GiB']*2**30:
                return
            time.sleep(5)

    def ready(self,server,arm):
        limit=min(time.monotonic()+self.plan['model_ready_wall_sec'],self.task_deadline)
        while True:
            self.guard('target_and_real_CAL_setup_'+arm)
            if server.poll() is not None:
                raise RuntimeError('model exited before READY; see server.log')
            path=self.job/f'{arm}-READY.json'
            if path.is_file():
                ready=read(path)
                if (ready.get('status')!='READY' or not ready.get('setup_RNG_restored')
                        or not ready.get('target_RNG_captured_before_draft_and_CAL')
                        or ready.get('full_real_CAL_rows')!=(0 if arm=='T' else 20237)
                        or ready.get('synthetic_queries')!=0
                        or ready.get('arm_target_budget')!=self.plan['job_target_forward_cap']-self.total_target_forwards
                        or ready.get('per_episode_target_cap')!=self.plan['per_episode_target_forward_cap']
                        or ready.get('O1_O2_O3') is not (arm!='T')
                        or ready.get('predictor_checkpoint')!=(None if arm=='T' else self.plan['selected'][arm]['checkpoint'])):
                    raise ValueError('model READY lacks target-only RNG isolation')
                if arm!='T' and ready['target_initial_rng']!=read(self.job/'T-READY.json')['target_initial_rng']:
                    raise ValueError('target-only initial CPU/CUDA RNG differs across arms')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{self.plan["port"]}/healthz',timeout=1) as response:
                        if response.status==200:
                            return ready
                except (OSError,TimeoutError):
                    pass
            if time.monotonic()>limit:
                raise TimeoutError('model/complete real CAL readiness exceeded 15 minutes')
            time.sleep(1)

    def arm(self,arm):
        rows=[r for r in self.plan['cases'] if r['arm']==arm]
        arm_budget=self.plan['job_target_forward_cap']-self.total_target_forwards
        if arm_budget<=0:raise RuntimeError('full-job target-forward cap exhausted before arm')
        root=self.job/'arms'/arm
        root.mkdir(parents=True,exist_ok=False)
        server=client=None
        self.admission()
        command=[self.plan['va_python'],'-m','torch.distributed.run','--nproc_per_node=1',
            '--master_port='+str(self.plan['master_port']),'--local-ranks-filter=0','--tee=3',
            str(OWN/'server_entry.py'),'--config-name','libero','--port',str(self.plan['port']),
            '--save_root',str(root/'server_output')]
        try:
            server=self.spawn(command,root/'server.log',self.environment(rows[0],root,arm_budget))
            self.model_active=True
            self.ready(server,arm)
            for row in rows:
                self.current=row['key']
                out=self.job/'episodes'/row['key']
                out.mkdir(parents=True,exist_ok=False)
                env=self.environment(row,out,arm_budget)
                env.update({key:str(self.plan['client_threads']) for key in
                            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
                cmd=[self.plan['client_python'],str(OWN/'client_entry.py'),'--port',str(self.plan['port']),
                    '--master-port',str(self.plan['master_port']),'--benchmarks','libero_10',
                    '--task-indices',str(row['task']),'--episodes-per-task','1','--episode-start',str(row['init']),
                    '--condition',row['key'],'--campaign-id',self.job.name,'--worker-id',row['key'],
                    '--trace-mode','off','--latency-benchmark','--out-dir',str(out/'client'),
                    '--fail-fast','--no-resume','--worktree',str(WORKTREE)]
                client=self.spawn(cmd,out/'client.log',env)
                limit=min(time.monotonic()+self.plan['per_episode_wall_sec'],self.task_deadline)
                while client.poll() is None:
                    self.guard('complete_LIBERO_task')
                    if (self.job/f'{arm}-MONITOR-FAILED.json').exists():
                        raise RuntimeError('server state/accounting monitor failed')
                    if server.poll() is not None or time.monotonic()>limit:
                        raise RuntimeError('target exit or 40-minute full-task deadline')
                    time.sleep(1)
                if client.returncode:
                    raise RuntimeError('complete DEV client exited nonzero; see client.log')
                (out/'FINISH_REQUESTED').write_text('Complete full task; drain and state snapshots follow\n')
                limit=min(time.monotonic()+180,self.task_deadline)
                while not (out/'FINISH_READY.json').exists():
                    self.guard('complete_task_state_and_drain')
                    if (self.job/f'{arm}-MONITOR-FAILED.json').exists() or server.poll() is not None or time.monotonic()>limit:
                        raise RuntimeError('complete DEV server failed state/accounting close')
                    time.sleep(.5)
                closed=cleanup_roots([client],self.pins,deadline=time.monotonic()+15,wait_seconds=.5)
                write(out/'CLIENT-CLEANUP.json',closed)
                if closed.get('all_roots_reaped') is not True:
                    raise RuntimeError('full task client subtree did not close')
                client=None
                counter=read(out/'COUNTER.json')
                forwards=counter['target_forwards']
                if not isinstance(forwards,int) or not 0<=forwards<=self.plan['per_episode_target_forward_cap']:
                    raise ValueError('per-episode target-forward cap differs')
                self.total_target_forwards+=forwards
                if self.total_target_forwards>self.plan['job_target_forward_cap']:
                    raise RuntimeError('full-job target-forward cap exceeded')
                self.completed.append(row['key'])
                self.next_sample=0
                self.guard('full_task_complete')
            self.current=None
        finally:
            processes=[v for v in (client,server) if v is not None]
            if processes:
                closed=cleanup_roots(processes,self.pins,deadline=min(time.monotonic()+30,self.final_deadline),wait_seconds=.5)
                write(root/'OWNED-CLEANUP.json',closed)
                if closed.get('all_roots_reaped') is not True:
                    raise RuntimeError('arm model/client subtree did not close')
            self.model_active=False
            self.next_sample=0

    def run(self):
        result=None
        failure=None
        try:
            for arm in self.plan['arms']:
                self.arm(arm)
            if self.completed!=[r['key'] for r in self.plan['cases']]:
                raise ValueError('all forty fixed complete episodes required')
            for arm in self.plan['arms']:
                if read(self.job/'arms'/arm/'OWNED-CLEANUP.json').get('all_roots_reaped') is not True:
                    raise ValueError('arm cleanup incomplete: '+arm)
            write(self.job/'COLLECTION.json',dict(status='PASS_COMPLETE_40_TF_USABLE_DEV_TASK_COLLECTION',
                completed=self.completed,full_episodes=len(self.plan['cases']),total_target_forwards=self.total_target_forwards,
                elapsed_sec=time.monotonic()-self.started,
                all_own_processes_cleaned=True,shared_GPU=True,performance_claim=False))
            from tf_usable_quality_v1.analyze import analyze
            result=analyze(self.job)
        except BaseException as exc:
            failure=dict(status='FAILED_COMPLETE_TF_USABLE_DEV_CAMPAIGN',error=repr(exc),
                traceback=traceback.format_exc(),stage=self.stage,completed=self.completed,
                partial_not_a_complete_DEV_result=True,elapsed_sec=time.monotonic()-self.started)
        finally:
            try:
                closed=cleanup_roots(self.roots,self.pins,deadline=min(time.monotonic()+30,self.final_deadline),wait_seconds=.5)
                write(self.job/'FINAL-OWNED-CLEANUP.json',closed)
                if closed.get('all_roots_reaped') is not True:
                    raise RuntimeError('final owned process tree did not close')
            except BaseException as exc:
                if failure is None:
                    failure=dict(status='FAILED_COMPLETE_TF_USABLE_DEV_CAMPAIGN',error=repr(exc),
                        traceback=traceback.format_exc(),stage='final_owned_cleanup',completed=self.completed,
                        partial_not_a_complete_DEV_result=True,elapsed_sec=time.monotonic()-self.started)
            finally:
                try:
                    self.nv.close()
                except BaseException as exc:
                    if failure is None:
                        failure=dict(status='FAILED_COMPLETE_TF_USABLE_DEV_CAMPAIGN',error=repr(exc),
                            traceback=traceback.format_exc(),stage='NVML_close',completed=self.completed,
                            partial_not_a_complete_DEV_result=True,elapsed_sec=time.monotonic()-self.started)
        markers=[str(f.relative_to(self.job)) for f in self.job.rglob('*FAILED*.json')]
        if (self.job/'STOP_REQUESTED').exists():
            markers.append('STOP_REQUESTED')
        if failure is None and time.monotonic()>self.final_deadline:
            failure=dict(status='FAILED_COMPLETE_TF_USABLE_DEV_CAMPAIGN',error='three-hour per-site wall deadline exceeded',
                stage='terminal_deadline',completed=self.completed,partial_not_a_complete_DEV_result=True)
        if failure is None and markers:
            failure=dict(status='FAILED_COMPLETE_TF_USABLE_DEV_CAMPAIGN',error='failure markers: '+repr(markers),
                stage='final_failure_scan',completed=self.completed,partial_not_a_complete_DEV_result=True)
        if failure is None and (result is None or not (self.job/'REPORT.md').is_file()
                                or not (self.job/'PAIRING-PIXELS.json').is_file()
                                or read(self.job/'PAIRING-PIXELS.json').get('physical_RNG_pairing_passed') is not True):
            failure=dict(status='FAILED_COMPLETE_TF_USABLE_DEV_CAMPAIGN',error='report/physical RNG pairing certificate absent',
                stage='final_report_gate',completed=self.completed,partial_not_a_complete_DEV_result=True)
        if failure is None and (self.job/'STOP_REQUESTED').exists():
            failure=dict(status='FAILED_COMPLETE_TF_USABLE_DEV_CAMPAIGN',error='STOP_REQUESTED before terminal result',
                stage='terminal_STOP_gate',completed=self.completed,partial_not_a_complete_DEV_result=True)
        if failure is not None:
            write(self.job/'FAILED.json',failure)
            raise RuntimeError(failure['error'])
        write(self.job/'status.json',dict(status='COMPLETE',completed=len(self.plan['cases']),planned=len(self.plan['cases']),performance_claim=False),replace=True)
        write(self.job/'RESULT.json',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--site',choices=('local','autodl'))
    for name in ('child_checkpoint','cont_checkpoint','mlp_checkpoint','child_training_dir',
                 'cont_training_dir','mlp_training_dir','bank','bank_result','reset_equivalence_result','target_checkpoint_path',
                 'target_checkpoint_id','registry','target_weight_stats','cal_data'):
        parser.add_argument('--'+name.replace('_','-'),required=False)
    args=parser.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='2'
    job=args.job.resolve()
    if args.prepare:
        required=('site','child_checkpoint','cont_checkpoint','mlp_checkpoint','child_training_dir',
                  'cont_training_dir','mlp_training_dir','bank','bank_result','reset_equivalence_result','target_checkpoint_path',
                  'target_checkpoint_id','registry','target_weight_stats','cal_data')
        if any(getattr(args,k) is None for k in required):
            parser.error('--prepare requires all explicit site/model/bank/CAL/provenance inputs')
        opts={k:getattr(args,k) for k in required}
        stats=read(opts['target_weight_stats'])
        opts['target_weight_stats']=stats.get('target_weight_stats',stats) if isinstance(stats,dict) else stats
        p=prepare(job,**opts)
        print(json.dumps(dict(status='PREPARED_CPU_ONLY',job=str(job),full_episodes=len(p['cases']),GPU_used=False)))
        return
    p=load(job)
    if any((job/f).exists() for f in ('COLLECTION.json','RESULT.json','FAILED.json','RUN-STARTED.json')):
        raise ValueError('one-shot fresh DEV campaign; no overwrite or automatic retry')
    for port in (p['port'],p['master_port']):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',port))
    locks=[ROOT/'.pipeline/.chunk-template-gpu.lock',RUNTIME/'.speculative_minimal_gpu.lock',
           RUNTIME/'.shared-prefix-GPU.lock',RUNTIME/'.target-trace-draft-dev-v1.lock',
           RUNTIME/'.target-trace-draft-followup-dev-v1.lock',RUNTIME/'.target-trace-draft-fresh-dev-v1.lock',
           RUNTIME/f'.tf-usable-quality-v1-{p["site"]}.lock']
    locks=[path for path in locks if path.parent.is_dir()]
    if len(locks)<1 or RUNTIME not in [path.parent for path in locks]:
        raise RuntimeError('site runtime GPU lock directory absent')
    opened=[]
    try:
        for path in locks:
            handle=path.open('a+')
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise RuntimeError('another local project GPU job holds '+str(path))
            opened.append(handle)
        write(job/'RUN-STARTED.json',dict(status='EXPLICIT_RUN',pid=os.getpid(),performance_claim=False,
             site=p['site'],global_started=p['global_started'],absolute_work_stop=p['absolute_work_stop'],
             absolute_cleanup_deadline=p['absolute_cleanup_deadline'],max_run_wall_sec=p['max_run_wall_sec']))
        def stop(_signal,_frame):
            raise SystemExit('signal requested owned cleanup')
        signal.signal(signal.SIGTERM,stop)
        try:
            Campaign(job,p).run()
        except BaseException as exc:
            if not (job/'FAILED.json').exists():
                write(job/'FAILED.json',dict(status='FAILED_FRESH_DEV_ADMISSION_OR_CONSTRUCTOR',
                    error=repr(exc),traceback=traceback.format_exc(),partial_not_a_complete_DEV_result=True))
            raise
    finally:
        for handle in reversed(opened):
            fcntl.flock(handle,fcntl.LOCK_UN)
            handle.close()


if __name__=='__main__':
    main()

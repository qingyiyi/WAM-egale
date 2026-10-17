"""Finite suite collection -> fixed TF training -> descriptive T/C/P coverage.

This is an actual job orchestrator, not a test harness. Every episode remains
complete and phase failures stop the chain; there is no automatic retry or
checkpoint selection. No credentials or asset digests are used here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from tf_suite_pipeline_v1.protocol import ROOT,PERMISSIONS,SCHEMA,read,require,write
from tf_suite_pipeline_v1.runner import process,matches,forensic_groups

GPU_PHASES=('T_POOL','PARENT_DATA','CURRENT_DATA','FINAL_0','FINAL_1')
STEPS=('T_POOL','PACK_TARGET','PARENT2000','PARENT_DATA','PACK_PARENT','CHILD2500',
       'CONT3000','CURRENT_DATA','PACK_CURRENT','ON3000','FINAL_0','FINAL_1')
FORWARD_CAPS=dict(T_POOL=801000,PARENT_DATA=120000,CURRENT_DATA=120000,FINAL_0=2200000,FINAL_1=3000000)

def stamp(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch,timezone.utc).isoformat()

class Campaign:
    def __init__(self,job):
        self.job=job.resolve(strict=True);self.p=read(self.job/'registration.json')
        p=self.p
        require(p['schema']=='tf-suite-campaign/v1' and p['permissions']==PERMISSIONS
            and p['source_code_commit'] not in ('','PENDING_NOT_EXECUTABLE'), 'frozen new campaign required')
        require(p['suite'] in ('object','goal','spatial') and p['gpu_index'] in (0,1)
            and p['episode_count_cap']==1560 and p['target_forward_attempt_cap']==6241000
            and p['research_storage_cap_bytes']<=5*2**30,'fixed complete suite and total budgets required')
        require(Path(p['root_path']).resolve()==ROOT,'actual running campaign root differs')
        self.work=datetime.fromisoformat(p['absolute_work_stop']).timestamp()
        self.hard=datetime.fromisoformat(p['absolute_cleanup_deadline']).timestamp()
        origin=datetime.fromisoformat(p['authorization_origin']).timestamp()
        require(origin<self.work<self.hard and self.hard-origin<=96*3600
            and self.hard-self.work>=900,'new finite96h lifetime with cleanup reserve')
        self.started=time.time();self.children=[];self.active={};self.complete=[];self.paused=False
        self.suite='libero_'+p['suite'];self.info=None

    def current(self,status,**fields):
        temp=self.job/'status.pending.json'
        temp.write_text(json.dumps(dict(schema='tf-suite-campaign-status/v1',status=status,suite=self.suite,
            at=stamp(),completed_steps=self.complete,active_steps=list(self.active),
            lifecycle_wall_sec=time.time()-datetime.fromisoformat(self.p['authorization_origin']).timestamp(),
            permissions=PERMISSIONS,**fields),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        os.replace(temp,self.job/'status.json')

    def guard(self):
        if (self.job/'STOP_REQUESTED').exists():raise RuntimeError('campaign hard stop requested')
        if time.time()>=self.work:raise TimeoutError('registered campaign absolute work stop')
        used=0
        for x in self.job.rglob('*'):
            try:
                if x.is_file() and not x.is_symlink():used+=x.stat().st_size
            except FileNotFoundError:
                continue  # atomic publication can remove a temporary name
        if used>self.p['research_storage_cap_bytes']:raise RuntimeError('suite all-phase5GiB storage cap')
        if shutil.disk_usage(self.job).free<5*2**30:raise RuntimeError('data disk5GiB reserve')

    def wait_inputs(self):
        require(not (self.job/'CAMPAIGN-STARTED.json').exists(),'one attempt; no implicit continuation')
        pin=process(os.getpid())
        write(self.job/'CAMPAIGN-STARTED.json',dict(pid=os.getpid(),birth_ticks=pin['birth_ticks'],
            pgid=pin['pgid'],uid=pin['uid'],at=stamp(),source_code_commit=self.p['source_code_commit'],
            GPU=0,scope='controller; target processes have separate pins and independent guards',permissions=PERMISSIONS))
        while True:
            self.guard()
            if (self.job/'STOP_AFTER_EPISODE').exists():self.paused=True;return False
            missing=[]
            for name in ('asset_ready','bank','prompts'):
                if not Path(self.p[name]).is_file():missing.append(name)
            for dep in self.p.get('wait_for_closed_jobs',[]):
                base=Path(dep)
                if not (base/'ALL-OWNED-CLEANUP.json').is_file() or not (base/'GUARD-FINAL.json').is_file():
                    missing.append('closed_owned_job:'+str(base));continue
                a,b=read(base/'ALL-OWNED-CLEANUP.json'),read(base/'GUARD-FINAL.json')
                if a.get('all_owned_groups_empty') is not True or b.get('status')!='CLEAN':
                    raise RuntimeError('prior owned job not clean; no new GPU admission')
            if not missing:break
            self.current('WAITING_REAL_ASSETS_BANK_OR_PRIOR_CLEANUP',missing=missing);time.sleep(10)
        raw=read(self.p['asset_ready']);model=Path(raw['model_path']).resolve(strict=True)
        require(raw['suite']==self.suite and raw['checkpoint_id']=='Wjjjh/lingbot-va-libero-'+self.p['suite']
            and raw['publisher_revision']==self.p['publisher_revision'],'registered publisher target differs')
        require((model/'transformer/diffusion_pytorch_model.safetensors').stat().st_size==10177831668,
            'complete registered transformer required; no digest')
        prompts=read(self.p['prompts'])[self.p['suite']]['public_task_prompts']
        self.info={k:raw[k] for k in ('suite','checkpoint_id','publisher_revision','norm_stat')}
        self.info.update(model_path=str(model),public_task_prompts=prompts)
        reg=self.job/'SUITE-REGISTRY.json'
        write(reg,dict(schema='independent-community-suite-registry/v1',suites={self.suite:dict(
            status='available',official_checkpoint=self.info['checkpoint_id'],local_path=str(model),
            publisher='Wjjjh',official_release=False,publisher_revision=self.info['publisher_revision'])}))
        write(self.job/'INPUT-MANIFEST.json',dict(suite_info=self.info,bank=self.p['bank'],prompts=self.p['prompts'],
            reset='full512 BF16 original mathematical path',no_digest=True,
            cryptographic_asset_identity='UNKNOWN_NOT_COMPUTED',independent_TEST=False,permissions=PERMISSIONS))
        return True

    def path(self,step):return self.job/'stages'/step

    def selected(self,condition):
        s=read(self.path(condition)/'selected.json')
        return dict(path=s['checkpoint'],architecture='transformer64',parameter_count=53070,
            condition=condition,metadata=s['metadata'])

    def phase_plan(self,step):
        p=self.p;j=self.path(step);j.mkdir(parents=True)
        phase='FINAL_EVAL' if step.startswith('FINAL_') else step
        sub=int(step[-1]) if phase=='FINAL_EVAL' else 0
        affinity=p['CPU_profiles'][sub]
        rows=[]
        if phase=='T_POOL':triples=[(t,i,'T') for i in range(20) for t in range(10)]
        elif phase=='PARENT_DATA':triples=[(t,i,'C') for i in range(20,23) for t in range(10)]
        elif phase=='CURRENT_DATA':triples=[(t,i,'C') for i in range(23,26) for t in range(10)]
        else:
            cases=list(range(0,250)) if sub==0 else list(range(250,500))
            triples=[(c%10,c//10,a) for c in cases for a in (('T','C','P'),('C','P','T'),('P','T','C'))[c%3]]
        for task,init,arm in triples:
            c=init*10+task;r=dict(row_id=len(rows),case_id=c,task=task,init=init,arm=arm,
                worker_index=0,out=str(j/'episodes'/f'case-{c:04d}'/arm))
            r['data_exposure']=('TARGET_FIT' if init<=17 else 'TARGET_CAL' if init<=19 else
                'PARENT_FIT' if init<=21 else 'PARENT_CAL' if init==22 else 'CURRENT_FIT' if init<=24 else
                'CURRENT_CAL' if init==25 else 'UNSEEN_BY_NEW_SUITE_TRAINING_NOT_CERTIFIED_TEST')
            if phase=='FINAL_EVAL' and arm=='T' and init<20:
                r['reuse_source']=dict(job=str(self.path('T_POOL')),out=str(self.path('T_POOL')/'episodes'/f'case-{c:04d}'/'T'))
            rows.append(r)
        plan=dict(schema=SCHEMA,job=str(j),root_path=str(ROOT),suite=self.suite,suite_info=self.info,
            phase=phase,rows=rows,max_GPU_workers=1,gpu_index=p['gpu_index'],port=p['base_port']+sub,
            master_port=p['base_master_port']+sub,server_threads_per_worker=2,client_threads_per_worker=2,
            CPU_affinity=affinity,va_python=p['va_python'],client_python=p['client_python'],
            LIBERO_CONFIG_PATH=p['LIBERO_CONFIG_PATH'],registry=str(self.job/'SUITE-REGISTRY.json'),
            bank=p['bank'],target_source_reference=p['target_source_reference'],
            mode='verifier_k2_b2_bonus',selector='prefix_then_cost',threshold=.007839680526450805,
            evaluation_seed=20260922,horizon=800,no_smoke=True,no_independent_tests=True,no_SHA=True,
            credential_storage=False,source_code_commit=p['source_code_commit'],frozen_permissions=PERMISSIONS,
            permissions=PERMISSIONS,authorization_origin=p['authorization_origin'],absolute_work_stop=p['absolute_work_stop'],
            absolute_cleanup_deadline=p['absolute_cleanup_deadline'],per_episode_wall_sec=1200,
            per_episode_forward_attempt_cap=4000,job_forward_attempt_cap=FORWARD_CAPS[step],
            new_evidence_cap_bytes=p['research_storage_cap_bytes'],remote_data_disk_floor_bytes=5*2**30,
            suite_storage_root=str(self.job/'compact'),collector_output_dir=str(self.job/'compact'/step),
            full_suite_descriptive_coverage=True,training_exposure_disclosed=True,
            formal_performance=False,physical_GPU_assignment_fixed_for_all_arms=True)
        if phase=='FINAL_EVAL':plan['case_ids']=cases
        if phase!='T_POOL':plan['selected']=self.selected({'PARENT_DATA':'PARENT2000','CURRENT_DATA':'CONT3000','FINAL_EVAL':'ON3000'}[phase])
        write(j/'plan.json',plan)
        write(j/'DEPLOYMENT-COMPLETE.json',dict(status='REAL_ASSETS_AND_FROZEN_SOURCE_AVAILABLE',
            at=stamp(),source_code_commit=p['source_code_commit'],asset_ready=p['asset_ready'],bank=p['bank'],
            no_independent_tests=True,no_SHA=True,actual_reset_audit='PENDING_INLINE_COMPLETE_EPISODE',permissions=PERMISSIONS))
        return j

    def env(self):
        e=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
        e['PYTHONPATH']=str(OWN.parent)+':'+e.get('PYTHONPATH','')
        return e

    def child(self,argv,step,affinity,timeout):
        self.guard();previous=os.sched_getaffinity(0);log=self.job/(step+'.log')
        with log.open('xb') as stream:
            try:
                os.sched_setaffinity(0,set(affinity))
                c=subprocess.Popen(list(map(str,argv)),stdout=stream,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,env=self.env(),cwd=ROOT,close_fds=True,start_new_session=True)
            finally:os.sched_setaffinity(0,previous)
        pin=process(c.pid);require(pin is not None and pin['pgid']==c.pid,'fresh owned controller identity')
        record=dict(proc=c,pin=pin,step=step,deadline=min(self.work,time.time()+timeout))
        self.children.append(record)
        write(self.job/('OWNED-'+step+'.json'),dict(**pin,step=step,owner_job=str(self.job),argv=list(map(str,argv))))
        return record

    def reap(self,record):
        if time.time()>record['deadline'] and record['proc'].poll() is None:
            if matches(record['pin']):os.killpg(record['pin']['pgid'],signal.SIGTERM)
            raise TimeoutError('bounded CPU or launcher phase '+record['step'])
        rc=record['proc'].poll()
        if rc is None:return False
        record['proc'].wait()
        if rc!=0:raise RuntimeError('registered phase failed, no retry: '+record['step']+' rc='+str(rc))
        return True

    def cpu_step(self,step):
        p=self.p;j=self.path(step);deadline=min(self.work,time.time()+7200)
        if step.startswith('PACK_'):
            deadline=min(self.work,time.time()+3600)
            source={'PACK_TARGET':'T_POOL','PACK_PARENT':'PARENT_DATA','PACK_CURRENT':'CURRENT_DATA'}[step]
            stage={'PACK_TARGET':'TARGET200','PACK_PARENT':'PARENT30','PACK_CURRENT':'CURRENT30'}[step]
            reg=dict(schema='tf-suite-data-pack-registration/v1',suite=p['suite'],source_stage=stage,
                cpu_threads=2,permissions=PERMISSIONS,deadline_epoch=deadline,output_limit_bytes=64*2**20,
                maximum_wall_seconds=3600)
            regfile=self.job/(step+'-registration.json');write(regfile,reg)
            argv=[p['va_python'],'-B',OWN/'training.py','--pack','--job',j,
                '--collection-job',self.path(source),'--registration',regfile]
        else:
            reg=dict(schema='tf-suite-stage-training-registration/v1',suite=p['suite'],condition=step,
                seed=20260923,cpu_threads=2,permissions=PERMISSIONS,deadline_epoch=deadline,
                output_limit_bytes=512*2**20,maximum_wall_seconds=7200,source_code_commit=p['source_code_commit'],
                target_columns=str(self.path('PACK_TARGET')/'INDEX.json'),
                target_checkpoint_id=self.info['checkpoint_id'],target_checkpoint_path=self.info['model_path'])
            if step!='PARENT2000':
                parent='PARENT2000' if step=='CHILD2500' else 'CHILD2500'
                observer='PACK_CURRENT' if step=='ON3000' else 'PACK_PARENT'
                reg.update(parent_checkpoint=str(self.path(parent)/'checkpoint.pt'),observer_columns=str(self.path(observer)/'INDEX.json'))
            regfile=self.job/(step+'-registration.json');write(regfile,reg)
            prepare=[p['va_python'],'-B',OWN/'training.py','--prepare','--job',j,'--registration',regfile]
            rec=self.child(prepare,step+'-prepare',p['training_CPU'],120)
            while not self.reap(rec):self.guard();time.sleep(1)
            argv=[p['va_python'],'-B',OWN/'training.py','--run','--job',j]
        record=self.child(argv,step,p['training_CPU'],3600 if step.startswith('PACK_') else 7200)
        while not self.reap(record):
            self.guard();self.current('RUNNING_FIXED_CPU_TRAINING_OR_PACK',stage=step);time.sleep(10)
        if step.startswith('PACK_'):require((j/'INDEX.json').is_file(),'data pack result missing')
        else:require((j/'checkpoint.pt').is_file() and (j/'selected.json').is_file(),'fixed final checkpoint missing')
        self.complete.append(step)

    def gpu_admit(self):
        self.guard()
        result=subprocess.run(['nvidia-smi','-i',str(self.p['gpu_index']),
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        if result.returncode:raise RuntimeError('actual GPU metadata unavailable')
        return int(result.stdout.strip())>=self.p.get('new_model_free_MiB',22000)

    def start_gpu(self,step):
        j=self.phase_plan(step)
        self.active[step]=j
        control=self.p['CPU_profiles'][int(step[-1]) if step.startswith('FINAL_') else 0]['control']
        rec=self.child([self.p['va_python'],'-B',OWN/'launch.py','--job',j],step+'-launch',control,180)
        while not self.reap(rec):self.guard();time.sleep(1)
        require((j/'LAUNCH-RESULT.json').is_file(),'actual runner/guard launch missing')

    def inspect_gpu(self,step):
        j=self.active[step]
        if (self.job/'STOP_AFTER_EPISODE').exists():
            flag=j/'STOP_AFTER_EPISODE'
            if not flag.exists():flag.write_text('user pause; complete current episode first\n')
        if not (j/'SUPERVISOR-RESULT.json').is_file():return False
        if not (j/'GUARD-FINAL.json').is_file():return False
        s,g=read(j/'SUPERVISOR-RESULT.json'),read(j/'GUARD-FINAL.json')
        require(g['status']=='CLEAN' and s['cleanup']['all_owned_groups_empty'] is True,
            'actual phase cleanup incomplete; no next GPU model')
        if s['status']=='PAUSED_AFTER_COMPLETE_EPISODE':self.paused=True;return True
        require(s['status']=='COMPLETE_RECORDED_SUITE_PHASE' and s['complete_rows']==s['planned_rows'],
            'phase PARTIAL/FAILED, preserve and stop without automatic retry')
        self.complete.append(step);return True

    def stop_gpu_jobs(self,reason):
        for j in self.active.values():
            flag=j/'STOP_REQUESTED'
            if not flag.exists():flag.write_text(reason+'\n')
        until=min(self.hard,time.time()+120)
        while time.time()<until:
            if all((j/'GUARD-FINAL.json').is_file() for j in self.active.values()):break
            time.sleep(1)

    def run(self):
        error=None
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('owned campaign signal')))
        signal.signal(signal.SIGINT,lambda *_:(_ for _ in ()).throw(InterruptedError('owned campaign interrupt')))
        try:
            if not self.wait_inputs():return 2
            for step in STEPS[:-2]:
                self.guard()
                if (self.job/'STOP_AFTER_EPISODE').exists():self.paused=True;break
                if step in GPU_PHASES:
                    while not self.gpu_admit():
                        if (self.job/'STOP_AFTER_EPISODE').exists():self.paused=True;break
                        self.guard();self.current('WAITING_ACTUAL_GPU_MEMORY',next=step);time.sleep(10)
                    if self.paused or (self.job/'STOP_AFTER_EPISODE').exists():self.paused=True;break
                    self.start_gpu(step)
                    while not self.inspect_gpu(step):
                        self.guard();self.current('RUNNING_COMPLETE_EPISODE_PHASE',stage=step);time.sleep(10)
                    del self.active[step]
                    if self.paused:break
                else:self.cpu_step(step)
            pending=['FINAL_0','FINAL_1'] if not self.paused else []
            while pending or self.active:
                self.guard()
                if (self.job/'STOP_AFTER_EPISODE').exists():self.paused=True;pending=[]
                for step in list(self.active):
                    if self.inspect_gpu(step):del self.active[step]
                if pending and not self.paused and self.gpu_admit():self.start_gpu(pending.pop(0))
                if pending or self.active:
                    self.current('FULL_500_CASE_THREE_ARM_EVALUATION',pending_shards=pending);time.sleep(10)
        except BaseException as exc:
            error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            self.stop_gpu_jobs('campaign hard failure: '+type(exc).__name__)
        finally:
            for r in self.children:
                if r['proc'].poll() is None and matches(r['pin']):
                    os.killpg(r['pin']['pgid'],signal.SIGTERM)
                    try:r['proc'].wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if matches(r['pin']):os.killpg(r['pin']['pgid'],signal.SIGKILL)
                        r['proc'].wait(timeout=10)
            proof=forensic_groups([r['pin'] for r in self.children])
            all_done=set(self.complete)==set(STEPS)
            status='COMPLETE_DESCRIPTIVE_SUITE_COVERAGE' if all_done and error is None else 'PAUSED_PARTIAL' if self.paused else 'FAILED_OR_PARTIAL_NO_RETRY'
            write(self.job/'CAMPAIGN-RESULT.json',dict(schema='tf-suite-campaign-terminal/v1',status=status,
                suite=self.suite,completed_steps=self.complete,uncompleted_steps=[s for s in STEPS if s not in self.complete],
                error=error,started_at=stamp(self.started),finished_at=stamp(),controller_processes=proof,
                GPU_phase_cleanup=[dict(job=str(j),guard=read(j/'GUARD-FINAL.json') if (j/'GUARD-FINAL.json').is_file() else 'UNKNOWN') for j in self.active.values()],
                independent_TEST=False,formal_performance=False,permissions=PERMISSIONS,
                instance_shutdown='NOT_REQUESTED_NOT_EXECUTED'))
            self.current(status,error=error)
        return 0 if all_done and error is None else 2

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);p=a.parse_args()
    return Campaign(p.job).run()

if __name__=='__main__':raise SystemExit(main())

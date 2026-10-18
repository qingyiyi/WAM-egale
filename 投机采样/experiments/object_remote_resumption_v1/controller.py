"""Finite all-remote Object continuation; no local GPU or foreign signals."""
from pathlib import Path
from datetime import datetime
import argparse
import copy
import os
import signal
import subprocess
import sys
import time
import traceback

OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from tf_suite_pipeline_v1.protocol import ROOT,PERMISSIONS,read,write,require,load
from tf_suite_pipeline_v1.runner import process,matches
from object_final_placement_v1.common import atomic,stamp,phase_clean,controller_pin,make_plan
from object_final_placement_v2.remote import released_primary_slot,gpu_metadata
from baai_suite_pipeline_v1.runtime import environment


class Controller:
    def __init__(self,job):
        self.job=job.resolve(strict=True);self.r=read(job/'registration.json')
        self.work=datetime.fromisoformat(self.r['absolute_work_stop']).timestamp()
        self.children=[];self.active={};self.finished={};self.current_slot=None
        require(self.r['schema']=='object-all-remote-resumption/v1' and self.r['permissions']==PERMISSIONS,
                'new bounded resource namespace required')
        require(ROOT.is_relative_to((Path.home()/'cxz').resolve()),'only ~/cxz')

    def guard(self):
        require(time.time()<self.work and not (self.job/'STOP_REQUESTED').exists(),'original M17 stop')
        used=2*self.r.get('shared_imported_bytes',0)
        for p in self.job.rglob('*'):
            try:
                if p.is_file() and not p.is_symlink():used+=p.stat().st_size
            except FileNotFoundError:continue
        require(used<=self.r['total_Object_storage_cap_bytes'],'original Object storage cap')

    def status(self,state,**extra):
        atomic(self.job/'status.json',dict(status=state,at=stamp(),active=list(self.active),completed=self.finished,
            local_GPU_jobs=0,permissions=PERMISSIONS,**extra))

    def free_slots(self):
        result=[]
        for gpu in (0,1):
            for slot in (0,1):
                if any(x['gpu']==gpu and x['slot']==slot for x in self.active.values()):continue
                primary=released_primary_slot(Path(self.r['primary_campaigns'][str(gpu)]),slot)
                if primary is None:continue
                same=[x for x in self.active.values() if x['gpu']==gpu]
                if len(primary)+len(same)>=2:continue
                if any(not (x['job']/'server-worker0/READY.json').is_file() for x in same):continue
                uuid,free,observed=gpu_metadata(gpu)
                require(uuid==self.r['GPU_UUIDs'][str(gpu)],'fixed GPU UUID')
                allowed=primary|{read(x['job']/'server-worker0/READY.json')['pid'] for x in same}
                if free>=22000 and observed.issubset(allowed):result.append((gpu,slot))
        return result

    def bind(self,p,job,gpu,slot,cap):
        p=copy.deepcopy(p);first=16+gpu*8+slot*4
        p.update(job=str(job),root_path=str(ROOT),gpu_index=gpu,
            CPU_affinity=dict(server=[first,first+1],client=[first+2,first+3],control=[40+gpu]),
            port=19900+gpu*10+slot,master_port=29700+gpu*10+slot,job_forward_attempt_cap=cap,
            new_evidence_cap_bytes=1024*2**20,
            source_code_commit=self.r['source_code_commit'],migration_schema='object-complete-row-resumption/v1')
        for row in p['rows']:
            row['out']=str(job/'episodes'/f'case-{row["case_id"]:04d}'/row['arm'])
        return p

    def launch(self,job,gpu,slot,name):
        self.guard()
        env=environment(self.r,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        launch=subprocess.run([self.r['va_python'],'-B',str(OWN/'launch.py'),'--job',str(job)],
            cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        write(job/'DISPATCH-RETURN.json',dict(returncode=launch.returncode,stdout=launch.stdout[-3000:],
            stderr=launch.stderr[-3000:],at=stamp(),permissions=PERMISSIONS))
        require(launch.returncode==0 and (job/'LAUNCH-RESULT.json').is_file(),'one real launch, no automatic retry')
        self.active[name]=dict(job=job,gpu=gpu,slot=slot)

    def refresh(self):
        for name,item in list(self.active.items()):
            if not phase_clean(item['job']):continue
            result=read(item['job']/'SUPERVISOR-RESULT.json')
            self.finished[name]=dict(status=result['status'],complete_rows=result['complete_rows'],
                planned_rows=result['planned_rows'],GPU=item['gpu'],new_episodes=result['actual_new_complete_episodes'])
            del self.active[name]
            require(result['status']=='COMPLETE_RECORDED_SUITE_PHASE'
                    and result['complete_rows']==result['planned_rows'],'preserve partial, no retry')

    def cpu(self,argv,name,timeout):
        self.guard();env=environment(self.r,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        old=os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0,set(self.r['training_CPU']))
            with (self.job/(name+'.log')).open('xb') as log:
                p=subprocess.Popen(list(map(str,argv)),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    env=env,cwd=ROOT,start_new_session=True)
        finally:os.sched_setaffinity(0,old)
        pin=process(p.pid);require(pin is not None,'owned CPU identity')
        self.children.append((p,pin));write(self.job/('OWNED-'+name+'.json'),pin)
        until=min(self.work,time.time()+timeout)
        while p.poll() is None:
            self.guard();require(time.time()<until,'bounded CPU step '+name);self.status('RUNNING_FIXED_CPU_'+name)
            time.sleep(2)
        require(p.wait()==0,'fixed CPU step failed, no retry: '+name)

    def pack(self,source,dest,stage,name):
        reg=self.job/(name+'-registration.json')
        write(reg,dict(schema='tf-suite-data-pack-registration/v1',suite='object',source_stage=stage,
            cpu_threads=2,permissions=PERMISSIONS,deadline_epoch=min(self.work,time.time()+3600),
            output_limit_bytes=64*2**20,maximum_wall_seconds=3600))
        self.cpu([self.r['va_python'],'-B',OWN/'training.py','--pack','--job',dest,
            '--collection-job',source,'--registration',reg],name,3600)

    def run(self):
        controller_pin(self.job,'all_remote_Object_fixed_continuation')
        self.pack(self.job/'TARGET_TRAINING_DATA',self.job/'PACK_TARGET','TARGET200','PACK_TARGET')
        while True:
            self.guard();slots=self.free_slots()
            if slots:self.current_slot=slots[0];break
            self.status('WAITING_NATURALLY_FREED_REMOTE_SLOT_CURRENT14',remaining_current_episodes=14)
            time.sleep(30)
        gpu,slot=self.current_slot
        current=self.job/'CURRENT_DATA';original=read(current/'plan.json')
        # This file is new prepared metadata, never a prior experiment result.
        atomic(current/'plan.json',self.bind(original,current,gpu,slot,120000))
        self.launch(current,gpu,slot,'CURRENT_DATA')
        while self.active:self.guard();self.refresh();self.status('RUNNING_REMAINING_CURRENT14');time.sleep(10)
        self.pack(current,self.job/'PACK_CURRENT','CURRENT30','PACK_CURRENT')
        prior=Path(self.r['prior_import']);reg=self.job/'ON3000-registration.json'
        write(reg,dict(schema='tf-suite-stage-training-registration/v1',suite='object',condition='ON3000',
            seed=20260923,cpu_threads=2,permissions=PERMISSIONS,deadline_epoch=min(self.work,time.time()+7200),
            output_limit_bytes=512*2**20,maximum_wall_seconds=7200,source_code_commit=self.r['source_code_commit'],
            target_columns=str(self.job/'PACK_TARGET/INDEX.json'),target_checkpoint_id='Wjjjh/lingbot-va-libero-object',
            target_checkpoint_path=self.r['model_path'],parent_checkpoint=str(prior/'stages/CHILD2500/checkpoint.pt'),
            observer_columns=str(self.job/'PACK_CURRENT/INDEX.json')))
        training=OWN.parent/'tf_suite_pipeline_v1/training.py';on=self.job/'ON3000'
        self.cpu([self.r['va_python'],'-B',training,'--prepare','--job',on,'--registration',reg],'ON3000-prepare',120)
        self.cpu([self.r['va_python'],'-B',training,'--run','--job',on],'ON3000',7200)
        chosen=read(on/'selected.json');selected=dict(path=chosen['checkpoint'],architecture='transformer64',
            parameter_count=53070,condition='ON3000',metadata=chosen['metadata'])
        write(self.job/'SELECTED-ON3000.json',selected)
        # New matched Target200 on this physical GPU is never used to retrain ON.
        base=self.job/'MATCHED_TARGET200';base.mkdir()
        p=read(self.job/'TARGET_TRAINING_DATA/plan.json')
        p=self.bind(p,base,gpu,slot,800000)
        p.pop('migration_readonly_completed_row_ids',None);p.pop('migration_readonly_row_results',None)
        p.update(suite_storage_root=str(self.job/'data'),collector_output_dir=str(self.job/'data/MATCHED_TARGET200'))
        write(base/'plan.json',p);write(base/'DEPLOYMENT-COMPLETE.json',dict(status='REGISTERED_MATCHED_REMOTE_TARGET200',
            not_training_data_for_ON=True,permissions=PERMISSIONS))
        self.launch(base,gpu,slot,'MATCHED_TARGET200')
        template=read(self.job/'FINAL-TEMPLATE.json')
        pending=[]
        for g in (0,1):
            for s in (0,1):
                start=200+g*150+s*75
                pending.append(dict(name=f'FINAL_{start}_{start+74}',gpu=g,slot=s,cases=list(range(start,start+75)),cap=900000))
        added_first200=False
        while pending or self.active:
            self.guard();self.refresh()
            if 'MATCHED_TARGET200' in self.finished and not added_first200:
                for first in (0,100):
                    pending.append(dict(name=f'FINAL_{first}_{first+99}',gpu=gpu,slot=None,
                                        cases=list(range(first,first+100)),cap=800000))
                added_first200=True
            for g,s in self.free_slots():
                if (g,s) not in self.free_slots():continue
                candidates=[x for x in pending if x['gpu']==g and x['slot'] in (None,s)]
                if not candidates:continue
                item=candidates[0];folder=self.job/item['name'];first=16+g*8+s*4
                config=dict(self.r,T_POOL_source=str(base))
                make_plan(config,folder,template,selected,item['cases'],g,
                    dict(server=[first,first+1],client=[first+2,first+3],control=[40+g]),
                    19900+g*10+s,29700+g*10+s,item['cap'])
                self.launch(folder,g,s,item['name']);pending.remove(item)
            self.status('RUNNING_OR_QUEUED_ALL_REMOTE_OBJECT500',pending=[x['name'] for x in pending],
                        first200_queued=added_first200,new_matched_Target200_is_additional_resource_debit=True)
            time.sleep(20 if pending or self.active else 0)
        write(self.job/'TERMINAL.json',dict(status='COMPLETE_DESCRIPTIVE_REMOTE_OBJECT500',
            completed=self.finished,permissions=PERMISSIONS,independent_TEST=False,formal_performance=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    obj=Controller(parser.parse_args().job)
    os.sched_setaffinity(0,{43})
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('owned CPU controller stopped')))
    error=None
    try:obj.run()
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc())
        for item in obj.active.values():
            flag=item['job']/'STOP_AFTER_EPISODE'
            if not flag.exists():flag.write_text('owned controller failed; close current full episode and preserve partial\n')
        if not (obj.job/'TERMINAL.json').exists():write(obj.job/'TERMINAL.json',dict(status='PARTIAL_REMOTE_OBJECT_NO_RETRY',**error,permissions=PERMISSIONS))
        raise
    finally:
        for proc,pin in obj.children:
            if proc.poll() is None and matches(pin):
                os.killpg(pin['pgid'],signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if matches(pin):os.killpg(pin['pgid'],signal.SIGKILL)
                    proc.wait(timeout=10)
        write(obj.job/'CONTROLLER-EXIT.json',dict(at=stamp(),error=error,
            cleanup='independent original guards; phase receipts required',foreign_signals=False,permissions=PERMISSIONS))


if __name__=='__main__':main()

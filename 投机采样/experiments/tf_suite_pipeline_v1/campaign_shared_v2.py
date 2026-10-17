"""New placement-only controller over frozen campaign.py (9aa66fb lineage).

One cross-suite flock serializes two leases per fixed physical GPU. A controller
exit never frees a possibly live model. Original math, rows, caps and producer
entrypoints are inherited unchanged. No model probes, digests or retries.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from tf_suite_pipeline_v1.campaign import Campaign as BaseCampaign, GPU_PHASES, stamp
from tf_suite_pipeline_v1.protocol import read,write,require,PERMISSIONS
from tf_suite_pipeline_v1.runner import process,matches,forensic_groups

LEASE_SCHEMA='tf-suite-shared-placement-lease/v2'
FIXED_GPU={'goal':0,'object':0,'spatial':1}
RESERVE_MiB=16*1024
MINIMUM_FREE_MiB=22000


def alive(pin):
    actual=process(pin['pid'])
    return bool(actual and actual['birth_ticks']==pin['birth_ticks'] and actual['uid']==pin['uid']
        and actual['state'] not in ('Z','X'))


def profile(gpu,slot):
    first=gpu*8+slot*4
    return dict(server=[first,first+1],client=[first+2,first+3],control=[16])


class Placement:
    def __init__(self,root,campaign):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        require(self.root.stat().st_uid==os.getuid(),'shared lease root must belong to current UID')
        self.history=self.root/'history';self.history.mkdir(exist_ok=True)
        self.campaign=campaign

    @contextmanager
    def locked(self):
        fd=os.open(self.root/'GLOBAL.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        require(os.fstat(fd).st_uid==os.getuid(),'shared placement lock UID differs')
        until=min(self.campaign.work,time.time()+30)
        try:
            while True:
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    require(time.time()<until,'bounded shared placement lock wait exceeded')
                    time.sleep(.1)
            yield
        finally:
            fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

    def path(self,gpu,slot):return self.root/f'gpu{gpu}-slot{slot}.json'

    def save(self,path,value,event):
        value=dict(value,updated_at=stamp())
        write(self.history/f"{value['lease_id']}-{time.time_ns()}-{event}.json",dict(event=event,lease=value))
        temp=path.with_name(path.name+f'.tmp-{os.getpid()}-{time.time_ns()}')
        with temp.open('x') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
        return value

    def check(self,path):
        require(not path.is_symlink() and path.stat().st_uid==os.getuid(),'foreign lease file UID/path')
        value=read(path)
        require(value['schema']==LEASE_SCHEMA and value['owner_uid']==os.getuid()
            and value['gpu_index'] in (0,1) and value['slot'] in (0,1)
            and path==self.path(value['gpu_index'],value['slot'])
            and value['controller_pin']['uid']==os.getuid() and value['permissions']==PERMISSIONS,
            'shared suite lease identity differs')
        owner=Path(value['campaign_job']).resolve()
        if value.get('phase_job'):
            phase=Path(value['phase_job']).resolve()
            require(phase.parent==owner/'stages' and phase.name in GPU_PHASES,'lease phase namespace differs')
        return value

    def owned_pins(self,phase):
        result=[]
        for file in phase.glob('OWNED-*.json'):
            pin=read(file)
            require(pin['owner_job']==str(phase) and pin['uid']==os.getuid()
                and pin['pid']==pin['pgid'],'phase owned lifetime namespace differs')
            result.append(pin)
        return result

    def clean(self,lease):
        if not lease.get('phase_job'):return False
        phase=Path(lease['phase_job'])
        try:
            all_closed=read(phase/'ALL-OWNED-CLEANUP.json')
            guard=read(phase/'GUARD-FINAL.json')
            if not (all_closed['all_owned_groups_empty'] is True and guard['status']=='CLEAN'
                and guard['all_owned_groups_empty'] is True
                and all_closed['permissions']==guard['permissions']==PERMISSIONS):return False
            pins=self.owned_pins(phase)
            if any(alive(pin) for pin in pins):return False
            forensic=forensic_groups(pins)
            return forensic['scan_complete'] and not any(g['members'] for g in forensic['groups'])
        except (OSError,ValueError,KeyError,TypeError):return False

    def ready(self,lease):
        if not lease.get('phase_job'):return False
        phase=Path(lease['phase_job'])
        try:
            receipt=read(phase/'server-worker0/READY.json')
            pin=read(phase/f"OWNED-{receipt['pid']}.json")
            return bool(receipt['schema']=='tf-suite-server-ready/v1' and receipt['status']=='READY'
                and receipt['physical_GPU_index']==lease['gpu_index'] and receipt['permissions']==PERMISSIONS
                and pin['role']=='server-0' and pin['owner_job']==str(phase)
                and pin['uid']==os.getuid() and matches(pin) and alive(pin))
        except (OSError,ValueError,KeyError,TypeError):return False

    def stale_unbound(self,lease):
        # No reclamation merely because the controller disappeared. The only
        # crash-only shortcut is an unbound reservation with no possible job
        # or still-owned work from that controller's campaign.
        if alive(lease['controller_pin']):return False
        if lease.get('phase_job') and Path(lease['phase_job']).exists():return False
        owner=Path(lease['campaign_job'])
        stages=owner/'stages'
        if stages.exists():
            for phase in stages.iterdir():
                if not phase.is_dir():continue
                for marker in ('LAUNCH-STARTED.json','RUN-STARTED.json','server-worker0/READY.json'):
                    if (phase/marker).exists():
                        hypothetical=dict(lease,phase_job=str(phase))
                        if not self.clean(hypothetical):return False
                try:
                    pins=self.owned_pins(phase)
                    if any(alive(pin) for pin in pins):return False
                    forensic=forensic_groups(pins)
                    if not forensic['scan_complete'] or any(g['members'] for g in forensic['groups']):return False
                except (OSError,ValueError,KeyError,TypeError):return False
        return True

    def remove(self,path,lease,reason):
        write(self.history/f"{lease['lease_id']}-{time.time_ns()}-RELEASED.json",
            dict(event='RELEASED',reason=reason,lease=lease,released_at=stamp(),permissions=PERMISSIONS))
        path.unlink()

    def refresh(self):
        active=[]
        for gpu in (0,1):
            for slot in (0,1):
                path=self.path(gpu,slot)
                if not path.exists():continue
                lease=self.check(path)
                if self.clean(lease):
                    self.remove(path,lease,'REAL_GUARD_CLEAN_AND_ALL_OWNED_GROUPS_EMPTY');continue
                if self.stale_unbound(lease):
                    self.remove(path,lease,'DEAD_CONTROLLER_NO_PHASE_JOB_AND_NO_OWNED_PHASE_WORK');continue
                actual_ready=self.ready(lease)
                state='READY' if actual_ready else 'PENDING'
                if lease['state']!=state:
                    lease=self.save(path,dict(lease,state=state),state)
                active.append(lease)
        return active

    def acquire(self):
        c=self.campaign;gpu=c.p['gpu_index']
        with self.locked():
            leases=self.refresh()
            occupied={x['slot'] for x in leases if x['gpu_index']==gpu}
            slots=[i for i in (0,1) if i not in occupied]
            if not slots:return None
            if (c.job/'STOP_AFTER_EPISODE').exists():c.paused=True;return None
            c.guard()
            result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            require(result.returncode==0,'actual shared GPU metadata unavailable')
            free=int(result.stdout.strip())
            reserved=sum(RESERVE_MiB for x in leases if x['gpu_index']==gpu and x['state']=='PENDING')
            if free-reserved<MINIMUM_FREE_MiB:return None
            pin=process(os.getpid())
            require(pin is not None and pin['uid']==os.getuid(),'actual suite controller lifetime required')
            identity=f"u{os.getuid()}-p{pin['pid']}-b{pin['birth_ticks']}-{time.time_ns()}"
            lease=dict(schema=LEASE_SCHEMA,lease_id=identity,state='PENDING',gpu_index=gpu,slot=slots[0],
                CPU_affinity=profile(gpu,slots[0]),owner_uid=os.getuid(),controller_pin=pin,
                campaign_job=str(c.job),suite=c.suite,phase_job=None,phase_step=None,created_at=stamp(),
                pending_model_reserve_MiB=RESERVE_MiB,admission_observed_free_MiB=free,
                admission_existing_pending_reserve_MiB=reserved,admission_effective_free_MiB=free-reserved,
                READY_memory_already_in_observed_free=True,permissions=PERMISSIONS)
            return self.save(self.path(gpu,slots[0]),lease,'RESERVED')

    def bind(self,lease,step):
        with self.locked():
            path=self.path(lease['gpu_index'],lease['slot']);current=self.check(path)
            require(current['lease_id']==lease['lease_id'] and alive(current['controller_pin']),
                'current placement lease/lifetime changed')
            require(current.get('phase_job') is None,'placement reservation already bound')
            return self.save(path,dict(current,phase_job=str(self.campaign.path(step)),phase_step=step),'BOUND_PHASE')

    def release_if_clean(self,lease):
        with self.locked():
            path=self.path(lease['gpu_index'],lease['slot'])
            if not path.exists():return True
            current=self.check(path)
            if current['lease_id']!=lease['lease_id']:return True
            if self.clean(current):
                self.remove(path,current,'REAL_GUARD_CLEAN_AND_ALL_OWNED_GROUPS_EMPTY');return True
            return False

    def cancel_unused(self,lease,*,never_launched=False):
        with self.locked():
            path=self.path(lease['gpu_index'],lease['slot'])
            if not path.exists():return True
            current=self.check(path)
            require(current['lease_id']==lease['lease_id'] and current['controller_pin']['pid']==os.getpid()
                and alive(current['controller_pin']),'only own current unlaunched reservation may be cancelled')
            if current.get('phase_job'):
                phase=Path(current['phase_job'])
                if not never_launched or any((phase/n).exists() for n in
                    ('LAUNCH-STARTED.json','RUN-STARTED.json','server-worker0/READY.json')) or self.owned_pins(phase):return False
            self.remove(path,current,'LIVE_CONTROLLER_CANCELLED_NEVER_LAUNCHED_RESERVATION');return True


class SharedCampaign(BaseCampaign):
    def __init__(self,job):
        super().__init__(job)
        require(self.p['gpu_index']==FIXED_GPU[self.p['suite']],'suite physical GPU must remain fixed for all arms')
        require(Path(self.p['placement_lease_root']).is_absolute(),'common absolute placement lease root required')
        self.placement=Placement(self.p['placement_lease_root'],self)
        self.pending_lease=None;self.phase_leases={};self.launch_initiated=set();self.cpu_lock_fd=None
        self.inspected_complete=set()
        require(len(self.p['training_CPU'])==2 and all(i in (16,17,18,19) for i in self.p['training_CPU']),
            'shared CPU training/report placement must stay within16..19')

    def gpu_admit(self):
        self.guard()
        if self.pending_lease is not None:return True
        self.pending_lease=self.placement.acquire()
        return self.pending_lease is not None

    def phase_plan(self,step):
        lease=self.phase_leases[step]
        profiles=self.p['CPU_profiles']
        self.p['CPU_profiles']=[lease['CPU_affinity'],lease['CPU_affinity']]
        try:job=super().phase_plan(step)
        finally:self.p['CPU_profiles']=profiles
        write(job/'PLACEMENT-LEASE.json',dict(schema='tf-suite-phase-placement/v2',lease=lease,
            changed_only='physical CPU placement and shared-resource admission; original math/data/caps unchanged',
            inherited_campaign_source='9aa66fb',permissions=PERMISSIONS))
        return job

    def start_gpu(self,step):
        require(self.pending_lease is not None,'GPU slot must be atomically reserved before launch')
        lease=self.placement.bind(self.pending_lease,step)
        self.pending_lease=None;self.phase_leases[step]=lease
        self.active[step]=self.path(step)  # before any launcher side effect
        job=self.phase_plan(step)
        self.launch_initiated.add(step)
        rec=self.child([self.p['va_python'],'-B',OWN/'launch.py','--job',job],step+'-launch',[16],180)
        while not self.reap(rec):self.guard();time.sleep(1)
        require((job/'LAUNCH-RESULT.json').is_file(),'actual leased runner/guard launch missing')

    def inspect_gpu(self,step):
        if step not in self.inspected_complete:
            complete=super().inspect_gpu(step)
            if not complete:return False
            self.inspected_complete.add(step)
        if not self.placement.release_if_clean(self.phase_leases[step]):return False
        return True

    def child(self,argv,step,affinity,timeout,*,cleanup_report=False):
        if step=='CPU-final-report':
            # Independent corrected audit; frozen campaign.py/report.py stay
            # unchanged and no report/certificate from an old job is rewritten.
            require(cleanup_report,'terminal descriptive report boundary required')
            argv=[self.p['va_python'],'-B',OWN/'report_v2.py','--campaign',self.job,
                '--out',self.job/'reports/final-v2']
            write(self.job/'REPORT-IMPLEMENTATION.json',dict(
                schema='tf-suite-report-selection/v2',report='report_v2.py',
                output=str(self.job/'reports/final-v2'),
                reason='nested counter fields, partial attempts and full coverage cost audit',
                old_report_source_unchanged=True,permissions=PERMISSIONS))
        # Inherited training math/CLI is untouched. Passing the flock FD keeps
        # its CPU slot held if this controller dies while its child survives.
        if self.cpu_lock_fd is None:
            return super().child(argv,step,affinity,timeout,cleanup_report=cleanup_report)
        if cleanup_report:require(time.time()+timeout<self.hard,'bounded CPU report must fit cleanup reserve')
        else:self.guard()
        previous=os.sched_getaffinity(0);log=self.job/(step+'.log')
        with log.open('xb') as stream:
            try:
                os.sched_setaffinity(0,set(affinity))
                child=subprocess.Popen(list(map(str,argv)),stdout=stream,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,env=self.env(),cwd=self.p['root_path'],close_fds=True,
                    pass_fds=(self.cpu_lock_fd,),start_new_session=True)
            finally:os.sched_setaffinity(0,previous)
        pin=process(child.pid)
        require(pin is not None and pin['pgid']==child.pid and pin['uid']==os.getuid(),'fresh leased CPU child identity')
        record=dict(proc=child,pin=pin,step=step,deadline=min(self.work,time.time()+timeout))
        self.children.append(record)
        write(self.job/('OWNED-'+step+'.json'),dict(**pin,step=step,owner_job=str(self.job),argv=list(map(str,argv))))
        return record

    def cpu_step(self,step):
        chosen=None
        while chosen is None:
            self.guard()
            if (self.job/'STOP_AFTER_EPISODE').exists():self.paused=True;return
            for slot in (0,1):
                fd=os.open(self.placement.root/f'CPU-training-{slot}.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
                require(os.fstat(fd).st_uid==os.getuid(),'CPU slot lock UID differs')
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:os.close(fd);continue
                chosen=(slot,fd);break
            if chosen is None:self.current('WAITING_SHARED_CPU_TRAINING_SLOT',next=step);time.sleep(2)
        slot,self.cpu_lock_fd=chosen
        original=self.p['training_CPU'];self.p['training_CPU']=[16+slot*2,17+slot*2]
        first_child=len(self.children)
        try:
            write(self.job/(step+'-CPU-PLACEMENT.json'),dict(schema='tf-suite-cpu-placement/v2',slot=slot,
                CPU_affinity=self.p['training_CPU'],lock_inherited_by_real_child=True,permissions=PERMISSIONS))
            return super().cpu_step(step)
        finally:
            fd=self.cpu_lock_fd;self.cpu_lock_fd=None;self.p['training_CPU']=original
            current=[r for r in self.children[first_child:] if r['proc'].poll() is None]
            # Explicit LOCK_UN would also unlock a surviving child's inherited
            # open-file description. Closing just our FD preserves its lease.
            if not current:fcntl.flock(fd,fcntl.LOCK_UN)
            os.close(fd)

    def stop_gpu_jobs(self,reason):
        for job in self.active.values():
            if job.is_dir() and not (job/'STOP_REQUESTED').exists():
                (job/'STOP_REQUESTED').write_text(reason+'\n')
        until=min(self.hard,time.time()+120)
        while time.time()<until:
            waiting=[step for step in self.active if step in self.launch_initiated
                and not self.placement.clean(self.phase_leases[step])]
            if not waiting:break
            time.sleep(1)

    def run(self):
        try:return super().run()
        finally:
            retained=[]
            if self.pending_lease is not None:
                try:self.placement.cancel_unused(self.pending_lease)
                except BaseException as exc:retained.append(dict(lease=self.pending_lease,error=repr(exc)))
            for step,lease in self.phase_leases.items():
                try:
                    released=(self.placement.release_if_clean(lease) if step in self.launch_initiated
                        else self.placement.cancel_unused(lease,never_launched=True))
                    if not released:retained.append(dict(step=step,lease=lease,reason='OWNED_MODEL_OR_CLEANUP_PROOF_REMAINS_UNKNOWN'))
                except BaseException as exc:retained.append(dict(step=step,lease=lease,error=repr(exc)))
            write(self.job/'SHARED-PLACEMENT-FINAL.json',dict(schema='tf-suite-shared-placement-terminal/v2',
                status='LEASES_RELEASED_ONLY_WITH_PROOF' if not retained else 'LEASES_RETAINED_FOR_LIVE_OR_UNKNOWN_OWNED_WORK',
                retained=retained,controller_exit_does_not_release_model=True,
                fixed_GPU=self.p['gpu_index'],permissions=PERMISSIONS))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    return SharedCampaign(args.job).run()


if __name__=='__main__':raise SystemExit(main())

"""Attach-only CPU clock for an existing TF quality supervisor; no NVML/CUDA.

No workload is launched. Only exact pinned identities or their proved descendants
may be signalled. Non-child exit observation never invents waitpid/returncode.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

EXP=Path(__file__).resolve().parent.parent
OWNERSHIP=EXP/'single_worker_chc_gpu_v4/ownership.py'
SPEC=importlib.util.spec_from_file_location('_tf_usable_clock_ownership',OWNERSHIP)
ownership=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=ownership;SPEC.loader.exec_module(ownership)
MAX_SECONDS=10800
RESERVE=300
ABSOLUTE='2026-10-16T12:48:35+08:00'
WORK='2026-10-16T12:43:35+08:00'

def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    os.replace(temp,path)
def key(p):return p.pid,p.start_ticks,p.uid,p.pgid,p.argv
def lifetime(p):return p.pid,p.start_ticks,p.uid,p.pgid
def proc_state(pid):
    text=Path(f'/proc/{pid}/stat').read_text();return text[text.rfind(')')+2:].split()[0]
def observation(pin):
    try:
        now=ownership.read_process(pin.pid)
        if lifetime(now)!=lifetime(pin):return dict(live=False,kind='original_lifetime_exited_PID_reused')
        if proc_state(pin.pid)=='Z':return dict(live=False,kind='observed_zombie_not_waitpid')
        return dict(live=True,kind='live_or_exec_same_lifetime')
    except (FileNotFoundError,ProcessLookupError):return dict(live=False,kind='observed_proc_absent_not_waitpid')
    except (OSError,RuntimeError) as e:return dict(live=True,kind='inspection_uncertain',error=repr(e))

def identify(job):
    job=Path(job).resolve(strict=True);record=read(job/'RUN-STARTED.json');plan=read(job/'plan.json')
    require(record.get('status')=='EXPLICIT_RUN' and record.get('pid',0)>1,'actual RUN-STARTED supervisor PID required')
    require(record.get('max_run_wall_sec')==plan.get('max_run_wall_sec')==MAX_SECONDS,'fixed per-site3h budget required')
    require(record.get('absolute_cleanup_deadline')==plan.get('absolute_cleanup_deadline')==ABSOLUTE
        and record.get('absolute_work_stop')==plan.get('absolute_work_stop')==WORK,'fixed absolute deadlines required')
    p=ownership.read_process(record['pid']);require(p.uid==os.geteuid() and p.argv,'same UID live quality supervisor required')
    cwd=Path(f'/proc/{p.pid}/cwd').resolve(strict=True)
    paths=[(cwd/word).resolve() if not Path(word).is_absolute() else Path(word).resolve() for word in p.argv]
    require((EXP/'tf_usable_quality_v1/run.py').resolve() in paths,'actual quality run.py source required')
    require('--job' in p.argv and p.argv.index('--job')+1<len(p.argv),'supervisor --job binding missing')
    argument=p.argv[p.argv.index('--job')+1];bound=(cwd/argument).resolve() if not Path(argument).is_absolute() else Path(argument).resolve()
    require(bound==job,'actual supervisor argv belongs to another job')
    require(key(ownership.read_process(p.pid))==key(p),'supervisor changed during attach inspection')
    hz=os.sysconf('SC_CLK_TCK');uptime=float(Path('/proc/uptime').read_text().split()[0]);mono=time.monotonic()
    age=uptime-p.start_ticks/hz;require(age>=0,'invalid actual process birth')
    # Conservative process-birth clock, not attach/RUN file creation time.
    origin=mono-age
    cumulative_path=job/'CUMULATIVE-SITE-BUDGET.json'
    predecessors=[]
    if cumulative_path.exists():
        cumulative=read(cumulative_path)
        require(cumulative.get('schema')=='tf-usable-cumulative-site-budget/v1'
            and cumulative.get('site')==plan.get('site')
            and cumulative.get('max_site_seconds')==MAX_SECONDS
            and cumulative.get('cleanup_reserve_seconds')==RESERVE,
            'cumulative same-site budget contract differs')
        previous=cumulative.get('predecessors')
        require(isinstance(previous,list) and previous,'nonempty predecessor proof required')
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        for prior in previous:
            prior_job=Path(prior['job']).resolve(strict=True)
            require(prior_job!=job,'current job cannot be its own predecessor')
            paths={k:Path(prior[k]).resolve(strict=True) for k in ('run_started','failed','cleanup')}
            require(paths['run_started']==prior_job/'RUN-STARTED.json'
                and paths['failed']==prior_job/'FAILED.json'
                and paths['cleanup']==prior_job/'FINAL-OWNED-CLEANUP.json',
                'predecessor raw proof paths differ')
            old_run=read(paths['run_started']);old_failed=read(paths['failed']);old_clean=read(paths['cleanup'])
            require(old_run.get('status')=='EXPLICIT_RUN' and old_run.get('site')==plan.get('site')
                and old_run.get('pid')==prior['supervisor_pid'] and 'FAILED' in str(old_failed.get('status',''))
                and old_clean.get('all_roots_reaped') is True,'predecessor FAILED/actual cleanup not proved')
            require(prior.get('SC_CLK_TCK')==hz and prior.get('boot_id')==boot
                and 0<prior['supervisor_start_ticks']<=p.start_ticks,
                'predecessor birth must use same boot and clock ticks')
            prior_mono=mono-(uptime-prior['supervisor_start_ticks']/hz)
            require(prior_mono<=mono-age and mono-prior_mono>=0,'predecessor budget cannot start later')
            # The root-authorized birth record preserves the original observed
            # live PID/start ticks. Raw RUN/FAILED/cleanup paths remain unchanged.
            require(prior.get('birth_observed_while_live') is True,
                'predecessor needs explicit original live-birth evidence declaration')
            live_receipt=Path(prior['observed_live_launch_receipt']).resolve(strict=True)
            require(live_receipt==prior_job/'REMOTE-LAUNCH.json','original detached live launch receipt required')
            launch=read(live_receipt);old_pin=launch.get('pin',{})
            require(launch.get('status')=='PINNED_DETACHED_EXPLICIT_REMOTE_LAUNCH'
                and old_pin.get('pid')==prior['supervisor_pid']
                and old_pin.get('start_ticks')==prior['supervisor_start_ticks']
                and old_pin.get('uid')==p.uid,'original observed launch pin differs from predecessor birth')
            if prior.get('started_at'):
                elapsed_wall=(datetime.now(timezone.utc)-datetime.fromisoformat(prior['started_at'])).total_seconds()
                require(abs(elapsed_wall-(mono-prior_mono))<10,
                    'predecessor wall time and same-boot process ticks disagree')
            origin=min(origin,prior_mono)
            predecessors.append(dict(job=str(prior_job),supervisor_pid=prior['supervisor_pid'],
                supervisor_start_ticks=prior['supervisor_start_ticks'],SC_CLK_TCK=hz,boot_id=boot,
                actual_birth_monotonic=prior_mono,raw_proofs={k:str(v) for k,v in paths.items()},
                raw_proof_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
                observed_live_launch_receipt=str(live_receipt),
                observed_live_launch_sha256=hashlib.sha256(live_receipt.read_bytes()).hexdigest()))
    absolute_remaining=(datetime.fromisoformat(ABSOLUTE)-datetime.now(timezone.utc)).total_seconds()
    hard=min(origin+MAX_SECONDS,mono+absolute_remaining)
    work=min(hard-RESERVE,mono+(datetime.fromisoformat(WORK)-datetime.now(timezone.utc)).total_seconds())
    return job,record,plan,p,dict(process_birth_monotonic=origin,hard_stop_monotonic=hard,
        work_stop_monotonic=work,actual_process_age_at_attach_sec=age,max_seconds=MAX_SECONDS,
        cleanup_reserve_seconds=RESERVE,absolute_work_stop=WORK,absolute_cleanup_deadline=ABSOLUTE,
        clock_origin='earliest verified same-site process birth via /proc ticks; never attach/retry time',
        current_process_birth_monotonic=mono-age,predecessors=predecessors,
        cumulative_budget_path=str(cumulative_path) if predecessors else None,
        cumulative_budget_sha256=hashlib.sha256(cumulative_path.read_bytes()).hexdigest() if predecessors else None)

class AttachedClock:
    def __init__(self,job):
        self.job,self.run_record,self.plan,self.root,self.clock=identify(job)
        self.pins={self.root.pid:self.root};self.notes=[];self.signals=[];self.samples=0
        self.external=None;self.error=None;self.manifest_seen=False;self.started=time.monotonic()
        self.origins={self.root.pid:'actual_RUN_STARTED_root'}

    def refresh(self):
        # Manifest is optional; current launcher only has RAM pins.
        manifest=self.job/'OWNED-PROCESSES.json'
        if manifest.exists():
            self.manifest_seen=True
            records=read(manifest);require(isinstance(records,list),'invalid optional ownership manifest')
            for record in records:
                pid=int(record['pid'])
                try:info=ownership.read_process(pid)
                except (FileNotFoundError,ProcessLookupError):continue
                expected=(pid,record['start_ticks'],record['uid'],record['process_group'],tuple(record['command']))
                if key(info)!=expected:continue
                if pid in self.pins and lifetime(self.pins[pid])!=lifetime(info):continue
                ancestry={k:v for k,v in self.pins.items() if k!=pid}
                if ownership.is_owned(info,ancestry):self.pins[pid]=info;self.origins[pid]='manifest_and_verified_live_ancestry'
        # Linux per-thread children is a /proc descendant scan without touching
        # unrelated process environments or command lines. Retain pins on reparent.
        queue=[p.pid for p in self.pins.values() if observation(p)['live']];seen=set()
        while queue:
            pid=queue.pop(0)
            if pid in seen:continue
            seen.add(pid)
            children=set()
            try:
                for task in Path(f'/proc/{pid}/task').iterdir():
                    try:children.update(int(x) for x in (task/'children').read_text().split())
                    except (FileNotFoundError,ProcessLookupError):continue
            except (FileNotFoundError,ProcessLookupError):continue
            for child in children:
                try:
                    info=ownership.read_process(child)
                    if info.uid!=self.root.uid or not info.argv:continue
                    old=self.pins.get(child)
                    if old is not None and lifetime(old)!=lifetime(info):continue
                    ancestry={k:v for k,v in self.pins.items() if k!=child}
                    if ownership.is_owned(info,ancestry):
                        self.pins[child]=info;self.origins[child]='verified_proc_live_ancestry';queue.append(child)
                except (FileNotFoundError,ProcessLookupError,RuntimeError,OSError):continue
        self.samples+=1

    def live(self):return [p for p in self.pins.values() if observation(p)['live']]

    def signal_one(self,pin,sig):
        if not observation(pin)['live']:return
        current=ownership.read_process(pin.pid)
        require(key(current)==key(pin) and ownership.is_owned(current,self.pins),'strict pinned signal identity mismatch')
        # The supervisor may share a process group with its non-owned invoking
        # shell. Signal a proved PID, never that group or a username/global GPU.
        fd=None
        try:
            if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
                fd=os.pidfd_open(pin.pid)
                require(key(ownership.read_process(pin.pid))==key(pin),'process changed after pidfd open')
                signal.pidfd_send_signal(fd,sig)
            else:
                require(key(ownership.read_process(pin.pid))==key(pin),'process changed before PID signal')
                os.kill(pin.pid,sig)
            self.signals.append(dict(pid=pin.pid,start_ticks=pin.start_ticks,uid=pin.uid,signal=int(sig),pidfd=fd is not None))
        except (ProcessLookupError,FileNotFoundError):return
        finally:
            if fd is not None:os.close(fd)

    def stop(self,reason):
        if self.external is None:self.external=reason
        stop=self.job/'STOP_REQUESTED'
        if not stop.exists():
            with stop.open('x') as f:f.write('tf_usable_clock_v1: '+reason+'\n')
        atomic(self.job/'CLOCK-STOP.json',dict(reason=reason,clock=self.clock,
            at=datetime.now(timezone.utc).isoformat(),no_quality_FINAL_or_FAILED_modified=True))
        self.refresh()
        self.signal_one(self.root,signal.SIGTERM)

    def cleanup(self):
        # First allow supervisor's original bounded cleanup. Never waitpid an
        # attached non-child. Every survivor is lifetime/argv validated again.
        natural=min(time.monotonic()+30,self.clock['hard_stop_monotonic']-45)
        while self.live() and time.monotonic()<natural:
            try:self.refresh()
            except (OSError,RuntimeError,ValueError) as e:self.notes.append('refresh during cleanup: '+repr(e))
            time.sleep(.25)
        for sig,span in ((signal.SIGTERM,10),(signal.SIGKILL,20)):
            for pin in reversed(list(self.pins.values())):
                try:self.signal_one(pin,sig)
                except (RuntimeError,OSError,ValueError) as e:self.notes.append('signal refused/uncertain: '+repr(e))
            end=min(time.monotonic()+span,self.clock['hard_stop_monotonic']-2)
            while self.live() and time.monotonic()<end:
                try:self.refresh()
                except (OSError,RuntimeError,ValueError) as e:self.notes.append('refresh after signal: '+repr(e))
                time.sleep(.25)
            if not self.live():break
        require(not self.live(),'observed matching owned live processes remain after bounded cleanup')

    def run(self):
        guard=self.job/'CLOCK-GUARD.json';require(not guard.exists(),'one attach attempt; no clock reset')
        me=ownership.read_process(os.getpid())
        atomic(guard,dict(schema='tf-usable-attached-clock/v1',status='RUNNING_ATTACHED_EXISTING_SUPERVISOR',
            job=str(self.job),supervisor_pin=asdict(self.root),guard_pin=asdict(me),clock=self.clock,
            nvml_independent=True,no_workload_launched=True,no_budget_reset=True,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),ownership_sha256=hashlib.sha256(OWNERSHIP.read_bytes()).hexdigest()))
        def interrupted(sig,_frame):self.external=self.external or f'guard received signal {sig}'
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            while True:
                self.refresh()
                atomic(self.job/'CLOCK-STATUS.json',dict(status='OBSERVING',samples=self.samples,
                    at=datetime.now(timezone.utc).isoformat(),pinned_processes=[asdict(p) for p in self.pins.values()],
                    root_observation=observation(self.root),clock=self.clock))
                if self.external:
                    self.stop(self.external);break
                if time.monotonic()>=self.clock['work_stop_monotonic']:
                    self.stop('independent actual3h-minus300 or absolute12:43:35 work cutoff');break
                if not observation(self.root)['live']:
                    if self.live():self.external='supervisor exited with observed live owned descendants'
                    break
                time.sleep(2)
        except BaseException as exc:
            self.error=dict(error=repr(exc),traceback=traceback.format_exc())
            try:self.stop('clock inspection error; stopping proved own supervisor')
            except BaseException as e:self.notes.append('stop error: '+repr(e))
        finally:
            try:
                if self.live():self.cleanup()
            except BaseException as exc:self.error=self.error or dict(error=repr(exc),traceback=traceback.format_exc())
            observed=[dict(pin=asdict(p),observation=observation(p)) for p in self.pins.values()]
            clean=not any(x['observation']['live'] for x in observed)
            result=dict(schema='tf-usable-attached-clock-result/v1',
                status='CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS' if clean and self.error is None and self.external is None else 'STOPPED_OR_FAILED_ATTACHED_CLOCK',
                job=str(self.job),all_observed_owned_live_processes_exited=clean,
                supervisor_exit_observation=observation(self.root),supervisor_returncode=None,waitpid_not_attempted=True,
                original_quality_FAILED_present=(self.job/'FAILED.json').exists(),original_quality_RESULT_present=(self.job/'RESULT.json').exists(),
                external_stop_reason=self.external,error=self.error,notes=self.notes,signals=self.signals,
                samples=self.samples,owned_manifest_was_optional_seen=self.manifest_seen,
                process_observations=observed,clock=self.clock,elapsed_guard_seconds=time.monotonic()-self.started,
                elapsed_quality_process_seconds=time.monotonic()-self.clock['process_birth_monotonic'],
                nvml_independent=True,no_workload_launched=True,no_foreign_signals=True,
                observation_scope='pinned supervisor and descendants proved while live; not unobserved historical processes')
            atomic(self.job/'CLOCK-RESULT.json',result)
            if not clean or self.error is not None:atomic(self.job/'CLOCK-FAILED.json',result)
        return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--inspect',action='store_true');mode.add_argument('--attach',action='store_true')
    args=parser.parse_args()
    if args.inspect:
        job,record,plan,pin,clock=identify(args.job)
        print(json.dumps(dict(status='PASS_CPU_INSPECT_ONLY_NOT_ATTACHED',job=str(job),supervisor_pin=asdict(pin),clock=clock,nvml_or_CUDA_imported=False),ensure_ascii=False));return
    result=AttachedClock(args.job).run();print(json.dumps({k:v for k,v in result.items() if k!='process_observations'},ensure_ascii=False),flush=True)
    if result['error'] is not None or not result['all_observed_owned_live_processes_exited']:raise SystemExit(1)

if __name__=='__main__':main()

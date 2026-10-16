"""CPU attach-only clock for four prospectively registered expanded-DEV shards."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import sys
import time

EXP=Path(__file__).resolve().parent.parent
ROOT=EXP.parent.parent
BASE=EXP/'tf_usable_clock_v1/watch.py'
BASE_SHA='d377fb3438e62c0e867c3550c3c781e84e84eae549cadceb208a391af3ab480d'
PROTOCOL=EXP/'draft_frozen_validation_v1/protocol.py'
RUN_SOURCE=PROTOCOL.parent/'run.py'
PROVENANCE='DRAFT-FROZEN-VALIDATION-CLOCK-PROVENANCE.json'
SHARDS=('init36','init37','init38','init39')
SHARD_SECONDS=14400
COHORT_SECONDS=28800
COHORT_WALL_SECONDS=43200
RESERVE=300
SHARD_BYTES=3*1024**3
COHORT_BYTES=5*1024**3
DISK_FLOOR=5*1024**3

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result

require(sha(BASE)==BASE_SHA,'original pin/ownership/cleanup source changed')
base=module(BASE,'_frozen_dev_clock_base')
protocol=module(PROTOCOL,'_frozen_dev_clock_protocol')

def fail_markers(path):
    return list(Path(path).rglob('*FAILED*.json'))+list(Path(path).rglob('*STOP*'))

def stamp(path):
    path=Path(path)
    return dict(path=str(path),sha256=sha(path))

def registry_contract(plan,job):
    # The original protocol performs the complete fixed-manifest, source,
    # checkpoint, today-terminal and predecessor/release admission validation.
    manifest=protocol._manifest(plan['manifest_path'])
    registration_path,reg=protocol._registration(plan['registration'],manifest)
    require(plan['shard'] in SHARDS and reg['shard_order']==list(SHARDS)
        and reg['shard_jobs'][plan['shard']]==str(job)
        and plan['registration_sha256']==sha(registration_path)
        and Path(reg['clock_wrapper_path']).resolve()==Path(__file__).resolve()
        and reg['clock_wrapper_sha256']==sha(__file__)
        and reg['clock_provenance_name']==PROVENANCE,
        'actual registration/shard/wrapper binding differs')
    require(reg['per_shard_max_seconds']==SHARD_SECONDS
        and reg['cohort_gpu_job_cap_seconds']==COHORT_SECONDS
        and reg['cohort_wall_cap_seconds']==COHORT_WALL_SECONDS
        and reg['cleanup_reserve_seconds']==RESERVE
        and reg['cohort_output_budget_GiB']==5 and reg['per_shard_output_budget_GiB']==3
        and reg['disk_floor_GiB']==5,'registered cohort resource contract differs')
    metadata=Path(reg['cohort_metadata_root']).resolve(strict=True)
    jobs={key:Path(value).resolve() for key,value in reg['shard_jobs'].items()}
    require(metadata==registration_path.parent and metadata.parent==protocol.RUNTIME
        and metadata not in jobs.values() and metadata!=protocol.RUNTIME
        and all(metadata not in p.parents and p not in metadata.parents for p in jobs.values()),
        'independent cohort metadata/shard roots required')
    require(plan['shard_jobs']==reg['shard_jobs'] and plan['max_run_wall_sec']==SHARD_SECONDS
        and plan['cleanup_reserve_sec']==RESERVE and plan['cohort_gpu_job_cap_seconds']==COHORT_SECONDS
        and plan['cohort_output_budget_GiB']==5 and plan['output_budget_GiB']==3
        and plan['disk_floor_GiB']==5 and plan['absolute_no_AutoDL'] is True,
        'registered resource limits and actual shard plan differ')
    return registration_path,reg,metadata,jobs

def prior_resources(plan,reg,registration_sha,boot):
    index=SHARDS.index(plan['shard']);charged=0.;anchor=None;proofs={}
    for shard in SHARDS[:index]:
        job=Path(reg['shard_jobs'][shard]).resolve(strict=True)
        require(not fail_markers(job),'prior failed/STOP shard cannot release another')
        names=('plan.json','COLLECTION.json','RESULT.json','FINAL-OWNED-CLEANUP.json',
               'CLOCK-RESULT.json','CLOCK-GUARD.json',PROVENANCE)
        docs={name:base.read(job/name) for name in names}
        old,c,g,v=docs['plan.json'],docs['CLOCK-RESULT.json'],docs['CLOCK-GUARD.json'],docs[PROVENANCE]
        state=c['clock'];elapsed=c['elapsed_quality_process_seconds']
        require(c['status']=='CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS'
            and c['all_observed_owned_live_processes_exited'] is True
            and c['error'] is None and c['external_stop_reason'] is None and not c['signals']
            and docs['FINAL-OWNED-CLEANUP.json']['all_roots_reaped'] is True,
            'prior clock terminal must be normally closed, not merely cleaned after STOP')
        require(state==g['clock']==v['clock'] and state['registration_sha256']==registration_sha
            and state['shard']==shard and state['boot_id']==boot
            and state['new_wrapper_sha256']==v['wrapper_source_sha256']==sha(__file__)
            and g['source_sha256']==v['base_sha256']==BASE_SHA
            and old['registration_sha256']==registration_sha,
            'prior shard clock/source/registration/boot chain differs')
        require(state['charged_gpu_job_seconds_before_shard']==charged
            and state['cohort_gpu_job_seconds_remaining']==COHORT_SECONDS-charged
            and state['effective_shard_seconds_cap']==min(SHARD_SECONDS,COHORT_SECONDS-charged)
            and type(elapsed) in (int,float) and math.isfinite(elapsed)
            and 0<elapsed<=state['effective_shard_seconds_cap'],
            'prior actual lifetime or cumulative charge unavailable')
        if anchor is None:anchor=state['cohort_anchor']
        require(state['cohort_anchor']==anchor,'cohort monotonic anchor reset between shards')
        for observed in c['process_observations']:
            pin=dict(observed['pin']);pin['argv']=tuple(pin['argv'])
            require(observed['observation']['live'] is False
                and not base.observation(base.ownership.Process(**pin))['live'],
                'prior matching pinned process still alive/uncertain')
        charged+=elapsed
        proofs.update({str(job/name):sha(job/name) for name in names})
    require(charged==plan['charged_gpu_job_seconds_before_shard']
        and COHORT_SECONDS-charged==plan['cohort_gpu_job_seconds_remaining']
        and COHORT_SECONDS-charged>RESERVE,'cohort remaining charge lacks active budget plus cleanup')
    return charged,anchor,proofs

def identify(job):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2,
            'CPU<=2 and CUDA_VISIBLE_DEVICES empty required')
    job=Path(job).resolve(strict=True)
    plan=protocol.load(job,verify=True)
    require(job.name.startswith('p0-draft-frozen-validation-v1-local-'+plan['shard'])
        and plan['schema']=='draft-frozen-validation-shard/v1'
        and plan['profile']=='expanded_DEV_init36_39','only registered expanded DEV shards accepted')
    reg_path,reg,metadata,jobs=registry_contract(plan,job)
    record=base.read(job/'RUN-STARTED.json')
    require(record['status']=='EXPLICIT_RUN' and record['pid']>1 and record['site']=='local'
        and record['shard']==plan['shard'] and record['max_run_wall_sec']==SHARD_SECONDS
        and record['absolute_work_stop']==reg['absolute_work_stop']
        and record['absolute_cleanup_deadline']==reg['absolute_cleanup_deadline']
        and record['charged_gpu_job_seconds_before_shard']==plan['charged_gpu_job_seconds_before_shard'],
        'actual shard RUN/registration resource identity differs')
    pin=base.ownership.read_process(record['pid'])
    require(pin.uid==os.geteuid() and pin.argv,'actual live same-UID shard supervisor required')
    cwd=Path(f'/proc/{pin.pid}/cwd').resolve(strict=True)
    paths=[(cwd/s).resolve() if not Path(s).is_absolute() else Path(s).resolve() for s in pin.argv]
    require(RUN_SOURCE.resolve() in paths and '--job' in pin.argv,'actual shard run.py argv required')
    idx=pin.argv.index('--job');require(idx+1<len(pin.argv),'actual --job missing')
    arg=Path(pin.argv[idx+1]);bound=arg.resolve() if arg.is_absolute() else (cwd/arg).resolve()
    require(bound==job and base.key(base.ownership.read_process(pin.pid))==base.key(pin),
            'actual shard process identity changed or belongs to another job')
    launch_path=job/'ROOT-DETACHED-LAUNCH.json';launch=base.read(launch_path);declared=launch['pin']
    require((declared['pid'],declared['start_ticks'],declared['uid'],declared['pgid'],tuple(declared['argv']))==base.key(pin)
        and launch['command']==list(pin.argv) and launch['start_new_session'] is True and pin.pgid==pin.pid,
        'root detached shard launch pin/session/command differs')
    hz=os.sysconf('SC_CLK_TCK');boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(launch['boot_id']==boot and launch['SC_CLK_TCK']==hz
        and launch['source_sha256']==sha(job/'ROOT-DETACHED-LAUNCH.py')
        and record['start_ticks']==pin.start_ticks and record['SC_CLK_TCK']==hz,
        'actual launch source/boot/tick proof differs')
    charged,anchor,proofs=prior_resources(plan,reg,sha(reg_path),boot)
    require(all(not (p/'RUN-STARTED.json').exists() for sid,p in jobs.items()
                if SHARDS.index(sid)>SHARDS.index(plan['shard'])),
            'a later shard has already started; sequential cohort accounting uncertain')
    mono=time.monotonic();uptime=float(Path('/proc/uptime').read_text().split()[0])
    age=uptime-pin.start_ticks/hz;require(age>=0,'invalid actual shard birth')
    origin=mono-age-.01;now=datetime.now(timezone.utc)
    start=datetime.fromisoformat(reg['cohort_started_at'])
    absolute_hard=datetime.fromisoformat(reg['absolute_cleanup_deadline'])
    absolute_work=datetime.fromisoformat(reg['absolute_work_stop'])
    require(start<=now and start<absolute_work<absolute_hard
        and (absolute_hard-start).total_seconds()<=COHORT_WALL_SECONDS
        and (absolute_hard-absolute_work).total_seconds()>=RESERVE,
        'registered twelve-hour wall/cleanup contract differs')
    if anchor is None:
        cohort_origin=mono-(now-start).total_seconds()-.01
        require(cohort_origin<=origin,'registration cohort starts after actual first supervisor')
        anchor=dict(registered_start_at=reg['cohort_started_at'],boot_id=boot,
            registration_sha256=sha(reg_path),origin_monotonic=cohort_origin,
            hard_monotonic=min(cohort_origin+COHORT_WALL_SECONDS,mono+(absolute_hard-now).total_seconds()),
            work_monotonic=mono+(absolute_work-now).total_seconds(),first_supervisor_pin=asdict(pin))
    require(anchor['boot_id']==boot and anchor['registration_sha256']==sha(reg_path)
        and anchor['registered_start_at']==reg['cohort_started_at']
        and anchor['origin_monotonic']<=origin,'actual same-boot cohort anchor differs')
    remaining=COHORT_SECONDS-charged;effective=min(SHARD_SECONDS,remaining)
    hard=min(origin+effective,anchor['hard_monotonic'],anchor['origin_monotonic']+COHORT_WALL_SECONDS,
             mono+(absolute_hard-now).total_seconds())
    work=min(hard-RESERVE,anchor['work_monotonic'],mono+(absolute_work-now).total_seconds())
    require(mono<work,'shard/cohort charged or absolute work budget exhausted')
    clock=dict(process_birth_monotonic=origin,current_process_birth_monotonic=origin,
        hard_stop_monotonic=hard,work_stop_monotonic=work,actual_process_age_at_attach_sec=age,
        max_seconds=SHARD_SECONDS,cleanup_reserve_seconds=RESERVE,
        absolute_work_stop=reg['absolute_work_stop'],absolute_cleanup_deadline=reg['absolute_cleanup_deadline'],
        clock_origin='actual registered shard birth; prior real charged lifetimes and first cohort monotonic anchor',
        morning_budget_inherited=False,shard=plan['shard'],boot_id=boot,SC_CLK_TCK=hz,
        charged_gpu_job_seconds_before_shard=charged,cohort_gpu_job_seconds_remaining=remaining,
        effective_shard_seconds_cap=effective,cohort_gpu_job_cap_seconds=COHORT_SECONDS,
        cohort_wall_cap_seconds=COHORT_WALL_SECONDS,cohort_anchor=anchor,
        registration=str(reg_path),registration_sha256=sha(reg_path),cohort_metadata_root=str(metadata),
        prior_shard_evidence_sha256=proofs,new_wrapper_sha256=sha(__file__),
        plan_sha256=sha(job/'plan.json'),root_detached_launch_sha256=sha(launch_path),
        root_launch_source_sha256=sha(job/'ROOT-DETACHED-LAUNCH.py'),original_base_clock_sha256=BASE_SHA,
        output_limit_bytes=SHARD_BYTES,cohort_output_limit_bytes=COHORT_BYTES,disk_floor_bytes=DISK_FLOOR)
    return job,record,plan,pin,clock

def actual_bytes(path):
    total=0
    for p in Path(path).rglob('*'):
        require(not p.is_symlink(),'unexpected artifact symlink: '+str(p))
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:continue
    return total

class ValidationClock(base.AttachedClock):
    def __init__(self,job):
        self.job,self.run_record,self.plan,self.root,self.clock=identify(job)
        self.pins={self.root.pid:self.root};self.notes=[];self.signals=[];self.samples=0
        self.external=None;self.error=None;self.manifest_seen=False;self.started=time.monotonic()
        self.origins={self.root.pid:'actual_RUN_STARTED_root'}
        self.next_space_check=0;self.space=None;self.record_space=False

    def refresh(self):
        super().refresh();now=time.monotonic()
        if now>=self.next_space_check:
            sizes={s:actual_bytes(p) if Path(p).exists() else 0 for s,p in self.plan['shard_jobs'].items()}
            metadata=actual_bytes(self.clock['cohort_metadata_root']);free=shutil.disk_usage(self.job).free
            self.space=dict(shard_bytes=sizes[self.plan['shard']],all_registered_shard_bytes=sizes,
                cohort_metadata_bytes=metadata,cohort_bytes=sum(sizes.values())+metadata,
                shard_cap_bytes=SHARD_BYTES,cohort_cap_bytes=COHORT_BYTES,
                disk_free_bytes=free,disk_floor_bytes=DISK_FLOOR,at_monotonic=now,interval_seconds=15)
            if self.record_space:base.atomic(self.job/'DRAFT-FROZEN-VALIDATION-CLOCK-SPACE.json',self.space)
            require(self.space['shard_bytes']<=SHARD_BYTES and self.space['cohort_bytes']<=COHORT_BYTES
                and free>=DISK_FLOOR,'registered shard/cohort/disk output boundary reached')
            require(not any((Path(p)/'RUN-STARTED.json').exists() for s,p in self.plan['shard_jobs'].items()
                if SHARDS.index(s)>SHARDS.index(self.plan['shard'])),
                'later shard started before current guard terminal; stop only own pinned tree')
            self.next_space_check=now+15

    def stop(self,reason):
        if reason=='independent actual3h-minus300 or absolute12:43:35 work cutoff':
            reason='registered shard birth/cohort charge/twelve-hour or absolute cutoff,300s cleanup reserved'
        return super().stop(reason)

    def run(self):
        path=self.job/PROVENANCE
        require(not path.exists() and not (self.job/'CLOCK-GUARD.json').exists(),
                'one actual attach; never reset shard or cohort budget')
        self.record_space=True;self.refresh()
        base.atomic(path,dict(schema='draft-frozen-validation-attached-clock-provenance/v1',
            job=str(self.job),supervisor_pin=asdict(self.root),clock=self.clock,space=self.space,
            wrapper_source_sha256=sha(__file__),base_source=str(BASE),base_sha256=BASE_SHA,
            ownership_sha256=sha(base.OWNERSHIP),registration=stamp(self.clock['registration']),
            source_provenance=stamp(PROTOCOL),no_workload_launched=True,no_GPU_or_NVML=True,
            inherited_unchanged_methods=['live','signal_one','cleanup'],morning_budget_inherited=False,
            no_process_group_signals=True,nonchild_waitpid_not_attempted=True))
        return super().run()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect',action='store_true');mode.add_argument('--attach',action='store_true')
    args=parser.parse_args();clock=ValidationClock(args.job)
    if args.inspect:
        clock.refresh()
        import json
        print(json.dumps(dict(status='PASS_ACTUAL_REGISTERED_DEV_SHARD_CPU_INSPECT_ONLY_NOT_ATTACHED',
            job=str(clock.job),supervisor_pin=asdict(clock.root),clock=clock.clock,space=clock.space,
            no_workload_launched=True,no_signals=True),ensure_ascii=False));return
    import json
    result=clock.run();print(json.dumps({k:v for k,v in result.items() if k!='process_observations'},ensure_ascii=False),flush=True)
    if result['error'] is not None or not result['all_observed_owned_live_processes_exited']:raise SystemExit(1)

if __name__=='__main__':main()

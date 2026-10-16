"""Attach-only today50 CPU watchdog: actual new birth4h, bounded absolute deadlines."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

EXP=Path(__file__).resolve().parent.parent
ROOT=EXP.parent.parent
BASE=EXP/'tf_usable_clock_v1/watch.py'
BASE_SHA='d377fb3438e62c0e867c3550c3c781e84e84eae549cadceb208a391af3ab480d'
RUN_SOURCE=EXP/'draft_freeze_quality_v1/run.py'
MAX_SECONDS=14400
RESERVE=300
WORK='2026-10-16T18:15:29+08:00'
HARD='2026-10-16T18:20:29+08:00'
CAP=3*1024**3
FLOOR=5*1024**3

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

require(sha(BASE)==BASE_SHA,'frozen ownership/cleanup source changed')
SPEC=importlib.util.spec_from_file_location('_draft_freeze_frozen_clock',BASE)
base=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=base;SPEC.loader.exec_module(base)

def contract(p):
    require(p.get('schema')=='draft-freeze-complete-dev/v1' and p.get('profile')=='today50'
        and p.get('site')=='local' and p.get('max_run_wall_sec')==MAX_SECONDS
        and p.get('cleanup_reserve_sec')==RESERVE and p.get('absolute_work_stop')==WORK
        and p.get('absolute_cleanup_deadline')==HARD and p.get('output_budget_GiB')==3
        and p.get('disk_floor_GiB')==5 and p.get('workers')==1,
        'new authorized today50 resource contract differs')
    rows=p['cases']
    require(len(rows)==len({r['key'] for r in rows})==50
        and Counter(r['group'] for r in rows)==dict(fresh_DEV=30,risk_DEV=16,repeatability=4)
        and p.get('planned_full_episodes')==50 and p.get('absolute_no_AutoDL') is True,
        'actual new50 local-only plan required')

def identify(job):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2,
            'CPU<=2 and CUDA_VISIBLE_DEVICES empty required')
    job=Path(job).resolve(strict=True)
    require(job.name.startswith('p0-draft-freeze-quality-v1-local-'),'only the new authorized job namespace')
    record=base.read(job/'RUN-STARTED.json');plan=base.read(job/'plan.json');contract(plan)
    require(plan['job']==str(job) and record.get('status')=='EXPLICIT_RUN'
        and record.get('site')=='local' and record.get('pid',0)>1
        and record.get('max_run_wall_sec')==MAX_SECONDS
        and record.get('absolute_work_stop')==WORK and record.get('absolute_cleanup_deadline')==HARD,
        'real new supervisor launch/plan binding differs')
    require(not (job/'CUMULATIVE-SITE-BUDGET.json').exists()
        and not (job/'RECOVERY-CLOCK-PROVENANCE.json').exists(),
        'new authorization must not inherit morning/recovery clocks')
    pin=base.ownership.read_process(record['pid'])
    require(pin.uid==os.geteuid() and pin.argv,'actual same-UID supervisor required')
    cwd=Path(f'/proc/{pin.pid}/cwd').resolve(strict=True)
    paths=[(cwd/s).resolve() if not Path(s).is_absolute() else Path(s).resolve() for s in pin.argv]
    require(RUN_SOURCE.resolve() in paths and '--job' in pin.argv,'actual new run.py required')
    index=pin.argv.index('--job');require(index+1<len(pin.argv),'missing actual --job argument')
    value=Path(pin.argv[index+1]);bound=value.resolve() if value.is_absolute() else (cwd/value).resolve()
    require(bound==job and base.key(base.ownership.read_process(pin.pid))==base.key(pin),
            'actual supervisor identity changed or belongs to another job')
    launch_path=job/'ROOT-DETACHED-LAUNCH.json';launch=base.read(launch_path);declared=launch['pin']
    require((declared['pid'],declared['start_ticks'],declared['uid'],declared['pgid'],tuple(declared['argv']))
        ==base.key(pin) and launch['command']==list(pin.argv) and launch['start_new_session'] is True
        and pin.pgid==pin.pid,'root detached launch pin/command/session proof differs')
    hz=os.sysconf('SC_CLK_TCK');boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(launch['boot_id']==boot and launch['SC_CLK_TCK']==hz,'root launch belongs to another boot/tick clock')
    require(launch.get('source_sha256')==sha(job/'ROOT-DETACHED-LAUNCH.py'),
            'actual root launch script source differs')
    require(sha(RUN_SOURCE)==plan['source_sha256'][str(RUN_SOURCE)]
        and sha(plan['preregistration'])==plan['preregistration_sha256'],
        'frozen actual run/preregistration source differs')
    mono=time.monotonic();uptime=float(Path('/proc/uptime').read_text().split()[0])
    age=uptime-pin.start_ticks/hz;require(age>=0,'invalid actual process birth')
    # /proc/uptime has centisecond precision; reserve one reported quantum.
    rounding_reserve=.01
    origin=mono-age-rounding_reserve
    now=datetime.now(timezone.utc)
    hard=min(origin+MAX_SECONDS,mono+(datetime.fromisoformat(HARD)-now).total_seconds())
    work=min(hard-RESERVE,mono+(datetime.fromisoformat(WORK)-now).total_seconds())
    require(mono<work,'new actual-birth/absolute work budget exhausted')
    clock=dict(process_birth_monotonic=origin,current_process_birth_monotonic=origin,
        hard_stop_monotonic=hard,work_stop_monotonic=work,actual_process_age_at_attach_sec=age,
        max_seconds=MAX_SECONDS,cleanup_reserve_seconds=RESERVE,absolute_work_stop=WORK,
        birth_clock_rounding_reserve_seconds=rounding_reserve,
        absolute_cleanup_deadline=HARD,clock_origin='actual new supervisor birth via /proc ticks; never attach time',
        morning_budget_inherited=False,predecessors=[],boot_id=boot,SC_CLK_TCK=hz,
        new_wrapper_sha256=sha(__file__),plan_sha256=sha(job/'plan.json'),
        root_detached_launch_sha256=sha(launch_path),root_launch_source_sha256=sha(job/'ROOT-DETACHED-LAUNCH.py'),
        original_base_clock_sha256=BASE_SHA,output_limit_bytes=CAP,disk_floor_bytes=FLOOR)
    return job,record,plan,pin,clock

def actual_bytes(job):
    total=0
    for p in Path(job).rglob('*'):
        require(not p.is_symlink(),'unexpected artifact symlink: '+str(p))
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:continue
    return total

class DraftFreezeClock(base.AttachedClock):
    def __init__(self,job):
        # Same initialized ownership state as frozen AttachedClock; only identify
        # is this namespace's new4h contract. Base globals/source stay untouched.
        self.job,self.run_record,self.plan,self.root,self.clock=identify(job)
        self.pins={self.root.pid:self.root};self.notes=[];self.signals=[];self.samples=0
        self.external=None;self.error=None;self.manifest_seen=False;self.started=time.monotonic()
        self.origins={self.root.pid:'actual_RUN_STARTED_root'}
        self.next_space_check=0;self.space=None;self.record_space=False

    def refresh(self):
        super().refresh()
        now=time.monotonic()
        if now>=self.next_space_check:
            self.space=dict(job_bytes=actual_bytes(self.job),cap_bytes=CAP,
                disk_free_bytes=shutil.disk_usage(self.job).free,disk_floor_bytes=FLOOR,
                at_monotonic=now,interval_seconds=15,new_job_only=True)
            if self.record_space:base.atomic(self.job/'DRAFT-FREEZE-CLOCK-SPACE.json',self.space)
            require(self.space['job_bytes']<=CAP and self.space['disk_free_bytes']>=FLOOR,
                    'new job3GiB or disk5GiB resource boundary reached')
            self.next_space_check=now+15

    def stop(self,reason):
        if reason=='independent actual3h-minus300 or absolute12:43:35 work cutoff':
            reason='independent actual4h-minus300 or absolute18:15:29 work cutoff'
        return super().stop(reason)

    def run(self):
        proof=self.job/'DRAFT-FREEZE-CLOCK-PROVENANCE.json'
        require(not proof.exists() and not (self.job/'CLOCK-GUARD.json').exists(),
                'one attach attempt; no resetting actual birth budget')
        self.record_space=True;self.refresh()
        base.atomic(proof,dict(schema='draft-freeze-attached-clock-provenance/v1',job=str(self.job),
            supervisor_pin=asdict(self.root),clock=self.clock,space=self.space,
            wrapper_source_sha256=sha(__file__),base_source=str(BASE),base_sha256=BASE_SHA,
            ownership_sha256=sha(base.OWNERSHIP),no_workload_launched=True,no_GPU_or_NVML=True,
            inherited_unchanged_methods=['live','signal_one','cleanup'],morning_budget_inherited=False,
            nonchild_waitpid_not_attempted=True,no_process_group_signals=True))
        return super().run()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect',action='store_true');mode.add_argument('--attach',action='store_true')
    args=parser.parse_args();clock=DraftFreezeClock(args.job)
    if args.inspect:
        clock.refresh()
        print(json.dumps(dict(status='PASS_ACTUAL_NEW50_CPU_INSPECT_ONLY_NOT_ATTACHED',
            job=str(clock.job),supervisor_pin=asdict(clock.root),clock=clock.clock,space=clock.space,
            no_signals=True,no_workload_launched=True),ensure_ascii=False));return
    result=clock.run();print(json.dumps({k:v for k,v in result.items() if k!='process_observations'},ensure_ascii=False),flush=True)
    if result['error'] is not None or not result['all_observed_owned_live_processes_exited']:raise SystemExit(1)

if __name__=='__main__':main()

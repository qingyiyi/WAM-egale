"""Attach-only independent actual-birth clock; reuse proven PID ownership."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time

OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN.parent))
from task4_direction_diagnostic_v1.protocol import load,read,sha,write

BASE=OWN.parent/'tf_usable_clock_v1/watch.py'
if sha(BASE)!='d377fb3438e62c0e867c3550c3c781e84e84eae549cadceb208a391af3ab480d':
    raise ValueError('frozen ownership clock source differs')
spec=importlib.util.spec_from_file_location('_task4_frozen_ownership_clock',BASE)
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)

def await_guard(job,p,self_pin):
    end=time.monotonic()+60
    while time.monotonic()<end:
        if (job/'STOP_REQUESTED').exists():raise RuntimeError('stopped before workload')
        path=job/'CLOCK-GUARD.json'
        if path.exists():
            r=read(path)
            if r['supervisor_pin']['pid']!=self_pin['pid'] or r['supervisor_pin']['start_ticks']!=self_pin['start_ticks']:
                raise ValueError('guard attached to different supervisor')
            for key in ('supervisor_pin','guard_pin'):
                pin=r[key];live=base.ownership.read_process(pin['pid'])
                if (live.start_ticks!=pin['start_ticks'] or live.uid!=pin['uid']
                        or list(live.argv)!=pin['argv'] or not base.observation(live)['live']):
                    raise ValueError('independent guard/root no longer exact/live')
            c=r['clock']
            if (c['max_seconds']!=2400 or c['cleanup_reserve_seconds']!=300
                    or c['plan_sha256']!=sha(job/'plan.json')
                    or c['diagnostic_clock_sha256']!=p['source_sha256'][str(OWN/'clock.py')]):
                raise ValueError('independent diagnostic guard contract differs')
            return
        time.sleep(.2)
    raise TimeoutError('independent clock admission timeout; no GPU worker launched')

class DiagnosticClock(base.AttachedClock):
    def __init__(self,job):
        if os.getenv('CUDA_VISIBLE_DEVICES')!='' or len(os.sched_getaffinity(0))>2:
            raise ValueError('independent clock CPU<=2, CUDA disabled')
        self.job=Path(job).resolve(strict=True);self.plan=load(self.job)
        self.run_record=read(self.job/'RUN-STARTED.json')
        self.root=base.ownership.read_process(self.run_record['pid'])
        if self.root.uid!=os.getuid() or str(OWN/'run.py') not in self.root.argv or str(self.job) not in self.root.argv:
            raise ValueError('actual diagnostic supervisor argv/job binding differs')
        launch=read(self.job/'ROOT-LAUNCH.json')['pin']
        actual=asdict(self.root);actual['argv']=list(actual['argv'])
        if launch!=actual:raise ValueError('actual detached root launch differs')
        hz=os.sysconf('SC_CLK_TCK');mono=time.monotonic()
        age=float(Path('/proc/uptime').read_text().split()[0])-self.root.start_ticks/hz
        if age<0:raise ValueError('invalid process birth')
        origin=mono-age-.01
        now=datetime.now(timezone.utc)
        hard=min(origin+2400,mono+(datetime.fromisoformat(self.plan['absolute_cleanup_deadline'])-now).total_seconds())
        work=min(hard-300,mono+(datetime.fromisoformat(self.plan['absolute_work_stop'])-now).total_seconds())
        if work<=mono:raise TimeoutError('diagnostic budget expired before attach')
        self.clock=dict(process_birth_monotonic=origin,hard_stop_monotonic=hard,work_stop_monotonic=work,
            max_seconds=2400,cleanup_reserve_seconds=300,absolute_work_stop=self.plan['absolute_work_stop'],
            absolute_cleanup_deadline=self.plan['absolute_cleanup_deadline'],plan_sha256=sha(self.job/'plan.json'),
            diagnostic_clock_sha256=sha(__file__),base_source_sha256=sha(BASE),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),SC_CLK_TCK=hz,
            clock_origin='actual supervisor birth; no retry/reset; includes setup/wait/failure')
        self.pins={self.root.pid:self.root};self.notes=[];self.signals=[];self.samples=0
        self.external=None;self.error=None;self.manifest_seen=False;self.started=mono
        self.origins={self.root.pid:'observed diagnostic supervisor birth'};self.next_space_check=0

    def refresh(self):
        super().refresh()
        if time.monotonic()>=self.next_space_check:
            size=sum(f.stat().st_size for f in self.job.rglob('*') if f.is_file())
            free=shutil.disk_usage(self.job).free
            if size>2**30 or free<5*2**30:raise RuntimeError('diagnostic storage cap/floor reached')
            self.next_space_check=time.monotonic()+15

    def stop(self,reason):
        if reason.startswith('independent actual3h'):reason='diagnostic actual40min-minus300 work cutoff'
        return super().stop(reason)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--attach',action='store_true',required=True);args=parser.parse_args()
    result=DiagnosticClock(args.job).run()
    if result['error'] is not None or not result['all_observed_owned_live_processes_exited']:raise SystemExit(1)

if __name__=='__main__':main()

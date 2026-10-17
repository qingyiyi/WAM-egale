"""Independent lifetime-pinned clock, fixed09:35 stop and09:40 GPU cleanup."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
import time

OWN=Path(__file__).resolve().parent;EXP=OWN.parent
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from tf_effect_campaign_continue_v2.protocol import load,read,require,sha
from tf_effect_campaign_v1 import clock as original


class Clock(original.base.AttachedClock):
    def __init__(self,job):
        require(os.getenv('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2,'CPU-only independent clock')
        self.job=Path(job).resolve();self.plan=load(self.job);self.run_record=read(self.job/'RUN-STARTED.json')
        self.root=original.base.ownership.read_process(self.run_record['pid'])
        require(self.root.uid==os.getuid() and str(OWN/'run.py') in self.root.argv and str(self.job) in self.root.argv,
            'exact own continuation supervisor binding required')
        saved=read(self.job/'ROOT-LAUNCH.json')['pin'];actual=asdict(self.root)
        require(all(actual[k]==saved[k] for k in ('pid','uid','pgid','start_ticks')) and list(self.root.argv)==saved['argv'],
            'root lifetime/UID/PGID/argv changed; mutable parent is not identity')
        mono=time.monotonic();age=float(Path('/proc/uptime').read_text().split()[0])-self.root.start_ticks/os.sysconf('SC_CLK_TCK')
        require(age>=0,'actual root birth required');origin=mono-age-.01;now=datetime.now(timezone.utc)
        hard=min(origin+self.plan['max_run_wall_sec'],mono+(datetime.fromisoformat(self.plan['absolute_cleanup_deadline'])-now).total_seconds())
        work=min(hard-self.plan['cleanup_reserve_sec'],mono+(datetime.fromisoformat(self.plan['absolute_work_stop'])-now).total_seconds())
        require(work>mono,'new bounded continuation window expired')
        self.clock=dict(process_birth_monotonic=origin,hard_stop_monotonic=hard,work_stop_monotonic=work,
            max_seconds=self.plan['max_run_wall_sec'],cleanup_reserve_seconds=self.plan['cleanup_reserve_sec'],
            absolute_work_stop=self.plan['absolute_work_stop'],absolute_cleanup_deadline=self.plan['absolute_cleanup_deadline'],
            plan_sha256=sha(self.job/'plan.json'),clock_sha256=sha(__file__),
            clock_origin='actual continuation birth plus new authorized fixed09:40; original costs debited and failure retained',
            predecessor_job=self.plan['predecessor_job'],cumulative_budget=self.plan['cumulative_budget'])
        self.pins={self.root.pid:self.root};self.notes=[];self.signals=[];self.samples=0
        self.external=None;self.error=None;self.manifest_seen=False;self.started=mono;self.origins={self.root.pid:'strict own continuation root'}
    def stop(self,reason):
        if reason.startswith('independent actual3h'):reason='registered pending-only continuation09:35 work deadline'
        return super().stop(reason)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    result=Clock(args.job).run()
    if result['error'] or not result['all_observed_owned_live_processes_exited']:raise SystemExit(1)


if __name__=='__main__':main()

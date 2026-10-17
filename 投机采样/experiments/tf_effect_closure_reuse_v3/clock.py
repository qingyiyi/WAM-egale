"""Parent reparenting is permitted; lifetime/UID/PGID/argv remain exact."""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,importlib.util,os,sys,time
from pathlib import Path
OWN=Path(__file__).resolve().parent;EXP=OWN.parent;sys.path.insert(0,str(EXP))
from tf_effect_closure_v1.protocol import load,read,sha,require
BASE=EXP/'tf_usable_clock_v1/watch.py'
spec=importlib.util.spec_from_file_location('_tf_effect_guard_existing_clock',BASE)
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)

class Clock(base.AttachedClock):
    def __init__(self,job):
        require(os.getenv('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2,'CPU<=2 and no CUDA')
        self.job=Path(job).resolve();self.plan=load(self.job);self.run_record=read(self.job/'RUN-STARTED.json')
        self.root=base.ownership.read_process(self.run_record['pid'])
        require(self.root.uid==os.getuid() and str(OWN/'run.py') in self.root.argv and str(self.job) in self.root.argv,'exact supervisor binding required')
        launch=read(self.job/'ROOT-LAUNCH.json')['pin'];actual=asdict(self.root)
        require(all(actual[k]==launch[k] for k in ('pid','start_ticks','uid','pgid')) and list(self.root.argv)==launch['argv'],'supervisor lifetime/UID/PGID/argv differ')
        mono=time.monotonic();age=float(Path('/proc/uptime').read_text().split()[0])-self.root.start_ticks/os.sysconf('SC_CLK_TCK')
        require(age>=0,'actual birth required');origin=mono-age-.01;now=datetime.now(timezone.utc)
        # The original qualification deadline includes the failed admission and
        # correction interval. The new process cannot reset that deadline.
        hard=min(origin+self.plan['max_run_wall_sec'],mono+(datetime.fromisoformat(self.plan['absolute_cleanup_deadline'])-now).total_seconds())
        work=hard-self.plan['cleanup_reserve_sec'];require(work>mono,'cumulative budget expired')
        self.clock=dict(process_birth_monotonic=origin,hard_stop_monotonic=hard,work_stop_monotonic=work,
            max_seconds=self.plan['max_run_wall_sec'],cleanup_reserve_seconds=self.plan['cleanup_reserve_sec'],
            plan_sha256=sha(self.job/'plan.json'),clock_sha256=sha(__file__),absolute_cleanup_deadline=self.plan['absolute_cleanup_deadline'],
            absolute_work_stop=self.plan['absolute_work_stop'],clock_origin='original registration absolute cap; predecessor failure and correction gap charged')
        self.pins={self.root.pid:self.root};self.notes=[];self.signals=[];self.samples=0
        self.external=None;self.error=None;self.manifest_seen=False;self.started=mono;self.origins={self.root.pid:'observed exact own root; parent may reparent'}
    def stop(self,reason):
        if reason.startswith('independent actual3h'):reason='registered cumulative TF effect work deadline'
        return super().stop(reason)

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);args=a.parse_args()
    r=Clock(args.job).run()
    if r['error'] or not r['all_observed_owned_live_processes_exited']:raise SystemExit(1)
if __name__=='__main__':main()

"""Attach-only full-episode clock using existing strict ownership signals."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,importlib.util,os,sys,time
OWN=Path(__file__).resolve().parent;EXP=OWN.parent;sys.path.insert(0,str(EXP))
from tf_effect_campaign_v1.protocol import load,read,sha,require
spec=importlib.util.spec_from_file_location('_tf_effect_campaign_proven_clock',EXP/'tf_usable_clock_v1/watch.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)

class Clock(base.AttachedClock):
    def __init__(self,job):
        require(os.getenv('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2,'CPU-only clock')
        self.job=Path(job).resolve();self.plan=load(self.job);self.run_record=read(self.job/'RUN-STARTED.json')
        self.root=base.ownership.read_process(self.run_record['pid'])
        require(self.root.uid==os.getuid() and str(OWN/'run.py') in self.root.argv and str(self.job) in self.root.argv,'exact own root binding required')
        old=read(self.job/'ROOT-LAUNCH.json')['pin'];now=asdict(self.root)
        require(all(now[k]==old[k] for k in ('pid','uid','pgid','start_ticks')) and list(self.root.argv)==old['argv'],'root lifetime/argv changed')
        mono=time.monotonic();age=float(Path('/proc/uptime').read_text().split()[0])-self.root.start_ticks/os.sysconf('SC_CLK_TCK')
        require(age>=0,'trustworthy actual birth required');origin=mono-age-.01
        hard=min(origin+self.plan['max_run_wall_sec'],mono+(datetime.fromisoformat(self.plan['absolute_cleanup_deadline'])-datetime.now(timezone.utc)).total_seconds())
        work=hard-self.plan['cleanup_reserve_sec'];require(work>mono,'campaign expired')
        self.clock=dict(process_birth_monotonic=origin,hard_stop_monotonic=hard,work_stop_monotonic=work,
            max_seconds=self.plan['max_run_wall_sec'],cleanup_reserve_seconds=self.plan['cleanup_reserve_sec'],
            absolute_work_stop=self.plan['absolute_work_stop'],absolute_cleanup_deadline=self.plan['absolute_cleanup_deadline'],
            plan_sha256=sha(self.job/'plan.json'),clock_sha256=sha(__file__),clock_origin='actual birth and original registration absolute deadline; no retry reset')
        self.pins={self.root.pid:self.root};self.notes=[];self.signals=[];self.samples=0
        self.external=None;self.error=None;self.manifest_seen=False;self.started=mono;self.origins={self.root.pid:'strict own root'}
    def stop(self,reason):
        if reason.startswith('independent actual3h'):reason='registered complete TF campaign deadline'
        return super().stop(reason)

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);args=a.parse_args()
    result=Clock(args.job).run()
    if result['error'] or not result['all_observed_owned_live_processes_exited']:raise SystemExit(1)
if __name__=='__main__':main()

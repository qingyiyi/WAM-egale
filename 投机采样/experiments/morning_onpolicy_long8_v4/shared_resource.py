"""Shared non-performance collection: finite capacity and owned-only cleanup."""
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
import ownership
from experiment import write,stamp

def charged_reservations(job, entries, *, failed_phase=False):
    total=0;rows=[];errors=[]
    for e in entries:
        charge=e['cap'];exact=False;observed=None;path=Path(e['out'])/'COUNTER-CLOSED.json'
        try:
            if path.exists():
                c=json.loads(path.read_text())
                if (c['status']!='CLOSED' or c['cap']!=e['cap'] or type(c['target_intents']) is not int
                    or not 0<=c['target_intents']<=e['cap'] or c['target_intents']!=c['host_returned']
                    or c['failed_forwards'] or c['active_forwards'] or c['request_active']):
                    raise ValueError('invalid closure')
                observed=c['target_intents'];charge=observed;exact=True
        except BaseException as exc:errors.append(dict(out=e['out'],error=repr(exc),charged_full_cap=True))
        failure_marker=any((Path(e['out'])/f).exists() for f in ('WORKER-FAILED.json','FINISH_FAILED.json'))
        if failed_phase or failure_marker:charge=e['cap'];exact=False
        total+=charge;rows.append({**e,'charged_target_intents':charge,'exact_closed_count':exact,
            'observed_valid_closed_intents':observed,'late_failure_full_charge':bool(failed_phase or failure_marker)})
    return dict(charged_target_intents=total,reservations=rows,counter_errors=errors,
                conservative_unclosed_reservations=True,failed_phase_all_reservations_charged_full=bool(failed_phase))

class SharedResources:
    @staticmethod
    def startup_ready(sample,foreign):
        # Foreign workers may coexist; their PIDs are observed, never signalled.
        return sample['memory_free_bytes']>=2*2**30

    def inspect(self,*,enforce_runtime=True):
        sample=self.nv.sample();identities=[];foreign=[]
        if sample['uuid']!=self.p['GPU_UUID']:raise ValueError('wrong physical GPU UUID')
        for pid in {r['pid'] for k in ('compute','graphics') for r in sample[k]}:
            try:info=ownership.read_process(pid)
            except FileNotFoundError:continue
            owned=ownership.is_owned(info,self.pins)
            key=(info.pid,info.start_ticks,info.uid,info.pgid)
            if self.start is None:self.baseline_GPU_identities.add(key)
            if owned:self.gpu_pins[pid]=info
            else:foreign.append(pid)
            identities.append(dict(pid=pid,start_ticks=info.start_ticks,uid=info.uid,pgid=info.pgid,owned=owned))
        self.latest_foreign=foreign
        write(self.job/'OWNED-GPU-PROCESSES.json',[asdict(v) for v in self.gpu_pins.values()],durable=True)
        with (self.job/'GPU-samples.jsonl').open('a') as f:
            f.write(json.dumps(dict(at=stamp(),sample=sample,process_identities=identities,
                 shared_quality_authorized=True,foreign_allowed=True,new_speed_claim_allowed=False))+'\n')
        if enforce_runtime and sample['memory_free_bytes']<self.p['GPU_runtime_free_floor_bytes']:
            raise RuntimeError('runtime free memory below2GiB; stop owned work only')
        return sample

    def idle(self,*,initial=False):
        limit=min(time.monotonic()+30,self.deadline-25 if self.deadline else float('inf'))
        quiet=0;self.pre_inference=True
        while quiet<3:
            self.guard();sample=self.inspect()
            ready=sample['memory_free_bytes']>=self.p['GPU_initial_free_bytes']
            quiet=quiet+1 if ready else 0
            with (self.job/'capacity-samples.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=stamp(),sample=sample,consecutive=quiet,required_capacity_samples=3,
                                        required_free_bytes=33*2**30,foreign_GPU_processes=self.latest_foreign,
                                        utilization_not_an_admission_metric=True))+'\n')
            if time.monotonic()>limit:raise TimeoutError('bounded shared GPU capacity check; no retry or queue')
            if quiet<3:time.sleep(1)
        if initial:
            import shutil,disk_guard
            disk_guard.entry(shutil.disk_usage(self.job).free,self.p)
        return sample

    def finish_shared_cleanup(self):
        sample=self.inspect(enforce_runtime=False);remaining=[]
        for pin in self.gpu_pins.values():
            try:now=ownership.read_process(pin.pid)
            except FileNotFoundError:continue
            if (now.pid,now.start_ticks,now.uid,now.pgid)==(pin.pid,pin.start_ticks,pin.uid,pin.pgid):
                remaining.append(dict(pid=now.pid,reason='pinned owned identity still alive'))
        for pid in {r['pid'] for k in ('compute','graphics') for r in sample[k]}:
            try:info=ownership.read_process(pid)
            except FileNotFoundError:continue
            key=(info.pid,info.start_ticks,info.uid,info.pgid)
            if ownership.is_owned(info,self.pins):
                remaining.append(dict(pid=pid,reason='pinned owned GPU identity'))
        if remaining:raise RuntimeError('GPU cleanup incomplete; foreign processes not signalled: '+repr(remaining))
        return dict(status='PASS_OWNED_CLEANUP',sample=sample,remaining_owned_GPU_pids=[],foreign_preserved=True)

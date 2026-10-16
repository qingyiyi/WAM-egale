"""Finite fresh E2E job. Two natural EGL workers per arm, arms serialized.

Do not call an old campaign/zero-episode guard. Only exact owned roots and
descendants can be signalled. Failures preserve files and charge reservations.
"""
import argparse
from dataclasses import asdict
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

from task_contract import (OWN, RESEARCH, ROOT, WORKTREE, VA, CLIENT,
                           ENABLED, APPROVED, plan, digest, write, stamp)
from resource_guard import NVML, Idle
import ownership
from launch_identity import pin as pin_launch, bootstrap_key
import disk_guard

STOP = False

def charged_reservations(job, entries):
    total, rows = 0, []
    for entry in entries:
        path = Path(entry['out'])/'COUNTER-CLOSED.json'
        charge = entry['cap']
        exact = False
        if path.exists():
            c = json.loads(path.read_text())
            if (c['status'] != 'CLOSED' or c['cap'] != entry['cap']
                    or type(c['target_intents']) is not int
                    or not 0 <= c['target_intents'] <= entry['cap']
                    or c['target_intents'] != c['host_returned']
                    or c['failed_forwards'] or c['active_forwards'] or c['request_active']):
                raise ValueError('invalid closed reservation')
            charge, exact = c['target_intents'], True
        total += charge
        rows.append({**entry,'charged_target_intents':charge,'exact_closed_count':exact})
    return dict(charged_target_intents=total,reservations=rows,
                conservative_unclosed_reservations=True)

class Campaign:
    def __init__(self,job):
        self.job=Path(job).resolve()
        self.p=plan(self.job,gpu=True)
        self.nv=NVML()
        self.roots=[]
        self.pins={}
        self.boots={}
        self.reservations=[]
        self.stage='pre_gpu_gate'
        self.start=None
        self.deadline=None
        self.next_sample=0
        self.completed=0
        self.model_launches=0
        path=OWN.parent/'prompt_kv_gpu_v4/exit_guard.py'
        spec=importlib.util.spec_from_file_location('e2e_corrected_exit_guard',path)
        self.exit=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.exit)

    def status(self,**extra):
        counts=charged_reservations(self.job,self.reservations)
        write(self.job/'status.json',dict(stage=self.stage,at=stamp(),supervisor_pid=os.getpid(),
              GPU_started=self.start is not None,completed_episodes=self.completed,
              full_episode_cap=4,reused_target_episodes=2,model_launches=self.model_launches,
              GPU_elapsed_seconds=0 if self.start is None else time.monotonic()-self.start,
              **counts,**extra))

    def inspect(self):
        sample=self.nv.sample()
        apps='\n'.join(str(r['pid']) for k in ('compute','graphics') for r in sample[k])
        classified=self.exit.classify(apps,self.pins,enabled=True)
        with (self.job/'GPU-samples.jsonl').open('a') as f:
            f.write(json.dumps(dict(at=stamp(),monotonic=time.monotonic(),sample=sample,
                                   **classified))+'\n')
        if classified['foreign']:
            raise RuntimeError('foreign GPU processes; no foreign signal: '+str(classified['foreign']))
        return sample

    def guard(self):
        if STOP or (self.job/'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED')
        if self.deadline is not None and time.monotonic() >= self.deadline-25:
            raise TimeoutError('bounded GPU stage deadline, cleanup reserve retained')
        disk_guard.runtime(shutil.disk_usage(self.job).free,0,self.p)
        if self.start is not None and time.monotonic() >= self.next_sample:
            used=0
            for file in self.job.rglob('*'):
                try:
                    if file.is_file():used+=file.stat().st_size
                except FileNotFoundError:pass
            disk_guard.runtime(shutil.disk_usage(self.job).free,used,self.p)
            self.inspect();self.status();self.next_sample=time.monotonic()+5

    def idle(self,*,initial=False):
        gate=Idle(samples=15,max_util=2,interval=1)
        limit=time.monotonic()+180
        while True:
            self.guard()
            sample=self.nv.sample()
            with (self.job/'idle-samples.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=stamp(),monotonic=time.monotonic(),sample=sample))+'\n')
            if gate.add(sample,time.monotonic()): break
            if time.monotonic()>limit:
                raise RuntimeError('GPU did not reach stable idle; no contention')
            time.sleep(1)
        if initial:disk_guard.entry(shutil.disk_usage(self.job).free,self.p)
        return sample

    def environment(self,arm,out):
        env={k:v for k,v in os.environ.items()
             if not k.startswith(('WAM_P0_','P0_','LINGBOT_'))}
        ref=self.p['frozen_reference']
        env.update({ENABLED:'1',APPROVED:'1','P0_SPEC_E2E_JOB':str(self.job),
              'P0_SPEC_E2E_OUT':str(out),'P0_SPEC_E2E_ARM':arm,
              'CUDA_VISIBLE_DEVICES':'0','LINGBOT_SEED':'20260922','PYTHONHASHSEED':'0',
              'PYTHONDONTWRITEBYTECODE':'1','LINGBOT_DISABLE_FSDP':'1',
              'LINGBOT_CACHE_MODE':'original','LINGBOT_ATTN_MODE':'torch','LINGBOT_NSYS_PROFILE':'off',
              'LINGBOT_VA_LIBERO_SUITE':'libero_10',
              'LINGBOT_VA_LIBERO_CHECKPOINT':ref['target_checkpoint_path'],
              'LINGBOT_VA_LIBERO_CHECKPOINT_ID':ref['target_checkpoint_id'],
              'LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY':ref['registry'],
              'WAM_VAE_DEVICE':'cuda','WAM_P0_PHASE_PROFILE':'0',
              'WAM_P0_ACTION_DRAFT_MODE':'off' if arm=='T' else 'verifier_k2_b2',
              'WAM_P0_ACTION_DRAFT_SELECTOR':'global',
              'WAM_P0_ACTION_DRAFT_THRESHOLD':str(ref['threshold']),
              'WAM_P0_ACTION_DRAFT_B2_CHECKPOINT':self.p['frozen_A_off'],
              'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','MUJOCO_EGL_DEVICE_ID':'0',
              'OMP_NUM_THREADS':'7','MKL_NUM_THREADS':'7','OPENBLAS_NUM_THREADS':'7',
              'NUMEXPR_NUM_THREADS':'7'})
        return env

    def reserve(self,out,cap,kind):
        if len(self.reservations)>=4:
            raise RuntimeError('model launch cap')
        if self.p['aggregate_prior_target_intents']+sum(r['cap'] for r in self.reservations)+cap>35000:
            raise RuntimeError('cumulative forward reservation cap')
        out.mkdir(parents=True,exist_ok=True)
        r=dict(out=str(out),cap=cap,kind=kind,ordinal=len(self.reservations)+1,
               at=stamp(),plan_sha256=digest(self.job/'plan.json'))
        write(out/'FORWARD-RESERVATION.json',r,exclusive=True,durable=True)
        self.reservations.append(r)
        write(self.job/'RESERVATIONS.json',self.reservations,durable=True)

    def spawn(self,cmd,log,env,*,model=False):
        self.guard()
        Path(log).parent.mkdir(parents=True,exist_ok=True)
        with Path(log).open('xb') as f:
            p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                    env=env,cwd=WORKTREE,start_new_session=True)
        self.roots.append(p)
        # Track Popen immediately, wait for /proc cmdline to settle. This is a
        # bounded metadata wait, not a process/workload retry.
        def boot(info):self.boots[p.pid]=info
        try:
            info=pin_launch(p,cmd,on_boot=boot)
            self.pins[p.pid]=info
        except BaseException:
            if p.poll() is None and p.pid in self.boots:
                info=ownership.read_process(p.pid)
                if bootstrap_key(info)!=bootstrap_key(self.boots[p.pid]):
                    raise RuntimeError('refuse PID-changed bootstrap cleanup')
                write(self.job/('LAUNCH-PIN-FAILED-'+str(p.pid)+'.json'),
                      dict(bootstrap=asdict(self.boots[p.pid]),actual=asdict(info),
                           expected_cmd=cmd,owned_child_PID_only_cleanup=True),exclusive=True,durable=True)
                os.kill(p.pid,signal.SIGTERM)
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    info=ownership.read_process(p.pid)
                    if bootstrap_key(info)!=bootstrap_key(self.boots[p.pid]):
                        raise RuntimeError('PID-changed bootstrap kill refused')
                    os.kill(p.pid,signal.SIGKILL);p.wait(timeout=3)
            raise
        write(self.job/'OWNED-PROCESSES.json',[asdict(v) for v in self.pins.values()],durable=True)
        if model:self.model_launches+=1
        return p

    def stop(self,p):
        if p.poll() is not None:return
        info=ownership.read_process(p.pid)
        if not ownership.is_owned(info,self.pins):
            raise RuntimeError('refuse unowned cleanup')
        # Check every live group member before signalling the owned group.
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:child=ownership.read_process(int(path.name))
            except (FileNotFoundError,ProcessLookupError):continue
            if child.pgid==info.pgid and not ownership.is_owned(child,self.pins):
                raise RuntimeError('group has unowned process; refuse group signal')
        try:os.killpg(info.pgid,signal.SIGTERM)
        except ProcessLookupError:return
        try:p.wait(timeout=12)
        except subprocess.TimeoutExpired:
            again=ownership.read_process(p.pid)
            if not ownership.is_owned(again,self.pins):raise RuntimeError('PID identity changed during cleanup')
            os.killpg(again.pgid,signal.SIGKILL);p.wait(timeout=8)

    def model_cmd(self,script,port,master,save):
        return [VA,'-m','torch.distributed.run','--nproc_per_node=1',
                '--master_port='+str(master),'--local-ranks-filter=0','--tee=3',
                str(OWN/script),'--config-name','libero','--port',str(port),'--save_root',str(save)]

    def probe(self):
        self.stage='GPU_full_chunk_bitwise_probe'
        out=self.job/'probe';self.reserve(out,400,'complete-chunk-D-S-bitwise')
        env=self.environment('D',out)
        p=self.spawn(self.model_cmd('probe.py',self.p['ports'][0],self.p['ports'][4],out/'model_output'),
                     out/'probe.log',env,model=True)
        try:
            while p.poll() is None:self.guard();time.sleep(2)
            if p.returncode!=0:raise RuntimeError('full chunk probe failed; inspect preserved log')
            r=json.loads((out/'PROBE-RESULT.json').read_text())
            if r['status']!='PASS_FULL_CHUNK_BITWISE':raise RuntimeError('full chunk numerical gate')
            charged_reservations(self.job,self.reservations)
        finally:self.stop(p)
        self.idle()

    def wave(self,arm):
        self.stage='full_E2E_'+arm
        servers=[];clients=[];audits=[];outs=[]
        try:
            for i,init in enumerate((3,4)):
                out=self.job/'campaigns'/arm/'workers'/('worker'+str(i))
                outs.append(out);self.reserve(out,4000,'full-episode-'+arm+'-init'+str(init))
                env=self.environment(arm,out)
                write(out/'launch-config.json',dict(arm=arm,init=init,parallel_workers=2,
                      server_port=self.p['ports'][i],master_port=self.p['ports'][i+2],
                      EGL={k:env[k] for k in ('MUJOCO_GL','PYOPENGL_PLATFORM','MUJOCO_EGL_DEVICE_ID','CUDA_VISIBLE_DEVICES')},
                      full_horizon=800,trace_mode='off',profile_OFF=True,teacher_OFF=True,
                      target=self.p['frozen_reference'],draft=self.p['frozen_A_off'],
                      draft_sha256=self.p['frozen_A_off_sha256'],plan_sha256=digest(self.job/'plan.json')),
                      exclusive=True,durable=True)
                server=self.spawn(self.model_cmd('server_entry.py',self.p['ports'][i],
                      self.p['ports'][i+2],out/'server_output'),out/'server.log',env,model=True)
                servers.append(server);(out/'server.pid').write_text(str(server.pid))
            limit=time.monotonic()+600
            ready=set()
            while len(ready)<2:
                self.guard()
                for i,s in enumerate(servers):
                    if s.poll() is not None:raise RuntimeError('model server exited before readiness')
                    if i in ready:continue
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:'+str(self.p['ports'][i])+'/healthz',timeout=1) as r:
                            if r.status==200 and (outs[i]/'READY.json').exists():ready.add(i)
                    except (OSError,TimeoutError):pass
                if time.monotonic()>limit:raise TimeoutError('parallel model readiness deadline')
                time.sleep(2)
            for i,init in enumerate((3,4)):
                out=outs[i];env=self.environment(arm,out)
                cmd=[CLIENT,str(OWN/'client_entry.py'),'--port',str(self.p['ports'][i]),
                     '--master-port',str(self.p['ports'][i+2]),'--benchmarks','libero_10',
                     '--task-indices','0','--episodes-per-task','1','--episode-start',str(init),
                     '--condition',arm,'--campaign-id',self.job.name+'-'+arm,
                     '--worker-id','worker'+str(i),'--trace-mode','off','--out-dir',str(out/'client'),
                     '--fail-fast','--no-resume','--latency-benchmark','--worktree',str(WORKTREE)]
                c=self.spawn(cmd,out/'client.log',{**env,'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'})
                clients.append(c);(out/'client.pid').write_text(str(c.pid))
                a=self.spawn(['/usr/bin/python3',str(RESEARCH/'scripts/p0_live_device_egl_audit.py'),
                     '--client-pid-file',str(out/'client.pid'),'--server-pid-file',str(out/'server.pid'),
                     '--server-log',str(out/'server.log'),'--out',str(out/'live_device_egl_audit.json'),
                     '--timeout','300'],out/'egl-audit.log',env)
                audits.append(a)
            while any(c.poll() is None for c in clients):
                self.guard()
                if any(s.poll() is not None for s in servers):raise RuntimeError('server died during episode')
                if any(c.poll() not in (None,0) for c in clients):raise RuntimeError('EGL client failed')
                if any(a.poll() not in (None,0) for a in audits):raise RuntimeError('EGL live-device audit failed')
                time.sleep(2)
            for c in clients:
                if c.returncode!=0:raise RuntimeError('client nonzero exit')
            while any(a.poll() is None for a in audits):self.guard();time.sleep(1)
            if any(a.returncode!=0 for a in audits):raise RuntimeError('EGL provenance failure')
            for out,init in zip(outs,(3,4)):
                rows=[json.loads(x) for x in (out/'client/episodes.jsonl').read_text().splitlines() if x.strip()]
                if (len(rows)!=1 or rows[0]['episode_index']!=init or rows[0]['status']!='completed'
                        or rows[0]['truncated_after_max_policy_chunks']
                        or not (rows[0]['success'] or rows[0]['horizon_reached'])):
                    raise RuntimeError('actual full-episode coverage failed')
                if (out/'client/chunk_events.jsonl').exists():raise RuntimeError('unexpected timed chunk diagnostics')
                self.completed+=1
                (out/'FINISH_REQUESTED').write_text('clients complete; outside episode timing\n')
            finish_limit=time.monotonic()+90
            while not all((out/'FINISH_READY.json').exists() for out in outs):
                self.guard()
                if any((out/'FINISH_FAILED.json').exists() for out in outs):raise RuntimeError('worker close failed')
                if time.monotonic()>finish_limit:raise TimeoutError('asynchronous artifacts/accounting close deadline')
                time.sleep(1)
            self.status()
        finally:
            for p in reversed(audits+clients+servers):self.stop(p)
        self.idle()

    def run(self):
        error=None
        cleanup=None
        try:
            self.status()
            self.idle(initial=True)
            for port in self.p['ports']:
                with socket.socket() as s:s.bind(('127.0.0.1',port))
            self.start=time.monotonic();self.deadline=self.start+5400
            write(self.job/'GPU-ATTEMPT.json',dict(at=stamp(),attempt=1,
                  monotonic_start=self.start,monotonic_deadline=self.deadline,
                  plan_sha256=digest(self.job/'plan.json'),GPU_seconds_cap=5400),exclusive=True,durable=True)
            # Accepted original full-chunk D/S probe and two target episodes
            # are frozen from v1. Shared/model/client code is byte-identical.
            for arm in ('D','S'):self.wave(arm)
            self.stage='GPU_collection_complete'
        except BaseException as exc:
            error=dict(error=repr(exc),traceback=traceback.format_exc(),failed_stage=self.stage)
            self.stage='failed'
        finally:
            try:
                for p in reversed(self.roots):self.stop(p)
                cleanup=self.nv.sample()
                owned_live=[]
                for pid,pin in self.pins.items():
                    try:info=ownership.read_process(pid)
                    except FileNotFoundError:continue
                    if ownership.is_owned(info,self.pins):owned_live.append(pid)
                if cleanup['compute'] or cleanup['graphics'] or cleanup['memory_used_bytes'] or owned_live:
                    raise RuntimeError('GPU cleanup not empty or owned live roots remain')
            except BaseException as exc:
                cleanup=dict(status='FAIL',error=repr(exc))
                if error is None:error=dict(error=repr(exc),failed_stage='cleanup')
            finally:self.nv.close()
        elapsed=0 if self.start is None else time.monotonic()-self.start
        charges=charged_reservations(self.job,self.reservations)
        final=dict(status='PASS_GPU_FULL_EPISODE_COLLECTION' if error is None else 'FAILED',
                   at=stamp(),completed_episodes=self.completed,model_launches=self.model_launches,
                   reused_target_episodes=2,full_episode_data_total=2+self.completed,
                   new_GPU_seconds=elapsed,aggregate_GPU_seconds=self.p['aggregate_prior_GPU_seconds']+elapsed,
                   aggregate_GPU_seconds_cap=18000,aggregate_target_intents=self.p['aggregate_prior_target_intents']+charges['charged_target_intents'],
                   aggregate_target_cap=35000,cleanup=cleanup,**charges)
        if error:final.update(error)
        write(self.job/('GPU-COLLECTION.json' if error is None else 'FAILED.json'),final,
              exclusive=True,durable=True)
        self.status(terminal_result=final['status'])
        if error:raise RuntimeError(error['error'])

def main():
    global STOP
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    args=ap.parse_args()
    for sig in (signal.SIGINT,signal.SIGTERM):
        def stop(*_):
            global STOP
            STOP=True
        signal.signal(sig,stop)
    # Same shared campaign lock as the original parallel evaluation runner.
    with (RESEARCH/'results/runtime/.speculative_minimal_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (RESEARCH/'results/runtime/.shared-prefix-GPU.lock').open('a') as prefix_lock:
            fcntl.flock(prefix_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            Campaign(args.job).run()

if __name__=='__main__':main()

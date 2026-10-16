"""One deployment fixture worker, then three paired EGL waves; frozen idle/ownership guards."""
import argparse,json,os,time,traceback,signal,socket,fcntl,urllib.request
from pathlib import Path
from experiment import ROOT,OWN,S0,WORKTREE,RESEARCH,VA,CLIENT,ENABLED,APPROVED,plan,digest,write,stamp,MODES,SELECTORS
from paths import load
old=load('o4_frozen_campaign',S0/'supervisor.py')

class Campaign(old.Campaign):
    def __init__(self,job):
        self.job=Path(job).resolve();self.p=plan(self.job,gpu=True)
        self.nv=old.NVML();self.roots=[];self.pins={};self.boots={};self.reservations=[]
        self.stage='pre_gpu_gate';self.start=self.deadline=None;self.next_sample=0
        self.completed=self.model_launches=0
        self.pre_inference=True;self.latest_foreign=[]
        self.exit=load('o4_frozen_exit',OWN.parent/'prompt_kv_gpu_v4/exit_guard.py')

    @staticmethod
    def startup_ready(sample,foreign):
        return not foreign and sample['utilization_percent']<=2

    def inspect(self):
        sample=self.nv.sample()
        apps='\n'.join(str(r['pid']) for kind in ('compute','graphics') for r in sample[kind])
        classified=self.exit.classify(apps,self.pins,enabled=True)
        self.latest_foreign=classified['foreign']
        provenance=[]
        for pid in self.latest_foreign:
            try:
                info=old.ownership.read_process(pid)
                provenance.append(dict(pid=pid,ppid=info.ppid,pgid=info.pgid,uid=info.uid,
                    start_ticks=info.start_ticks,executable=info.argv[0] if info.argv else None,
                    command_sha256=__import__('hashlib').sha256(repr(info.argv).encode()).hexdigest(),
                    known_inductor_worker=any('compile_worker' in a for a in info.argv)))
            except FileNotFoundError:pass
        with (self.job/'GPU-samples.jsonl').open('a') as f:
            f.write(json.dumps(dict(at=stamp(),monotonic=time.monotonic(),sample=sample,
                pre_inference=self.pre_inference,foreign_provenance=provenance,**classified))+'\n')
        if self.latest_foreign and not self.pre_inference:
            raise RuntimeError('foreign GPU processes during formal episode; no foreign signal: '+str(self.latest_foreign))
        return sample

    def reserve(self,out,cap,kind):
        if len(self.reservations)>=self.p['model_worker_cap']:
            raise RuntimeError('finite model launch cap')
        if self.p['aggregate_prior_target_intents']+sum(r['cap'] for r in self.reservations)+cap>self.p['aggregate_target_cap']:
            raise RuntimeError('finite aggregate forward cap')
        out.mkdir(parents=True,exist_ok=True)
        r=dict(out=str(out),cap=cap,kind=kind,ordinal=len(self.reservations)+1,
               at=stamp(),plan_sha256=digest(self.job/'plan.json'))
        write(out/'FORWARD-RESERVATION.json',r,exclusive=True,durable=True)
        self.reservations.append(r)
        write(self.job/'RESERVATIONS.json',self.reservations,durable=True)

    def idle(self,*,initial=False):
        gate=old.Idle(samples=15,max_util=2,interval=1)
        limit=time.monotonic()+(self.p['initial_resource_wait_seconds'] if initial else 180)
        if initial:self.stage='waiting_for_verified_idle_zero_models'
        while True:
            self.guard();sample=self.nv.sample()
            with (self.job/'idle-samples.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=stamp(),monotonic=time.monotonic(),sample=sample))+'\n')
            if gate.add(sample,time.monotonic()):break
            if initial:self.status(resource_wait=True,latest_idle_sample=sample)
            if time.monotonic()>limit:raise RuntimeError('GPU stable idle unavailable; no model launch or contention')
            time.sleep(1)
        if initial:
            import shutil,disk_guard
            disk_guard.entry(shutil.disk_usage(self.job).free,self.p)
        return sample

    def fixed(self):
        self.stage='single_model_bonus_selector_numerical_oracles'
        self.pre_inference=False
        out=self.job/'fixed/worker0'
        self.reserve(out,self.p['fixed_target_cap'],'single-model-fixed-complete-chunks')
        env=self.environment('DEPTH',out)
        proc=self.spawn(self.model_cmd('fixed_worker.py',self.p['ports'][0],
            self.p['ports'][2],out/'server_output'),out/'server.log',env,model=True)
        try:
            while proc.poll() is None:self.guard();time.sleep(1)
            if proc.returncode!=0 or not (out/'WORKER-RESULT.json').exists():
                raise RuntimeError('fixed bonus integrity/performance worker failed')
            result=json.loads((out/'WORKER-RESULT.json').read_text())
            if result['status']!='PASS_FIXED_ACCEPTANCE_ORACLES' or result['chunks']!=8:
                raise RuntimeError('fixed complete chunk coverage failed')
        finally:
            from cleanup import roots
            closed=roots([proc],self.pins,deadline=min(time.monotonic()+25,self.deadline),wait_seconds=1.)
            write(self.job/'FIXED-CLEANUP.json',closed,exclusive=True,durable=True)
        self.idle()

    def status(self,**extra):
        counts=old.charged_reservations(self.job,self.reservations)
        write(self.job/'status.json',dict(stage=self.stage,at=stamp(),supervisor_pid=os.getpid(),
            GPU_started=self.start is not None,completed_episodes=self.completed,full_episode_cap=6,
            model_launches=self.model_launches,GPU_elapsed_seconds=0 if self.start is None else time.monotonic()-self.start,
            **counts,**extra))

    def environment(self,arm,out):
        env=super().environment(arm,out)
        env.update({ENABLED:'1',APPROVED:'1','BONUS_ACCEPT_INIT':str(6+int(out.name[-1])),
            'WAM_P0_ACTION_DRAFT_MODE':MODES[arm],
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'})
        for k in list(env):
            if k.startswith(('CHUNK_ROWS_','CHUNK_SCALAR_','CHUNK_TEMPLATE_')):env.pop(k)
        return env

    def model_cmd(self,script,port,master,save):
        return [VA,'-m','torch.distributed.run','--nproc_per_node=1',
            '--master_port='+str(master),'--local-ranks-filter=0','--tee=3',
            str(OWN/script),'--config-name','libero','--port',str(port),'--save_root',str(save)]

    def wave(self,wave_index):
        specs=self.p['assignments'][wave_index]
        self.pre_inference=True
        self.stage='bonus_quality_balanced_wave_'+str(wave_index)
        servers=[];clients=[];audits=[];outs=[]
        try:
            for i,item in enumerate(specs):
                arm,init=item['arm'],item['init']
                out=self.job/'campaigns'/arm/'workers'/('worker'+str(init-6))
                outs.append(out);self.reserve(out,item['cap'],'full-episode-'+arm+'-init'+str(init))
                env=self.environment(arm,out)
                write(out/'launch-config.json',dict(arm=arm,init=init,parallel_workers=2,effective_selector=SELECTORS[arm],bonus_enabled=(arm!='T'),speed_comparison_deferred=True,chunk_events_ON=True,
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
            self.stage='ready_models_waiting_before_any_episode'
            limit=min(time.monotonic()+180,self.deadline-25);quiet=0
            while quiet<15:
                self.guard();sample=self.inspect()
                quiet=quiet+1 if self.startup_ready(sample,self.latest_foreign) else 0
                for out in outs:
                    c=json.loads((out/'counter-state.json').read_text())
                    if c['target_intents'] or c['request_active']:raise RuntimeError('unexpected inference before client admission')
                if time.monotonic()>limit:raise TimeoutError('pre-client owned-only stable GPU admission')
                time.sleep(1)
            self.pre_inference=False;self.next_sample=0;self.guard()
            for i,item in enumerate(specs):
                arm,init=item['arm'],item['init']
                out=outs[i];env=self.environment(arm,out)
                cmd=[CLIENT,str(OWN/'client_entry.py'),'--port',str(self.p['ports'][i]),
                     '--master-port',str(self.p['ports'][i+2]),'--benchmarks','libero_10',
                     '--task-indices','0','--episodes-per-task','1','--episode-start',str(init),
                     '--condition',arm,'--campaign-id',self.job.name+'-'+arm,
                     '--worker-id','worker'+str(init-6),'--trace-mode','off','--out-dir',str(out/'client'),
                     '--fail-fast','--no-resume','--worktree',str(WORKTREE)]
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
            for out,item in zip(outs,specs):
                init=item['init']
                rows=[json.loads(x) for x in (out/'client/episodes.jsonl').read_text().splitlines() if x.strip()]
                if (len(rows)!=1 or rows[0]['episode_index']!=init or rows[0]['status']!='completed'
                        or rows[0]['truncated_after_max_policy_chunks']
                        or not (rows[0]['success'] or rows[0]['horizon_reached'])):
                    raise RuntimeError('actual full-episode coverage failed')
                if not (out/'client/chunk_events.jsonl').exists():raise RuntimeError('missing required quality chunk events')
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
            from cleanup import roots
            closed=roots(audits+clients+servers,self.pins,deadline=min(time.monotonic()+25,self.deadline),wait_seconds=1.)
            write(self.job/f'WAVE-{wave_index}-CLEANUP.json',closed,exclusive=True,durable=True)
        self.idle()

    def run(self):
        error=cleanup=None
        try:
            self.status();self.idle(initial=True)
            for port in self.p['ports']:
                with socket.socket() as s:s.bind(('127.0.0.1',port))
            self.start=time.monotonic();self.deadline=self.start+self.p['GPU_hard_limit_seconds']
            write(self.job/'GPU-ATTEMPT.json',dict(at=stamp(),attempt=1,monotonic_start=self.start,
                monotonic_deadline=self.deadline,plan_sha256=digest(self.job/'plan.json'),
                GPU_seconds_cap=self.p['GPU_hard_limit_seconds']),exclusive=True,durable=True)
            self.fixed()
            for wave_index in (0,1,2):self.wave(wave_index)
            self.stage='GPU_collection_complete'
        except BaseException as exc:
            error=dict(error=repr(exc),traceback=traceback.format_exc(),failed_stage=self.stage)
            write(self.job/'STOP_REQUESTED',dict(at=stamp(),reason=repr(exc)))
            self.stage='failed'
        finally:
            try:
                from cleanup import roots
                closed=roots(self.roots,self.pins,deadline=min(time.monotonic()+25,self.deadline or time.monotonic()+25),wait_seconds=1.)
                write(self.job/'CLEANUP.json',closed,exclusive=True,durable=True)
                cleanup=self.nv.sample()
                if not all(p.poll() is not None for p in self.roots):
                    raise RuntimeError('owned process cleanup incomplete')
                if self.start is not None and (cleanup['compute'] or cleanup['graphics'] or cleanup['memory_used_bytes']):
                    raise RuntimeError('GPU/owned cleanup not empty')
            except BaseException as exc:
                cleanup=dict(status='FAIL',error=repr(exc));error=error or dict(error=repr(exc),failed_stage='cleanup')
            finally:self.nv.close()
        seconds=0 if self.start is None else time.monotonic()-self.start
        charges=old.charged_reservations(self.job,self.reservations)
        final=dict(status='PASS_GPU_FULL_EPISODE_COLLECTION' if error is None else 'FAILED',at=stamp(),
            completed_episodes=self.completed,model_launches=self.model_launches,reused_target_episodes=0,
            full_episode_data_total=self.completed,new_GPU_seconds=seconds,
            aggregate_GPU_seconds=self.p['aggregate_prior_GPU_seconds']+seconds,aggregate_GPU_seconds_cap=18000,
            aggregate_target_intents=self.p['aggregate_prior_target_intents']+charges['charged_target_intents'],
            aggregate_target_cap=self.p['aggregate_target_cap'],cleanup=cleanup,owned_roots_exited=all(p.poll() is not None for p in self.roots),**charges)
        if error:final.update(error)
        write(self.job/('GPU-COLLECTION.json' if error is None else 'FAILED.json'),final,exclusive=True,durable=True)
        self.status(terminal_result=final['status'])
        if error:raise RuntimeError(error['error'])
        print(json.dumps(final,ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);args=ap.parse_args()
    def stop(*_):old.STOP=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    with (ROOT/'.pipeline/.chunk-template-gpu.lock').open('a') as local, (RESEARCH/'results/runtime/.speculative_minimal_gpu.lock').open('a') as task, (RESEARCH/'results/runtime/.shared-prefix-GPU.lock').open('a') as prefix:
        for lock in (local,task,prefix):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Campaign(args.job).run()

if __name__=='__main__':main()

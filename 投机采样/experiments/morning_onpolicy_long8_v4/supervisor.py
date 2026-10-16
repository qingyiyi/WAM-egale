"""One fresh numeric fixture and one paired EGL wave per isolated job."""
import argparse,json,os,time,traceback,signal,socket,fcntl,urllib.request
from pathlib import Path
from experiment import ROOT,OWN,S0,WORKTREE,RESEARCH,VA,CLIENT,ENABLED,APPROVED,plan,digest,write,stamp,MODES,SELECTORS
from paths import load
from shared_resource import SharedResources,charged_reservations
from retry_guard import conservative_model_launches
old=load('o4_frozen_campaign',S0/'supervisor.py')

class Campaign(SharedResources,old.Campaign):
    def __init__(self,job):
        self.job=Path(job).resolve();self.p=plan(self.job,gpu=True)
        self.nv=old.NVML();self.roots=[];self.pins={};self.boots={};self.reservations=[]
        self.stage='pre_gpu_gate';self.start=self.deadline=None;self.next_sample=0
        self.completed=self.model_launches=0
        self.client_admission_started=False
        self.episode_identity_audits=[]
        self.pre_inference=True;self.latest_foreign=[]
        self.baseline_GPU_identities=set();self.gpu_pins={}
        self.exit=load('o4_frozen_exit',OWN.parent/'prompt_kv_gpu_v4/exit_guard.py')



    def reserve(self,out,cap,kind):
        if len(self.reservations)>=self.p['model_worker_cap']:
            raise RuntimeError('finite model launch cap')
        if self.p['aggregate_prior_model_launches']+len(self.reservations)+1>self.p['aggregate_model_launches_cap']:
            raise RuntimeError('aggregate conservative model launch cap')
        if sum(r['cap'] for r in self.reservations)+cap>self.p['target_cap']:
            raise RuntimeError('finite wave forward cap')
        if self.p['aggregate_prior_target_intents']+sum(r['cap'] for r in self.reservations)+cap>self.p['aggregate_target_cap']:
            raise RuntimeError('finite aggregate forward cap')
        out.mkdir(parents=True,exist_ok=True)
        r=dict(out=str(out),cap=cap,kind=kind,ordinal=len(self.reservations)+1,
               at=stamp(),plan_sha256=digest(self.job/'plan.json'))
        write(out/'FORWARD-RESERVATION.json',r,exclusive=True,durable=True)
        self.reservations.append(r)
        write(self.job/'RESERVATIONS.json',self.reservations,durable=True)


    def fixed(self):
        self.stage='new_cold_install_four_native_OFF_ON_oracles'
        self.pre_inference=False
        out=self.job/'fixed/worker0';self.reserve(out,320,'new-cold-four-complete-native-chunks')
        env=self.environment('COLLECT',out)
        proc=self.spawn(self.model_cmd('fixed_worker.py',self.p['ports'][0],self.p['ports'][2],out/'server_output'),out/'server.log',env,model=True)
        try:
            while proc.poll() is None:self.guard();time.sleep(1)
            if proc.returncode!=0 or not (out/'WORKER-RESULT.json').exists():
                raise RuntimeError('new cold native numeric worker failed')
            result=json.loads((out/'WORKER-RESULT.json').read_text())
            cold=json.loads((out/'COLD-INSTALL-PROOF.json').read_text())
            if (result['status']!='PASS_FIXED_TEACHER_TRAIN_ORACLES' or result['chunks']!=4 or
                not result['new_native_cold_install_proof'] or result['previous_four_GPU_oracles_reused'] or
                not result['same_candidate_teacher_OFF_ON_exact'] or not result['teacher_variants_all30_output_routes_RNG_exact'] or
                cold['status']!='PASS_NATIVE_COLD_INSTALL_ORIGINAL_RESET' or not cold['constructor_had_no_action_per_frame']):
                raise RuntimeError('new cold/teacher full50 numeric proof incomplete')
            write(self.job/'NEW-NATIVE-ORACLE-PASS.json',dict(status='PASS_NEW_COLD_SOURCE_FOUR_NATIVE_ORACLES',
                complete_chunks=4,target_intents=result['target_intents'],worker_result_sha256=digest(out/'WORKER-RESULT.json'),
                cold_proof_sha256=digest(out/'COLD-INSTALL-PROOF.json'),previous_GPU_oracles_not_reused=True),exclusive=True,durable=True)
        finally:
            from cleanup import roots
            closed=roots([proc],self.pins,deadline=min(time.monotonic()+25,self.deadline),wait_seconds=1.)
            write(self.job/'FIXED-CLEANUP.json',closed,exclusive=True,durable=True)
        self.idle()

    def guard(self):
        from datetime import datetime
        if datetime.now().astimezone()>=datetime.fromisoformat(self.p['GPU_absolute_stop_at']):
            raise TimeoutError('overnight absolute GPU stop deadline')
        return super().guard()

    def status(self,**extra):
        counts=charged_reservations(self.job,self.reservations)
        write(self.job/'status.json',dict(stage=self.stage,at=stamp(),supervisor_pid=os.getpid(),
            GPU_started=self.start is not None,completed_episodes=self.completed,full_episode_cap=self.p['full_episode_cap'],
            model_launches=self.model_launches,charged_model_launches=conservative_model_launches(self.model_launches,self.reservations),
            GPU_elapsed_seconds=0 if self.start is None else time.monotonic()-self.start,
            **counts,**extra))

    def environment(self,arm,out):
        env=super().environment(arm,out)
        item=next((x for wave in self.p['assignments'] for x in wave if x['id']==out.name),None)
        env.update({ENABLED:'1',APPROVED:'1','WAM_P0_ACTION_DRAFT_MODE':MODES[arm],
            'WAM_P0_ACTION_DRAFT_B2_CHECKPOINT':self.p['new_checkpoint'],
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'})
        if item is not None:
            env.update(DRAFT_TRAIN_ID=item['id'],DRAFT_TRAIN_TASK=str(item['task']),DRAFT_TRAIN_INIT=str(item['init']))
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
                out=self.job/'campaigns'/arm/'workers'/item['id']
                outs.append(out);self.reserve(out,item['cap'],'full-episode-'+item['id'])
                env=self.environment(arm,out)
                write(out/'launch-config.json',dict(arm=arm,init=init,parallel_workers=2,effective_selector=SELECTORS[arm],bonus_enabled=(arm!='T'),speed_comparison_deferred=True,chunk_events_ON=True,
                      server_port=self.p['ports'][i],master_port=self.p['ports'][i+2],
                      EGL={k:env[k] for k in ('MUJOCO_GL','PYOPENGL_PLATFORM','MUJOCO_EGL_DEVICE_ID','CUDA_VISIBLE_DEVICES')},
                      full_horizon=800,trace_mode='off',profile_OFF=True,teacher_OFF=(self.p['stage']!='collection'),
                      target=self.p['frozen_reference'],draft=self.p['new_checkpoint'],
                      draft_sha256=self.p['new_checkpoint_sha256'],plan_sha256=digest(self.job/'plan.json')),
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
            while quiet<3:
                self.guard();sample=self.inspect()
                quiet=quiet+1 if self.startup_ready(sample,self.latest_foreign) else 0
                for out in outs:
                    c=json.loads((out/'counter-state.json').read_text())
                    if c['target_intents'] or c['request_active']:raise RuntimeError('unexpected inference before client admission')
                if time.monotonic()>limit:raise TimeoutError('shared free-memory admission failed')
                time.sleep(1)
            self.pre_inference=False;self.next_sample=0;self.stage='full_episodes_wave_'+str(wave_index);self.guard()
            # Set before the first Popen: a partial pair is never retried in
            # full under already used task/init keys.
            write(self.job/'CLIENT-ADMISSION-INTENT.json',dict(at=stamp(),wave_index=wave_index,
                assignment_ids=[x['id'] for x in specs],reason='before_first_client_spawn'),
                exclusive=True,durable=True)
            self.client_admission_started=True
            for i,item in enumerate(specs):
                arm,init=item['arm'],item['init']
                out=outs[i];env=self.environment(arm,out)
                cmd=[CLIENT,str(OWN/'client_entry.py'),'--port',str(self.p['ports'][i]),
                     '--master-port',str(self.p['ports'][i+2]),'--benchmarks','libero_10',
                     '--task-indices',str(item['task']),'--episodes-per-task','1','--episode-start',str(init),
                     '--condition',arm,'--campaign-id',self.job.name+'-'+arm,
                     '--worker-id',item['id'],'--trace-mode','off','--out-dir',str(out/'client'),
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
                repro=json.loads((out/'client/repro_metadata.json').read_text())
                if (len(rows)!=1 or rows[0]['task_index']!=item['task'] or rows[0]['episode_index']!=init or rows[0]['status']!='completed'
                        or rows[0]['truncated_after_max_policy_chunks']
                        or not (rows[0]['success'] or rows[0]['horizon_reached'])
                        or repro['command_count']!=rows[0]['env_steps']
                        or repro['actual_EGL_initialization_attempts']!=1):
                    raise RuntimeError('actual full-episode coverage failed')
                if not (out/'client/chunk_events.jsonl').exists():raise RuntimeError('missing required quality chunk events')
                (out/'FINISH_REQUESTED').write_text('clients complete; outside episode timing\n')
            finish_limit=time.monotonic()+90
            while not all((out/'FINISH_READY.json').exists() for out in outs):
                self.guard()
                if any((out/'FINISH_FAILED.json').exists() for out in outs):raise RuntimeError('worker close failed')
                if time.monotonic()>finish_limit:raise TimeoutError('asynchronous artifacts/accounting close deadline')
                time.sleep(1)
            for out in outs:
                labels=json.loads((out/'LABELS-SUMMARY.json').read_text())
                if labels['errors'] or labels['saved_trees']<=0 or labels['extra_teacher_forwards']!=0:
                    raise ValueError('teacher error/empty/missing same-query labels')
                markers=[json.loads(f.read_text()) for f in (out/'teacher').glob('CHUNK-*.json')]
                ep=json.loads((out/'client/episodes.jsonl').read_text().splitlines()[0])
                if len(markers)!=ep['infer_chunks'] or any(not m['complete'] or m['attempted_trees']!=m['saved_trees'] for m in markers):
                    raise ValueError('teacher CHUNK markers incomplete')
                if sum(m['saved_trees'] for m in markers)!=labels['saved_trees']:
                    raise ValueError('teacher summary/marker counts mismatch')
            # Treat a wave as complete only after all files bind to the same
            # task/init/attempt, generator and actual verifier query identity.
            import torch
            from identity_gate import validate as validate_identity
            for out,item in zip(outs,specs):
                episode=json.loads((out/'client/episodes.jsonl').read_text().splitlines()[0])
                repro=json.loads((out/'client/repro_metadata.json').read_text())
                events=[json.loads(line) for line in (out/'client/chunk_events.jsonl').read_text().splitlines() if line.strip()]
                labels=json.loads((out/'LABELS-SUMMARY.json').read_text())
                marker_sources=sorted((out/'teacher').glob('CHUNK-*.json'))
                markers=[json.loads(f.read_text()) for f in marker_sources]
                sources=sorted((out/'teacher').glob('*.pt'))
                payloads=[torch.load(f,map_location='cpu',weights_only=False) for f in sources]
                audit=validate_identity(item,episode,repro,labels,markers,payloads,events,
                    plan_sha256=digest(self.job/'plan.json'),
                    draft_sha256=self.p['new_checkpoint_sha256'],
                    target_checkpoint_id=self.p['frozen_reference']['target_checkpoint_id'],
                    threshold=self.p['frozen_reference']['threshold'])
                audit_path=out/'EPISODE-IDENTITY-AUDIT.json'
                write(audit_path,dict(**audit,
                    label_files=[dict(path=str(f),sha256=digest(f)) for f in sources],
                    chunk_marker_files=[dict(path=str(f),sha256=digest(f)) for f in marker_sources],
                    client_files=[dict(path=str(out/'client'/name),sha256=digest(out/'client'/name))
                        for name in ('episodes.jsonl','repro_metadata.json','chunk_events.jsonl',
                                     'commands.npy','initial_state.npz','final_state.npz','run_metadata.json')]),
                    exclusive=True,durable=True)
                self.episode_identity_audits.append(dict(id=item['id'],task=item['task'],init=item['init'],
                    path=str(audit_path),sha256=digest(audit_path)))
            if len(self.episode_identity_audits)!=len(specs):
                raise RuntimeError('both identity audits required before committing wave')
            self.completed+=len(specs)
            self.status()
        finally:
            from cleanup import roots
            closed=roots(audits+clients+servers,self.pins,deadline=min(time.monotonic()+25,self.deadline),wait_seconds=1.)
            write(self.job/f'WAVE-{wave_index}-CLEANUP.json',closed,exclusive=True,durable=True)
        # One wave per job; the next wave performs a fresh admission check.

    def run(self):
        error=cleanup=None
        try:
            self.status();self.idle(initial=True)
            for port in self.p['ports']:
                with socket.socket() as sock:sock.bind(('127.0.0.1',port))
            from datetime import datetime
            remaining=(datetime.fromisoformat(self.p['GPU_absolute_stop_at'])-datetime.now().astimezone()).total_seconds()
            if remaining<=25:raise TimeoutError('insufficient time before absolute stop')
            self.start=time.monotonic();self.deadline=self.start+min(self.p['GPU_hard_limit_seconds'],remaining)
            write(self.job/'GPU-ATTEMPT.json',dict(at=stamp(),attempt=1,monotonic_start=self.start,
                monotonic_deadline=self.deadline,plan_sha256=digest(self.job/'plan.json'),
                GPU_seconds_cap=self.p['GPU_hard_limit_seconds']),exclusive=True,durable=True)
            self.fixed()
            for index in range(len(self.p['assignments'])):self.wave(index)
            if self.completed!=self.p['full_episode_cap']:raise RuntimeError('paired full TRAIN episodes required')
            self.stage='GPU_collection_complete'
        except BaseException as exc:
            error=dict(error=repr(exc),traceback=traceback.format_exc(),failed_stage=self.stage)
            write(self.job/'STOP_REQUESTED',dict(at=stamp(),reason=repr(exc)))
            self.stage='failed'
        finally:
            try:
                from cleanup import roots
                now=time.monotonic();end=min(now+20,self.deadline or now+20)
                closed=roots(self.roots,self.pins,deadline=end,wait_seconds=1.)
                write(self.job/'CLEANUP.json',closed,exclusive=True,durable=True)
                cleanup=self.finish_shared_cleanup()
                if not all(p.poll() is not None for p in self.roots):raise RuntimeError('owned roots remain')
            except BaseException as exc:
                cleanup=dict(status='FAIL',error=repr(exc));error=error or dict(error=repr(exc),failed_stage='cleanup')
            try:self.nv.close()
            except BaseException as exc:error=error or dict(error=repr(exc),failed_stage='NVML_close')
        seconds=0 if self.start is None else time.monotonic()-self.start
        if seconds>self.p['GPU_hard_limit_seconds']:
            error=error or dict(error='actual elapsed exceeded wave hard cap',failed_stage='deadline')
        from datetime import datetime
        if self.start is not None and datetime.now().astimezone()>datetime.fromisoformat(self.p['GPU_absolute_stop_at']):error=error or dict(error='actual absolute GPU stop deadline exceeded',failed_stage='deadline')
        charges=charged_reservations(self.job,self.reservations,failed_phase=error is not None)
        if charges['counter_errors']:
            error=error or dict(error=repr(charges['counter_errors']),failed_stage='counter_closure')
            charges=charged_reservations(self.job,self.reservations,failed_phase=True)
        from retry_guard import inspect_workers
        observation=inspect_workers(self.job,self.p['assignments'],
                                    client_admission_started=self.client_admission_started)
        charged_models=conservative_model_launches(self.model_launches,self.reservations)
        if error is None and (charged_models!=self.p['model_worker_cap'] or
                              len(self.episode_identity_audits)!=self.p['full_episode_cap']):
            error=dict(error='successful wave lacks exact model/audit closure',failed_stage='terminal_closure')
            charges=charged_reservations(self.job,self.reservations,failed_phase=True)
        retry_eligible=False
        final=dict(status='PASS_GPU_FULL_EPISODE_COLLECTION' if error is None else 'FAILED',at=stamp(),
            completed_episodes=self.completed,model_launches=self.model_launches,
            charged_model_launches=charged_models,
            new_GPU_seconds=seconds,aggregate_GPU_seconds=self.p['aggregate_prior_GPU_seconds']+seconds,
            aggregate_target_intents=self.p['aggregate_prior_target_intents']+charges['charged_target_intents'],
            aggregate_model_launches=self.p['aggregate_prior_model_launches']+charged_models,
            aggregate_GPU_seconds_cap=self.p['aggregate_GPU_seconds_cap'],aggregate_target_cap=self.p['aggregate_target_cap'],cleanup=cleanup,
            owned_roots_exited=all(p.poll() is not None for p in self.roots),
            episode_identity_audits=self.episode_identity_audits,
            retry_observation=observation,retry_eligible=retry_eligible,
            blocked_retry_reason=None if error is None else 'v4 requires a new reviewed recovery plan after any FAILED wave',
            shared_quality_authorized=True,exclusive_required=False,performance_deferred=True,no_deployment=True,
            plan_sha256=digest(self.job/'plan.json'),**charges)
        if error:final.update(error)
        write(self.job/('GPU-COLLECTION.json' if error is None else 'FAILED.json'),final,exclusive=True,durable=True)
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

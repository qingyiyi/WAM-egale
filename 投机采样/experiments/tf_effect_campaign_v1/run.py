"""Owned, single-model complete-episode driver, no formal timing or retry."""
import argparse,fcntl,os,signal,socket,sys,time,traceback,shutil,urllib.request
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
OWN=Path(__file__).resolve().parent;sys.path.insert(0,str(OWN.parent))
from tf_effect_campaign_v1.protocol import EXP,RT,ROOT,read,write,load,prepare,environment,require,sha

def execute(job,p):
    from tf_usable_quality_v1.legacy import launcher
    Base,cleanup=launcher()
    class Driver:
        def __init__(self):self.roots=[];self.pins={}
    d=Driver();born=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19])/os.sysconf('SC_CLK_TCK')
    end=min(born+p['max_run_wall_sec'],time.monotonic()+(datetime.fromisoformat(p['absolute_cleanup_deadline'])-datetime.now(timezone.utc)).total_seconds())
    work=end-p['cleanup_reserve_sec'];require(work>time.monotonic(),'campaign expired')
    write(job/'RUN-STARTED.json',dict(status='EXPLICIT_RUN',pid=os.getpid(),max_run_wall_sec=p['max_run_wall_sec'],
        birth_monotonic=born,absolute_work_stop=p['absolute_work_stop'],absolute_cleanup_deadline=p['absolute_cleanup_deadline']))
    until=min(time.monotonic()+60,work)
    while not (job/'CLOCK-GUARD.json').exists():
        require(time.monotonic()<until,'independent clock not attached');time.sleep(.2)
    g=read(job/'CLOCK-GUARD.json');require(g['supervisor_pin']['pid']==os.getpid() and g['clock']['plan_sha256']==sha(job/'plan.json'),'wrong clock')
    env0=environment(p);completed=[];total=0;current=None;seq=0;last_probe=0;last_size=0
    def spawn(cmd,out,env,affinity=None):
        if not (job/'SPAWN-ATTEMPTED.json').exists():write(job/'SPAWN-ATTEMPTED.json',dict(status='OWNED_CHILD_LAUNCH_ATTEMPTED',pid=os.getpid()))
        previous=os.sched_getaffinity(0)
        try:
            if affinity is not None:os.sched_setaffinity(0,set(affinity))
            proc=Base.spawn(d,cmd,out,env)
        finally:os.sched_setaffinity(0,previous)
        write(job/f'OWNED-{proc.pid}.json',asdict(d.pins[proc.pid]));return proc
    def probe():
        nonlocal seq
        seq+=1;out=job/f'GPU-{seq:04d}.json'
        child=spawn([p['reference_plan']['va_python'],'-B',str(EXP/'wam_quality_diagnostic_v1/resource_probe.py'),'--out',str(out)],job/f'probe-{seq:04d}.log',dict(env0,CUDA_VISIBLE_DEVICES=''))
        stop=min(time.monotonic()+12,end-30)
        while child.poll() is None and time.monotonic()<stop:time.sleep(.05)
        require(child.poll() is not None and child.returncode==0 and out.exists(),'metadata probe failed or timeout')
        return read(out)
    def guard(stage,model_active=False):
        nonlocal last_probe,last_size
        require(time.monotonic()<work,'registered work deadline')
        require(not (job/'STOP_REQUESTED').exists(),'STOP_REQUESTED')
        if time.monotonic()-last_probe>=30:
            require(shutil.disk_usage(job).free>=p['disk_floor_bytes'],'disk floor')
            last_size=sum(f.stat().st_size for f in job.rglob('*') if f.is_file());require(last_size<=p['storage_cap_bytes'],'storage cap')
            sample=probe();last_probe=time.monotonic()
            if model_active:require(sample['memory_free_bytes']>=p['runtime_free_GiB']*2**30,'runtime GPU memory floor')
            write(job/'status.json',dict(status='RUNNING_COMPLETE_EPISODES',stage=stage,current=current,
                completed=len(completed),planned=len(p['cases']),completed_keys=completed,total_target_forwards=total,
                elapsed_sec=time.monotonic()-born,output_bytes=last_size,performance_claim=False),replace=True)
    failure=None
    try:
        for arm in p['arms']:
            current=None;guard('arm_admission')
            require(probe()['memory_free_bytes']>=p['initial_free_GiB']*2**30,'insufficient model headroom')
            arm_budget=p['job_target_forward_cap']-total;require(arm_budget>0,'target budget exhausted')
            root=job/'arms'/arm;root.mkdir(parents=True,exist_ok=False)
            env=dict(env0,TF_EFFECT_ARM=arm,TF_EFFECT_ARM_BUDGET=str(arm_budget),
                WAM_P0_ACTION_DRAFT_MODE='off' if arm=='T' else p['mode'],
                WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=p['selected']['TF_CONT' if arm=='T' else arm]['checkpoint'],
                MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
            if arm in p['gpu_cal_jobs']:
                cal_job=Path(p['gpu_cal_jobs'][arm]);cal_job.parent.mkdir(parents=True,exist_ok=True)
                cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=20','120',p['reference_plan']['va_python'],'-B',
                    str(EXP/'tf_effect_train_v1/gpu_actual_cal.py'),'--prepare','--job',str(cal_job),
                    '--training-dir',p['selected'][arm]['training_dir'],'--condition',arm,'--wall-cap-seconds','900']
                prep=spawn(cmd,root/'cal-prepare.log',dict(env,CUDA_VISIBLE_DEVICES=''),p['server_affinity'])
                while prep.poll() is None:guard('new_CAL_CPU_prepare');time.sleep(.2)
                require(prep.returncode==0 and (cal_job/'PREPARED.json').exists(),'new CAL preparation failed')
            server=client=None
            try:
                server=spawn([p['reference_plan']['va_python'],'-B',str(OWN/'server_entry.py'),
                    '--config-name','libero','--port',str(p['port']),'--save_root',str(root/'server_output')],root/'server.log',env,p['server_affinity'])
                limit=min(time.monotonic()+p['model_ready_wall_sec'],work)
                while True:
                    guard('model_setup_'+arm,True);require(server.poll() is None,'server exited before ready')
                    require(time.monotonic()<limit,'model ready deadline')
                    if (job/f'{arm}-READY.json').exists():
                        ready=read(job/f'{arm}-READY.json')
                        require(ready['status']=='READY' and ready['setup_RNG_restored'] and ready['target_RNG_captured_before_draft_and_CAL'],'missing setup RNG proof')
                        if arm!='T':
                            require(ready['O1_O2_O3'] is True and ready['predictor_checkpoint']==p['selected'][arm]['checkpoint'],'wrong production model/O3')
                            ref=job/'T-READY.json' if (job/'T-READY.json').exists() else Path(p['reference_plan']['reference_target_READY'])
                            require(ready['target_initial_rng']==read(ref)['target_initial_rng'],'target initialization RNG differs')
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{p["port"]}/healthz',timeout=1) as r:
                                if r.status==200:break
                        except (OSError,TimeoutError):pass
                    time.sleep(.5)
                for row in [r for r in p['cases'] if r['arm']==arm]:
                    current=row['key'];out=job/'episodes'/current;out.mkdir(parents=True,exist_ok=False)
                    client_env=dict(env,TF_EFFECT_CASE=current)
                    cmd=[p['reference_plan']['client_python'],'-B',str(OWN/'client_entry.py'),'--port',str(p['port']),'--master-port',str(p['master_port']),
                        '--benchmarks','libero_10','--task-indices',str(row['task']),'--episodes-per-task','1',
                        '--episode-start',str(row['init']),'--condition',current,'--campaign-id',job.name,'--worker-id',current,
                        '--trace-mode','off','--latency-benchmark','--out-dir',str(out/'client'),'--fail-fast','--no-resume',
                        '--worktree',str(ROOT/'投机采样/wam-p0-worktree')]
                    client=spawn(cmd,out/'client.log',client_env,p['client_affinity']);limit=min(time.monotonic()+p['per_episode_wall_sec'],work)
                    while client.poll() is None:
                        guard('complete_LIBERO_episode',True)
                        require(server.poll() is None,'target server exited during episode')
                        require(not (job/f'{arm}-MONITOR-FAILED.json').exists(),'server monitor failed')
                        require(time.monotonic()<limit,'registered complete episode timeout')
                        time.sleep(.5)
                    require(client.returncode==0,'complete LIBERO client nonzero; see client.log')
                    (out/'FINISH_REQUESTED').write_text('Complete full task; drain and account\n')
                    limit=min(time.monotonic()+180,work)
                    while not (out/'FINISH_READY.json').exists():
                        guard('episode_full_accounting',True)
                        require(server.poll() is None and not (job/f'{arm}-MONITOR-FAILED.json').exists(),'server monitor failed after episode')
                        require(time.monotonic()<limit,'full episode close deadline');time.sleep(.2)
                    closed=cleanup([client],d.pins,deadline=min(time.monotonic()+15,end),wait_seconds=.5)
                    write(out/'CLIENT-CLEANUP.json',closed);require(closed['all_roots_reaped'],'client cleanup failed');client=None
                    count=read(out/'COUNTER.json')['target_forwards']
                    require(isinstance(count,int) and 0<count<=p['per_episode_target_forward_cap'],'episode target cap')
                    total+=count;require(total<=p['job_target_forward_cap'],'full target cap')
                    completed.append(current);last_probe=0;guard('complete_episode_terminal',True)
            finally:
                roots=[x for x in (client,server) if x is not None]
                if roots:
                    closed=cleanup(roots,d.pins,deadline=min(time.monotonic()+30,end),wait_seconds=.5)
                    write(root/'OWNED-CLEANUP.json',closed);require(closed['all_roots_reaped'],'arm cleanup failed')
        expected=[r['key'] for a in p['arms'] for r in p['cases'] if r['arm']==a]
        require(completed==expected,'not all registered episodes complete')
        write(job/'COLLECTION.json',dict(status='COMPLETE_REGISTERED_FULL_EPISODES',full_episodes=len(completed),
            completed=completed,total_target_forwards=total,collect=p['collect'],elapsed_sec=time.monotonic()-born,
            performance_claim=False,permissions=p['permissions']))
    except BaseException as exc:failure=dict(error=repr(exc),traceback=traceback.format_exc(),current=current,completed=completed)
    finally:
        closed=cleanup(d.roots,d.pins,deadline=min(time.monotonic()+30,end),wait_seconds=.5)
        write(job/'FINAL-OWNED-CLEANUP.json',closed);require(closed['all_roots_reaped'],'owned cleanup incomplete')
    if failure:write(job/'FAILED.json',dict(status='FAILED_OR_PARTIAL_COMPLETE_EPISODE_CAMPAIGN',**failure))
    write(job/'SUPERVISOR-RESULT.json',dict(status='FAILED_OR_PARTIAL' if failure else 'COMPLETE',failure=failure,
        completed=completed,full_episodes=len(completed),elapsed_sec=time.monotonic()-born,total_target_forwards=total,
        observed_pins=[asdict(x) for x in d.pins.values()],cleanup=closed,permissions=p['permissions'],performance_claim=False))
    if failure:raise RuntimeError(failure['error'])

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--prepare',action='store_true');a.add_argument('--collect',action='store_true');a.add_argument('--audit',type=Path);a.add_argument('--training',type=Path);a.add_argument('--trigger',type=Path);args=a.parse_args()
    if args.prepare:
        p=prepare(args.job,collect=args.collect,audit=args.audit,training=args.training,trigger=args.trigger);print(dict(status='REGISTERED',job=p['job'],full_episodes=len(p['cases'])));return
    job=args.job.resolve();p=load(job);require(not (job/'RUN-STARTED.json').exists(),'one attempt; no overwrite')
    locks=[]
    try:
        for port in (p['port'],p['master_port']):
            with socket.socket() as s:s.bind(('127.0.0.1',port))
        for name in ('.speculative_minimal_gpu.lock','.shared-prefix-GPU.lock','.target-trace-draft-dev-v1.lock'):
            f=(RT/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit('owned stop')))
        execute(job,p)
    except BaseException as exc:
        # Admission can fail before execute's workload try/finally. Preserve a
        # concrete terminal receipt without inventing successful child cleanup.
        failure=dict(status='FAILED_ADMISSION_OR_UNCAUGHT_CAMPAIGN',error=repr(exc),traceback=traceback.format_exc(),
            permissions=p['permissions'],performance_claim=False)
        if not (job/'FAILED.json').exists():write(job/'FAILED.json',failure)
        no_child_attempt=not (job/'SPAWN-ATTEMPTED.json').exists()
        if no_child_attempt and not (job/'FINAL-OWNED-CLEANUP.json').exists():
            write(job/'FINAL-OWNED-CLEANUP.json',dict(all_roots_reaped=True,no_child_launch_attempted=True,scope='admission before first owned child launch'))
        if not (job/'SUPERVISOR-RESULT.json').exists():
            write(job/'SUPERVISOR-RESULT.json',dict(status='FAILED_OR_PARTIAL',failure=failure,
                completed=[] if no_child_attempt else None,full_episodes=0 if no_child_attempt else None,
                cleanup=read(job/'FINAL-OWNED-CLEANUP.json') if (job/'FINAL-OWNED-CLEANUP.json').exists() else 'UNKNOWN_SEE_INDEPENDENT_CLOCK',
                permissions=p['permissions'],performance_claim=False))
        raise
    finally:
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
if __name__=='__main__':main()

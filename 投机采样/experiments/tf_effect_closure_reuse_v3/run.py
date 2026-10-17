"""CPU supervisor, one GPU worker, exact ownership cleanup, no auto retry."""
import argparse,fcntl,os,signal,sys,time,traceback,shutil
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
OWN=Path(__file__).resolve().parent;sys.path.insert(0,str(OWN.parent))
from tf_effect_closure_v1.protocol import EXP,RT,load,read,write,require,environment,prepare_qualification,sha

def execute(job,p):
    from tf_usable_quality_v1.legacy import launcher
    Base,cleanup=launcher()
    class Driver:
        def __init__(self):self.roots=[];self.pins={}
    d=Driver();begin=time.monotonic();born=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19])/os.sysconf('SC_CLK_TCK')
    end=min(born+p['max_run_wall_sec'],begin+(datetime.fromisoformat(p['absolute_cleanup_deadline'])-datetime.now(timezone.utc)).total_seconds())
    work=end-p['cleanup_reserve_sec'];require(work>begin,'job expired')
    write(job/'RUN-STARTED.json',dict(status='EXPLICIT_RUN',pid=os.getpid(),max_run_wall_sec=p['max_run_wall_sec'],
        born_monotonic=born,work_monotonic=work,hard_monotonic=end,performance_claim=False))
    guard_end=min(work,time.monotonic()+60)
    while not (job/'CLOCK-GUARD.json').exists():
        require(time.monotonic()<guard_end,'independent guard not attached');time.sleep(.2)
    g=read(job/'CLOCK-GUARD.json');require(g['supervisor_pin']['pid']==os.getpid() and g['clock']['plan_sha256']==sha(job/'plan.json'),'guard bound to wrong job')
    env=environment(p);seq=0
    def spawn(cmd,out,e):
        proc=Base.spawn(d,cmd,out,e);write(job/f'OWNED-{proc.pid}.json',asdict(d.pins[proc.pid]));return proc
    def probe():
        nonlocal seq
        seq+=1;out=job/f'GPU-{seq:04d}.json'
        proc=spawn([p['reference_plan']['va_python'],'-B',str(EXP/'wam_quality_diagnostic_v1/resource_probe.py'),'--out',str(out)],job/f'probe-{seq:04d}.log',dict(env,CUDA_VISIBLE_DEVICES=''))
        until=min(time.monotonic()+12,end-30)
        while proc.poll() is None and time.monotonic()<until:time.sleep(.05)
        require(proc.poll() is not None and proc.returncode==0 and out.exists(),'metadata probe failed/timed out')
        return read(out)
    failure=None
    try:
        require(probe()['memory_free_bytes']>=p['initial_free_GiB']*2**30,'22GiB headroom unavailable')
        require(shutil.disk_usage(job).free>=p['disk_floor_bytes'],'disk floor')
        require(p['kind']=='qualification','other phases use separately reviewed campaign driver')
        worker=spawn([p['reference_plan']['va_python'],'-B',str(OWN/'replay_worker.py'),'--job',str(job)],job/'worker.log',env)
        next_probe=time.monotonic()+45
        while worker.poll() is None:
            require(time.monotonic()<work,'work deadline reached')
            require(not (job/'STOP_REQUESTED').exists(),'STOP_REQUESTED')
            require(shutil.disk_usage(job).free>=p['disk_floor_bytes'],'disk floor')
            require(sum(f.stat().st_size for f in job.rglob('*') if f.is_file())<=p['storage_cap_bytes'],'storage cap')
            if time.monotonic()>=next_probe:
                require(probe()['memory_free_bytes']>=.5*2**30,'runtime GPU memory floor');next_probe=time.monotonic()+45
            time.sleep(.5)
        require(worker.returncode==0 and (job/'WORKER-RESULT.json').exists() and not (job/'WORKER-FAILED.json').exists(),'worker failed; see evidence')
    except BaseException as exc:failure=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        closed=cleanup(d.roots,d.pins,deadline=min(time.monotonic()+30,end),wait_seconds=.5)
        write(job/'FINAL-OWNED-CLEANUP.json',closed)
        require(closed['all_roots_reaped'],'own workers remain')
    if failure:write(job/'FAILED.json',dict(status='FAILED_OR_PARTIAL',**failure))
    write(job/'SUPERVISOR-RESULT.json',dict(status='FAILED_OR_PARTIAL' if failure else 'COMPLETE',failure=failure,
        elapsed_sec=time.monotonic()-born,cleanup=closed,observed_pins=[asdict(x) for x in d.pins.values()],
        output_bytes=sum(f.stat().st_size for f in job.rglob('*') if f.is_file()),permissions=p['permissions'],performance_claim=False))
    if failure:raise RuntimeError(failure['error'])

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--prepare',action='store_true');args=a.parse_args()
    if args.prepare:
        p=prepare_qualification(args.job);print(dict(status='REGISTERED',job=p['job']));return
    job=args.job.resolve();p=load(job);require(not (job/'RUN-STARTED.json').exists(),'one attempt only')
    locks=[]
    try:
        for name in ('.speculative_minimal_gpu.lock','.shared-prefix-GPU.lock'):
            f=(RT/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit('owned stop')))
        execute(job,p)
    finally:
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
if __name__=='__main__':main()

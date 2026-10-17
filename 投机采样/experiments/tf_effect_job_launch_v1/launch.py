"""Launch exactly one registered full campaign and attach its CPU clock."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'投机采样/experiments';sys.path.insert(0,str(EXP))
from tf_effect_campaign_v1.protocol import load,require,write

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);args=a.parse_args()
    require(os.getenv('CUDA_VISIBLE_DEVICES')=='','launcher must not allocate CUDA')
    job=args.job.resolve();p=load(job)
    require(not any((job/n).exists() for n in ('RUN-STARTED.json','ROOT-LAUNCH.json','FAILED.json','COLLECTION.json')),'fresh registered job only; never rerun')
    own=EXP/'tf_effect_campaign_v1'
    s=importlib.util.spec_from_file_location('_tf_effect_launcher_ownership',EXP/'single_worker_chc_gpu_v4/ownership.py')
    module=importlib.util.module_from_spec(s);sys.modules[s.name]=module;s.loader.exec_module(module)
    root=clock=None;pin=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        os.sched_setaffinity(0,set(p['CPU_affinity']))
        cmd=[p['reference_plan']['va_python'],'-B',str(own/'run.py'),'--job',str(job)]
        with (job/'supervisor.log').open('x') as log:
            root=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        until=time.monotonic()+40
        while not (job/'RUN-STARTED.json').exists() and root.poll() is None and time.monotonic()<until:time.sleep(.1)
        require(root.poll() is None and (job/'RUN-STARTED.json').exists(),'supervisor failed before clock admission; inspect log')
        observed=module.read_process(root.pid);require(observed.argv==tuple(cmd) and observed.pgid==root.pid and observed.uid==os.getuid(),'exact spawned root required')
        pin=asdict(observed);pin['argv']=list(pin['argv'])
        write(job/'ROOT-LAUNCH.json',dict(status='EXPLICIT_DETACHED_OWNED_LAUNCH',pin=pin,command=cmd,start_new_session=True,
            launcher_source=str(Path(__file__).resolve()),launcher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        os.sched_setaffinity(0,{0,1})
        cmd=[sys.executable,'-B',str(own/'clock.py'),'--job',str(job)]
        with (job/'clock.log').open('x') as log:
            clock=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        until=time.monotonic()+15
        while not (job/'CLOCK-GUARD.json').exists() and clock.poll() is None and time.monotonic()<until:time.sleep(.1)
        require((job/'CLOCK-GUARD.json').exists() and clock.poll() is None,'clock failed to attach')
        cp=asdict(module.read_process(clock.pid));cp['argv']=list(cp['argv']);write(job/'CLOCK-LAUNCH.json',dict(pin=cp))
        print(json.dumps(dict(status='RUNNING_OWNED_CAMPAIGN_WITH_CLOCK',supervisor=root.pid,clock=clock.pid,job=str(job)),ensure_ascii=False))
    except BaseException as exc:
        if root is not None and root.poll() is None and pin is not None:
            now=module.read_process(root.pid)
            if all(getattr(now,k)==pin[k] for k in ('pid','uid','start_ticks','pgid')) and list(now.argv)==pin['argv']:
                os.kill(root.pid,signal.SIGTERM)
                try:root.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
        if not (job/'LAUNCH-FAILED.json').exists():write(job/'LAUNCH-FAILED.json',dict(status='FAILED_EXPLICIT_LAUNCH',error=repr(exc),no_foreign_signals=True))
        raise
if __name__=='__main__':main()

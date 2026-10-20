"""M38 bounded parent; all child processes have owned identity and cleanup."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

BASE=Path(__file__).resolve().parent
WORK=Path('/root/cxz/ops/persistent-tile-20261019/workspace')
OPS=WORK/'算子研究'
UUID='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'
def write(name,obj):
    p=BASE/(name+'.pending');p.write_text(json.dumps(obj,indent=2)+'\n');p.replace(BASE/name)
def identity(pid):
    return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),
                pgid=os.getpgid(pid),uid=Path(f'/proc/{pid}').stat().st_uid)
def matches(x):
    try:return identity(x['pid'])==x
    except (FileNotFoundError,ProcessLookupError):return False
def contexts():
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    return [r.split(',')[1].strip() for r in s.splitlines() if r.split(',')[0].strip()==UUID]
def main():
    os.sched_setaffinity(0,{4,5});assert not (BASE/'REGISTRATION.json').exists(),'namespace reuse prohibited'
    start=time.monotonic();now=datetime.now(timezone.utc)
    reg=dict(goal='explain/profile current single-layer kernel; six real action signatures and one static-queue counterfactual',
        user_authorization='latest explicit request to prioritize explanation and profiling of slow operator',
        T0=now.isoformat(),work_stop=(now+timedelta(seconds=2400)).isoformat(),
        cleanup_deadline=(now+timedelta(seconds=2700)).isoformat(),GPU_uuid=UUID,CPU=[4,5],
        lifecycle_cap_seconds=2700,full_block_cap=240,primitive_cap=600,storage_cap_bytes=512*1024**2,
        robot_episodes=0,full30=0,training=0,no_automatic_retry=True,
        input_manifest='six existing input-{short,middle,long}-B{2,14}-action-layer0-first.pt fixtures',
        source_manifest=json.loads((BASE/'SOURCE-MANIFEST.json').read_text()),
        engineering_quality_screen=False,independent_test_complete=False,
        noninferiority_established=False,formal_performance_permission=False)
    write('REGISTRATION.json',reg)
    source=[OPS/'wam_ops/optimized/csrc/whole_layer.cu',OPS/'wam_ops/optimized/whole_layer.py',
            WORK/'投机采样/wam-p0-worktree/wan_va/modules/model.py']
    for d in (OPS/'wam_ops/baseline',OPS/'wam_ops/optimized/csrc/vendor'):
        source.extend(p for p in d.rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cuh','.h'))
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source};write('FROZEN-BEFORE.json',frozen)
    for sub in ('tmp','cache','cache/cuda','cache/torch','build'):(BASE/sub).mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(OPS),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',CUDA_VISIBLE_DEVICES=UUID,
        TMPDIR=str(BASE/'tmp'),XDG_CACHE_HOME=str(BASE/'cache'),CUDA_CACHE_PATH=str(BASE/'cache/cuda'),
        TORCH_EXTENSIONS_DIR=str(BASE/'cache/torch'),TORCH_CUDA_ARCH_LIST='8.0',
        CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4')
    state=dict(status='RUNNING',parent=identity(os.getpid()),jobs=[]);write('STATUS.json',state)
    def run(name,cmd,cap,cpu=False):
        assert datetime.now(timezone.utc)+timedelta(seconds=cap+60)<datetime.fromisoformat(reg['work_stop'])
        if not cpu:assert not contexts(),'foreign context before worker'
        t=time.monotonic()
        with (BASE/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(cmd,cwd=OPS,env=dict(env,CUDA_VISIBLE_DEVICES='' if cpu else UUID),
                stdout=log,stderr=log,start_new_session=True)
            owned=identity(child.pid);write(name+'-OWNED.json',owned);timeout=False
            try:rc=child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timeout=True
                if matches(owned):os.killpg(owned['pgid'],signal.SIGTERM)
                try:rc=child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if matches(owned):os.killpg(owned['pgid'],signal.SIGKILL)
                    rc=child.wait(timeout=20)
        rec=dict(name=name,returncode=rc,timed_out=timeout,owned=owned,root_reaped=True,
            lifecycle_seconds=time.monotonic()-t,remaining_NVML=contexts())
        write(name+'-CLEANUP.json',rec);state['jobs'].append(rec);write('STATUS.json',state)
        assert rc==0 and not timeout and not rec['remaining_NVML'],name+' did not cleanly finish'
    try:
        assert not contexts(),'GPU busy; no context started'
        py='/root/cxz/envs/va/bin/python'
        run('build-1',[py,str(BASE/'build_static.py'),'build_ext','--build-lib',str(BASE),'--build-temp',str(BASE/'build')],900,cpu=True)
        run('profile-1',[py,'-B',str(BASE/'worker.py'),'--registration',str(BASE/'REGISTRATION.json'),'--output',str(BASE/'profile-1')],900)
        result=json.loads((BASE/'profile-1/RESULT.json').read_text());state['status']=result['status']
        state['full_block_calls']=result['full_block_calls'];state['primitive_calls']=result['primitive_calls']
    except BaseException as e:
        state.update(status='STOPPED_FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
    finally:
        state.update(at_UTC=datetime.now(timezone.utc).isoformat(),lifecycle_seconds=time.monotonic()-start,
            remaining_NVML=contexts(),foreign_processes_signalled=False,
            frozen_unchanged={str(p):hashlib.sha256(p.read_bytes()).hexdigest()==frozen[str(p)] for p in source},
            storage_bytes=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()))
        write('STATUS.json',state)
if __name__=='__main__':main()

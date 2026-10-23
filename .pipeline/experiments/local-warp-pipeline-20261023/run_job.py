"""Bounded single child; original campaign ledger and GPU1 are read-only."""
from pathlib import Path
import os, sys, json, time, datetime, subprocess, signal, traceback, fcntl
ROOT=Path(__file__).resolve().parent
job=Path(sys.argv[1]).resolve(); reg=json.loads(job.read_text()); out=job.parent
global_reg=json.loads((ROOT/'JOB.json').read_text())
GPU=global_reg['gpu']; WORK=Path('/root/cxz/ops/persistent-tile-20261019/workspace')
os.sched_setaffinity(0,{4,5})
def identity(pid):
    return dict(pid=pid,birth=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
def alive(who):
    try:return identity(who['pid'])==who
    except (FileNotFoundError,ProcessLookupError):return False
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not (out/'STATUS.json').exists(), 'Do not overwrite an attempt'
start=time.time(); state=dict(status='RUNNING',start=start,registration=reg,owned_guard=identity(os.getpid()))
dump(out/'STATUS.json',state)
lock=(ROOT/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    ledger=json.loads((ROOT/'RESOURCE-LEDGER.json').read_text())
    cap=reg['cap_seconds']; gpu=reg['gpu']
    assert datetime.datetime.fromisoformat(global_reg['deadline']).timestamp()-start>cap+60
    used=ledger.get('GPU_lifecycle_seconds',0)
    assert not gpu or used+cap+60<=min(global_reg['gpu_cap_seconds'],global_reg['cumulative_cap_seconds']-global_reg['prior_cumulative_seconds']-ledger.get('CPU_compile_seconds',0))
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    assert size<global_reg['storage_cap_bytes']
    if gpu:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        state['prelaunch_apps']=apps
        assert not any(x.startswith(GPU) for x in apps.splitlines()), 'GPU0 occupied'
    for d in ['tmp','cache','build']:(out/d).mkdir(exist_ok=True)
    cuda='/root/cxz/ops/persistent-tile-20261019/cuda-12.4'
    env=dict(os.environ,CUDA_HOME=cuda,LD_LIBRARY_PATH=cuda+'/lib64:/root/cxz/envs/va/lib/python3.12/site-packages/torch/lib:'+os.environ.get('LD_LIBRARY_PATH',''),
        PYTHONPATH=str(WORK/'算子研究'),CUDA_VISIBLE_DEVICES=GPU if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.0',
        TMPDIR=str(out/'tmp'),XDG_CACHE_HOME=str(out/'cache'),CUDA_CACHE_PATH=str(out/'cache/cuda'),PYTHONDONTWRITEBYTECODE='1',
        WAM_CONTEXT_TILE_M='128',WAM_FA2_BUILD_ROOT=str(out/'build'),WAM_RUN_DIR=str(out))
    with (out/'worker.log').open('x') as log:
        child=subprocess.Popen(reg['command'],cwd=reg.get('cwd',str(WORK/'算子研究')),env=env,stdout=log,stderr=log,start_new_session=True)
    state['owned_child']=identity(child.pid);state['child_start']=time.time();dump(out/'STATUS.json',state)
    try:rc=child.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        state['timeout']=True
        if alive(state['owned_child']):os.killpg(state['owned_child']['pgid'],signal.SIGTERM)
        try:rc=child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if alive(state['owned_child']):os.killpg(state['owned_child']['pgid'],signal.SIGKILL)
            rc=child.wait(timeout=10)
    state.update(returncode=rc,child_end=time.time(),owned_exited=not alive(state['owned_child']))
    state['status']='COMPLETE' if rc==0 and not state.get('timeout') else 'FAILED_OR_TIMEOUT'
except BaseException as exc:
    state.update(status='FAILED_PREPARATION',error=repr(exc),traceback=traceback.format_exc())
finally:
    state['end']=time.time();state['lifecycle_seconds']=state['end']-start
    dump(out/'STATUS.json',state)
    ledger=json.loads((ROOT/'RESOURCE-LEDGER.json').read_text());ledger['jobs'].append(state)
    ledger['GPU_lifecycle_seconds']=sum(x['lifecycle_seconds'] for x in ledger['jobs'] if x['registration']['gpu'])
    ledger['CPU_compile_seconds']=sum(x['lifecycle_seconds'] for x in ledger['jobs'] if not x['registration']['gpu'])
    ledger['cumulative_charged_seconds']=global_reg['prior_cumulative_seconds']+ledger['GPU_lifecycle_seconds']+ledger['CPU_compile_seconds']
    dump(ROOT/'RESOURCE-LEDGER.json',ledger)
print(json.dumps({k:state[k] for k in ['status','lifecycle_seconds']},indent=2),flush=True)
sys.exit(state['status']!='COMPLETE')

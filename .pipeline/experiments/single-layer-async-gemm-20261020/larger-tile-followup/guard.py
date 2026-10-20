"""Bounded two-variant whole-block experiment; never retries a failed artifact."""
from pathlib import Path
import datetime,json,os,signal,subprocess,time,traceback,shutil
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
PY='/root/cxz/envs/va/bin/python'
reg=json.loads((B/'REGISTRATION.json').read_text());UUID=reg['GPU'];os.sched_setaffinity(0,{4,5})
deadline=min(datetime.datetime.fromisoformat(reg['start']).timestamp()+reg['new_wall_cap_seconds'],datetime.datetime.fromisoformat(reg['absolute_deadline']).timestamp())
state=dict(status='RUNNING',start=time.time(),deadline=deadline,jobs=[],variants={},GPU_lifecycle_seconds=0,full_block_calls=0)
for name in ('tmp','cache'):(B/name).mkdir(exist_ok=True)
def save():(B/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
def identity(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
def same(row):
 try:return identity(row['pid'])==row
 except (FileNotFoundError,ProcessLookupError):return False
def disk_bytes():return sum(p.stat().st_size for p in B.rglob('*') if p.is_file())
def count_calls():return sum(json.loads(p.read_text()).get('full_block_calls',0) for p in B.glob('*/*RESULT.json'))
try:
 for variant,tile in [('cooperative64',64),('context128',128)]:
  V=B/variant;V.mkdir(exist_ok=False);state['variants'][variant]='RUNNING'
  env=dict(os.environ,CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',WAM_FA2_BUILD_ROOT=str(V/'build'),TORCH_CUDA_ARCH_LIST='8.0',TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'),CUDA_CACHE_PATH=str(B/'cache/cuda'),MAX_JOBS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R.parents[1]),GPU_PRELAUNCH_EMPTY='1',WAM_RUN_DIR=str(V),WAM_CONTEXT_TILE_M=str(tile))
  jobs=[('build',[PY,str(R/'build_whole_layer.py'),'build_ext','--build-lib',str(R),'--build-temp',str(V/'build-temp')],600,False,0),('gate',[PY,'-B',str(B/'gate.py')],240,True,72),('timing-1',[PY,'-B',str(B/'timing.py'),'1'],180,True,57),('timing-2',[PY,'-B',str(B/'timing.py'),'2'],180,True,57)]
  for name,cmd,cap,gpu,calls in jobs:
   if name.startswith('timing') and not json.loads((V/'CUTLASS-GATE-RESULT.json').read_text()).get('all_pass'):raise RuntimeError('Numerical gate failed; timing prohibited')
   state['full_block_calls']=count_calls();state['storage_bytes']=disk_bytes();save()
   if state['full_block_calls']+calls>reg['new_full_block_cap']:raise RuntimeError('Block-call cap')
   if state['storage_bytes']>reg['storage_cap_bytes']:raise RuntimeError('Storage cap')
   cap=min(cap,int(deadline-time.time()-60))
   if gpu:
    cap=min(cap,int(reg['new_GPU_lifecycle_cap_seconds']-state['GPU_lifecycle_seconds']-30))
    contexts=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    if any(x.startswith(UUID) for x in contexts.splitlines()):raise RuntimeError('GPU0 occupied, stopped before launch')
   if cap<=0:raise TimeoutError('Insufficient lifecycle budget')
   with (V/(name+'.log')).open('x') as log:worker=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=log,env=dict(env,CUDA_VISIBLE_DEVICES=UUID if gpu else ''),start_new_session=True)
   row=dict(variant=variant,name=name,owned=identity(worker.pid),start=time.time(),gpu=gpu,cap_seconds=cap);state['jobs'].append(row);save()
   try:rc=worker.wait(timeout=cap)
   except subprocess.TimeoutExpired:
    row['timeout']=True
    if same(row['owned']):os.killpg(row['owned']['pgid'],signal.SIGTERM)
    try:rc=worker.wait(timeout=15)
    except subprocess.TimeoutExpired:
     if same(row['owned']):os.killpg(row['owned']['pgid'],signal.SIGKILL)
     rc=worker.wait(timeout=15)
   row.update(returncode=rc,end=time.time(),lifecycle_seconds=time.time()-row['start'],root_reaped=True,root_identity_alive=same(row['owned']))
   if gpu:state['GPU_lifecycle_seconds']+=row['lifecycle_seconds']
   state['full_block_calls']=count_calls();save()
   if rc!=0 or row.get('timeout'):raise RuntimeError(variant+'/'+name+' failed; no automatic retry')
   if name=='gate' and not json.loads((V/'CUTLASS-GATE-RESULT.json').read_text()).get('all_pass'):raise RuntimeError(variant+' finite gate failed')
   if name.startswith('timing') and json.loads((V/f'cutlass-{name}-RESULT.json').read_text())['status'].startswith('FAILED'):raise RuntimeError(variant+'/'+name+' reported failure')
  shutil.copy2(next(R.glob('wam_whole_fa2_cuda*.so')),V/'selected-build.so')
  state['variants'][variant]='COMPLETE';save()
 state['status']='COMPLETE'
except BaseException as e:state.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:
 state.update(end=time.time(),lifecycle_seconds=time.time()-state['start'],full_block_calls=count_calls(),storage_bytes=disk_bytes());save()

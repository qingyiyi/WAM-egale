"""Run finite measurement phases; cleanup only PID+birth-matched children."""
from pathlib import Path
import os,json,time,datetime,subprocess,signal,traceback
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
PY='/root/cxz/envs/va/bin/python'
GPU='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'
os.sched_setaffinity(0,{4,5})
reg=json.loads((B/'JOB.json').read_text())
assert reg['execution_authorization']=='USER_APPROVED_CROSS_KV_PRECOMPUTE_AND_STAGE_TIMING', 'New window must be explicitly approved'
deadline=datetime.datetime.fromisoformat(reg['deadline']).timestamp()
assert reg['cumulative_prior_GPU_seconds'] >= 7699.251076945541
start=time.time();state=dict(status='RUNNING',start=start,deadline=deadline,jobs=[])
for n in ['cache','tmp','build','generated']:(B/n).mkdir(exist_ok=True)
env=dict(os.environ,LD_LIBRARY_PATH='/root/cxz/ops/persistent-tile-20261019/cuda-12.4/lib64:/root/cxz/envs/va/lib/python3.12/site-packages/torch/lib:'+os.environ.get('LD_LIBRARY_PATH',''),CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',
 PYTHONPATH=str(R.parents[1]),TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
 TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'),CUDA_CACHE_PATH=str(B/'cache/cuda'),PYTHONDONTWRITEBYTECODE='1',
 WAM_CONTEXT_TILE_M='128',WAM_FA2_BUILD_ROOT=str(B/'active-build'),WAM_RUN_DIR=str(B),GPU_PRELAUNCH_EMPTY='1')
def ident(pid):
 f=Path(f'/proc/{pid}/stat');return dict(pid=pid,birth_ticks=int(f.read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
def alive(x):
 try:return ident(x['pid'])==x
 except (FileNotFoundError,ProcessLookupError):return False
def save():(B/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
try:
 assert not (B/'STATUS.json').exists(),'Existing attempt cannot restart'
 state['status']='RUNNING';state['owned_guard']=ident(os.getpid());save()
 for name,cmd,cap,gpu in reg['steps']:
  size=sum(p.stat().st_size for p in B.rglob('*') if p.is_file())
  if size>reg['new_storage_cap_bytes']:raise RuntimeError('Storage cap exceeded')
  remaining=deadline-time.time()-300
  if remaining<cap:raise TimeoutError('Insufficient registered lifecycle reserve')
  if gpu:
   used=sum(x.get('lifecycle_seconds',0) for x in state['jobs'] if x['gpu'])
   if used+cap>reg['GPU_lifecycle_cap_seconds'] or reg['cumulative_prior_GPU_seconds']+used+cap>reg['cumulative_cap_seconds']:raise TimeoutError('GPU total lifecycle reserve exhausted')
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   if any(x.startswith(GPU) for x in apps.splitlines()):raise RuntimeError('GPU0 occupied')
  with (B/f'{name}.log').open('x') as log:
   p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES=GPU if gpu else ''),stdout=log,stderr=log,start_new_session=True)
  row=dict(name=name,owned=ident(p.pid),start=time.time(),gpu=gpu);state['jobs'].append(row);save()
  try:rc=p.wait(timeout=cap)
  except subprocess.TimeoutExpired:
   row['timeout']=True
   if alive(row['owned']):os.killpg(row['owned']['pgid'],signal.SIGTERM)
   try:rc=p.wait(timeout=15)
   except subprocess.TimeoutExpired:
    if alive(row['owned']):os.killpg(row['owned']['pgid'],signal.SIGKILL)
    rc=p.wait(timeout=15)
  row.update(returncode=rc,end=time.time(),lifecycle_seconds=time.time()-row['start'],root_reaped=True,identity_alive=alive(row['owned']));save()
  if rc or row.get('timeout'):raise RuntimeError(name+' failed')
 state['status']='COMPLETE_FINITE_PRECOMPUTE_VALIDATION'
except BaseException as e:
 state.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:
 state.update(end=time.time(),lifecycle_seconds=time.time()-start);save()

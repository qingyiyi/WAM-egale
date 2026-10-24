"""Wait for fixed training; run one bounded eight-policy numerical job on GPU1."""
from pathlib import Path
import os,json,datetime,time,subprocess,fcntl,signal,traceback
B=Path(__file__).resolve().parent;R=B.parent;os.sched_setaffinity(0,{18})
GPU='GPU-4acf84ff-31ce-dead-c2a4-2a122b123b8c';deadline=datetime.datetime.fromisoformat('2026-10-21T09:30:00+08:00').timestamp()
state=dict(status='WAITING_PARENT_AND_Q32_BUILD',start=time.time(),GPU_lifecycle_seconds=0.)
def save():
 p=B/'INTERFACE-GUARD-r3.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
def identity(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
def same(x):
 try:return identity(x['pid'])==x
 except (FileNotFoundError,ProcessLookupError):return False
save();gpu_start=None;p=None
try:
 while True:
  s=json.loads((B/'STATUS.json').read_text())
  if s['status']=='COMPLETE_PARENT_AND_CAL_WAITING_INTERFACE_REVIEW':break
  if s['status'].startswith('FAILED'):raise RuntimeError('Parent prerequisite failed')
  if time.time()+3600+900>deadline:raise TimeoutError('No full interface reserve')
  time.sleep(30)
 assert json.loads((B/'Q32-BUILD-STATUS.json').read_text())['status']=='BUILT_Q32_NOT_VALIDATED'
 lease=(R/'GPU1-LEASE.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ledger=json.loads((R/'GPU-LEDGER.json').read_text());assert ledger['consumed_seconds']+3600<=ledger['cap_seconds']
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert not any(x.startswith(GPU) for x in apps.splitlines())
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
  TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'),CUDA_CACHE_PATH=str(B/'cache/cuda'),HF_HOME='/root/cxz/robotwin/cache/hf',HF_HUB_OFFLINE='1',
  MASTER_ADDR='127.0.0.1',MASTER_PORT='34981',WORLD_SIZE='1',RANK='0',LOCAL_RANK='0')
 gpu_start=time.time()
 with (B/'interface-gate-r3.log').open('x') as f:p=subprocess.Popen(['/root/cxz/envs/va/bin/python',str(B/'interface_gate_r3.py')],cwd=B,env=env,stdout=f,stderr=f,start_new_session=True)
 state.update(status='RUNNING_EIGHT_COMPLETE_POLICY_INTERFACE',GPU_start=gpu_start,owned=identity(p.pid));save()
 try:rc=p.wait(timeout=2700)
 except subprocess.TimeoutExpired:
  state['timeout']=True
  if same(state['owned']):os.killpg(state['owned']['pgid'],signal.SIGTERM)
  try:rc=p.wait(timeout=15)
  except subprocess.TimeoutExpired:
   if same(state['owned']):os.killpg(state['owned']['pgid'],signal.SIGKILL)
   rc=p.wait(timeout=15)
 state.update(returncode=rc,root_reaped=True,identity_alive=same(state['owned']))
 result=json.loads((B/'interface-gate-r3/RESULT.json').read_text());state['result']=result['status'];state['teacher_attempted']=result['teacher_attempted']
 assert rc==0 and result['status']=='PASS_FINITE_EIGHT_COMPLETE_POLICY_CALLS_NOT_ROBOT_QUALITY'
 state['status']='COMPLETE_INTERFACE_GATE'
except BaseException as e:state.update(status='FAILED_OR_BLOCKED_INTERFACE',error=repr(e),traceback=traceback.format_exc())
finally:
 state['end']=time.time();state['wall_seconds']=time.time()-state['start']
 if gpu_start:
  state['GPU_lifecycle_seconds']=time.time()-gpu_start
  with (R/'GPU-LEDGER.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads((R/'GPU-LEDGER.json').read_text());ledger['consumed_seconds']+=state['GPU_lifecycle_seconds'];ledger['jobs'].append(dict(namespace=str(B/'interface-gate-r3'),**state))
   tmp=R/'GPU-LEDGER.json.tmp';tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(R/'GPU-LEDGER.json')
 save()

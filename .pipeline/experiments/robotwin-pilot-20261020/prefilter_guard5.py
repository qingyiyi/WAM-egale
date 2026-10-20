"""Wait for this run's dependency terminal, then launch one bounded real expert job."""
from pathlib import Path
from datetime import datetime
import json,os,time,subprocess,signal,traceback
R=Path('/root/cxz/robotwin');J=R/'runs/prefilter-control-5';J.mkdir(exist_ok=False);os.sched_setaffinity(0,{18})
auth=json.loads((R/'runs/AUTHORIZATION.json').read_text());cpu_deadline=datetime.fromisoformat(auth['CPU_preparation_deadline']).timestamp();uuid=auth['GPU_uuid'];state=dict(status='WAITING_OWNED_DEPENDENCY',start=time.time(),GPU_lifecycle_seconds=0,children=[])
def save():(J/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
def identity(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
def same(rec):
 try:return identity(rec['pid'])==rec
 except (FileNotFoundError,ProcessLookupError):return False
try:
 while True:
  if time.time()>cpu_deadline-900:raise TimeoutError('original CPU preparation deadline; no new GPU job')
  p=R/'runs/oidn-compatibility-2/STATUS.json'
  if p.exists():
   d=json.loads(p.read_text())
   if d['status']=='PRIVATE_OIDN_COMPATIBILITY_READY':break
   if d['status'].startswith('FAILED'):raise RuntimeError('dependency failed; no automatic retry: '+d.get('error',''))
  save();time.sleep(5)
 env={**os.environ,**json.loads((R/'runs/CLIENT-ENVIRONMENT.json').read_text()),'CUDA_VISIBLE_DEVICES':uuid,'PYTHONPATH':str(R/'runs'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','CUDA_CACHE_PATH':str(R/'cache/cuda'),'WARP_CACHE_PATH':str(R/'cache/warp')}
 contexts=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 if any(line.startswith(uuid) for line in contexts.splitlines()):raise RuntimeError('GPU1 occupied; not starting over foreign context')
 gpu_start=time.time();reservation=dict(goal='six fixed tasks, 19 expert-valid scenes per task; no policy or teacher',GPU=uuid,start=gpu_start,deadline=gpu_start+3600,cleanup_deadline=gpu_start+3900,worker_cap=3600,teacher_cap=0,policy_episodes=0,expert_attempt_cap=1200,source='pinned official client main/config and experts; wrapper records roles/physical/RNG',expected_output=str(R/'runs/prefilter-2/STATUS.json'),parent_GPU_lifecycle_cap=86400,storage_cap=2*2**30)
 (J/'RESERVATION.json').write_text(json.dumps(reservation,indent=2))
 with (J/'worker.log').open('x') as out:
  proc=subprocess.Popen([str(R/'envs/client/bin/python'),'-B',str(R/'runs/prefilter.py'),str(R/'runs/prefilter-2')],cwd=R,env=env,stdout=out,stderr=out,start_new_session=True)
  owned=identity(proc.pid);state.update(status='RUNNING_GPU_EXPERT_PREPARATION',root_owned=owned,GPU_start=gpu_start);save();seen={proc.pid:owned}
  while proc.poll() is None and time.time()<reservation['deadline']:
   log=(J/'worker.log').read_text(errors='replace')[-16000:]
   if 'OIDN Error:' in log and 'illegal memory access' in log:
    terminal=R/'runs/prefilter-2/STATUS.json'
    if terminal.exists():
     d=json.loads(terminal.read_text());d.update(status='STOPPED_INFRA_OIDN_ILLEGAL_MEMORY_ACCESS',reason='new private OIDN still failed before policy/teacher');terminal.write_text(json.dumps(d,indent=2))
    state['explicit_infrastructure_stop']=True
    if same(owned):os.killpg(owned['pgid'],signal.SIGTERM)
    break
   for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
     rec=identity(int(p.name))
     if rec['pgid']==owned['pgid']:seen[rec['pid']]=rec
    except (FileNotFoundError,ProcessLookupError):pass
   state['children']=list(seen.values());save();time.sleep(2)
  if proc.poll() is None:
   state['timeout']=True
   if same(owned):os.killpg(owned['pgid'],signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:
    if same(owned):os.killpg(owned['pgid'],signal.SIGKILL)
    proc.wait(timeout=15)
  if proc.poll() is None:proc.wait(timeout=15)
  # Only captured, still matching owned processes may be cleaned.
  cleaned=[]
  for rec in seen.values():
   if rec['pid']==proc.pid:continue
   if same(rec):os.kill(rec['pid'],signal.SIGTERM);cleaned.append(rec)
  state.update(returncode=proc.returncode,root_reaped=True,cleanup_signalled=cleaned,GPU_lifecycle_seconds=time.time()-gpu_start)
 terminal=R/'runs/prefilter-2/STATUS.json';state['worker_terminal']=json.loads(terminal.read_text())['status'] if terminal.exists() else 'MISSING_TERMINAL'
 state['status']='COMPLETE_EXPERT_PREPARATION' if state['worker_terminal']=='COMPLETE_FIXED_SCENES_NO_POLICY_QUALITY_RESULT' and proc.returncode==0 else 'STOPPED_FAILED_OR_PARTIAL'
 ledger=json.loads((R/'runs/GPU-LEDGER.json').read_text());ledger['consumed_seconds']+=state['GPU_lifecycle_seconds'];ledger['jobs'].append(dict(namespace=str(J),**state));(R/'runs/GPU-LEDGER.json').write_text(json.dumps(ledger,indent=2))
except BaseException as e:state.update(status='STOPPED_DEPENDENCY_OR_RESOURCE',error=repr(e),traceback=traceback.format_exc())
finally:state.update(end=time.time(),lifecycle_seconds=time.time()-state['start']);save()

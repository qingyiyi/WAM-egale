"""Bounded real matrix supervisor, one owned GPU tree; new admission after r1 resource interruption."""
import ast,json,os,signal,socket,subprocess,sys,time,traceback,threading,queue,hashlib,pwd
from pathlib import Path
OWN=Path(__file__).resolve().parent;ROOT=OWN.parents[2]
sys.path.insert(0,str(OWN.parent/'single_worker_chc_gpu_v4'))
from resource_guard import NVML,Idle
import ownership
sys.path.append(str(OWN.parent/'speculative_e2e_v3'))
from launch_identity import pin,bootstrap_key
PRIOR=Path('/var/tmp/cxz-wam-quality-20261014/p0-megaverifier-profile-v2-20261014-marker-fix/plan.json')
def bounded(fn,seconds=5):
 q=queue.Queue()
 def work():
  try:q.put((True,fn()))
  except BaseException as e:q.put((False,e))
 threading.Thread(target=work,daemon=True).start()
 try:ok,value=q.get(timeout=seconds)
 except queue.Empty:raise TimeoutError('NVML timeout; independent wallclock cleanup remains active')
 if not ok:raise value
 return value
def lifetime(info):
 return (info.pid,info.pgid,info.start_ticks,info.uid)
def main():
 out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
 p=json.loads(PRIOR.read_text());ref=p['frozen_reference']
 provenance={};administrative_changes=[]
 for path,old_digest in p['source_sha256'].items():
  actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  if path in (str(ROOT/'.pipeline/config.yaml'),str(ROOT/'AGENTS.md')):
   administrative_changes.append(dict(path=path,old=old_digest,current=actual,reason='authorized HW upgrade; not a math/runtime dependency'))
  else:assert actual==old_digest,'frozen implementation drift: '+path
  provenance[path]=actual
 for item in p['target_weight_inventory']:
  st=Path(item['path']).stat();assert(st.st_size,st.st_mtime_ns)==(item['bytes'],item['mtime_ns'])
 for item in [p['short'],p['long']]+p['middle_prefix_artifacts']:
  assert Path(item['path']).is_file()
 for name in ('full_matrix.py','run_matrix_r2.py'):
  ast.parse((OWN/name).read_text());provenance[str(OWN/name)]=hashlib.sha256((OWN/name).read_bytes()).hexdigest()
 (out/'SOURCE-PROVENANCE.json').write_text(json.dumps(dict(files=provenance,administrative_exceptions=administrative_changes,whole_model_hash=False),indent=2)+'\n')
 (out/'SOURCE-PLAN.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
 plan=dict(scope='real frozen short/middle/long full30 verifier mechanism, not formal robot speed',GPU_seconds_including_cleanup=1800,cleanup_seconds=60,CPU_worker=7,target_forward_cap=1800,complete_chunks=3,episode_cap=0,fixture_and_output_cap_bytes=3*2**30,inputs=str(PRIOR),single_process_samples=6,retries=0,new_TF_formal_performance=False,existing_original_FAILED_unchanged=True)
 (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
 nv=None;idle=Idle();rows=[];start=time.monotonic();child=None;reason=None;gpu_start=None;identity=None;bootstrap=None;pins={};cleanup_events=[]
 def remember(info):
  nonlocal bootstrap
  bootstrap=info
  pins[info.pid]=info
 def refresh():
  for d in Path('/proc').iterdir():
   if not d.name.isdigit():continue
   try:
    info=ownership.read_process(int(d.name))
    if ownership.is_owned(info,pins):pins[info.pid]=info
   except (FileNotFoundError,PermissionError,ProcessLookupError,RuntimeError):continue
 def live():
  found=[]
  for pid,known in pins.items():
   try:
    info=ownership.read_process(pid)
    if lifetime(info)==lifetime(known):
     raw=Path('/proc',str(pid),'stat').read_text();state=raw[raw.rfind(')')+2:].split()[0]
     if state!='Z':found.append(pid)
   except (FileNotFoundError,ProcessLookupError):pass
   except RuntimeError:found.append(pid)
  return found
 def signal_owned(sig):
  refresh()
  for pid in live():
   try:
    current=ownership.read_process(pid);known=pins[pid]
    if lifetime(current)!=lifetime(known):raise RuntimeError('refuse changed/unpinned PID')
    os.kill(pid,sig);cleanup_events.append(dict(pid=pid,signal=int(sig)))
   except (FileNotFoundError,ProcessLookupError):pass
 try:
  nv=bounded(NVML)
  for _ in range(60):
   sample=bounded(nv.sample);rows.append(sample)
   if idle.add(sample,time.monotonic()):break
   time.sleep(1)
  if len(idle.samples)<15:raise RuntimeError('no clean GPU window')
  with socket.socket() as sock:sock.bind(('127.0.0.1',19391))
  env={k:v for k,v in os.environ.items() if not k.startswith(('WAM_P0_','P0_','LINGBOT_'))}
  env.update(CUDA_VISIBLE_DEVICES='0',WAM_SHAPE_OUT=str(out),OMP_NUM_THREADS='7',MKL_NUM_THREADS='7',OPENBLAS_NUM_THREADS='7',NUMEXPR_NUM_THREADS='7',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0',LINGBOT_SEED='20260922',LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',LINGBOT_ATTN_MODE='torch',LINGBOT_NSYS_PROFILE='off',LINGBOT_VA_LIBERO_SUITE='libero_10',LINGBOT_VA_LIBERO_CHECKPOINT=ref['target_checkpoint_path'],LINGBOT_VA_LIBERO_CHECKPOINT_ID=ref['target_checkpoint_id'],LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=ref['registry'],WAM_VAE_DEVICE='cuda',WAM_P0_PHASE_PROFILE='0',WAM_P0_ACTION_DRAFT_MODE='off',WAM_P0_ACTION_DRAFT_SELECTOR='prefix_then_cost',WAM_P0_ACTION_DRAFT_THRESHOLD=str(ref['threshold']),WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=p['frozen_A_off'],PREFIX_ACCEPT_ENABLED='1',PREFIX_ACCEPT_GPU_APPROVED='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='19391',RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',PYTHONPATH=':'.join(map(str,[OWN,OWN.parent/'prefix_acceptance_v1',ROOT/'投机采样/wam-p0-worktree',ROOT/'投机采样/scripts'])))
  with (out/'worker.log').open('x') as log:
   cmd=['/rjs/cxz/miniconda3/envs/va/bin/python','-B',str(OWN/'full_matrix.py'),'--config-name','libero','--port','19390','--save_root',str(out/'server')]
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT/'投机采样/wam-p0-worktree')
   gpu_start=time.monotonic()
   identity=pin(child,cmd,on_boot=remember)
   pins[identity.pid]=identity
   (out/'OWNED-PROCESS.json').write_text(json.dumps(vars(identity),indent=2)+'\n')
   while child.poll() is None:
    refresh();sample=bounded(nv.sample);sample['wall_epoch']=time.time();rows.append(sample)
    foreign=[];provenance=[]
    for item in [p for kind in ('compute','graphics') for p in sample[kind]]:
     try:
      info=ownership.read_process(item['pid']);owned=ownership.is_owned(info,pins)
      if owned:pins[info.pid]=info
      else:foreign.append(item)
      provenance.append(dict(pid=info.pid,ppid=info.ppid,uid=info.uid,user=pwd.getpwuid(info.uid).pw_name,start_ticks=info.start_ticks,executable=info.argv[0] if info.argv else None,owned=owned))
     except FileNotFoundError:
      provenance.append(dict(pid=item['pid'],exited_before_inspection=True,previously_owned=item['pid'] in pins))
      if item['pid'] not in pins:foreign.append(item)
    sample['process_provenance']=provenance
    if foreign:raise RuntimeError('foreign GPU process appeared: '+repr(foreign))
    if time.monotonic()-gpu_start>1740:raise TimeoutError('1800s GPU cap incl60s cleanup')
    if sum(x.stat().st_size for x in out.rglob('*') if x.is_file())>3*2**30:raise RuntimeError('3GiB output cap')
    time.sleep(1)
  if child.returncode:raise RuntimeError('worker exit '+str(child.returncode))
  result=json.loads((out/'MATRIX-RESULT.json').read_text())
  assert result['status']=='COMPLETE_THREE_LENGTH_FULL_VERIFIER_MECHANISM'
 except BaseException as e:reason=repr(e)
 finally:
  try:
   if child is not None and child.poll() is None:
    (out/'STOP_REQUESTED').write_text(reason or 'stop')
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
     signal_owned(signal.SIGTERM)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:signal_owned(signal.SIGKILL);child.wait(timeout=5)
   refresh()
   if live():
    signal_owned(signal.SIGTERM);limit=time.monotonic()+5
    while live() and time.monotonic()<limit:time.sleep(.2)
    if live():signal_owned(signal.SIGKILL)
    limit=time.monotonic()+5
    while live() and time.monotonic()<limit:time.sleep(.2)
   if live():raise RuntimeError('owned descendants remain alive: '+repr(live()))
  except BaseException as e:reason=(reason or '')+'; cleanup: '+repr(e)
  final=None
  if nv is not None:
   try:final=bounded(nv.sample)
   except BaseException as e:final=dict(inspection_error=repr(e));reason=reason or repr(e)
   try:bounded(nv.close)
   except BaseException as e:reason=reason or repr(e)
  result=dict(status='COMPLETE_MECHANISM_MATRIX' if reason is None else 'FAILED_OR_BLOCKED',reason=reason,elapsed_sec=time.monotonic()-start,GPU_wall_sec=0 if gpu_start is None else time.monotonic()-gpu_start,owned_pid=child.pid if child else None,own_process_reaped=child is None or child.poll() is not None,owned_tree=[vars(x) for x in pins.values()],cleanup_events=cleanup_events,all_owned_live_exited=not live(),final_resources=final,samples=rows,formal_robot_performance=False)
  (out/'SUPERVISOR-RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
if __name__=='__main__':main()

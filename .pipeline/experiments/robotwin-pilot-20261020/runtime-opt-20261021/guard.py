"""Fixed full-episode DEV queue, single leased GPU, no retries."""
from pathlib import Path
import datetime,fcntl,json,os,signal,subprocess,sys,time,traceback
os.sched_setaffinity(0,{18,19})
J=Path(__file__).resolve().parent;R=Path('/root/cxz/robotwin');B=R/'runs/tf-migration-20261021'
reg=json.loads((J/'PLAN.json').read_text());assert reg.get('execution_authorization')=='APPROVED_RUNTIME_OPTIMIZATION_20261021';auth=json.loads((R/'runs/AUTHORIZATION.json').read_text())
assert not (J/'STATUS.json').exists(),'Existing attempt cannot restart'
start=time.time();deadline=min(start+reg['execution_lifecycle_cap_seconds'],datetime.datetime.fromisoformat(reg['must_finish_before']).timestamp());reserve=reg['cleanup_reserve_seconds']
state=dict(status='PREPARING_RUNTIME_OPTIMIZATION',start=start,deadline=deadline,episodes=[],pair_audits=[],owned=[],GPU_lifecycle_seconds=0,formal_performance_permission=False,independent_test_complete=False)
workers=[];seen={};gpu_start=None

def save():
 f=J/'STATUS.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(f)
def identity(pid):
 s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,birth_ticks=int(s[19]),pgid=os.getpgid(pid))
def same(o):
 try:return identity(o['pid'])==o
 except (FileNotFoundError,ProcessLookupError):return False

def discover():
 groups={o['pgid'] for o in seen.values()}
 for f in Path('/proc').iterdir():
  if not f.name.isdigit():continue
  try:
   row=identity(int(f.name))
   if row['pgid'] in groups:seen[row['pid']]=row
  except (FileNotFoundError,ProcessLookupError):pass

def launch(cmd,env,log):
 with log.open('x') as f:q=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=f,start_new_session=True)
 workers.append(q);row=identity(q.pid);seen[q.pid]=row;state['owned']=list(seen.values());save();return q

def cleanup():
 discover()
 for row in seen.values():
  if same(row):
   try:os.kill(row['pid'],signal.SIGTERM)
   except ProcessLookupError:pass
 limit=time.time()+15
 for worker in workers:
  if worker.poll() is None:
   try:worker.wait(timeout=max(.1,limit-time.time()))
   except subprocess.TimeoutExpired:pass
 for row in seen.values():
  if same(row):
   try:os.kill(row['pid'],signal.SIGKILL)
   except ProcessLookupError:pass
 for worker in workers:worker.wait(timeout=15)

def audit_variant(case):
 name=case['episode_id'];variant=case['variant'];baseline='open_microwave-valid16-BASELINE'
 if variant=='BASELINE':return
 fields={}
 for rel in ['commands.jsonl','OBS-AUDIT.jsonl','COMMAND-STATE-AUDIT.jsonl']:
  fields[rel]=(J/name/rel).read_bytes()==(J/baseline/rel).read_bytes()
 fields['all_server_RPC']=(J/'server'/name/'ALL-RPC-AUDIT.jsonl').read_bytes()==(J/'server'/baseline/'ALL-RPC-AUDIT.jsonl').read_bytes()
 left=json.loads((J/baseline/'STATUS.json').read_text());right=json.loads((J/name/'STATUS.json').read_text())
 fields['terminal_and_steps']=all(left.get(k)==right.get(k) for k in ['status','success','environment_commands','environment_advances','policy_chunks'])
 result=dict(variant=variant,fields=fields,pass_all=all(fields.values()),scope='One exposed complete episode; all saved digests exact, no raw all-image/KV replay claim')
 state.setdefault('variant_audits',[]).append(result);(J/(variant+'-AUDIT.json')).write_text(json.dumps(result,indent=2)+'\n');save()


try:
 save();assert time.time()<datetime.datetime.fromisoformat(reg['prepare_deadline']).timestamp()
 assert json.loads((J/'CPU-READY.json').read_text())['status']=='PASS_FIXED_INPUT_AND_DEPENDENCY_REVIEW'
 assert all(c['task']=='open_microwave' for c in json.loads((J/'INPUT-MANIFEST.json').read_text()))
 lock=(R/'runs/GPU1-LEASE.lock').open('a');lock2=(R/'runs/GPU1.lease').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock2,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ledger=json.loads((R/'runs/GPU-LEDGER.json').read_text());assert ledger['consumed_seconds']+deadline-time.time()<=auth['GPU_lifecycle_cap_seconds']
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(x.startswith(reg['GPU_uuid']) for x in apps.splitlines()),'GPU1 occupied'
 queue=json.loads((J/'INPUT-MANIFEST.json').read_text());assert len(queue)==4 and all(c['valid_index']==16 and c['arm']=='T' for c in queue)
 common={**os.environ,'CUDA_VISIBLE_DEVICES':reg['GPU_uuid'],'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','XDG_CACHE_HOME':str(R/'cache'),'HF_HOME':str(R/'cache/hf'),'HF_HUB_OFFLINE':'1','TMPDIR':str(R/'tmp'),'CUDA_CACHE_PATH':str(R/'cache/cuda'),'TORCHINDUCTOR_CACHE_DIR':str(R/'cache/inductor'),'TORCH_EXTENSIONS_DIR':str(R/'cache/torch_extensions'),'MPLCONFIGDIR':str(R/'cache/matplotlib')}
 client_env={**common,**json.loads((R/'runs/CLIENT-ENVIRONMENT.json').read_text()),'CUDA_VISIBLE_DEVICES':reg['GPU_uuid'],'PYTHONPATH':str(J)+os.pathsep+str(B)+os.pathsep+str(R/'runs'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
 assert not any('/compat/' in x for x in client_env['LD_LIBRARY_PATH'].split(':'))
 model_env={**common,'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'1','MASTER_ADDR':'127.0.0.1','MASTER_PORT':'34981','PYTHONPATH':str(R/'source/lingbot-va-pinned')}
 gpu_start=time.time();state.update(status='LOADING_MODEL',GPU_start=gpu_start,prior_GPU_lifecycle_seconds=ledger['consumed_seconds']);save()
 server=launch(['/root/cxz/envs/va/bin/python','-B',str(J/'server.py'),str(J/'server')],model_env,J/'server.log')
 ready_deadline=min(time.time()+900,deadline-reg['per_episode_seconds']-reserve)
 while True:
  if server.poll() is not None:raise RuntimeError(f'Server exited before ready {server.returncode}')
  f=J/'server/STATUS.json'
  if f.exists():
   d=json.loads(f.read_text());state['server_stage']=d['status']
   if d['status']=='READY':break
   if d['status']=='FAILED':raise RuntimeError(d.get('error'))
  if time.time()>=ready_deadline:raise TimeoutError('Model setup deadline')
  save();time.sleep(5)
 for case in queue:
  if case['variant']=='COMBINED' and not (len(state.get('variant_audits',[]))==2 and all(a['pass_all'] for a in state['variant_audits'])):
   state.update(status='COMPLETE_SINGLE_VARIANTS_COMBINED_BLOCKED');break
  if (J/'STOP-AFTER-EPISODE').exists():state.update(status='PAUSED_BY_REQUEST_AFTER_COMPLETE_EPISODE');break
  if time.time()+reg['per_episode_seconds']+reserve>=deadline:state.update(status='PARTIAL_BUDGET_NO_NEXT_EPISODE');break
  name=case['episode_id'];f=J/f'{name}-CASE.json';f.write_text(json.dumps(case,indent=2)+'\n')
  state.update(status='RUNNING_COMPLETE_DEV_EPISODE',active_case=name,active_pair=case['pair_id'],active_arm=case['arm']);save()
  worker=launch([str(R/'envs/client/bin/python'),'-B',str(J/'client.py'),str(f),str(J/name)],client_env,J/f'{name}.log');end=time.time()+reg['per_episode_seconds']
  while worker.poll() is None:
   discover()
   if server.poll() is not None:raise RuntimeError('Server exited during full episode')
   if time.time()>=end:raise TimeoutError('Full episode timeout; no retry')
   sd=json.loads((J/'server/STATUS.json').read_text())
   if sd['status']=='FAILED':raise RuntimeError('Server failure: '+sd.get('error',''))
   if sd['teacher_attempted']>reg['new_teacher_attempts_cap']:raise RuntimeError('Teacher cap')
   size=sum(f.stat().st_size for f in J.rglob('*') if f.is_file())
   if size>reg['storage_cap_bytes']:raise RuntimeError('DEV storage cap')
   state.update(owned=list(seen.values()),GPU_lifecycle_seconds=time.time()-gpu_start,teacher_attempted=sd['teacher_attempted'],storage_bytes=size);save();time.sleep(5)
  terminal=json.loads((J/name/'STATUS.json').read_text())
  state['episodes'].append(dict(episode_id=name,pair_id=case['pair_id'],arm=case['arm'],task=case['task'],valid_index=case['valid_index'],status=terminal['status'],success=terminal.get('success'),initial_checks=terminal.get('initial_checks'),actual_commands=terminal['environment_commands'],policy_chunks=terminal['policy_chunks'],failure_penalty=terminal.get('failure_penalty'),horizon=terminal.get('task_step_limit'),lifecycle_seconds=terminal['lifecycle_seconds'],client_returncode=worker.returncode));save()
  if worker.returncode!=0 or not terminal['status'].startswith('COMPLETE_'):
   if case['variant']!='BASELINE' and 'Frozen initial scene/observation mismatch' in terminal.get('error',''):
    state.setdefault('variant_audits',[]).append(dict(variant=case['variant'],pass_all=False,reason='initial equality failed before actions; no retry'))
    save();continue
   raise RuntimeError('Episode infrastructure failure; no automatic retry')
  audit_variant(case)
 else:state.update(status='COMPLETE_ENGINEERING_VARIANTS_NOT_QUALITY_TEST')
except BaseException as error:state.update(status='FAILED_OR_PARTIAL',error=repr(error),traceback=traceback.format_exc())
finally:
 cleanup()
 f=J/'server/STATUS.json'
 if f.exists():
  d=json.loads(f.read_text());state.update(teacher_attempted=d['teacher_attempted'],teacher_returned=d['teacher_returned'],policy_chunks=d['policy_chunks'])
 state.update(end=time.time(),wall_seconds=time.time()-start,remaining_owned=[o for o in seen.values() if same(o)],returncodes=[q.returncode for q in workers])
 if gpu_start is not None:
  state['GPU_lifecycle_seconds']=time.time()-gpu_start
  with (R/'runs/GPU-LEDGER.lock').open('a') as lf:
   fcntl.flock(lf,fcntl.LOCK_EX);ledger=json.loads((R/'runs/GPU-LEDGER.json').read_text());ledger['consumed_seconds']+=state['GPU_lifecycle_seconds'];ledger['jobs'].append({'namespace':str(J),**state});tmp=R/'runs/GPU-LEDGER.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');tmp.replace(R/'runs/GPU-LEDGER.json')
 save()

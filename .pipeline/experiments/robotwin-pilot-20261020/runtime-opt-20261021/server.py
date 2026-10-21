"""Two exposed DEV scenes with original Target inference, no draft adapter."""
from pathlib import Path
import json,os,random,sys,time,traceback,types
from engineering import digest
import numpy as np
import torch
os.sched_setaffinity(0,{20,21})
R=Path('/root/cxz/robotwin');sys.path.insert(0,str(R/'runs/tf-migration-20261021'))
from common import setup
J=Path(sys.argv[1]);J.mkdir(exist_ok=False)
plan=json.loads((J.parent/'PLAN.json').read_text())
allowed={c['episode_id']:c for c in json.loads((J.parent/'INPUT-MANIFEST.json').read_text())}
torch.set_num_threads(2);torch.set_num_interop_threads(1)
state=dict(status='LOADING',start=time.time(),teacher_attempted=0,teacher_returned=0,policy_chunks=0)
cur=dict(case=None,episode_forwards=0,policy_chunks=0,counts={})
def save():
 p=J/'STATUS.json';p.with_suffix('.tmp').write_text(json.dumps(state,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
def rng():return dict(CPU=torch.get_rng_state(),CUDA=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state())
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,np.ndarray):return x.copy()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return type(x)(cpu(v) for v in x)
 return x
try:
 source,s=setup(J)
 assert s._infer.__func__ is source.VA_Server._infer
 real_forward=s.transformer.forward
 def forward(*args,**kwargs):
  if state['teacher_attempted']>=plan['new_teacher_attempts_cap'] or cur['episode_forwards']>=plan['per_episode_teacher_attempts']:raise RuntimeError('Registered teacher cap')
  state['teacher_attempted']+=1;cur['episode_forwards']+=1
  batch=int(args[0]['noisy_latents'].shape[0]);key=f"{'action' if kwargs.get('action_mode',False) else 'video'}_B{batch}_update{kwargs.get('update_cache',0)}"
  cur['counts'][key+'_attempted']=cur['counts'].get(key+'_attempted',0)+1
  result=real_forward(*args,**kwargs);state['teacher_returned']+=1
  cur['counts'][key+'_returned']=cur['counts'].get(key+'_returned',0)+1
  if state['teacher_attempted']%25==0:save()
  return result
 s.transformer.forward=forward
 original_infer=s.infer
 def infer(_s,request):
  if request.get('reset'):
   case=request['_audit_case'];assert case in allowed
   out=J/case;out.mkdir(exist_ok=False)
   cur.update(case=case,out=out,episode_forwards=0,policy_chunks=0,prefix=[])
   random.seed(plan['eval_seed']);np.random.seed(plan['eval_seed']);torch.manual_seed(plan['eval_seed']);torch.cuda.manual_seed_all(plan['eval_seed'])
   torch.save(rng(),out/'RNG-before-reset.pt')
  route='reset' if request.get('reset') else 'cache' if request.get('compute_kv_cache') else 'policy'
  cur['counts']={};before=cpu(rng()) if cur['policy_chunks']<2 else None
  req=cpu(request) if cur['policy_chunks']<2 else None
  request_identity=digest({k:v for k,v in request.items() if k!='_audit_case'});rng_before=digest(rng())
  start=time.perf_counter();result=original_infer(request);elapsed=time.perf_counter()-start
  with (cur['out']/'ALL-RPC-AUDIT.jsonl').open('a') as audit:
   audit.write(json.dumps(dict(route=route,request=request_identity,result=digest(result),RNG_before=rng_before,RNG_after=digest(rng()),counts=cur['counts']))+'\n')
  if route=='policy':
   assert cur['counts'].get('action_B2_update0_returned')==50
   assert cur['counts'].get('action_B2_update1_returned')==1
   if cur['policy_chunks']<2:
    cur['prefix'].append(dict(input=req,output=cpu(result['action']),RNG_before=before,RNG_after=cpu(rng()),counts=cur['counts'].copy()))
    torch.save(cur['prefix'],cur['out']/'FIRST-TWO-POLICY.pt')
   cur['policy_chunks']+=1;state['policy_chunks']+=1
  with (J/'requests.jsonl').open('a') as f:f.write(json.dumps(dict(case=cur['case'],route=route,counts=cur['counts'],elapsed=elapsed))+'\n')
  state.update(status='SERVING',active_case=cur['case'],active_route=route,active_chunk=cur['policy_chunks']);save()
  return result
 s.infer=types.MethodType(infer,s)
 state.update(status='READY',unpatched_original_infer=True,bank='same full512 BF16',no_draft_runtime=True);save()
 source.run_async_server_mode(s,0,'127.0.0.1',34986)
except BaseException as error:
 state.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise

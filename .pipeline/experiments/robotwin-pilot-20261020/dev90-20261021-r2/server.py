"""Fixed DEV90: same frozen model, explicit canonical T / S / SST routing.

Scientific math is imported from the already validated migration runtime.
Only observations/counters/route live here; no training rows are collected.
"""
from pathlib import Path
import ast,contextlib,hashlib,json,os,random,sys,time,traceback,types
os.sched_setaffinity(0,{20,21})
B=Path('/root/cxz/robotwin/runs/tf-migration-20261021');sys.path.insert(0,str(B))
J=Path(sys.argv[1]);J.mkdir(exist_ok=False)
import numpy as np
import torch
from common import setup,R
from route import Router
plan=json.loads((J.parent/'PLAN.json').read_text());cases=json.loads((J.parent/'INPUT-MANIFEST.json').read_text())
allowed={c['episode_id']:c for c in cases}
torch.set_num_threads(2);torch.set_num_interop_threads(1)
state=dict(status='LOADING',start=time.time(),teacher_attempted=0,teacher_returned=0,policy_chunks=0)
cur=dict(case=None,chunk=0,episode_forwards=0,counts={},action={},scheduler_calls=0)
def save():
 f=J/'STATUS.json';t=f.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(f)
def rng():
 return dict(CPU=torch.get_rng_state().clone(),CUDA=[x.clone() for x in torch.cuda.get_rng_state_all()],python=random.getstate(),numpy=np.random.get_state())
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,np.ndarray):return x.copy()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return type(x)(cpu(v) for v in x)
 return x

def digest(x):
 a=x.detach().contiguous().cpu();return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.view(torch.uint8).numpy().tobytes()).hexdigest())
def kv_digest(s):
 result=[]
 for block in s.transformer.blocks:
  c=block.attn1.attn_caches[s.cache_name];slots=c['mask'].nonzero().squeeze(-1)
  result.append({k:digest(c[k][:,slots] if k in ('k','v') else c[k]) for k in ('k','v','id','mask','is_pred')})
 return result
try:
 assert json.loads((R/'runs/current18-followup-20261021/CAL/RESULT.json').read_text())['status']=='PASS_FINITE_GPU_ADAPTER_DECODE_NOT_CLOSED_LOOP'
 source,s=setup(J)
 from eval_runtime import EvalRuntime
 rt=EvalRuntime(s,source,plan['checkpoint'],plan['tau'],J/'adapter')
 assert rt.metadata['condition']=='ON3000'
 # Time the whole original action loop or speculative action call, with its
 # terminal cache forward, as one boundary. Postprocess/decode remains policy.
 path=J/'adapter/canonical-infer-with-spec-route.py';tree=ast.parse(path.read_text());fn=tree.body[0]
 matches=[]
 for node in ast.walk(fn):
  if hasattr(node,'body') and isinstance(node.body,list):
   for i,stmt in enumerate(node.body):
    if isinstance(stmt,ast.If) and ast.unparse(stmt.test)=='self._robotwin_spec':matches.append((node,i,stmt))
 assert len(matches)==1
 parent,index,stmt=matches[0]
 parent.body[index]=ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_dev_action_scope',ctx=ast.Load()),args=[],keywords=[]))],body=[stmt])
 tree=ast.fix_missing_locations(tree);path=J/'adapter/dev-action-timed-infer.py';path.write_text(ast.unparse(tree)+'\n')
 ns=dict(vars(source));exec(compile(tree,str(path),'exec'),ns);s._infer=types.MethodType(ns['_infer'],s)
 @contextlib.contextmanager
 def action_scope(_s):
  torch.cuda.synchronize();start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  try:yield
  finally:
   b.record();b.synchronize();cur['action']=dict(wall_seconds=time.perf_counter()-start,GPU_event_seconds=a.elapsed_time(b)/1000)
 s._dev_action_scope=types.MethodType(action_scope,s)
 real_step=s.action_scheduler.step
 def step(*args,**kwargs):
  cur['scheduler_calls']+=1;return real_step(*args,**kwargs)
 s.action_scheduler.step=step
 real_forward=s.transformer.forward
 def forward(*args,**kwargs):
  if state['teacher_attempted']>=plan['new_teacher_attempts_cap'] or cur['episode_forwards']>=plan['per_episode_teacher_attempts']:raise RuntimeError('Registered teacher cap')
  state['teacher_attempted']+=1;cur['episode_forwards']+=1
  action=bool(kwargs.get('action_mode',False));batch=int(args[0]['noisy_latents'].shape[0]);key=f"{'action' if action else 'video'}_B{batch}_update{kwargs.get('update_cache',0)}"
  cur['counts'][key+'_attempted']=cur['counts'].get(key+'_attempted',0)+1
  result=real_forward(*args,**kwargs);state['teacher_returned']+=1;cur['counts'][key+'_returned']=cur['counts'].get(key+'_returned',0)+1
  if state['teacher_attempted']%25==0:save()
  return result
 s.transformer.forward=forward;canonical_infer=s.infer
 def infer(_s,request):
  if request.get('reset'):
   name=request['_audit_case'];assert name in allowed
   case=allowed[name];out=J/name;out.mkdir(exist_ok=False)
   cur.update(case=name,chunk=0,episode_forwards=0,router=Router(case['arm']),prefix=[],out=out)
   random.seed(plan['eval_seed']);np.random.seed(plan['eval_seed']);torch.manual_seed(plan['eval_seed']);torch.cuda.manual_seed_all(plan['eval_seed'])
   torch.save(rng(),out/'RNG-before-reset.pt')
  route='reset' if request.get('reset') else 'cache' if request.get('compute_kv_cache') else 'policy'
  is_policy=route=='policy';scheduled=cur['router'].next() if is_policy else None
  cur.update(counts={},action={},scheduler_calls=0);rt.begin_chunk(is_policy and scheduled=='S')
  capture=cur['chunk']<2
  before=rng() if capture else None
  req=cpu({k:v for k,v in request.items() if k!='_audit_case'}) if capture else None
  torch.cuda.synchronize();start=time.perf_counter();result=canonical_infer(request);torch.cuda.synchronize();elapsed=time.perf_counter()-start
  audit=rt.end_chunk() if is_policy else None
  counts=dict(cur['counts']);g=0
  if is_policy:
   assert counts.get('action_B2_update1_returned')==1,counts
   if scheduled=='S':
    g=audit['counters']['committed_scheduler_update_count'];assert g==50
    assert audit['counters']['tree_count']==len(audit['trees'])
    workspace=dict(rt.workspace_counts)
   else:
    assert counts.get('action_B2_update0_returned')==50 and counts.get('action_B14_update0_returned',0)==0,counts
    assert cur['scheduler_calls']==50,cur['scheduler_calls'];g=50;workspace=None
   assert cur['action'], 'Action timing boundary did not execute'
   if capture:
    cur['prefix'].append(dict(route=route,policy_chunk=cur['chunk'],scheduled_route=scheduled,input=req,output=cpu(result['action']),RNG_before=before,RNG_after=rng(),effective_KV_digest=kv_digest(s),counts=counts,counters=audit['counters'],trees=audit['trees']))
    torch.save(cur['prefix'],cur['out']/'FIRST-TWO-POLICY.pt')
   cur['router'].commit_returned();cur['chunk']+=1;state['policy_chunks']+=1
  elif capture:
   cur['prefix'].append(dict(route=route,policy_chunk=cur['chunk'],input=req,RNG_before=before,RNG_after=rng(),counts=counts))
  record=dict(schema='robotwin-dev90-request/v1',case=cur['case'],pair_id=allowed[cur['case']]['pair_id'],arm=allowed[cur['case']]['arm'],route=route,policy_chunk_index=cur['chunk']-1 if is_policy else cur['chunk'],scheduled_route=scheduled,actual_route=scheduled,returned_policy_chunks=cur['router'].returned,counts=counts,policy_or_reset_cache_wall_seconds=elapsed,action=cur['action'],valid_commits_g=g,all_scheduler_step_calls=cur['scheduler_calls'],audit=audit,workspace=workspace if is_policy else None,time_kind='instrumented exploratory; not formal performance',GPU_uuid=plan['GPU_uuid'])
  with (J/'requests.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  rt.rows=[];rt.tree_records=[];state.update(status='SERVING',active_case=cur['case'],active_route=route,active_chunk=cur['chunk'],last_request=record);save()
  return result
 s.infer=types.MethodType(infer,s);state.update(status='READY',checkpoint=plan['checkpoint'],tau=plan['tau'],bank='full512 BF16 reused',metadata=rt.metadata);save()
 source.run_async_server_mode(s,0,'127.0.0.1',34976)
except BaseException as e:
 state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise

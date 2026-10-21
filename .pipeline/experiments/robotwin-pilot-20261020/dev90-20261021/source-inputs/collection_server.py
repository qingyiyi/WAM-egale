"""Fixed PARENT2000 or CONT3000 visited-data collector; original full robot policy."""
from pathlib import Path
import os,sys,json,time,random,types,traceback
os.sched_setaffinity(0,{20,21})
import numpy as np
import torch
from common import setup,R
B=Path(__file__).resolve().parent;J=Path(sys.argv[1]);J.mkdir(exist_ok=True)
plan=json.loads((J.parent/'PLAN.json').read_text());cases=json.loads((J.parent/'INPUT-MANIFEST.json').read_text())
torch.set_num_threads(2);torch.set_num_interop_threads(1)
state=dict(status='LOADING',start=time.time(),teacher_attempted=0,teacher_returned=0,policy_chunks=0)
current=dict(case=None,chunk=0,episode_forwards=0,route=None,counts={})
def save():
 p=J/'STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
try:
 assert json.loads((B/'interface-gate-r3/RESULT.json').read_text())['status']=='PASS_FINITE_EIGHT_COMPLETE_POLICY_CALLS_NOT_ROBOT_QUALITY'
 source,s=setup(J)
 from runtime import Runtime
 tau=json.loads((B/'CALIBRATION.json').read_text())['tau']
 runtime=Runtime(s,source,plan['checkpoint'],tau,J/'adapter')
 assert runtime.metadata['condition']==plan['generator']
 canonical_forward=s.transformer.forward;canonical_infer=s.infer
 allowed={c['task']+f"-valid{c['valid_index']:02d}":c for c in cases}
 def forward(*args,**kwargs):
    assert state['teacher_attempted']<plan['teacher_cap'] and current['episode_forwards']<8000
    state['teacher_attempted']+=1;current['episode_forwards']+=1
    action=bool(kwargs.get('action_mode',False));n=int(args[0]['noisy_latents'].shape[0])
    key=f"{'action' if action else 'video'}_B{n}_update{kwargs.get('update_cache',0)}"
    current['counts'][key+'_attempted']=current['counts'].get(key+'_attempted',0)+1
    out=canonical_forward(*args,**kwargs)
    state['teacher_returned']+=1;current['counts'][key+'_returned']=current['counts'].get(key+'_returned',0)+1
    if state['teacher_attempted']%25==0:save()
    return out
 s.transformer.forward=forward
 def infer(_s,request):
    if request.get('reset'):
        name=request['_audit_case'];assert name in allowed
        current.update(case=name,chunk=0,episode_forwards=0)
        (J/name).mkdir(exist_ok=False)
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.save(dict(CPU=torch.get_rng_state(),CUDA=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state()),J/name/'RNG-before-reset.pt')
    route='reset' if request.get('reset') else 'cache' if request.get('compute_kv_cache') else 'policy'
    current.update(route=route,counts={});runtime.begin_chunk(route=='policy');start=time.time()
    result=canonical_infer(request);torch.cuda.synchronize();elapsed=time.time()-start
    audit=runtime.end_chunk() if route=='policy' else None
    if route=='policy':
        state['last_returned_policy']=dict(case=current['case'],chunk=current['chunk'],counts=dict(current['counts']),audit=audit)
        save() # Persist counters even if the following contract fails.
        assert audit['counters']['committed_scheduler_update_count']==50
        assert current['counts'].get('action_B2_update1_returned')==1
        rows=runtime.rows;assert rows
        torch.save(dict(schema='robotwin-reached-spec-query/v1',rows=rows,decoded_action=result['action'],
            case=current['case'],case_manifest=allowed[current['case']],chunk=current['chunk'],
            generator=runtime.metadata,checkpoint=plan['checkpoint'],tau=tau,
            full_conditions_for_teacher_requery=False,source_kind='selected reached root/parent only'),J/current['case']/f"chunk-{current['chunk']:04d}.pt")
        current['chunk']+=1;state['policy_chunks']+=1
    with (J/'requests.jsonl').open('a') as f:f.write(json.dumps(dict(case=current['case'],route=route,chunk=current['chunk'],
        counts=current['counts'],elapsed_seconds=elapsed,audit=audit,time_kind='instrumented collection, not formal performance'))+'\n')
    runtime.rows=[];runtime.tree_records=[]
    state.update(status='SERVING',active_case=current['case'],active_route=route,active_chunk=current['chunk']);save()
    return result
 s.infer=types.MethodType(infer,s);state.update(status='READY',generator=runtime.metadata,bank='full512 BF16 reused',tau=tau);save()
 source.run_async_server_mode(s,0,'127.0.0.1',34976)
except BaseException as e:
 state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise

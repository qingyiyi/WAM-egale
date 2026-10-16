"""Fixed-fixture full verifier Graph baseline, not a deployable policy engine.

Only immutable fixture metadata admission is hoisted. Arithmetic is the frozen
WanAttention body and either original SDPA or the existing shared native kernel.
All private bindings are restored; full production cache lifecycle is not claimed.
"""
import json,os,sys,time,types,statistics,traceback,gc
from pathlib import Path
import full_matrix as entry
from full_matrix import ROOT,EXP,OWN,write,load

def run(s,source,out,p):
 import torch
 import adapter,state_check
 sys.path.append(str(EXP/'shared_prefix_torch_v2'))
 import _shared_torch_v1 as native
 snap=load('graph_matrix_snapshot',EXP/'fixed_load_cost/snapshot.py')
 src=Path(os.environ['WAM_GRAPH_FIXTURE_ROOT'])
 terminal=json.loads((src/'MATRIX-RESULT.json').read_text())
 assert terminal['status']=='COMPLETE_THREE_LENGTH_FULL_VERIFIER_MECHANISM'
 s.infer(dict(reset=True,prompt=None));m=s.transformer
 result=dict(status='RUNNING',production_engine=False,new_MegaTransformer=False,scope='static frozen fixture full30 Graph strong baseline only',rows=[],guard_hoisting='fixed fixture metadata only, not production admission contract',target_forwards=0)
 start=time.monotonic()
 def check():
  if time.monotonic()-start>1650 or (out/'STOP_REQUESTED').exists():raise TimeoutError('Graph fixture stop')
  if result['target_forwards']>1800:raise RuntimeError('Graph forward cap')
 def dev(x):
  if isinstance(x,torch.Tensor):return x.to(s.device)
  if isinstance(x,dict):return {k:dev(v) for k,v in x.items()}
  return x
 def subset(x,idx):
  if isinstance(x,torch.Tensor) and x.ndim and x.shape[0]==14:return x[idx].contiguous()
  if isinstance(x,dict):return {k:subset(v,idx) for k,v in x.items()}
  return x
 def equal(a,b):
  a=a.contiguous();b=b.to(a.device).contiguous();d=(a.float()-b.float()).abs()
  return dict(bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),max_abs=float(d.max()),different=int(torch.count_nonzero(d)))
 def measure(fn,reps=6):
  values=[]
  for _ in range(reps):
   check();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize()
   t=time.perf_counter();a.record();y=fn();b.record();b.synchronize()
   values.append(dict(cuda_ms=a.elapsed_time(b),wall_ms=1000*(time.perf_counter()-t)))
  return dict(samples=values,median_cuda_ms=statistics.median(x['cuda_ms'] for x in values),median_wall_ms=statistics.median(x['wall_ms'] for x in values)),y
 try:
  for label in ('short','middle','long'):
   data=torch.load(src/label/'real-verifier-fixture.pt',map_location='cpu',weights_only=False)
   for b,c in zip(m.blocks,data['history']):snap.restore_cache(b.attn1.attn_caches[s.cache_name],c)
   refs=state_check.references(m,s.cache_name);rng=snap.fingerprint(snap.rng(True))
   slots=[c['slots'].to(s.device) for c in data['history']]
   originals=[b.attn1.forward for b in m.blocks]
   try:
    for query in data['queries']:
     inp=dev(query['input']);kw=dev(query['kwargs']);kw.pop('input_dict',None)
     for n in (1,3,7):
      idx=torch.tensor(list(range(n))+list(range(7,7+n)),device=s.device)
      si=subset(inp,idx);sk=subset(kw,idx);rows=sk['candidate_cache_rows']
      assert torch.equal(rows,torch.tensor([0]*n+[1]*n,device=s.device))
      # Real original same-shape oracle before any private bindings.
      gold=m(si,**sk);result['target_forwards']+=1
      if n==7:assert equal(gold,query['output'])['bitwise'],'frozen original B14 query no longer reproduced'
      engines=('dense','existing_shared') if n==7 else ('dense',)
      for engine in engines:
       patches=[]
       rope_original={name:getattr(m.rope,name) for name in ('f_freqs_base','h_freqs_base','w_freqs_base')}
       for name,value in rope_original.items():setattr(m.rope,name,value.to(s.device))
       for i,block in enumerate(m.blocks):
        a=block.attn1;cache=a.attn_caches[s.cache_name];valid=slots[i]
        assert valid.numel()+16==data['total_KV']
        def history(a,q,k,v,c,name,r,update,_cache=cache,_slots=valid):
         # These identities/scalars are fixed before graph capture. No dynamic
         # boolean CUDA checks, tensor arithmetic or precision changes added.
         assert c is _cache and name==s.cache_name and r is rows and update==0
         if engine=='existing_shared':return native.shared(q,_cache['k'],_cache['v'],k,v,_slots,rows)
         hk=_cache['k'][:,_slots];hv=_cache['v'][:,_slots]
         return a.attn_op(q,torch.cat((hk.index_select(0,rows),k),1),torch.cat((hv.index_select(0,rows),v),1))
        a.forward=types.MethodType(adapter._private_forward(originals[i],history),a);patches.append(a)
       try:
        def eager():
         result['target_forwards']+=1;return m(si,**sk)
        candidate=eager();admit=equal(candidate,gold);assert admit['bitwise'],'static metadata admission changed output'
        eager_timing,_=measure(eager)
        versions=[(x,x._version) for x in list(m.parameters())+[v for c in state_check.pools(m,s.cache_name) for v in c.values()]+list(si.values())+[rows]+slots+[getattr(m.rope,name) for name in rope_original] if isinstance(x,torch.Tensor)]
        # Capture one actual full forward on a warmed private stream.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
         for _ in range(2):eager()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph();capture_start=time.perf_counter()
        with torch.cuda.graph(graph):y=eager()
        torch.cuda.synchronize();capture_ms=1000*(time.perf_counter()-capture_start)
        def replay():
         result['target_forwards']+=1;graph.replay();return y
        graph_timing,actual=measure(replay)
        parity=equal(actual,gold);assert parity['bitwise'],'graph changed same-shape model output'
        assert all(x._version==v for x,v in versions),'fixed input/parameter/cache mutated'
        assert state_check.equal(m,refs,s.cache_name) and snap.fingerprint(snap.rng(True))==rng
        row=dict(length=label,total_KV=data['total_KV'],query=query['index'],n_v=n,engine=engine,eager=eager_timing,graph=graph_timing,capture_wall_ms_excluded=capture_ms,oracle=admit,graph_vs_original=parity,all_valid_KV_RNG_unchanged=True,inputs_resident=True,input_update_and_graph_rebuild_not_measured=True)
        result['rows'].append(row);print(json.dumps(dict(length=label,n=n,query=query['index'],engine=engine,graph_ms=graph_timing['median_wall_ms'])),flush=True)
        del graph,y,actual,candidate,versions;gc.collect();torch.cuda.empty_cache()
       finally:
        for a,orig in zip(patches,originals):a.forward=orig
        for name,value in rope_original.items():setattr(m.rope,name,value)
     del gold,inp,kw,si,sk
   finally:
    for b,orig in zip(m.blocks,originals):b.attn1.forward=orig
   del refs,data;gc.collect();torch.cuda.empty_cache()
  result.update(status='COMPLETE_STATIC_FIXTURE_GRAPH_MATRIX',elapsed_sec=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),formal_robot_performance=False)
 except BaseException as exc:
  result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());raise
 finally:write(out/'MATRIX-RESULT.json',result)

if __name__=='__main__':
 entry.run=run;entry.main()

"""Separate instrumented whole-layer diagnostic; not production latency."""
from pathlib import Path
import os,json,time,traceback,importlib.util,statistics,subprocess
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
from wam_ops.optimized.native import NativeBlock
B=Path(__file__).resolve().parent
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
normal=load(next(B.parent.glob('wam_whole_fa2_cuda*.so')),'normal.wam_whole_fa2_cuda')
diag=load(next((B/'profile').glob('wam_cached_profile_cuda*.so')),'wam_cached_profile_cuda')
start=time.time();out=dict(status='RUNNING',rows=[],full_calls=0,formal_performance_permission=False)
def bump(n=1):out['full_calls']+=n;assert out['full_calls']<=9000
def save():(B/'PROFILE-RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
try:
 block=load_block('cuda');native=NativeBlock(block);binding=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   f=BlockFixture(FIXTURES/f'input-{length}-B14-action-layer0-first.pt',device_tree(decode(torch.load(FIXTURES/f'input-{length}-B14-action-layer0-first.pt',map_location='cpu',mmap=True,weights_only=False))))
   a,k=f.arguments();f.install_cache(block);ref=native(*a,**k,debug=True);bump();f.install_cache(block)
   cache=block.attn1.attn_caches[k['cache_name']];before=visible_snapshot(cache);cl=[cache[x] for x in ['k','v','id','mask','is_pred']]
   binding.prepare_context(a[1],precompute=True);saved_kv=[x.clone() for x in binding.context_kv];saved_inputs=[x.clone() for x in a];rng=[torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()]
   common=(binding.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),binding.eps)
   for cached in [False,True]:
    label='cached' if cached else 'grouped_uncached';extra=(binding.context_rows,binding.context_roots,binding.context_kv if cached else [])
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([x for x in apps.splitlines() if x.startswith('GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726')])==1
    z=diag.forward(*common,True,*extra);bump();checks={key:compare(ref[key].reshape(-1),z[key].reshape(-1)) for key in ref.keys()&z.keys() if key not in ['resources','cache_info']};assert all(v['bitwise_pass'] for v in checks.values());resources=z['resources'].tolist()
    graphs={};values={}
    for arm,ext in [('normal',normal),('profile',diag)]:
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):y=ext.forward(*common,False,*extra)
     bump();graphs[arm]=g;values[arm]=y
    for _ in range(600):
     for g in graphs.values():g.replay();bump()
    torch.cuda.synchronize();samples=[];raw=[];stage_records=[]
    for repeat in range(8):
     for arm in (['normal','profile'] if repeat%2==0 else ['profile','normal']):
      e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record(stream);graphs[arm].replay();bump();e1.record(stream);e1.synchronize();samples.append(dict(repeat=repeat,arm=arm,ms=e0.elapsed_time(e1)))
     y=values['profile'];ticks=y['trace_ticks'].cpu().tolist();barriers=y['trace_barriers'].cpu().tolist();counts=y['trace_counts'].cpu().tolist();assert len(set(counts))==1 and counts[0]==(31 if cached else 36),counts
     raw.append(dict(repeat=repeat,ticks=ticks,barriers=barriers,counts=counts))
     for sid in range(27):
      measures=[]
      for cta,stages in enumerate(ticks):
       begin,end=stages[sid]
       if not begin:continue
       assert end>=begin
       waits=[(en,ex) for en,ex in barriers[cta][:counts[cta]] if en>=begin and ex<=end]
       wait=sum(ex-en for en,ex in waits);measures.append(dict(cta=cta,total_us=(end-begin)/1000,grid_wait_us=wait/1000,outside_grid_us=(end-begin-wait)/1000))
      stage_records.append(dict(repeat=repeat,stage=sid,skipped=not measures,cta0=measures[0] if measures else None,mean_CTA_wait_us=statistics.mean(x['grid_wait_us'] for x in measures) if measures else None,max_CTA_outside_grid_us=max((x['outside_grid_us'] for x in measures),default=None)))
    for arm,y in values.items():assert compare(ref['output'],y['output'])['bitwise_pass']
    assert state_equal(before,visible_snapshot(cache))['pass_all'] and all(torch.equal(x,y) for x,y in zip(saved_inputs,a)) and all(torch.equal(x,y) for x,y in zip(saved_kv,binding.context_kv))
    assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
    tape=f'PROFILE-TAPE-{length}-{label}.json';(B/tape).write_text(json.dumps(raw,separators=(',',':'))+'\n')
    row=dict(case=length,mode=label,profile_intermediates=checks,resources=resources,samples=samples,median_ms={arm:statistics.median(x['ms'] for x in samples if x['arm']==arm) for arm in graphs},stages=stage_records,tape=tape,grid_barriers=counts[0],finite_precision_and_state=True)
    out['rows'].append(row);save();del graphs,values,y,z
   del ref,f,a,k,cl,cache,before,saved_inputs,saved_kv;torch.cuda.empty_cache()
 out['status']='COMPLETE_SEPARATE_STAGE_DIAGNOSTIC'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error',''),out['full_calls'],flush=True)
if out['status']!='COMPLETE_SEPARATE_STAGE_DIAGNOSTIC':raise SystemExit(1)

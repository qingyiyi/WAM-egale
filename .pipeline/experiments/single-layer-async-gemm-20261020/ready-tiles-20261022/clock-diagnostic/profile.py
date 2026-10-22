"""Clock64诊断：CTA累计cycles，不冒充GPU墙钟占比；单层真实输入。"""
from pathlib import Path
import os,time,json,traceback,importlib.util,statistics,subprocess
import torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot,tensors
from tools.replay import device_tree
from tools.metrics import compare,state_equal,byte_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;P=B.parent
GPU='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'
def clock_state():
 return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=uuid,clocks.sm,clocks.mem,utilization.gpu','--format=csv,noheader'],text=True).strip()
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
start=time.time();r=dict(status='RUNNING',full_block_calls=0,rows=[],scope='per-CTA cycles, not wall time fractions; derived instrumentation only')
def save():(B/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
def load(path,name):
 sp=importlib.util.spec_from_file_location(name+'.wam_whole_fa2_cuda',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
try:
 base=load(next(P.glob('wam_whole_fa2_cuda*.so')),'ready');ins=load(next(B.glob('wam_whole_fa2_cuda*.so')),'clock')
 block=load_block('cuda');bind=WholeLayerBlock(block);stream=torch.cuda.Stream();expected=json.loads((B/'EXPECTED-TILES.json').read_text())
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments();f.install_cache(block)
   cache=block.attn1.attn_caches[k['cache_name']];cl=[cache[x] for x in ('k','v','id','mask','is_pred')];before=visible_snapshot(cache)
   inputs={n:x.clone() for n,x in tensors((a,k))};cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
   def call(ext):
    r['full_block_calls']+=1;assert r['full_block_calls']<2000
    return ext.forward(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps,True,True)
   original=call(base);z=call(ins);torch.cuda.synchronize()
   common=original.keys()&z.keys()-{'resources','logical_slots'}
   checks={n:compare(original[n].reshape(-1),z[n].reshape(-1)) for n in common};assert all(x['bitwise_pass'] for x in checks.values()),checks
   prefix=int(z['cache_info'][0].item());group=2 if prefix%16==0 and (prefix+95)//64==(prefix+79)//64 else 1
   ids=k['candidate_cache_rows'].tolist()
   if any(ids[cfg*7+i]!=ids[cfg*7+i+1] for cfg in range(2) for i in range(0,6,2)):group=1
   key=f"{int(a[3] is not None)}-{int(bind.weights[1].numel()>0)}-{((7+group-1)//group)*24}"
   want=expected[key];blocks=int(z['resources'][0].item());graphs={};values={}
   for label,ext in [('base',base),('instrumented',ins)]:
    for _ in range(2):call(ext)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):out=call(ext)
    graphs[label]=g;values[label]=out
   for _ in range(100):
    for g in graphs.values():g.replay();r['full_block_calls']+=1
   torch.cuda.synchronize();clock_before=clock_state();samples=[];profiles=[]
   for repeat in range(10):
    for label in (['base','instrumented'] if repeat%2==0 else ['instrumented','base']):
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     begin.record(stream);graphs[label].replay();r['full_block_calls']+=1;end.record(stream);end.synchronize()
     samples.append(dict(repeat=repeat,label=label,ms=begin.elapsed_time(end)))
     if label=='instrumented':
      x=values[label]['ready_profile'][:blocks].cpu();counts=x[:,:,1].sum(0).tolist();cycles=x[:,:,0].sum(0).tolist()
      assert counts[:28]==want,(length,counts[:28],want)
      assert counts[29]==sum(want) and counts[28]==sum(want)+counts[30]+blocks
      profiles.append(dict(repeat=repeat,cycles=cycles,counts=counts))
   ms={name:statistics.median(t['ms'] for t in samples if t['label']==name) for name in values}
   state=state_equal(before,visible_snapshot(cache));assert state['pass_all']
   unchanged=all(byte_equal(inputs[n],x) for n,x in tensors((a,k)));rng=byte_equal(cpu,torch.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state());assert unchanged and rng
   row=dict(case=length,clock_before=clock_before,clock_after=clock_state(),precision=checks,cache=state,inputs_unchanged=unchanged,RNG_unchanged=rng,resources=z['resources'].tolist(),expected_tiles=want,profiles=profiles,samples=samples,median_ms=ms,instrumentation_overhead_percent=100*(ms['instrumented']/ms['base']-1))
   r['rows'].append(row);save();del original,z,a,k,f,cache,cl,inputs,before,values,graphs;torch.cuda.empty_cache()
 r['status']='COMPLETE_FINITE_CLOCK_DIAGNOSTIC'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['elapsed_seconds']=time.time()-start;save()
print(r['status'],r.get('error',''),flush=True)
if r['status']!='COMPLETE_FINITE_CLOCK_DIAGNOSTIC':raise SystemExit(1)

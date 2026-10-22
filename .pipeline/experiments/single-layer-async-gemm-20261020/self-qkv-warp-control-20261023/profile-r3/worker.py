"""Paired finite software timeline; no GPU hardware counters."""
from pathlib import Path
import os,sys,json,time,traceback,importlib.util,statistics,subprocess
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
oldbase=B.parent.parent/'ffn-pipeline-inline-20261022'
old=load(oldbase/'tail-k64/wam_whole_fa2_cuda.cpython-312-x86_64-linux-gnu.so','old.wam_whole_fa2_cuda')
new=load(B.parent/'r3-isolated/wam_whole_fa2_cuda.cpython-312-x86_64-linux-gnu.so','new.wam_whole_fa2_cuda')
d4=load(oldbase/'current-stage-profile/r2-existing-trace/wam_current_stage_cuda.cpython-312-x86_64-linux-gnu.so','old.wam_current_stage_cuda')
d8=load(B/'wam_current_stage_cuda.cpython-312-x86_64-linux-gnu.so','new.wam_current_stage_cuda')
start=time.time();out=dict(status='RUNNING',rows=[],full_calls=0,formal_performance_permission=False,boundary='CTA0 software intervals include queues and barriers, not hardware counters or exact compute-only times')
def bump(n=1):out['full_calls']+=n;assert out['full_calls']<=8000
def save():(B/f'RESULT-{sys.argv[1]}.json').write_text(json.dumps(out,indent=2)+'\n')
try:
 block=load_block('cuda');binding=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([x for x in apps.splitlines() if x.startswith('GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726')])==1
   path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
   a,k=f.arguments();f.install_cache(block);orig=block(*a,**k);bump();f.install_cache(block)
   cache=block.attn1.attn_caches[k['cache_name']];before=visible_snapshot(cache);cl=[cache[x] for x in ['k','v','id','mask','is_pred']]
   binding.prepare_context(a[1],precompute=True);saved_kv=[x.clone() for x in binding.context_kv];saved_inputs=[x.clone() for x in a];rng=[torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()]
   common=(binding.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),binding.eps);extra=(binding.context_rows,binding.context_roots,binding.context_kv)
   z=old.forward(*common,True,*extra);bump();checks={};resources={}
   for label,ext in [('old',old),('new',new),('diag4',d4),('diag8',d8)]:
    d=ext.forward(*common,True,*extra);bump();cs={key:compare(z[key].reshape(-1),d[key].reshape(-1)) for key in z.keys()&d.keys() if key not in ['resources','cache_info','logical_slots','attention_meta','ffn_pipeline_state','ffn_pipeline_timestamps']};assert all(v['bitwise_pass'] for v in cs.values());checks[label]=cs;resources[label]=d['resources'].tolist()
   assert compare(orig,z['output'])['bitwise_pass'];graphs={};values={}
   for label,ext,enabled in [('old',old,None),('new',new,None),('d4_off',d4,False),('d4_on',d4,True),('d8_off',d8,False),('d8_on',d8,True)]:
    def call():return ext.forward(*common,False,*extra,*([] if enabled is None else [enabled]))
    for _ in range(2):call();bump()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):v=call()
    bump();graphs[label]=g;values[label]=v
   for _ in range(100):
    for g in graphs.values():g.replay();bump()
   samples=[];tapes={'d4_on':[],'d8_on':[]}
   for repeat in range(24):
    names=list(graphs);offset=repeat%6;names=names[offset:]+names[:offset]
    if repeat%2:names=names[::-1]
    for label in names:
     e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record(stream)
     for _ in range(10):graphs[label].replay();bump()
     e1.record(stream);e1.synchronize();samples.append(dict(repeat=repeat,arm=label,us=e0.elapsed_time(e1)*100))
    for label in tapes:tapes[label].append(values[label]['phase_ticks'].cpu().tolist())
   for y in values.values():assert compare(orig,y['output'])['bitwise_pass']
   stages={}
   for sid in range(22):
    med={label:statistics.median((t[sid][1]-t[sid][0])/1000 for t in tt) for label,tt in tapes.items()}
    stages[str(sid)]=med
   assert state_equal(before,visible_snapshot(cache))['pass_all'];assert all(torch.equal(x,y) for x,y in zip(saved_inputs,a));assert all(torch.equal(x,y) for x,y in zip(saved_kv,binding.context_kv));assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
   row=dict(case=length,fixture=str(path),precision=checks,resources=resources,samples=samples,full_median_us={arm:statistics.median(x['us'] for x in samples if x['arm']==arm) for arm in graphs},phase_tapes=tapes,stages=stages,finite_state_pass=True);out['rows'].append(row);save()
   del graphs,values,z,d,orig,f,a,k,cache,cl,before;torch.cuda.empty_cache()
 out['status']='COMPLETE_FINITE_PAIRED_PHASE_DIAGNOSTIC'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error',''),out['full_calls'],flush=True)
if out['status']!='COMPLETE_FINITE_PAIRED_PHASE_DIAGNOSTIC':raise SystemExit(1)

"""Three real inputs, one-process engineering screen; not formal latency claim."""
import os,json,time,traceback,importlib.util,sys
from pathlib import Path
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode
from tools.replay import device_tree
from tools.metrics import compare
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;label=sys.argv[1];os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);t=time.time();r=dict(status='RUNNING',rows=[],samples=[],full_block_calls=0,scope='three-input preliminary screen, one process',production_admitted=False)
def save():(B/(label+'-RESULT.json')).write_text(json.dumps(r,indent=2)+'\n')
try:
 p=next((B/'source-aligned').glob('wam_whole_fa2_cuda*.so'));sp=importlib.util.spec_from_file_location('wam_whole_fa2_cuda',p);old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
 block=load_block('cuda');new=WholeLayerBlock(block)
 work=Path(load_block.__code__.co_filename).resolve().parents[2];video=work/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
 def call(name,args,kw):
  r['full_block_calls']+=1
  if name=='new':return new(*args,**kw,debug=True,audit=False)
  cache=block.attn1.attn_caches.get(kw.get('cache_name','pos'));values=[] if cache is None else [cache[k] for k in ('k','v','id','mask','is_pred')]
  return old.forward(new.weights,*args,values,kw.get('candidate_cache_rows'),kw.get('update_cache',0),new.eps,False,True)
 with torch.inference_mode():
  for kind,batch,length in [('action',2,'short'),('action',14,'long'),('video',2,'long')]:
   p=(FIXTURES if kind=='action' else video)/f'input-{length}-B{batch}-{kind}-layer0-first.pt';d=decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False));f=BlockFixture(p,device_tree(d));args,kw=f.arguments();case=f'{length}-B{batch}-{kind}';f.install_cache(block);a=call('old',args,kw);f.install_cache(block);z=call('new',args,kw);torch.cuda.synchronize();cmp=compare(a['output'],z['output']);r['rows'].append(dict(case=case,vs_aligned=cmp,resources=z['resources'].tolist()));save()
   if not cmp['bitwise_pass']:raise RuntimeError('preliminary output mismatch: '+case)
   for repeat in range(2):
    for name in (('old','new') if repeat%2==0 else ('new','old')):
     f.install_cache(block);torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();h=time.perf_counter();z=call(name,args,kw);end.record();end.synchronize();r['samples'].append(dict(case=case,name=name,repeat=repeat,ms=start.elapsed_time(end),host_ms=1000*(time.perf_counter()-h)))
   save();del a,z,d,f,args,kw
 r['status']='COMPLETE_PRELIMINARY_NOT_FORMAL'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-t;save()

from pathlib import Path
import os,sys,json,time,traceback,importlib.util
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode
from tools.replay import device_tree
from tools.metrics import compare
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);t=time.time();r=dict(status='RUNNING',full_block_calls=0,rows=[],profiling_only=True)
def save():(B/'PROFILE-RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 p=next(B.glob('wam_stage_profile_cuda*.so'));spec=importlib.util.spec_from_file_location('wam_stage_profile_cuda',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 block=load_block('cuda');binding=WholeLayerBlock(block,bf16_split_reduction=True);native=NativeBlock(block)
 work=Path(load_block.__code__.co_filename).resolve().parents[2];video=work/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
 def callprof(args,kw):
  cache=block.attn1.attn_caches.get(kw.get('cache_name','pos'));values=[] if cache is None else [cache[k] for k in ('k','v','id','mask','is_pred')]
  return mod.forward(binding.weights,*args,values,kw.get('candidate_cache_rows'),kw.get('update_cache',0),binding.eps,True,True)
 with torch.inference_mode():
  for kind,batch,length in [('action',2,'short'),('action',14,'long'),('video',2,'long')]:
   p=(FIXTURES if kind=='action' else video)/f'input-{length}-B{batch}-{kind}-layer0-first.pt';d=decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False));f=BlockFixture(p,device_tree(d));args,kw=f.arguments()
   f.install_cache(block);ref=native(*args,**kw,debug=True);r['full_block_calls']+=1
   rows=[]
   for repeat in range(3):
    f.install_cache(block);x=callprof(args,kw);r['full_block_calls']+=1;torch.cuda.synchronize();ticks=x['stage_ticks'].cpu().tolist()
    rows.append(dict(repeat=repeat,vs_native=compare(ref['output'],x['output']),stage_ticks_ns=ticks,resources=x['resources'].tolist()))
   r['rows'].append(dict(case=f'{length}-B{batch}-{kind}',profiles=rows));save();del ref,x,args,kw,f,d
 r['status']='COMPLETE_PROFILE' if all(y['vs_native']['bitwise_pass'] for x in r['rows'] for y in x['profiles']) else 'PROFILE_OUTPUT_MISMATCH'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-t;save()

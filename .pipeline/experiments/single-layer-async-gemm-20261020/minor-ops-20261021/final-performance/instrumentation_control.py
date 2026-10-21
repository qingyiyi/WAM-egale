from pathlib import Path
import os,json,time,traceback,statistics,importlib.util
import torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;S=B.parent/'stage-current';os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
t0=time.time();r=dict(status='RUNNING',full_block_calls=0,rows=[],boundary='paired complete eager block, diagnostic of instrumentation cost, not official performance')
def save():(B/'INSTRUMENTATION-CONTROL.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 spec=importlib.util.spec_from_file_location('wam_stage_simple_cuda',next(S.glob('wam_stage_simple_cuda*.so')));own=importlib.util.module_from_spec(spec);spec.loader.exec_module(own);block=load_block('cuda');selected=WholeLayerBlock(block)
 with torch.inference_mode():
  for length in ['short','middle','long']:
   path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments();f.install_cache(block);ref=selected(*a,**k,debug=True);r['full_block_calls']+=1;torch.cuda.synchronize();cache=block.attn1.attn_caches[k['cache_name']];before=visible_snapshot(cache);cs=[cache[n] for n in ['k','v','id','mask','is_pred']];common=(selected.weights,*a,cs,k['candidate_cache_rows'],k.get('update_cache',0),selected.eps);row=dict(case=length,samples=[])
   for repeat in range(4):
    vals={}
    for which in (['unmarked','marked'] if repeat%2==0 else ['marked','unmarked']):
     aa=torch.cuda.Event(enable_timing=True);zz=torch.cuda.Event(enable_timing=True);aa.record();v=selected(*a,**k,debug=True) if which=='unmarked' else own.forward(*common,-1,[],[]);zz.record();zz.synchronize();r['full_block_calls']+=1;assert compare(ref['output'],v['output'])['bitwise_pass'];vals[which]=aa.elapsed_time(zz)
    row['samples'].append(dict(repeat=repeat,**vals))
   row['cache_unchanged']=state_equal(before,visible_snapshot(cache));assert row['cache_unchanged']['pass_all'];r['rows'].append(row);save();del f,a,k,ref,cache,cs,v;torch.cuda.empty_cache()
 r['status']='COMPLETE_PAIRED_INSTRUMENTATION_CONTROL'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-t0;save()
print(r['status'],r.get('error',''),flush=True)
if r['status']!='COMPLETE_PAIRED_INSTRUMENTATION_CONTROL':raise SystemExit(1)

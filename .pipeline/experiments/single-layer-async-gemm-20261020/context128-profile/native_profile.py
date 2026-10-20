"""Remaining 12 registered full-block calls: native kernel traces vs current output."""
from pathlib import Path
import json,os,time,traceback
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
t0=time.time();r={'status':'RUNNING','full_block_calls':0,'rows':[],'profiling_only':True}
def save():(B/'NATIVE-PROFILE-RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 block=load_block('cuda');native=NativeBlock(block);new=WholeLayerBlock(block)
 with torch.inference_mode():
  for length in ['short','middle','long']:
   p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments();row={'case':length,'fixture':str(p),'input_shapes':[list(x.shape) for x in a if isinstance(x,torch.Tensor)],'traces':[]}
   f.install_cache(block);r['full_block_calls']+=1;out=new(*a,**k,debug=True,audit=False);torch.cuda.synchronize();expected=out['output'].clone();ecache=visible_snapshot(block.attn1.attn_caches[k['cache_name']])
   for i in range(3):
    f.install_cache(block)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     r['full_block_calls']+=1;ref=native(*a,**k,debug=True);torch.cuda.synchronize()
    c=compare(expected,ref['output']);ks=state_equal(ecache,visible_snapshot(block.attn1.attn_caches[k['cache_name']]));assert c['bitwise_pass'] and ks['pass_all']
    path=B/f'native-{length}-{i}.trace.json';prof.export_chrome_trace(str(path));events=json.loads(path.read_text())['traceEvents'];kernels=[{k:e.get(k) for k in ['name','dur','ts','args']} for e in events if e.get('cat')=='kernel'];(B/f'native-{length}-{i}.kernels.json').write_text(json.dumps(kernels,indent=2)+'\n');row['traces'].append({'repeat':i,'output':c,'cache':ks,'kernel_count':len(kernels),'kernel_file':str(B/f'native-{length}-{i}.kernels.json')})
   r['rows'].append(row);save();del f,a,k,out,ref,expected,ecache
 assert r['full_block_calls']==12;r['status']='COMPLETE_NATIVE_PROFILE_FINITE_OUTPUT_CHECK'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-t0;save()

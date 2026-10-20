import json,os,time,traceback
from pathlib import Path
import torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot,tensors
from tools.replay import device_tree
from tools.metrics import compare,byte_equal,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
t0=time.time();r=dict(status='RUNNING',full_block_calls=0,primitive_calls=0,rows=[],production_admitted=False)
def save():(B/'NORM-RESULT-2.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 block=load_block('cuda');native=NativeBlock(block);new=WholeLayerBlock(block)
 work=Path(load_block.__code__.co_filename).resolve().parents[2]
 video=work/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
 with torch.inference_mode():
  for length in ['short','middle','long']:
   path=video/f'input-{length}-B2-video-layer0-first.pt';data=decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False));f=BlockFixture(path,device_tree(data));args,kw=f.arguments()
   cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state();inputs={n:t.clone() for n,t in tensors((args,kw))}
   f.install_cache(block);ref=native(*args,**kw,debug=True);r['full_block_calls']+=1
   f.install_cache(block);x=new(*args,**kw,debug=True,audit=True);r['full_block_calls']+=1
   # Same-input affine LN gives an independent boundary even if upstream differs.
   expected=block.norm2(x['residual1']);r['primitive_calls']+=1
   row=dict(case=length+'-B2-video',same_input_norm2=compare(expected.reshape(-1),x['norm2'].reshape(-1)),whole=compare(ref['output'],x['output']),stages={})
   for k in ('norm1','self_q_linear','residual1','norm2','cross_attention','residual2','norm3','ffn1','gelu','ffn2'):
    row['stages'][k]=compare(ref[k].reshape(-1),x[k].reshape(-1)) if k in ref else dict(status='REFERENCE_FIELD_NOT_SAVED')
   row['inputs_unchanged']=all(byte_equal(inputs[n],t) for n,t in tensors((args,kw)));row['RNG_unchanged']=byte_equal(cpu,torch.random.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state())
   row['resources']=x['resources'].tolist();r['rows'].append(row);save();del data,f,args,kw,inputs,ref,x,expected
 torch.cuda.synchronize();r['status']='COMPLETE_NORM_BOUNDARY_NOT_ADMISSION'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-t0;save()

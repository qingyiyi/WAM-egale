import json,os,time,traceback
from pathlib import Path
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode
from tools.replay import device_tree
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);t=time.time();r=dict(status='RUNNING',full_block_calls=0,rows=[])
def save():(B/'DEFAULT-TRACE-RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 block=load_block('cuda');new=WholeLayerBlock(block);work=Path(load_block.__code__.co_filename).resolve().parents[2];video=work/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
 with torch.inference_mode():
  for kind,batch in [('action',2),('action',14),('video',2)]:
   for length in ['short','middle','long']:
    assert time.time()-t<250
    p=(FIXTURES if kind=='action' else video)/f'input-{length}-B{batch}-{kind}-layer0-first.pt';d=decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False));f=BlockFixture(p,device_tree(d));args,kw=f.arguments();f.install_cache(block)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
     r['full_block_calls']+=1;z=new(*args,**kw,audit=False);torch.cuda.synchronize()
    dest=B/f'default-{length}-B{batch}-{kind}.trace.json';prof.export_chrome_trace(str(dest));events=json.loads(dest.read_text())['traceEvents'];kernels=[x['name'] for x in events if x.get('cat')=='kernel'];r['rows'].append(dict(fixture=str(p),kernels=kernels,default_tile_pipeline=new.tile_pipeline,pass_single_kernel=len(kernels)==1));save()
    if len(kernels)!=1:raise RuntimeError('default launch contract')
    del f,args,kw,z,d
 r['status']='COMPLETE_DEFAULT_NINE_SINGLE_KERNEL_TRACES'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r.update(end=time.time(),lifecycle_seconds=time.time()-t);save()

from pathlib import Path
import json,os,time,traceback
import torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture
from tools.replay import device_tree
from wam_ops.optimized.native import NativeBlock
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);t=time.time();r=dict(full_block_calls=0,primitive_calls=0,status='RUNNING')
try:
 with torch.inference_mode():
  block=load_block('cuda');f=BlockFixture.load(FIXTURES,'short',2);f=BlockFixture(f.path,device_tree(f.data));f.install_cache(block);a,k=f.arguments();d=NativeBlock(block)(*a,**k,debug=True);r['full_block_calls']+=1
  x=d['modulated1'];block.attn1.to_q(x);r['primitive_calls']+=1;torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:y=block.attn1.to_q(x);torch.cuda.synchronize();r['primitive_calls']+=1
  p.export_chrome_trace(str(B/'Q-TRACE.json'));events=json.loads((B/'Q-TRACE.json').read_text())['traceEvents'];r['kernels']=[dict(name=e['name'],duration_us=e['dur']) for e in events if e.get('cat')=='kernel'];r['status']='COMPLETE_ACTUAL_DEFAULT_Q_KERNEL_IDENTIFICATION'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-t;(B/'Q-PROFILE-RESULT.json').write_text(json.dumps(r,indent=2)+'\n')

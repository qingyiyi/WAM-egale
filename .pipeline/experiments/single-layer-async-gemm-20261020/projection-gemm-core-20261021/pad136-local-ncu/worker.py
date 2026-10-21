"""真实B14上下文投影的Torch/当前8warp硬件计数；只作profile，不作速度表。"""
from pathlib import Path
import os,json,time,importlib.util,subprocess,traceback
os.sched_setaffinity(0,{4,5})
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode
from tools.replay import device_tree
B=Path(__file__).resolve().parent;start=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(1)
out=dict(status='RUNNING',scopes=[],formal_performance_permission=False,boundary='hot real projection; NCU kernel replay, no cache/clock control; distinct from remote timing')
try:
 sp=importlib.util.spec_from_file_location('wam_projection_schedule_cuda',next(B.glob('wam_projection_schedule_cuda*.so')));own=importlib.util.module_from_spec(sp);sp.loader.exec_module(own);out['resources']=own.setup()
 block=load_block('cuda');p=FIXTURES/'input-middle-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));args,kw=f.arguments();x=args[1]
 q=torch.zeros(1,device='cuda',dtype=torch.int32);empty=torch.empty(0,device='cuda',dtype=torch.int32)
 with torch.inference_mode():
  for name,linear in [('K',block.attn2.to_k),('V',block.attn2.to_v)]:
   w,b=linear.weight.detach(),linear.bias.detach();expected=torch.nn.functional.linear(x,w,b);y=torch.empty_like(expected)
   own.run(x,w,b,y,q,0,empty,empty);torch.cuda.synchronize();assert torch.equal(y,expected),(name,'gate')
   for label,fn in [('torch_'+name,lambda:torch.nn.functional.linear(x,w,b)),('own_'+name,lambda:own.run(x,w,b,y,q,0,empty,empty))]:
    for _ in range(12):z=fn()
    torch.cuda.synchronize();assert torch.equal(z if label.startswith('torch') else y,expected)
    before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert all(int(pid)==os.getpid() for pid in before.splitlines() if pid.strip()),before
    torch.cuda.nvtx.range_push('measure_'+label);z=fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    out['scopes'].append(dict(name=label,exact_before_and_after=torch.equal(z if label.startswith('torch') else y,expected),input_shape=list(x.shape),weight_shape=list(w.shape)))
 out.update(status='COMPLETE_FINITE_PROFILE',fixture=str(p),targeted_launches=4)
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;(B/'WORKER-RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['status'],out.get('error'),flush=True)
if out['status']!='COMPLETE_FINITE_PROFILE':raise SystemExit(1)

"""Real middle fixture, active full layer and same-source GEMM stages. NCU only."""
from pathlib import Path
import os,json,importlib.util,sys,hashlib,time
os.sched_setaffinity(0,{4,5})
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;P=B.parent/'ffn1-tail-control-20261021/profile';N=B.parent/'minor-ops-20261021/stage-current'
torch.set_num_threads(2);torch.set_num_interop_threads(1)
def ext(name,path):
 p=next(path.glob(name+'*.so'));s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
names=['modulation','norm1','modulated1','self_q_linear','self_k_linear','self_v','self_q_norm','self_k_norm','self_q_rope','self_k_rope','self_attention','self_out','residual1','norm2','cross_q_linear','cross_k_linear','cross_v','cross_q','cross_k','cross_attention','cross_out','residual2','norm3','modulated3','ffn1','gelu','ffn2','output']
start=time.time();out={'status':'RUNNING','kind':'NCU_DIAGNOSTIC_NOT_TIMING','scopes':[]}
with torch.inference_mode():
 block=load_block('cuda');native=NativeBlock(block);owned=WholeLayerBlock(block)
 path=FIXTURES/'input-middle-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
 args,kw=f.arguments();cache=f.install_cache(block);ref=native(*args,**kw,debug=True);f.install_cache(block);v=owned(*args,**kw,debug=True);torch.cuda.synchronize()
 assert all(torch.equal(ref[k].reshape(-1),v[k].reshape(-1)) for k in ref.keys()&v.keys())
 cache=block.attn1.attn_caches[kw['cache_name']];snap=visible_snapshot(cache);cl=[cache[k] for k in ('k','v','id','mask','is_pred')];a=[v[k] for k in names];meta=[v['logical_slots'],v['cache_info']];valid=int(meta[1][0])
 old=ext('wam_stage_native_cuda',N);ours=ext('wam_stage_resource_cuda',P)
 common=(owned.weights,*args,cl,kw['candidate_cache_rows'],kw.get('update_cache',0),owned.eps)
 nc=(owned.weights,*args,a,cl,kw['candidate_cache_rows'],meta[0],valid,owned.eps)
 work=[('full_owned',lambda:owned(*args,**kw,debug=True),'output')]
 for stage,key in [(16,'cross_k_linear'),(25,'ffn1'),(27,'ffn2')]:
  work.extend([(f'native_{stage}',lambda st=stage:old.stage(st,*nc),key),(f'owned_{stage}',lambda st=stage:ours.forward(*common,st,a,meta),key)])
 for label,fn,key in work:
  for _ in range(12):value=fn()
  torch.cuda.synchronize();expected=v[key];actual=value[key] if isinstance(value,dict) else value
  assert torch.equal(expected.reshape(-1),actual.reshape(-1)),label
  print('BEGIN_NCU_SCOPE',label,flush=True);torch.cuda.nvtx.range_push('measure_'+label)
  value=fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop();print('END_NCU_SCOPE',label,flush=True)
  out['scopes'].append(dict(label=label,finite_output_exact=True))
 assert state_equal(snap,visible_snapshot(cache))['pass_all']
 out.update(status='COMPLETE_NCU_REAL_FIXTURE',lifecycle_seconds=time.time()-start,fixture=str(path),profiled_calls=len(work),all_calls=2+13*len(work))
(B/'NCU-WORKER-RESULT.json').write_text(json.dumps(out,indent=2)+'\n')

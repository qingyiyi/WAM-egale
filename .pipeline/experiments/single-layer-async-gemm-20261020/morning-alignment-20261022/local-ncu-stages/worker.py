from pathlib import Path
import os,json,importlib.util,time
os.sched_setaffinity(0,{4,5})
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
import wam_ops.optimized.whole_layer as wl
B=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_num_interop_threads(1)
p=next(B.glob('wam_whole_fa2_cuda*.so'));assert __import__('hashlib').sha256(p.read_bytes()).hexdigest()==json.loads((B/'JOB.json').read_text())['module_sha256'];sp=importlib.util.spec_from_file_location('selected.wam_whole_fa2_cuda',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);wl.extension=lambda:m
import wam_ops.optimized.native as native_module
ref=next(B.glob('wam_optimized_block_cuda*.so'))
assert __import__('hashlib').sha256(ref.read_bytes()).hexdigest()==json.loads((B/'REFERENCE-TRANSFER.json').read_text())['sha256']
sp=importlib.util.spec_from_file_location('reference.wam_optimized_block_cuda',ref);rm=importlib.util.module_from_spec(sp);sp.loader.exec_module(rm)
assert getattr(rm,'persistent_scheduler',None)=='stage_local_static_stride_CTA'
assert torch.version.git_version==native_module.TORCH_COMMIT
native_module.extension=lambda:rm
def ext(name):
 p=next(B.glob(name+'*.so'));entry=next(x for x in json.loads((B/'DIAGNOSTIC-BINARIES.json').read_text()) if x['name']==p.name)
 assert __import__('hashlib').sha256(p.read_bytes()).hexdigest()==entry['sha256']
 spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
stage,orig=ext('wam_stage_resource_cuda'),ext('wam_stage_native_cuda')
names=['modulation','norm1','modulated1','self_q_linear','self_k_linear','self_v','self_q_norm','self_k_norm','self_q_rope','self_k_rope','self_attention','self_out','residual1','norm2','cross_q_linear','cross_k_linear','cross_v','cross_q','cross_k','cross_attention','cross_out','residual2','norm3','modulated3','ffn1','gelu','ffn2','output']
selected=[(3,'self_q_linear'),(11,'self_attention'),(19,'cross_k'),(25,'ffn1'),(26,'gelu'),(27,'ffn2')]
start=time.time();out=dict(status='RUNNING',kind='NCU_ISOLATED_STAGE_DIAGNOSTIC_NOT_EMBEDDED_TIMING',scopes=[])
with torch.inference_mode():
 block=load_block('cuda');native=NativeBlock(block);own=WholeLayerBlock(block)
 p=FIXTURES/'input-middle-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments()
 f.install_cache(block);n=native(*a,**k,debug=True);f.install_cache(block);v=own(*a,**k,debug=True);torch.cuda.synchronize()
 assert all(torch.equal(n[t].reshape(-1),v[t].reshape(-1)) for t in n.keys()&v.keys())
 cache=block.attn1.attn_caches[k['cache_name']];snap=visible_snapshot(cache);cl=[cache[x] for x in ('k','v','id','mask','is_pred')];common=(own.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),own.eps)
 aa=[v[x] for x in names];meta=[v['logical_slots'],v['cache_info']];valid=int(meta[1][0].item());work=[]
 for sid,key in selected:
  work.append(('own_'+key,lambda sid=sid:stage.forward(*common,sid,aa,meta),key))
  if sid!=11:work.append(('native_'+key,lambda sid=sid:orig.stage(sid,own.weights,*a,aa,cl,k['candidate_cache_rows'],meta[0],valid,own.eps),key))
 for label,fn,key in work:
  for _ in range(4):z=fn()
  torch.cuda.synchronize();actual=z[key] if isinstance(z,dict) else z
  assert torch.equal(v[key].reshape(-1),actual.reshape(-1)),label
  torch.cuda.nvtx.range_push('measure_'+label);z=fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop();out['scopes'].append(dict(label=label,output_exact=True))
 assert state_equal(snap,visible_snapshot(cache))['pass_all'];out.update(status='COMPLETE_NCU_FINITE_STAGES',profiled_calls=len(work),full_and_stage_calls=2+5*len(work),resources=v['resources'].tolist(),lifecycle_seconds=time.time()-start)
(B/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n')

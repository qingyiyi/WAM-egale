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
start=time.time();out=dict(status='RUNNING',kind='NCU_DIAGNOSTIC_NOT_TIMING',scopes=[])
with torch.inference_mode():
 block=load_block('cuda');native=NativeBlock(block);own=WholeLayerBlock(block);p=FIXTURES/'input-middle-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))))
 a,k=f.arguments();f.install_cache(block);n=native(*a,**k,debug=True);f.install_cache(block);v=own(*a,**k,debug=True);torch.cuda.synchronize();assert all(torch.equal(n[t].reshape(-1),v[t].reshape(-1)) for t in n.keys()&v.keys());snap=visible_snapshot(block.attn1.attn_caches[k['cache_name']])
 work=[('whole_final',lambda:own(*a,**k,debug=True),v['output']),('native_selfq',lambda:torch.nn.functional.linear(v['modulated1'],own.weights[3],own.weights[4]),v['self_q_linear'])]
 work += [('native_ffn1',lambda:torch.nn.functional.linear(v['modulated3'],own.weights[23],own.weights[24]),v['ffn1']),('native_ffn2',lambda:torch.nn.functional.linear(v['gelu'],own.weights[25],own.weights[26]),v['ffn2'])]
 for label,fn,expected in work:
  for _ in range(12):z=fn()
  torch.cuda.synchronize();actual=z['output'] if isinstance(z,dict) else z;assert torch.equal(expected.reshape(-1),actual.reshape(-1)),label
  torch.cuda.nvtx.range_push('measure_'+label);z=fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop();out['scopes'].append(dict(label=label,output_exact=True))
 assert state_equal(snap,visible_snapshot(block.attn1.attn_caches[k['cache_name']]))['pass_all'];out.update(status='COMPLETE_NCU_FINITE_REAL_INPUT',profiled_calls=len(work),full_and_stage_calls=2+13*len(work),resources=v['resources'].tolist(),lifecycle_seconds=time.time()-start)
(B/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n')

from pathlib import Path
import os,json,importlib.util,torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import state_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;j=json.loads((B/'JOB.json').read_text());torch.set_num_threads(2);torch.set_num_interop_threads(1);os.sched_setaffinity(0,{4,5})
def load(p,name):
 s=importlib.util.spec_from_file_location(name+'.wam_whole_fa2_cuda',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
mods={k:load(v,k) for k,v in j['binaries'].items()}
block=load_block('cuda');bind=WholeLayerBlock(block)
with torch.inference_mode():
 f=BlockFixture(FIXTURES/'input-middle-B14-action-layer0-first.pt',device_tree(decode(torch.load(FIXTURES/'input-middle-B14-action-layer0-first.pt',map_location='cpu',mmap=True,weights_only=False))))
 a,k=f.arguments();f.install_cache(block);ref=block(*a,**k);f.install_cache(block);c=block.attn1.attn_caches[k['cache_name']];snap=visible_snapshot(c);cl=[c[x] for x in ('k','v','id','mask','is_pred')];bind.prepare_context(a[1],precompute=True)
 args=(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps,False,bind.context_rows,bind.context_roots,bind.context_kv)
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();inputs=[v.clone() for v in a]
 for m in mods.values():assert torch.equal(m.forward(*args)['output'],ref)
 for _ in range(100):
  for m in mods.values():m.forward(*args)
 torch.cuda.synchronize()
 for name,m in mods.items():
  torch.cuda.nvtx.range_push(name);m.forward(*args);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
 assert state_equal(snap,visible_snapshot(c))['pass_all'] and torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(a,inputs))
 (B/'FINITE-STATE.json').write_text(json.dumps(dict(output_bitwise=True,cache_unchanged=True,rng_unchanged=True,inputs_unchanged=True,scope='one real B14 middle whole layer',formal_performance_permission=False),indent=2))

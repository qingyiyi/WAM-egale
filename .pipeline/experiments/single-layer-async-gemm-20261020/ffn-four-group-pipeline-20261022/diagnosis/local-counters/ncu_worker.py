"""One real middle-KV FFN input; counter collection only, no reported timing."""
from pathlib import Path
import sys,os,importlib.util,torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode
from tools.replay import device_tree
from wam_ops.optimized.whole_layer import WholeLayerBlock,extension
B=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_num_interop_threads(1);os.sched_setaffinity(0,{4,5})
spec=importlib.util.spec_from_file_location('wam_ffn_diagnostic_cuda',next(B.glob('wam_ffn_diagnostic_cuda*.so')));diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
block=load_block('cuda');bind=WholeLayerBlock(block)
with torch.inference_mode():
 f=BlockFixture(FIXTURES/'input-middle-B14-action-layer0-first.pt',device_tree(decode(torch.load(FIXTURES/'input-middle-B14-action-layer0-first.pt',map_location='cpu',mmap=True,weights_only=False))))
 a,k=f.arguments();f.install_cache(block);cache=block.attn1.attn_caches[k['cache_name']];cl=[cache[x] for x in ('k','v','id','mask','is_pred')];bind.prepare_context(a[1],precompute=True)
 args=(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps);extra=(bind.context_rows,bind.context_roots,bind.context_kv)
 y=extension().forward(*args,True,*extra);mod=y['modulated3'].view_as(a[0]);res=y['residual2'].view_as(a[0]);gate=(block.scale_shift_table[None]+a[2].float()).select(2,5)
 def ours():return diag.forward(*args,False,*extra,False,[mod,res])
 def original():return (res.float()+block.ffn(mod).float()*gate).to(mod.dtype)
 assert torch.equal(ours()["output"],original()),"local finite FFN output gate"
 for _ in range(100):ours();original()
 torch.cuda.synchronize()
 for name,fn in [('CUSTOM_FFN',ours),('TORCH_FFN',original)]:
  torch.cuda.nvtx.range_push(name);fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop()

"""NCU-only three real GEMMs; no speed measurement or synthetic input."""
import json,os,sys
from pathlib import Path
OWN=Path(__file__).resolve().parent;ROOT=OWN.parents[2]
def main():
 os.sched_setaffinity(0,{1,2})
 import torch
 from safetensors import safe_open
 torch.set_num_threads(2)
 obs=ROOT/'投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/observation/layer00-tensors.pt'
 model=ROOT/'models/lingbot-va-posttrain-libero-long-modelscope-20260922/transformer'
 payload=torch.load(obs,map_location='cpu',weights_only=True)
 x=payload['original_out'].cuda().flatten(2,3)
 wm=json.loads((model/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map'];params=[]
 for s in ('weight','bias'):
  key='blocks.0.attn1.to_out.0.'+s
  with safe_open(str(model/wm[key]),framework='pt',device='cpu') as f:params.append(f.get_tensor(key).cuda())
 rows=[]
 for n in (1,3,7):
  xs=x[list(range(n))+list(range(7,7+n))].contiguous()
  for _ in range(2):torch.nn.functional.linear(xs,*params)
  torch.cuda.synchronize();torch.cuda.nvtx.range_push('SHAPE_PROFILE')
  y=torch.nn.functional.linear(xs,*params)
  torch.cuda.nvtx.range_pop();torch.cuda.synchronize()
  rows.append(dict(n_v=n,shape=list(xs.shape),MNK=[32*n,3072,3072],finite=bool(torch.isfinite(y).all())))
 Path(sys.argv[1]).write_text(json.dumps(dict(status='COMPLETE_REAL_GEMM_PROFILE_INPUTS',rows=rows,source=str(obs),timing_is_profiler_only=True),indent=2)+'\n')
if __name__=='__main__':main()

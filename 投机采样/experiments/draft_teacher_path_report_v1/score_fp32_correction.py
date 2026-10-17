"""Correct ONLY CPU-derived scores to cast each BF16 operand before subtraction.

The original model/GPU scores are never rewritten. V1 CPU analysis is retained.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from draft_teacher_path_diagnostic_v1.protocol import read,write
import torch
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
r=ROOT/'投机采样/results/runtime';a=r/'p0-draft-teacher-path-diagnostic-v1-20261016-r1';b=r/'p0-draft-timestep-causal-v1-20261016-r2'
x=read(b/'CPU-ROOT-ANALYSIS.json');p=read(a/'plan.json')
s=torch.load(a/'REAL-STAGES.pt',map_location='cpu',weights_only=True,mmap=True)
tf=torch.load(p['arms']['TF_CONT']['trace'],map_location='cpu',weights_only=False)
first=next(z for z in tf['rounds'] if z['kind']=='speculative_tree')
base=s['outputs']['T'][0].T[:7].reshape(112).float()
packed=s['outputs']['TF_CONT'][0].T[:7].reshape(112).float()
delta=abs(first['sigma_path'][1]-first['sigma_path'][0])
x['CPU_root_first_costs_B2']=[float(delta*torch.linalg.vector_norm(z.float()-base)) for z in first['first_directions']]
x['CPU_root_first_costs_B14']=[float(delta*torch.linalg.vector_norm(z.float()-packed)) for z in first['first_directions']]
x['teacher_numeric_difference_local_norm']=float(delta*torch.linalg.vector_norm(base-packed))
x['score_arithmetic']='CPU: cast both operands to FP32, subtract in FP32, norm FP32; not new production calibration'
x['supersedes']='CPU-ROOT-ANALYSIS.json CPU-derived three norm fields only; BF16 subtraction was used in V1. Other dtype/difference/guard facts unchanged.'
x['CPU_B14_cost_minus_recorded_GPU']=[c-v for c,v in zip(x['CPU_root_first_costs_B14'],x['recorded_first_costs'])]
write(b/'CPU-ROOT-ANALYSIS-v2.json',x)
write(b/'CPU-ANALYSIS-v1-SUPERSEDED.json',dict(status='V1_CPU_SCORE_ARITHMETIC_SUPERSEDED',
    replacement='CPU-ROOT-ANALYSIS-v2.json',old_file_preserved=True,old_GPU_scores_unchanged=True,
    correction='cast each BF16 direction to FP32 before subtraction; no acceptance threshold or quality field changes',CUDA_initialized=False))
import json
print(json.dumps({k:x[k] for k in ('CPU_root_first_costs_B2','CPU_root_first_costs_B14','CPU_B14_cost_minus_recorded_GPU')},ensure_ascii=False))

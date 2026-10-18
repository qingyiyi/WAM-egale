"""CPU-only independent saved full-output comparison; no model hashes."""
from pathlib import Path
import os,json
os.sched_setaffinity(0,{4,5})
import torch
import numpy as np
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[3]
job=ROOT/'投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r2/process1'
result=json.loads((job/'RESULT.json').read_text())
rows=[]
def eq(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(eq(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b
for query in result['complete_model_queries']:
    label,index=query['length'],query['query']
    gold=torch.load(job/f'{label}-q{index}-torch-complete-output.pt',map_location='cpu',weights_only=False)
    for engine in ['torch','standalone','shared','guarded']:
        v=torch.load(job/f'{label}-q{index}-{engine}-complete-output.pt',map_location='cpu',weights_only=False)
        rows.append(dict(scope='full30 raw output',length=label,query=index,engine=engine,byte_equal=eq(v,gold),GPU_record_agrees=eq(v,gold)==query['engines'][engine]['raw']['byte_equal']))
for case in result['complete_policy_replays']:
    label=case['length'];a=torch.load(job/f'{label}-torch-full-policy-return.pt',map_location='cpu',weights_only=False);b=torch.load(job/f'{label}-aligned-full-policy-return.pt',map_location='cpu',weights_only=False)
    rows.append(dict(scope='complete frozen policy return',length=label,byte_equal=eq(a,b),GPU_record_agrees=eq(a,b)==case['output_byte_equal']))
out=dict(status='COMPLETE_CPU_INDEPENDENT_SAVED_OUTPUT_AUDIT',rows=rows,all_equal=all(x['byte_equal'] for x in rows),all_GPU_records_agree=all(x['GPU_record_agrees'] for x in rows),CUDA_initialized=torch.cuda.is_initialized(),state_KV_audit_scope='live GPU/CPU tree comparison performed in worker; full KV not duplicated for offline audit')
assert not out['CUDA_initialized']
(Path(__file__).parent/'CPU-SAVED-OUTPUT-AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),all_equal=out['all_equal'],all_GPU_records_agree=out['all_GPU_records_agree'],CUDA_initialized=out['CUDA_initialized'])))

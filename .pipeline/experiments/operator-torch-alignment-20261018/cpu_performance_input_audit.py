"""CPU-only identity check of new timing inputs against already validated inputs."""
from pathlib import Path
import os,json
os.sched_setaffinity(0,{6,7})
import torch
import numpy as np
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[3]
RUNTIME=ROOT/'投机采样/results/runtime'
new=RUNTIME/'p0-operator-torch-alignment-v2-20261018-r3/process1'
old=RUNTIME/'p0-operator-kv-matrix-local-v1-20261018-r6/process1'
def eq(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(eq(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b
rows=[]
for path in sorted(new.glob('input-*.pt')):
    x=torch.load(path,map_location='cpu',weights_only=False,mmap=True);y=torch.load(old/path.name,map_location='cpu',weights_only=False,mmap=True)
    rows.append(dict(scope='all encoded real input/gold tensors and metadata',name=path.name,all_bytes_equal=eq(x,y)))
for label,indices in [('short',[0,18]),('middle',[0,20]),('long',[0,20])]:
    fixture=torch.load(RUNTIME/f'p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4/{label}/real-verifier-fixture.pt',map_location='cpu',weights_only=False,mmap=True)
    for query in fixture['queries']:
        path=new/f'{label}-query{query["index"]}-original-output.pt'
        rows.append(dict(scope='new timing capture full30 Torch oracle vs frozen original',name=path.name,all_bytes_equal=eq(torch.load(path,map_location='cpu',weights_only=False),query['output'])))
result=dict(status='COMPLETE_CPU_TIMING_INPUT_IDENTITY_AUDIT',rows=rows,all_equal=all(r['all_bytes_equal'] for r in rows),CUDA_initialized=torch.cuda.is_initialized(),scope='9 representative performance inputs +6 original complete outputs; not all54 intermediate input tensors saved, not timed engine output revalidation')
assert len(rows)==15 and result['all_equal'] and not result['CUDA_initialized']
(Path(__file__).parent/'performance/CPU-INPUT-IDENTITY-AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(checks=len(rows),all_equal=result['all_equal'],CUDA_initialized=result['CUDA_initialized'])))

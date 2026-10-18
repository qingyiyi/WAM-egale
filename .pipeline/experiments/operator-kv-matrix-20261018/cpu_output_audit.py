"""Independent CPU audit of the24 saved full-model raw outputs; no model/CUDA/hash."""
from pathlib import Path
import json
import math
import os

ROOT=Path('/rjs/cxz/论文探索')
OUT=Path(__file__).resolve().parent


def main():
    os.sched_setaffinity(0,{7})
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    rt=ROOT/'投机采样/results/runtime'
    job=rt/'p0-operator-kv-validation-shared-v1-20261018-r8'
    gpu=json.loads((job/'process1/RESULT.json').read_text())
    fixtures=rt/'p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4'
    rows=[];issues=[]
    for label in ('short','middle','long'):
        reference=torch.load(fixtures/label/'real-verifier-fixture.pt',map_location='cpu',weights_only=False,mmap=True)
        for query in reference['queries']:
            gold=query['output'].contiguous()
            recorded=next(q for q in gpu['complete_model_queries'] if q['length']==label and q['query']==query['index'])
            for engine in ('torch','standalone','shared','guarded'):
                path=job/f'process1/{label}-q{query["index"]}-{engine}-complete-output.pt'
                output=torch.load(path,map_location='cpu',weights_only=False).contiguous()
                diff=output.float()-gold.float()
                data=dict(length=label,query=query['index'],engine=engine,
                    byte_equal=bool(torch.equal(output.view(torch.uint8),gold.view(torch.uint8))),
                    max_abs=float(diff.abs().max()),rel_l2=float(diff.norm()/gold.float().norm().clamp_min(1e-30)),
                    allclose_02=bool(torch.allclose(output,gold,atol=.02,rtol=.02)),source=str(path))
                expected=recorded['engines'][engine]['raw']
                same=(data['byte_equal']==expected['byte_equal'] and data['allclose_02']==expected['allclose_02']
                    and data['max_abs']==expected['max_abs']
                    and math.isclose(data['rel_l2'],expected['rel_l2'],rel_tol=2e-6,abs_tol=1e-9))
                data['GPU_record_consistent']=same;rows.append(data)
                if not same:issues.append(dict(case=[label,query['index'],engine],CPU=data,GPU=expected))
    result=dict(status='CPU24_OUTPUT_AUDIT_COMPLETE' if not issues else 'CPU_OUTPUT_AUDIT_MISMATCH',
        rows=rows,issues=issues,CUDA_initialized=torch.cuda.is_initialized(),model_loaded=False,new_hashes=False,
        scope='saved full model raw outputs only; no CPU execution of WAM, no production accept calibration')
    assert not result['CUDA_initialized']
    (OUT/'CPU-SAVED-OUTPUT-AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],outputs=len(rows),issues=issues,CUDA_initialized=False),ensure_ascii=False))


if __name__=='__main__':main()

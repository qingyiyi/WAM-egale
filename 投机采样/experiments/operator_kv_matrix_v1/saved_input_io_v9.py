"""Replay the saved performance inputs; check output and all passed input bytes, no timing."""
from pathlib import Path
import ast
import json
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    job=Path(sys.argv[1]).resolve(strict=True);plan=json.loads((job/'REGISTRATION.json').read_text())
    os.sched_setaffinity(0,{4,5});started=time.monotonic()
    import torch
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20,0)
    old=ROOT/'wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/operator_lab/wam_transformer_tile'
    sys.path.insert(0,str(old));from wam_transformer_tile import torch_sdpa,cuda_sdpa_standalone
    sys.path.insert(0,str(ROOT/'投机采样/experiments/shared_prefix_torch_v2'))
    sys.path.insert(0,str(ROOT/'投机采样/experiments/wam_candidate_tile_v1'))
    import _shared_torch_v1 as native
    import _wam_candidate_tile as grouped
    tree=ast.parse((BASE/'worker_r5.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    fn=[n for n in run.body if isinstance(n,ast.FunctionDef) and n.name in ('bytes_equal','error')]
    namespace={'torch':torch};exec(compile(ast.Module(body=fn,type_ignores=[]),'<existing-output-error-metrics-no-model-import>','exec'),namespace)
    equal,metrics=namespace['bytes_equal'],namespace['error']
    def decode(x):
        if isinstance(x,dict) and 'encoding' in x:
            value=x['value']
            return value.repeat_interleave(7,dim=0) if x['encoding']=='two_CFG_groups_repeated7' else value
        if isinstance(x,dict):return {k:decode(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return type(x)(decode(v) for v in x)
        return x
    index=json.loads(Path(plan['saved_inputs_manifest']).read_text())
    outcome=dict(status='RUNNING_SAVED_INPUT_IO',timing_performed=False,rows=[],CUDA_memory_fraction_cap=.20)
    try:
        for item in index['rows']:
            assert time.monotonic()-started<plan['worker_active_cap_seconds']
            saved=decode(torch.load(item['path'],map_location='cpu',weights_only=False,mmap=True))
            q=saved['self_q'].cuda().contiguous();gold=saved['self_gold'].cuda().contiguous()
            tensors={'q':q};batch=q.shape[0]
            if batch==14:
                slots=saved['history_slots'].cuda();shape=saved['pool_shape']
                hk=torch.empty(shape,device='cuda',dtype=q.dtype);hv=torch.empty_like(hk)
                hk[:,slots]=saved['history_k'].cuda();hv[:,slots]=saved['history_v'].cuda()
                ck=saved['private_k'].cuda().contiguous();cv=saved['private_v'].cuda().contiguous()
                mapping=saved['candidate_cache_rows'].cuda()
                k=torch.cat((hk[:,slots].index_select(0,mapping),ck),dim=1)
                v=torch.cat((hv[:,slots].index_select(0,mapping),cv),dim=1)
                tensors.update(history_k_valid=hk[:,slots],history_v_valid=hv[:,slots],
                    private_k=ck,private_v=cv,slots=slots,mapping=mapping)
            else:k=saved['self_k'].cuda().contiguous();v=saved['self_v'].cuda().contiguous()
            tensors.update(k=k,v=v)
            reference={name:value.clone() for name,value in tensors.items()}
            descriptors={name:(tuple(t.shape),tuple(t.stride()),t.dtype,t.device,t.data_ptr()) for name,t in tensors.items()}
            functions=dict(torch=lambda:torch_sdpa(q,k,v),standalone=lambda:cuda_sdpa_standalone(q,k,v))
            if batch==14:
                group=2 if (len(slots)+32+63)//64==(len(slots)+16+63)//64 else 1
                functions.update(shared=lambda:native.shared(q,hk,hv,ck,cv,slots,mapping),
                    G1=lambda:grouped.grouped(q,hk,hv,ck,cv,slots,1),
                    guarded=lambda:grouped.grouped(q,hk,hv,ck,cv,slots,group))
            engines={}
            before_cpu=torch.get_rng_state();before_cuda=torch.cuda.get_rng_state()
            for name,function in functions.items():
                assert time.monotonic()-started<plan['worker_active_cap_seconds']
                value=function();stats=metrics(value,gold)
                unchanged=all(equal(tensors[n],ref) and descriptors[n]==(tuple(tensors[n].shape),tuple(tensors[n].stride()),tensors[n].dtype,tensors[n].device,tensors[n].data_ptr()) for n,ref in reference.items())
                if batch==14:
                    unchanged=unchanged and equal(hk[:,slots],reference['history_k_valid']) and equal(hv[:,slots],reference['history_v_valid'])
                engines[name]=dict(output=stats,all_passed_input_bytes_and_descriptors_unchanged=unchanged,
                    shape_dtype_device_correct=value.shape==gold.shape and value.dtype==gold.dtype and value.device==gold.device)
            unchanged_rng=torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
            outcome['rows'].append(dict(source=item['path'],identity=saved['identity'],engines=engines,RNG_unchanged=unchanged_rng,
                invalid_uninitialized_pool_slots_compared=False))
            write(job/'RESULT.json',outcome)
            print('SAVED_INPUT_VALIDATED',len(outcome['rows']),saved['identity'],flush=True)
            del tensors,reference,descriptors,functions,k,v,q,gold,saved
            torch.cuda.empty_cache()
        assert len(outcome['rows'])==9
        outcome.update(status='COMPLETE_NINE_REAL_SAVED_INPUT_IO_VALIDATION',wall_seconds=time.monotonic()-started)
    except BaseException as exc:
        outcome.update(status='PARTIAL_SAVED_INPUT_IO',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started)
        raise
    finally:write(job/'RESULT.json',outcome)


if __name__=='__main__':main()

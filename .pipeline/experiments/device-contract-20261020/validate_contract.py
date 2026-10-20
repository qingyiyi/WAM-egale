"""Four real same-input GEMM contracts, finite, no tuning or robot episodes."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import time
import traceback

import torch
from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture
from tools.metrics import byte_equal, compare
from tools.replay import device_tree
from tools.resources import resource_for_gpu
from wam_ops.optimized.native import NativeBlock


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reg=json.loads(a.registration.read_text())
    if torch.version.git_version!='0fabc3ba44823f257e70ce397d989c8de5e362c1':
        raise RuntimeError('pinned Torch required before creating context')
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
    obs=resource_for_gpu(reg['GPU_uuid'])
    if obs['pids']:raise RuntimeError('GPU busy, no context created')
    a.output.mkdir(exist_ok=False);start=time.monotonic()
    result=dict(status='RUNNING',rows=[],full_block_calls=0,primitive_calls=0,
                full30=0,robot_episodes=0,resources=[obs],production_admitted=False)
    expected=None
    def save():
        q=a.output/'RESULT.pending';q.write_text(json.dumps(result,indent=2)+'\n')
        q.replace(a.output/'RESULT.json')
    def guard():
        nonlocal expected
        if time.time()>datetime.fromisoformat(reg['work_stop']).timestamp()-60:
            raise TimeoutError('work deadline and cleanup reserve')
        if time.monotonic()-start>600:raise TimeoutError('600-second worker cap')
        r=resource_for_gpu(reg['GPU_uuid']);result['resources'].append(r)
        current=set(r['pids'].splitlines())
        if expected is None:
            if len(current)!=1:raise RuntimeError('sole NVML context cannot be identified')
            expected=current
        if current!=expected:raise RuntimeError('foreign context appeared; no further calls')
    try:
        ext=importlib.import_module('wam_ops.optimized.wam_contract_cuda')
        block=load_block('cuda');base=NativeBlock(block);guard()
        with torch.inference_mode():
            for batch in (2,14):
                fixture=BlockFixture.load(FIXTURES,'short',batch)
                fixture=BlockFixture(fixture.path,device_tree(fixture.data));fixture.install_cache(block)
                args,kwargs=fixture.arguments()
                debug=base(*args,**kwargs,debug=True);result['full_block_calls']+=1
                torch.cuda.synchronize()
                for name,x,layer in [('Q',debug['modulated1'],block.attn1.to_q),
                                     ('FFN2',debug['gelu'],block.ffn.net[2])]:
                    guard();x=x.reshape(-1,x.shape[-1]).contiguous()
                    original=x.clone();w=layer.weight.clone();bias=layer.bias.clone()
                    cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state()
                    gold=layer(x);result['primitive_calls']+=1
                    new=ext.forward(x,layer.weight,layer.bias);result['primitive_calls']+=1
                    torch.cuda.synchronize()
                    # Repeat only establishes deterministic behavior of this fixed candidate.
                    again=ext.forward(x,layer.weight,layer.bias);result['primitive_calls']+=1
                    torch.cuda.synchronize()
                    result['rows'].append(dict(batch=batch,name=name,fixture=str(fixture.path),
                        M=x.size(0),N=layer.weight.size(0),K=x.size(1),
                        result=compare(gold,new),repeat=compare(new,again),
                        immutable=byte_equal(original,x) and byte_equal(w,layer.weight) and byte_equal(bias,layer.bias),
                        RNG_unchanged=byte_equal(cpu,torch.random.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state()),
                        teacher_input='frozen native debug, not old WMMA output'))
                    save()
                del debug,fixture,args,kwargs
        result['status']=('COMPLETE_FOUR_PRIMITIVE_CONTRACTS_PASS' if
            all(r['result']['bitwise_pass'] and r['immutable'] and r['RNG_unchanged']
                and r['repeat']['bitwise_pass'] for r in result['rows']) else
            'COMPLETE_FINITE_NUMERICAL_NO_GO_NO_ASYNC_INTEGRATION')
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result.update(lifecycle_seconds=time.monotonic()-start,
                      at_UTC=datetime.now(timezone.utc).isoformat());save()

if __name__=='__main__':main()

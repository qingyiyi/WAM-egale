"""真实完整block的kernel分解；独立profiler结果，不进入性能主表。"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import traceback

import torch
from torch.profiler import profile, ProfilerActivity

from .block_loader import FIXTURES, load_block
from .fixtures import BlockFixture, visible_snapshot
from .metrics import compare, state_equal
from .replay import device_tree, resource, write
from wam_ops.install import install_block


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();reg=json.loads(args.registration.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();obs=resource()
    result=dict(status='RUNNING',rows=[],resources=[obs],block_forward_intents=0,
        scope='Torch profiler CUDA kernel durations, separate from timing; no NCU or clock changes')
    def save():write(args.output/'RESULT.json',result)
    try:
        if obs['pids'] or obs['gpu'].split(',')[0]!=reg['physical_gpu_uuid']:
            raise RuntimeError('profile requires natural idle')
        block=load_block()
        with torch.inference_mode():
            for length,batch in (('middle',14),('long',14),('long',2)):
                f=BlockFixture.load(FIXTURES,length,batch);f=BlockFixture(f.path,device_tree(f.data))
                arguments,kwargs=f.arguments();cache=f.install_cache(block);oracle=state=None
                for engine in ('original','baseline','optimized'):
                    obs=resource();result['resources'].append(obs)
                    if set(obs['pids'].splitlines())-{str(os.getpid())}:
                        raise RuntimeError('foreign PID entered profile')
                    context=nullcontext() if engine=='original' else install_block(block,engine)
                    with context:
                        result['block_forward_intents']+=1;output=block(*arguments,**kwargs)
                        torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                            for _ in range(3):
                                result['block_forward_intents']+=1;output=block(*arguments,**kwargs)
                            torch.cuda.synchronize()
                    if oracle is None:oracle=output.clone();state=visible_snapshot(cache)
                    numeric=compare(oracle,output);kv=state_equal(state,visible_snapshot(cache))
                    aggregate=defaultdict(lambda:dict(calls=0,total_us=0.))
                    for event in prof.events():
                        if event.device_type==torch.autograd.DeviceType.CUDA:
                            entry=aggregate[event.name];entry['calls']+=1
                            entry['total_us']+=event.device_time_total
                    result['rows'].append(dict(case=f'{length}-B{batch}',engine=engine,
                        output=numeric,state=kv,kernels=[dict(name=name,**entry) for name,entry in aggregate.items()]))
                    save()
                    if not numeric['bitwise_pass'] or not kv['pass_all']:raise RuntimeError('profile output/state gate')
                block.attn1.attn_caches={}
        result['status']='FINITE_COMPLETE_BLOCK_KERNEL_PROFILE'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['lifecycle_seconds']=time.monotonic()-start;save()


if __name__=='__main__':main()

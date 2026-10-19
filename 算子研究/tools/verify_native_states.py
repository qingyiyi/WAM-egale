"""真实Q128及从真实cache连续合法write1/2/0导出的状态边界。"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch

from .block_loader import FIXTURES, load_block
from .fixtures import BlockFixture, decode, tensors, visible_snapshot
from .metrics import byte_equal, compare, state_equal
from .replay import device_tree, resource, write
from wam_ops.baseline.native import NativeBlock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video-fixtures',type=Path)
    p.add_argument('--only-kind',choices=('action','video'))
    p.add_argument('--optimized',action='store_true')
    args=p.parse_args();reg=json.loads(args.registration.read_text())
    if not reg.get('shared_correctness_authorized'):raise PermissionError('correctness sharing not registered')
    args.output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();obs=resource();fields=obs['gpu'].split(',')
    if fields[0].strip()!=reg['physical_gpu_uuid'] or float(fields[2])<4000:raise RuntimeError('GPU identity/free memory')
    result=dict(status='RUNNING',rows=[],resources=[obs],block_forward_intents=0,
        sharing='CORRECTNESS_ONLY_NO_PERFORMANCE',formal_performance_permission=False)
    def save():write(args.output/'RESULT.json',result)
    def check():
        if time.monotonic()-start>480:raise TimeoutError('state worker lifecycle')
        obs=resource();result['resources'].append(obs)
        if float(obs['gpu'].split(',')[2])<1500:raise RuntimeError('free memory reserve')
    try:
        block=load_block();native=NativeBlock(block,fused_ffn1=args.optimized)
        with torch.inference_mode():
            for kind in ('action','video'):
                if args.only_kind and args.only_kind!=kind:continue
                for length in ('short','middle','long'):
                    check()
                    root=args.video_fixtures if kind=='video' and args.video_fixtures else FIXTURES
                    path=root/f'input-{length}-B2-{kind}-layer0-first.pt'
                    raw=decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))
                    if raw['identity']!=(length,2,kind,'first',0):raise RuntimeError('fixture identity')
                    fixture=BlockFixture(path,device_tree(raw));arguments,kwargs=fixture.arguments()
                    inputs={k:x.clone() for k,x in tensors((arguments,kwargs))}
                    cache=fixture.install_cache(block)
                    result['block_forward_intents']+=1;out=block(*arguments,**kwargs)
                    oracle=out.clone();oracle_state=visible_snapshot(cache)
                    fixture.install_cache(block)
                    result['block_forward_intents']+=1;candidate=native(*arguments,**kwargs)
                    row=dict(case=f'{length}-B2-{kind}',scope='saved real read0',
                        historical=compare(fixture.data['block_gold'],oracle),output=compare(oracle,candidate),
                        state=state_equal(oracle_state,visible_snapshot(block.attn1.attn_caches[kwargs['cache_name']])))
                    row['passed']=row['historical']['bitwise_pass'] and row['output']['bitwise_pass'] and row['state']['pass_all']
                    result['rows'].append(row);save()
                    # Same saved query, evolved cache. This is a counterfactual
                    # API replay, not a newly observed robot/policy trajectory.
                    fixture.install_cache(block);gold=[]
                    cpu_rng,gpu_rng=torch.random.get_rng_state(),torch.cuda.get_rng_state()
                    for update in (1,2,0):
                        kw=dict(kwargs,update_cache=update)
                        result['block_forward_intents']+=1;out=block(*arguments,**kw)
                        gold.append((out.clone(),visible_snapshot(block.attn1.attn_caches[kwargs['cache_name']])) )
                    fixture.install_cache(block)
                    for index,update in enumerate((1,2,0)):
                        kw=dict(kwargs,update_cache=update)
                        result['block_forward_intents']+=1;candidate=native(*arguments,**kw)
                        output=compare(gold[index][0],candidate)
                        state=state_equal(gold[index][1],visible_snapshot(block.attn1.attn_caches[kwargs['cache_name']]))
                        row=dict(case=f'{length}-B2-{kind}-write-sequence-{index}-{update}',
                            scope='derived legal state evolution [1,2,0] from saved query',output=output,state=state,
                            inputs_unchanged=all(byte_equal(inputs[k],x) for k,x in tensors((arguments,kwargs))),
                            RNG_unchanged=byte_equal(cpu_rng,torch.random.get_rng_state()) and byte_equal(gpu_rng,torch.cuda.get_rng_state()))
                        row['passed']=output['bitwise_pass'] and state['pass_all'] and row['inputs_unchanged'] and row['RNG_unchanged']
                        result['rows'].append(row);save()
                    del fixture,raw,inputs,gold;block.attn1.attn_caches={}
        result['status']='ALL_SAVED_VIDEO_AND_DERIVED_STATE_GATES_PASS' if all(r['passed'] for r in result['rows']) else 'NUMERIC_OR_STATE_REJECTED'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(lifecycle_seconds=time.monotonic()-start);save()


if __name__=='__main__':main()

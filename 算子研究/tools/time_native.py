"""完整动态block同输入交错计时；包含真实KV管理，不是attention core。

original_A/B为同实现A/A对照；只允许自然空闲，不测试机器人性能。
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import torch

from .block_loader import FIXTURES, load_block
from .fixtures import BlockFixture, decode, tensors, visible_snapshot
from .metrics import byte_equal, compare, state_equal
from .replay import device_tree, resource, write
from wam_ops.install import install_block

ENGINES=('original_A','baseline','original_B','optimized')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video-fixtures',type=Path)
    p.add_argument('--wait-idle-seconds',type=int,default=0)
    args=p.parse_args();reg=json.loads(args.registration.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();block=load_block('cpu');obs=resource()
    fields=[x.strip() for x in obs['gpu'].split(',')]
    result=dict(status='ADMISSION',rows=[],resources=[obs],block_forward_intents=0,
        scope='complete dynamic native/original block incl KV pack/sync; no CUDA Graph, no robot or policy timing',
        rounds=7,calls_per_sample=5,warmups_per_engine=2,
        formal_performance_permission=False)
    def save():write(args.output/'RESULT.json',result)
    def guard():
        obs=resource();result['resources'].append(obs)
        if set(obs['pids'].splitlines())-{str(os.getpid())}:
            raise RuntimeError('FOREIGN_GPU_PID_STOP_TIMING')
        if time.monotonic()-start>480:
            raise TimeoutError('timing worker lifecycle')
        if time.time()>datetime.fromisoformat(reg['absolute_work_stop']).timestamp()-60:
            raise TimeoutError('absolute reserve')
    save()
    wait_stop=time.monotonic()+args.wait_idle_seconds
    while obs['pids'] and time.monotonic()<wait_stop:
        result['status']='WAITING_NO_CUDA_CONTEXT';save()
        time.sleep(min(5,max(0,wait_stop-time.monotonic())))
        obs=resource();fields=[x.strip() for x in obs['gpu'].split(',')]
        result['resources'].append(obs)
    if obs['pids'] or fields[0]!=reg['physical_gpu_uuid'] or float(fields[2])<4000:
        result['status']='GPU_BUSY_OR_IDENTITY_MISMATCH_NO_CONTEXT';save()
        raise RuntimeError(result['status'])
    try:
        block=block.to('cuda')
        cases=[(length,batch,'action',FIXTURES) for length in ('short','middle','long') for batch in (2,14)]
        if args.video_fixtures:
            cases += [(length,2,'video',args.video_fixtures) for length in ('short','middle','long')]
        with torch.inference_mode():
            for length,batch,kind,root in cases:
                guard()
                path=root/f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                raw=decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))
                fixture=BlockFixture(path,device_tree(raw));arguments,kwargs=fixture.arguments()
                cache=fixture.install_cache(block);initial=visible_snapshot(cache)
                inputs={k:x.clone() for k,x in tensors((arguments,kwargs))}
                cpu_rng,gpu_rng=torch.random.get_rng_state(),torch.cuda.get_rng_state()
                def restore():
                    for k in ('id','mask','is_pred'):cache[k].copy_(initial[k])
                    for k in ('k','v'):cache[k].index_copy_(1,initial['slots'],initial[k])
                def installation(engine):
                    return nullcontext() if engine.startswith('original_') else install_block(block,engine)
                def call():
                    result['block_forward_intents']+=1
                    return block(*arguments,**kwargs)
                oracle=oracle_state=None
                row=dict(case=f'{length}-B{batch}-{kind}',numeric=[],raw=[],protocol='7 rotated rounds;5 calls/sample;2 warmups;A/A')
                try:
                    row['hardware_before']=subprocess.check_output(['nvidia-smi',
                        '--query-gpu=clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu',
                        '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                except (OSError,subprocess.SubprocessError) as exc:
                    row['hardware_before']='UNKNOWN: '+repr(exc)
                result['rows'].append(row)
                for engine in ENGINES:
                    restore()
                    with installation(engine):out=call()
                    if oracle is None:oracle=out.clone();oracle_state=visible_snapshot(cache)
                    check=dict(engine=engine,output=compare(oracle,out),
                        historical=compare(fixture.data['block_gold'],out),
                        state=state_equal(oracle_state,visible_snapshot(cache)))
                    check['passed']=check['output']['bitwise_pass'] and check['historical']['bitwise_pass'] and check['state']['pass_all']
                    row['numeric'].append(check);save()
                    if not check['passed']:raise RuntimeError('pre timing numeric/state gate')
                    with installation(engine):
                        for _ in range(2):out=call()
                    if not byte_equal(oracle,out):raise RuntimeError('warmup output')
                for index in range(7):
                    guard()
                    order=ENGINES[index%4:]+ENGINES[:index%4]
                    for engine in order:
                        restore()
                        with installation(engine):
                            torch.cuda.synchronize()
                            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            wall=time.perf_counter();begin.record()
                            for _ in range(5):out=call()
                            end.record();end.synchronize()
                            wall_ms=(time.perf_counter()-wall)*1000/5
                            event_ms=begin.elapsed_time(end)/5
                        if not byte_equal(oracle,out):raise RuntimeError('timed output gate')
                        if not state_equal(oracle_state,visible_snapshot(cache))['pass_all']:
                            raise RuntimeError('timed valid KV/metadata gate')
                        row['raw'].append(dict(round=index,engine=engine,event_ms=event_ms,wall_ms=wall_ms));save()
                medians={e:statistics.median(x['event_ms'] for x in row['raw'] if x['engine']==e) for e in ENGINES}
                row['median_event_ms']=medians
                row['baseline_over_original']=medians['baseline']/medians['original_A']
                row['optimized_over_original']=medians['optimized']/medians['original_A']
                row['AA_ratio']=medians['original_B']/medians['original_A']
                row['inputs_unchanged']=all(byte_equal(inputs[k],x) for k,x in tensors((arguments,kwargs)))
                row['RNG_unchanged']=byte_equal(cpu_rng,torch.random.get_rng_state()) and byte_equal(gpu_rng,torch.cuda.get_rng_state())
                if not row['inputs_unchanged'] or not row['RNG_unchanged']:raise RuntimeError('timing immutable/RNG')
                save();del fixture,raw,inputs,initial,oracle,oracle_state;block.attn1.attn_caches={};gc.collect()
            guard();result['status']='COMPLETE_DYNAMIC_BLOCK_FINITE_PARITY_AND_TIMING'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(lifecycle_seconds=time.monotonic()-start);save()


if __name__=='__main__':main()

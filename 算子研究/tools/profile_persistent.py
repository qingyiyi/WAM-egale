"""Profile the same complete single block; never a robot/30-layer test."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['timeline','counters'],default='timeline')
    p.add_argument('--case')
    p.add_argument('--engine',choices=['tiled','persistent','tiled_ffn','persistent_ffn'])
    args=p.parse_args();reg=json.loads(args.registration.read_text())
    root=Path(reg['source_workspace']);sys.path.insert(0,str(root/'算子研究'))
    import torch
    from tools.block_loader import FIXTURES,load_block
    from tools.fixtures import BlockFixture,decode,visible_snapshot
    from tools.metrics import byte_equal,state_equal
    from tools.replay import device_tree,write
    from tools.resources import resource_for_gpu
    from wam_ops.install import install_block
    args.output.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(reg['CPU_allowed']));torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic()
    result=dict(status='RUNNING',kind=args.kind,rows=[],block_forward_intents=0,resources=[],timing='PROFILER_NOT_FORMAL_PERFORMANCE',GPU_clock_control='none',cache_control='none_for_NCU',math_source_changed=False)
    def save():write(args.output/'RESULT.json',result)
    def guard():
        if time.time()>datetime.fromisoformat(reg['absolute_work_stop']).timestamp()-60:raise TimeoutError('absolute stop reserve')
        if result['block_forward_intents']>=reg['block_per_process_cap']:raise RuntimeError('forward cap')
        obs=resource_for_gpu(reg['physical_gpu_uuid']);result['resources'].append(obs)
        if set(obs['pids'].splitlines())!=set(result.get('nvml_context_pids',[])):raise RuntimeError('GPU context set changed')
    try:
        guard();block=load_block();torch.cuda.synchronize()
        obs=resource_for_gpu(reg['physical_gpu_uuid']);ids=obs['pids'].splitlines()
        if len(ids)!=1 or torch.cuda.memory_allocated()==0:raise RuntimeError('sole new GPU context not established')
        result['nvml_context_pids']=ids;result['container_pid']=os.getpid();result['PID_identity_note']='NVML host namespace differs; sole context observed after empty-GPU initialization'
        engines=['tiled','persistent','tiled_ffn','persistent_ffn']
        if args.engine:engines=[args.engine]
        cases=[(kind,b,length) for kind,b in [('action',2),('action',14),('video',2)] for length in ['short','middle','long']]
        if args.case:cases=[x for x in cases if f'{x[2]}-B{x[1]}-{x[0]}'==args.case]
        if not cases:raise ValueError('unknown registered case')
        if args.kind=='counters' and (len(cases)!=1 or len(engines)!=1):raise ValueError('NCU needs one complete block case/engine')
        with torch.inference_mode():
            for kind,b,length in cases:
                guard();case=f'{length}-B{b}-{kind}'
                base=FIXTURES if kind=='action' else root/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
                path=base/f'input-{length}-B{b}-{kind}-layer0-first.pt'
                f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
                arguments,kwargs=f.arguments();cache=f.install_cache(block);initial=visible_snapshot(cache)
                def restore():
                    for k in ['id','mask','is_pred']:cache[k].copy_(initial[k])
                    for k in ['k','v']:cache[k].index_copy_(1,initial['slots'],initial[k])
                result['block_forward_intents']+=1;gold=block(*arguments,**kwargs);gold_state=visible_snapshot(cache)
                if not byte_equal(gold,f.data['block_gold']):raise RuntimeError('original saved gold mismatch')
                for engine in engines:
                    guard();options=dict(execution_mode='persistent' if engine.startswith('persistent') else 'tiled',fused_ffn1=engine.endswith('_ffn'))
                    with install_block(block,'optimized',**options):
                        for warm in range(2):
                            restore();result['block_forward_intents']+=1;out=block(*arguments,**kwargs)
                            if not byte_equal(out,gold):raise RuntimeError('warmup output mismatch')
                        for repetition in range(2 if args.kind=='timeline' else 1):
                            restore();torch.cuda.synchronize();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                            tag=f'{case}-{engine}-{repetition}';row=dict(case=case,engine=engine,repetition=repetition,input_path=str(path))
                            if args.kind=='timeline':
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
                                    with torch.profiler.record_function('M33_COMPLETE_BLOCK'):
                                        result['block_forward_intents']+=1;out=block(*arguments,**kwargs)
                                    torch.cuda.synchronize()
                                trace=args.output/(tag+'.trace.json');prof.export_chrome_trace(str(trace))
                                events=json.loads(trace.read_text())['traceEvents']
                                kernels=[x for x in events if x.get('cat')=='kernel' and x.get('ph')=='X']
                                runtime=[x for x in events if x.get('cat')=='cuda_runtime' and x.get('ph')=='X']
                                row['kernels']=[dict(name=x['name'],duration_us=x['dur'],start_us=x['ts'],args=x.get('args',{})) for x in kernels]
                                row['CUDA_runtime']=[dict(name=x['name'],duration_us=x['dur'],start_us=x['ts']) for x in runtime]
                                row['kernel_sum_us']=sum(x['dur'] for x in kernels)
                                row['GPU_span_us']=max(x['ts']+x['dur'] for x in kernels)-min(x['ts'] for x in kernels)
                                row['trace_file']=trace.name
                            else:
                                torch.cuda.cudart().cudaProfilerStart()
                                result['block_forward_intents']+=1;out=block(*arguments,**kwargs);torch.cuda.synchronize()
                                torch.cuda.cudart().cudaProfilerStop()
                            row['output_bitwise']=byte_equal(out,gold)
                            row['valid_KV']=state_equal(gold_state,visible_snapshot(cache))['pass_all']
                            row['RNG_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                            row['passed']=row['output_bitwise'] and row['valid_KV'] and row['RNG_unchanged'];result['rows'].append(row);save()
                            if not row['passed']:raise RuntimeError('profile output/state/RNG failed')
                    guard()
                del f,arguments,kwargs,initial,cache,gold,gold_state,out
                block.attn1.attn_caches={};gc.collect()
        result['status']='PROFILE_COMPLETE_WITH_FINITE_OUTPUT_STATE_PARITY'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc));raise
    finally:
        result['lifecycle_seconds']=time.monotonic()-start;save()


if __name__=='__main__':main()

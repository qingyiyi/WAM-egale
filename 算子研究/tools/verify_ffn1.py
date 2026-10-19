"""收缩后的FFN1+GELU原语：中间量及真实完整block严格门检/交错计时。"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from .block_loader import FIXTURES,load_block
from .fixtures import BlockFixture,tensors,visible_snapshot
from .metrics import byte_equal,compare,state_equal
from .replay import device_tree,resource,write
from wam_ops.optimized.ffn import FusedFFN1
from wam_ops.optimized.prepared import PreparedBlock,extension


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',required=True,type=Path)
    p.add_argument('--profile-result',required=True,type=Path)
    p.add_argument('--ffn-inputs',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    reg=json.loads(args.registration.read_text())
    before=resource()
    if before['pids'] or before['gpu'].split(',')[0].strip()!=reg['physical_gpu_uuid']:
        raise RuntimeError('GPU_BUSY_NO_CONTEXT_CREATED')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0,{4,5})
    start=time.monotonic()
    result=dict(status='RUNNING',rows=[],full_block_forwards=0,primitive_calls=0,
        started_at=datetime.now().astimezone().isoformat(),resources=[before],
        scope='FFN1 Linear+BF16 rounding+GELU fused; FFN2 remains canonical Torch; six block0 fixtures',
        persistent_scheduler=False,production_admitted=False,formal_performance_permission=False)
    def save(): write(args.output/'RESULT.json',result)
    def check():
        if time.monotonic()-start>540 or result['full_block_forwards']>=reg['full_block_forward_cap']:
            raise TimeoutError('bounded FFN1 validation cap')
        sample=resource();result['resources'].append(sample)
        if set(sample['pids'].splitlines())-{str(os.getpid())}:
            raise RuntimeError('FOREIGN_GPU_PID_STOP_MEASUREMENT')
    try:
        torch.cuda.set_device(0)
        block=load_block()
        result['kernel_metadata']=extension('wam_ffn_cuda').metadata()
        profiles=json.loads(args.profile_result.read_text())['rows']
        with torch.inference_mode():
            for old in profiles:
                check()
                length,batch,*_=old['case'].split('-')
                batch=int(batch[1:])
                fixture=BlockFixture.load(FIXTURES,length,batch)
                fixture=BlockFixture(fixture.path,device_tree(fixture.data))
                arguments,kwargs=fixture.arguments()
                cache=fixture.install_cache(block)
                initial=visible_snapshot(cache)
                immutable={n:t.clone() for n,t in tensors((arguments,kwargs))}
                cpu_rng,gpu_rng=torch.random.get_rng_state(),torch.cuda.get_rng_state()
                def restore():
                    for key in ('id','mask','is_pred'): cache[key].copy_(initial[key])
                    for key in ('k','v'): cache[key].index_copy_(1,initial['slots'],initial[key])
                def call():
                    result['full_block_forwards']+=1
                    return block(*arguments,**kwargs)
                oracle=call().clone()
                expected=visible_snapshot(cache)
                row=dict(case=old['case'],original_historical=compare(fixture.data['block_gold'],oracle))
                result['rows'].append(row)
                if not row['original_historical']['bitwise_pass']:
                    raise RuntimeError('single-block loader differs from saved original GPU gold')
                prepared=PreparedBlock(block,arguments,kwargs,cache)
                prepared.fill()
                fused=FusedFFN1(block.ffn.net[0],arguments[0].shape)
                ff=torch.load(args.ffn_inputs/(old['case']+'-ffn.pt'),mmap=True,map_location='cpu',weights_only=False)
                x=ff['input'].to('cuda')
                expected_ff=block.ffn.net[0](x)
                actual_ff=fused.forward(x)
                result['primitive_calls']+=1
                row['FFN1_GELU_output']=compare(expected_ff,actual_ff)
                if not row['FFN1_GELU_output']['bitwise_pass']:
                    row['status']='REJECTED_PRIMITIVE_NO_BLOCK_INSTALL'
                    save();continue
                parameter_copy=fused.weight.detach().clone()
                bias_copy=fused.bias.detach().clone()
                graphs,outputs={},{}
                row['admissions']={}
                def validate(out):
                    state=state_equal(expected,visible_snapshot(cache))
                    inputs=all(byte_equal(immutable[n],t) for n,t in tensors((arguments,kwargs)))
                    rng=byte_equal(cpu_rng,torch.random.get_rng_state()) and byte_equal(gpu_rng,torch.cuda.get_rng_state())
                    numeric=compare(oracle,out)
                    return dict(output=numeric,state=state,inputs_unchanged=inputs,RNG_unchanged=rng,
                        passed=numeric['bitwise_pass'] and state['pass_all'] and inputs and rng)
                engine=old['strongest_static_baseline']['engine']
                row['strong_baseline_engine']=engine
                for arm in ('strong_baseline','FFN1_GELU'):
                    restore()
                    context=fused.install() if arm=='FFN1_GELU' else nullcontext()
                    with prepared.activate(engine,True),context:
                        eager=call()
                        row['admissions'][arm]=validate(eager)
                        if not row['admissions'][arm]['passed']:
                            raise RuntimeError('complete block eager numerical/state gate')
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream): call();call()
                        torch.cuda.current_stream().wait_stream(stream)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph): outputs[arm]=call()
                        graphs[arm]=graph
                        if arm=='FFN1_GELU': result['primitive_calls']+=4
                    graphs[arm].replay();result['full_block_forwards']+=1
                    if arm=='FFN1_GELU': result['primitive_calls']+=1
                    torch.cuda.synchronize()
                    row['admissions'][arm+'_graph']=validate(outputs[arm])
                    if not row['admissions'][arm+'_graph']['passed']:
                        raise RuntimeError('complete block Graph numerical/state gate')
                raw=[]
                for repeat in range(15):
                    for arm in (('strong_baseline','FFN1_GELU') if repeat%2==0 else ('FFN1_GELU','strong_baseline')):
                        torch.cuda.synchronize()
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        wall=time.perf_counter();begin.record();graphs[arm].replay();end.record();end.synchronize()
                        result['full_block_forwards']+=1
                        if arm=='FFN1_GELU': result['primitive_calls']+=1
                        raw.append(dict(repeat=repeat,arm=arm,cuda_ms=begin.elapsed_time(end),
                            wall_ms=1000*(time.perf_counter()-wall)))
                        if not byte_equal(outputs[arm],oracle): raise RuntimeError('repeated complete block output changed')
                row.update(status='SIX_FIXTURE_LOCAL_GATE_PASSED',raw_timing=raw,
                    median_ms={a:statistics.median(s['cuda_ms'] for s in raw if s['arm']==a) for a in graphs},
                    parameters_unchanged=byte_equal(parameter_copy,fused.weight) and byte_equal(bias_copy,fused.bias),
                    post_state=validate(outputs['FFN1_GELU']))
                row['candidate_over_strong_baseline']=row['median_ms']['FFN1_GELU']/row['median_ms']['strong_baseline']
                if not row['post_state']['passed'] or not row['parameters_unchanged']: raise RuntimeError('post state/parameter gate')
                print(json.dumps(dict(case=row['case'],ratio=row['candidate_over_strong_baseline'],median=row['median_ms'])),flush=True)
                save()
                block.attn1.attn_caches={}
                del graphs,outputs,graph,fused,prepared,fixture,cache,immutable,parameter_copy,bias_copy
                gc.collect();torch.cuda.empty_cache()
        result.update(status='COMPLETE_BOUNDED_FFN1_GELU_AND_BLOCK_MATRIX',
            all_six_strict_gate_passed=len(result['rows'])==6 and all(r.get('status')=='SIX_FIXTURE_LOCAL_GATE_PASSED' for r in result['rows']),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='FAILED_OR_PARTIAL',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        result.update(lifecycle_seconds=time.monotonic()-start,finished_at=datetime.now().astimezone().isoformat())
        save()


if __name__=='__main__':main()

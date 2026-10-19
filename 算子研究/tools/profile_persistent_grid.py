"""One registered 108->216 CTA counterfactual on three complete B14 blocks."""
import argparse
from datetime import datetime
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',type=Path,required=True)
    args=p.parse_args();reg=json.loads(args.registration.read_text());root=Path(reg['source_workspace']);sys.path.insert(0,str(root/'算子研究'))
    import torch
    from tools.block_loader import FIXTURES,load_block
    from tools.fixtures import BlockFixture,decode,visible_snapshot
    from tools.metrics import byte_equal,state_equal
    from tools.replay import device_tree,write
    from tools.resources import resource_for_gpu
    from wam_ops.install import install_block
    import wam_ops.optimized.native as native
    args.output.mkdir(exist_ok=False);os.sched_setaffinity(0,set(reg['CPU_allowed']));torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();result=dict(status='RUNNING',rows=[],block_forward_intents=0,resources=[],kind='ONE_GRID_COUNTERFACTUAL_NOT_WHOLE_LAYER_PERSISTENT',profile_time_not_formal=True)
    def save():write(args.output/'RESULT.json',result)
    def guard():
        if time.time()>datetime.fromisoformat(reg['absolute_work_stop']).timestamp()-60:raise TimeoutError('stop reserve')
        obs=resource_for_gpu(reg['physical_gpu_uuid']);result['resources'].append(obs)
        if set(obs['pids'].splitlines())!=set(result.get('context_pids',[])):raise RuntimeError('GPU context set changed')
    original_extension=native.extension
    try:
        guard();block=load_block();torch.cuda.synchronize();obs=resource_for_gpu(reg['physical_gpu_uuid']);result['context_pids']=obs['pids'].splitlines()
        if len(result['context_pids'])!=1:raise RuntimeError('cannot associate sole context')
        old=original_extension();files=list(args.variant.glob('m33_grid216_cuda*.so'))
        if len(files)!=1:raise RuntimeError('variant missing/ambiguous')
        # CPython caches single-phase extensions by their init name.  The
        # diagnostic DSO must have its own PyInit symbol, not just a new path.
        spec=importlib.util.spec_from_file_location('m33_grid216_cuda',files[0]);new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
        if not hasattr(new,'m33_attention_resources'):raise RuntimeError('diagnostic extension identity mismatch')
        result['variant_library']=str(files[0]);engines=[('static',old,'tiled'),('P108',old,'persistent'),('P216',new,'persistent')]
        with torch.inference_mode():
            for length in ['short','middle','long']:
                guard();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
                arguments,kwargs=f.arguments();cache=f.install_cache(block);initial=visible_snapshot(cache)
                def restore():
                    for k in ['id','mask','is_pred']:cache[k].copy_(initial[k])
                    for k in ['k','v']:cache[k].index_copy_(1,initial['slots'],initial[k])
                result['block_forward_intents']+=1;gold=block(*arguments,**kwargs);gold_state=visible_snapshot(cache)
                if not byte_equal(gold,f.data['block_gold']):raise RuntimeError('saved gold mismatch')
                row=dict(case=f'{length}-B14-action',input_path=str(path),profiles=[],raw_timing=[]);result['rows'].append(row)
                def checked_forward():
                    result['block_forward_intents']+=1
                    if result['block_forward_intents']>reg['block_per_process_cap']:raise RuntimeError('forward cap')
                    return block(*arguments,**kwargs)
                def check(out):
                    if not byte_equal(out,gold) or not state_equal(gold_state,visible_snapshot(cache))['pass_all']:raise RuntimeError('output/KV mismatch')
                for label,lib,mode in engines:
                    native.extension=lambda lib=lib:lib
                    with install_block(block,'optimized',execution_mode=mode,fused_ffn1=True):
                        for _ in range(2):restore();out=checked_forward();check(out)
                        for rep in range(2):
                            restore();torch.cuda.synchronize()
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                                out=checked_forward();torch.cuda.synchronize()
                            check(out);trace=args.output/f'{length}-{label}-{rep}.trace.json';prof.export_chrome_trace(str(trace))
                            events=json.loads(trace.read_text())['traceEvents'];kernels=[x for x in events if x.get('cat')=='kernel' and x.get('ph')=='X']
                            row['profiles'].append(dict(engine=label,repetition=rep,kernels=[dict(name=x['name'],duration_us=x['dur'],args=x.get('args',{})) for x in kernels],kernel_sum_us=sum(x['dur'] for x in kernels),trace_file=trace.name,output_valid_KV_pass=True))
                if length=='short':
                    # Prime static variant's opt-in attributes before querying
                    # theoretical occupancy; avoids the old pre-opt-in zero trap.
                    native.extension=lambda:new
                    restore()
                    with install_block(block,'optimized',execution_mode='tiled',fused_ffn1=True):out=checked_forward()
                    check(out);result['resource_schema']=['static_regs','persistent_regs','dynamic_smem_bytes','static_theoretical_CTAs_per_SM','persistent_theoretical_CTAs_per_SM']
                    result['resources_after_optin']=dict(attention=new.m33_attention_resources(),ffn1=new.m33_ffn_resources(),measured_occupancy='UNKNOWN:NCU permission denied')
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for rep in range(8):
                    for label,lib,mode in engines[rep%3:]+engines[:rep%3]:
                        native.extension=lambda lib=lib:lib;restore()
                        with install_block(block,'optimized',execution_mode=mode,fused_ffn1=True):
                            torch.cuda.synchronize();begin.record();out=checked_forward();end.record();end.synchronize()
                        check(out);row['raw_timing'].append(dict(round=rep,engine=label,event_ms=begin.elapsed_time(end)))
                row['event_medians_ms']={label:statistics.median(x['event_ms'] for x in row['raw_timing'] if x['engine']==label) for label,_,_ in engines}
                row['RNG_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                if not row['RNG_unchanged']:raise RuntimeError('RNG changed')
                guard();save();del f,arguments,kwargs,initial,cache,gold,gold_state,out;block.attn1.attn_caches={};gc.collect()
        result['status']='THREE_CASE_GRID_COUNTERFACTUAL_COMPLETE'
    except BaseException as exc:result.update(status='FAILED_OR_PARTIAL',error=repr(exc));raise
    finally:
        native.extension=original_extension;result['native_binding_restored']=native.extension is original_extension;result['lifecycle_seconds']=time.monotonic()-start;save()


if __name__=='__main__':main()

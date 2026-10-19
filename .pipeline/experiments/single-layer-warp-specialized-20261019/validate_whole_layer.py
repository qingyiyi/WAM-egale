"""Real complete single-block replay; separate structural/numeric/state gates.

This does not admit the experimental arithmetic, evaluate robots, or run 30 layers.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture, decode, tensors, visible_snapshot
from tools.metrics import byte_equal, compare, state_equal
from tools.replay import device_tree
from tools.resources import resource_for_gpu
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock

WORK = Path(load_block.__code__.co_filename).resolve().parents[2]
VIDEO = WORK/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
DAG_ORDER=('modulation','norm1','modulated1','self_q_norm','self_k_norm','self_v',
    'self_q_rope','self_k_rope','self_attention','self_out','residual1','norm2',
    'cross_q','cross_k','cross_v','cross_attention','cross_out','residual2',
    'norm3','modulated3','ffn1','gelu','ffn2','output')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timing', action='store_true')
    p.add_argument('--reference-native',choices=('optimized','baseline'),default='optimized')
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    if torch.version.git_version != '0fabc3ba44823f257e70ce397d989c8de5e362c1':
        raise RuntimeError('wrong interpreter/Torch; no GPU context may be created')
    os.sched_setaffinity(0, {4, 5})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    before = resource_for_gpu(reg['physical_gpu_uuid'])
    if before['pids']:
        raise RuntimeError('GPU_BUSY_NO_CONTEXT_CREATED')
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(status='PREPARING', rows=[], timing=[], resources=[before],
        complete_block_attempts=0, primitive_diagnostic_calls=0, full30_calls=0, robot_episodes=0,
        production_admitted=False, original_bitwise_gate_unchanged=True,
        numerical_route='NEW_WMMA_AND_ONLINE_ATTENTION_REQUIRES_SEPARATE_ADMISSION',
        timing_scope='MATCHED_UNCACHED_SINGLE_BLOCK_EAGER_RESEARCH_ONLY',
        strong_cached_Graph_comparison='NOT_YET_MEASURED',
        formal_performance_permission=False)
    result['source_sha256']={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (WORK/'算子研究/wam_ops/optimized/csrc/whole_layer.cu',
                  WORK/'算子研究/wam_ops/optimized/whole_layer.py')}
    result['validation_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    expected_context = None

    def save():
        q = args.output/'RESULT.pending'
        q.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        q.replace(args.output/'RESULT.json')

    def guard():
        nonlocal expected_context
        if time.monotonic()-start > 600:
            raise TimeoutError('600 second GPU worker lifecycle')
        if time.time() > datetime.fromisoformat(reg['absolute_work_stop']).timestamp()-60:
            raise TimeoutError('work deadline reserve')
        if result['complete_block_attempts'] >= 500:
            raise RuntimeError('registered per-worker call limit')
        obs = resource_for_gpu(reg['physical_gpu_uuid'])
        result['resources'].append(obs)
        current = set(obs['pids'].splitlines())
        if expected_context is None:
            if len(current) != 1:
                raise RuntimeError('cannot identify sole post-init NVML context')
            expected_context = current
        if current != expected_context:
            raise RuntimeError('FOREIGN_CONTEXT_NO_FURTHER_TIMING')

    def call(fn):
        result['complete_block_attempts'] += 1
        return fn()

    try:
        block = load_block(device='cuda')
        block.requires_grad_(True)
        guard()
        original_class = type(block)
        parameter_identity = {k:(v.data_ptr(),v.requires_grad) for k,v in block.named_parameters()}
        native = (NativeBlock(block, mode='sequential') if args.reference_native=='optimized' else
                  importlib.import_module('wam_ops.baseline.native').NativeBlock(block))
        result['reference_native']=args.reference_native
        whole = WholeLayerBlock(block)
        with torch.inference_mode():
            for kind,batch in [('action',2),('action',14),('video',2)]:
                for length in ('short','middle','long'):
                    guard()
                    path=(FIXTURES if kind=='action' else VIDEO)/f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                    raw=decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))
                    fixture=BlockFixture(path,device_tree(raw))
                    arguments,kwargs=fixture.arguments()
                    immutable={n:t.clone() for n,t in tensors((arguments,kwargs))}
                    cpu_rng,gpu_rng=torch.random.get_rng_state(),torch.cuda.get_rng_state()
                    fixture.install_cache(block)
                    gold=call(lambda:block(*arguments,**kwargs));torch.cuda.synchronize()
                    gold_state=visible_snapshot(block.attn1.attn_caches[kwargs['cache_name']])
                    fixture.install_cache(block)
                    base=call(lambda:native(*arguments,**kwargs,debug=True));torch.cuda.synchronize()
                    fixture.install_cache(block)
                    observed=call(lambda:whole(*arguments,**kwargs,debug=True));torch.cuda.synchronize()
                    whole_state=visible_snapshot(block.attn1.attn_caches[kwargs['cache_name']])
                    stats=observed['tile_stats'].cpu().tolist()
                    active_stats=[dict(stage=i,expected=x[0],completed=x[1],duplicate=x[2])
                                  for i,x in enumerate(stats) if any(x)]
                    checks={k:compare(v,observed[k].reshape_as(v)) for k,v in base.items()
                            if k in observed and k!='execution_flags'}
                    row=dict(case=f'{length}-B{batch}-{kind}',source=str(path),
                        original_vs_native=compare(gold,base['output']),
                        original_vs_whole=compare(gold,observed['output']),
                        historical_gold_vs_original=compare(fixture.data['block_gold'],gold),
                        intermediates=checks,
                        first_different_intermediate=next((k for k in DAG_ORDER if k in checks and not checks[k]['bitwise_pass']),None),
                        intermediate_order='explicit dependency order, not map alphabetical order',
                        state=state_equal(gold_state,whole_state),
                        cache_info=observed['cache_info'].cpu().tolist(),
                        resources=observed['resources'].tolist(),
                        tile_stats=active_stats,
                        tile_exact_once=all(x['expected']==x['completed'] and x['duplicate']==0 for x in active_stats),
                        immutable=all(byte_equal(immutable[n],v) for n,v in tensors((arguments,kwargs))),
                        RNG_unchanged=byte_equal(cpu_rng,torch.random.get_rng_state()) and byte_equal(gpu_rng,torch.cuda.get_rng_state()),
                        parameter_flags_and_addresses_unchanged=parameter_identity=={k:(v.data_ptr(),v.requires_grad) for k,v in block.named_parameters()},
                        original_class_unchanged=type(block) is original_class)
                    row['original_contract_passed']=(row['original_vs_whole']['bitwise_pass'] and
                        row['original_vs_native']['bitwise_pass'] and row['historical_gold_vs_original']['bitwise_pass'] and
                        row['state']['pass_all'] and row['tile_exact_once'] and row['immutable'] and
                        row['RNG_unchanged'] and row['parameter_flags_and_addresses_unchanged'])
                    # Same-input controls distinguish propagation from arithmetic
                    # order in the first projection and the missing old FFN2 gate.
                    q_control=block.attn1.to_q(observed['modulated1'])
                    f_control=block.ffn.net[2](observed['gelu'])
                    result['primitive_diagnostic_calls']+=2
                    row['same_input_GEMM_controls']=dict(
                        first_Q=compare(q_control,observed['self_q_linear'].reshape_as(q_control)),
                        FFN2=compare(f_control,observed['ffn2'].reshape_as(f_control)))
                    del q_control,f_control
                    result['rows'].append(row);save()

                    # Kernel trace is correctness/structure evidence, never timing.
                    if not args.timing:
                        fixture.install_cache(block)
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
                            call(lambda:whole(*arguments,**kwargs,debug=True));torch.cuda.synchronize()
                        kernels=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                        row['kernel_trace']=dict(names=kernels,count=len(kernels),
                            one_whole_layer_kernel=len(kernels)==1 and 'whole_layer' in kernels[0])
                        prof.export_chrome_trace(str(args.output/f'{row["case"]}-TRACE.json'))
                        save()

                    if batch==2 and not args.timing:
                        # Three legal cache transitions from this same saved input.
                        fixture.install_cache(block);expected=[]
                        for update in (1,2,0):
                            kw=dict(kwargs,update_cache=update)
                            out=call(lambda:block(*arguments,**kw));torch.cuda.synchronize()
                            expected.append((out.clone(),visible_snapshot(block.attn1.attn_caches[kw['cache_name']])))
                        fixture.install_cache(block);row['cache_transitions']=[]
                        for update,(output,state) in zip((1,2,0),expected):
                            kw=dict(kwargs,update_cache=update)
                            current=call(lambda:whole(*arguments,**kw));torch.cuda.synchronize()
                            row['cache_transitions'].append(dict(update=update,output=compare(output,current),
                                state=state_equal(state,visible_snapshot(block.attn1.attn_caches[kw['cache_name']]))))
                        save()

                    if args.timing and row['original_vs_whole']['finite_pass'] and row['tile_exact_once']:
                        # A failed strict gate stays failed. These samples characterize
                        # the experimental implementation; they are not a valid speedup.
                        modes={'Torch':lambda:block(*arguments,**kwargs),
                               'native':lambda:native(*arguments,**kwargs),
                               'whole':lambda:whole(*arguments,**kwargs,audit=False)}
                        for fn in modes.values():
                            fixture.install_cache(block);call(fn);torch.cuda.synchronize()
                        for repeat in range(5):
                            names=list(modes);names=names[repeat%3:]+names[:repeat%3]
                            for name in names:
                                guard();fixture.install_cache(block)
                                torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                a.record();t=time.perf_counter();call(modes[name]);b.record();b.synchronize()
                                result['timing'].append(dict(case=row['case'],mode=name,repeat=repeat,
                                    GPU_stream_ms=a.elapsed_time(b),host_complete_ms=(time.perf_counter()-t)*1000,
                                    numeric_admitted=row['original_contract_passed'] if name=='whole' else True))
                        save()
                    del observed,base,gold,gold_state,whole_state,fixture,arguments,kwargs,raw,immutable
                    gc.collect()
            result['status']=('COMPLETE_FINITE_NUMERIC_PASS' if all(r['original_contract_passed'] for r in result['rows'])
                              else 'COMPLETE_STRUCTURAL_RESEARCH_UNADMITTED_NUMERIC')
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result['lifecycle_seconds']=time.monotonic()-start
        result['finished_at']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()

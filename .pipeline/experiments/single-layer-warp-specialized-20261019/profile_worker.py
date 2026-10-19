"""Finite generated-kernel diagnostic; original numerical gate stays FAILED."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,tensors,visible_snapshot
from tools.metrics import byte_equal,state_equal
from tools.replay import device_tree
from tools.resources import resource_for_gpu
from wam_ops.optimized.whole_layer import WholeLayerBlock,extension

BASE=Path('/root/cxz/ops/whole-layer-warp-20261020')
WORK=Path(load_block.__code__.co_filename).resolve().parents[2]
VIDEO=WORK/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
LABELS=['table','norm1','modulate1','Q_proj','K_proj','V_proj','Q_RMS','K_RMS',
    'Q_RoPE','K_RoPE','private_KV','self_attention','self_output_proj','residual1',
    'norm2','cross_Q_proj','cross_K_proj','cross_V_proj','cross_Q_RMS','cross_K_RMS',
    'cross_attention','cross_output_proj','residual2','norm3','modulate3','FFN1','GELU','FFN2','output_residual']
GEMM={3,4,5,12,15,16,17,21,25,27}


def module(kind):
    name='wam_whole_diagnostic_'+kind
    paths=list((BASE/'diagnostic').glob(name+'*.so'));assert len(paths)==1
    spec=importlib.util.spec_from_file_location(name,paths[0]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('profile','pad40'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args();reg=json.loads(args.registration.read_text());uuid=reg['physical_gpu_uuid']
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2)
    assert torch.version.git_version=='0fabc3ba44823f257e70ce397d989c8de5e362c1'
    assert not resource_for_gpu(uuid)['pids'],'GPU must be released before context'
    args.output.mkdir(exist_ok=False);start=time.monotonic()
    result=dict(status='PREPARING',rows=[],timing=[],complete_block_calls=0,
        original_precision_admitted=False,full30=0,episodes=0,mode=args.mode)
    expected=None
    def save():(args.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    def guard():
        nonlocal expected
        assert time.monotonic()-start<240,'worker lifecycle cap'
        obs=resource_for_gpu(uuid);current=set(obs['pids'].splitlines())
        if expected is None:assert len(current)==1;expected=current
        assert current==expected,'foreign context; no further work'
    try:
        block=load_block('cuda');binding=WholeLayerBlock(block);guard()
        libs={'v1':extension(),'profile':module('profile')}
        if args.mode=='pad40':libs['pad40']=module('pad40')
        with torch.inference_mode():
            for kind,batch in [('action',2),('action',14),('video',2)]:
                for length in ('short','middle','long'):
                    guard();root=FIXTURES if kind=='action' else VIDEO
                    path=root/f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                    raw=decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))
                    fixture=BlockFixture(path,device_tree(raw));arguments,kwargs=fixture.arguments()
                    cpu,gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state()
                    immutable={k:v.clone() for k,v in tensors((arguments,kwargs))}
                    case=f'{length}-B{batch}-{kind}'
                    def run(name,audit=True):
                        result['complete_block_calls']+=1
                        cache=block.attn1.attn_caches[kwargs['cache_name']]
                        return libs[name].forward(binding.weights,*arguments,
                            [cache[k] for k in ('k','v','id','mask','is_pred')],
                            kwargs.get('candidate_cache_rows'),kwargs['update_cache'],binding.eps,audit)
                    observed={};states={}
                    for name in libs:
                        fixture.install_cache(block);observed[name]=run(name);torch.cuda.synchronize()
                        states[name]=visible_snapshot(block.attn1.attn_caches[kwargs['cache_name']])
                    gold=observed['v1'];checks={}
                    for name,value in observed.items():
                        if name=='v1':continue
                        checks[name]=dict(intermediates={k:byte_equal(v,value[k]) for k,v in gold.items()
                            if k not in ('tile_stats','resources')},
                            state=state_equal(states['v1'],states[name]),
                            cache_info=value['cache_info'].cpu().tolist(),resources=value['resources'].tolist())
                        checks[name]['match_v1']=all(checks[name]['intermediates'].values()) and checks[name]['state']['pass_all']
                    row=dict(case=case,checks=checks,
                        RNG_unchanged=byte_equal(cpu,torch.random.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state()),
                        inputs_unchanged=all(byte_equal(immutable[k],v) for k,v in tensors((arguments,kwargs))))
                    for name,value in observed.items():
                        if name=='v1':continue
                        ticks=value['stage_ticks'].cpu().tolist();total=ticks[40][1]-ticks[40][0]
                        assert total>0 and all(ticks[i][1]>=ticks[i][0]>0 for i in range(29))
                        row[name+'_stage_ns']={label:ticks[i][1]-ticks[i][0] for i,label in enumerate(LABELS)}
                        row[name+'_kernel_ns']=total
                        row[name+'_GEMM_ns']=sum(ticks[i][1]-ticks[i][0] for i in GEMM)
                        row[name+'_unassigned_ns']=total-sum(ticks[i][1]-ticks[i][0] for i in range(29))
                    result['rows'].append(row);save()
                    if not all(v['match_v1'] for v in checks.values()) or not row['RNG_unchanged'] or not row['inputs_unchanged']:
                        raise RuntimeError('diagnostic changed existing v1 arithmetic/state; no subsequent case')
                    if args.mode=='pad40':
                        for repeat,names in enumerate([('profile','pad40'),('pad40','profile')]):
                            for name in names:
                                guard();fixture.install_cache(block);torch.cuda.synchronize()
                                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                a.record();t=time.perf_counter();run(name,audit=False);b.record();b.synchronize()
                                result['timing'].append(dict(case=case,name=name,repeat=repeat,
                                    stream_ms=a.elapsed_time(b),host_ms=(time.perf_counter()-t)*1000))
                        if length=='short' and batch==14:
                            row['trace_names']={}
                            for name in ('profile','pad40'):
                                fixture.install_cache(block)
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                                    run(name,audit=False);torch.cuda.synchronize()
                                names=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                                row['trace_names'][name]=names
                                assert len(names)==1 and ('diagnostic_'+name) in names[0],'extension identity/launch mismatch'
                        save()
                    del observed,gold,states,checks,fixture,raw,arguments,kwargs,immutable
        result['status']='COMPLETE_DIAGNOSTIC_MATCHES_UNADMITTED_V1'
        selected=[r for r in result['rows'] if 'B14-action' in r['case']]
        result['B14_GEMM_kernel_fraction']=sum(r['profile_GEMM_ns'] for r in selected)/sum(r['profile_kernel_ns'] for r in selected)
        result['padding_trigger']=result['B14_GEMM_kernel_fraction']>=.5
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result['lifecycle_seconds']=time.monotonic()-start;save()


if __name__=='__main__':main()

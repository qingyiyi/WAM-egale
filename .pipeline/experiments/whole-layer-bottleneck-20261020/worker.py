"""Real six-shape cost attribution. A failed numerical prototype stays unadmitted."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,tensors,visible_snapshot
from tools.metrics import byte_equal,compare,state_equal
from tools.replay import device_tree
from tools.resources import resource_for_gpu
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock,extension

HERE=Path(__file__).resolve().parent
LABELS=['table','norm1','modulate1','Q_proj','K_proj','V_proj','Q_RMS','K_RMS',
        'Q_RoPE','K_RoPE','private_KV','self_attention','self_output_proj','residual1',
        'norm2','cross_Q_proj','cross_K_proj','cross_V_proj','cross_Q_RMS','cross_K_RMS',
        'cross_attention','cross_output_proj','residual2','norm3','modulate3','FFN1','GELU','FFN2','output_residual']

def load_dso(root,name):
    found=list(root.glob(name+'*.so'));assert len(found)==1
    s=importlib.util.spec_from_file_location(name,found[0]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m

def kernels(prof):
    return [dict(name=e.name,us=e.time_range.elapsed_us()) for e in prof.events()
            if e.device_type==torch.autograd.DeviceType.CUDA]

def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reg=json.loads(a.registration.read_text());a.output.mkdir(exist_ok=False)
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.version.git_version=='0fabc3ba44823f257e70ce397d989c8de5e362c1'
    assert not resource_for_gpu(reg['GPU_uuid'])['pids'],'foreign GPU before context'
    start=time.monotonic();expected=None
    result=dict(status='RUNNING',rows=[],timing=[],stages=[],primitive_timing=[],resources=[],
        full_block_calls=0,primitive_calls=0,robot_episodes=0,full30=0,
        numeric_admitted=False,formal_performance_permission=False,
        timing_scope='six real action signatures; uncached eager complete block',
        strong_cached_graph='NOT_MEASURED',hardware_counters='UNKNOWN_NCU_PERMISSION_PREVIOUSLY_DENIED')
    def save():
        q=a.output/'RESULT.pending';q.write_text(json.dumps(result,indent=2)+'\n');q.replace(a.output/'RESULT.json')
    def guard():
        nonlocal expected
        assert time.monotonic()-start<900,'worker lifecycle cap'
        assert time.time()<datetime.fromisoformat(reg['work_stop']).timestamp()-60,'deadline reserve'
        assert result['full_block_calls']<=240 and result['primitive_calls']<=600
        obs=resource_for_gpu(reg['GPU_uuid']);result['resources'].append(obs)
        cur=set(obs['pids'].splitlines())
        if expected is None:assert len(cur)==1;expected=cur
        assert cur==expected,'foreign GPU context; stop new calls'
    try:
        block=load_block('cuda');native=NativeBlock(block);binding=WholeLayerBlock(block);guard()
        libs=dict(whole=extension(),profile=load_dso(Path('/root/cxz/ops/whole-layer-warp-20261020/diagnostic'),'wam_whole_diagnostic_profile'),
                  static=load_dso(HERE,'wam_whole_m38_static'))
        with torch.inference_mode():
            for batch in (2,14):
                for length in ('short','middle','long'):
                    guard();fixture=BlockFixture.load(FIXTURES,length,batch)
                    fixture=BlockFixture(fixture.path,device_tree(fixture.data));args,kw=fixture.arguments()
                    case=f'{length}-B{batch}';cpu,gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state()
                    immutable={n:v.clone() for n,v in tensors((args,kw))}
                    def call(name,audit=False):
                        assert result['full_block_calls']<240
                        result['full_block_calls']+=1
                        if name=='Torch':return block(*args,**kw)
                        if name=='native':return native(*args,**kw,debug=True)
                        cache=block.attn1.attn_caches[kw['cache_name']]
                        return libs[name].forward(binding.weights,*args,[cache[k] for k in ('k','v','id','mask','is_pred')],
                            kw.get('candidate_cache_rows'),kw['update_cache'],binding.eps,audit)
                    values={};states={}
                    for name in ('Torch','native','whole','profile','static'):
                        fixture.install_cache(block);values[name]=call(name,True);torch.cuda.synchronize()
                        states[name]=visible_snapshot(block.attn1.attn_caches[kw['cache_name']])
                    row=dict(case=case,fixture=str(fixture.path),input_shape=list(args[0].shape),context_shape=list(args[1].shape),
                        native_vs_Torch=compare(values['Torch'],values['native']['output']),
                        whole_vs_Torch=compare(values['Torch'],values['whole']['output']),checks={},traces={},primitive_profile=[])
                    for name in ('profile','static'):
                        stats=values[name]['tile_stats'].cpu().tolist()
                        row['checks'][name]=dict(match_whole=all(byte_equal(v,values[name][k]) for k,v in values['whole'].items()
                            if k not in ('tile_stats','resources')),state=state_equal(states['whole'],states[name]),
                            resources=values[name]['resources'].tolist(),
                            exact_once=all(x[0]==x[1] and x[2]==0 for x in stats))
                    row['whole_resources']=values['whole']['resources'].tolist()
                    row['inputs_unchanged']=all(byte_equal(immutable[n],v) for n,v in tensors((args,kw)))
                    row['RNG_unchanged']=byte_equal(cpu,torch.random.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state())
                    result['rows'].append(row);save()
                    assert row['native_vs_Torch']['bitwise_pass'],'reference changed'
                    assert row['inputs_unchanged'] and row['RNG_unchanged']
                    assert all(c['match_whole'] and c['state']['pass_all'] and c['exact_once'] for c in row['checks'].values()),'ablation changed prototype arithmetic/state'
                    # Warm-up and event timings contain no profiler or tile audit.
                    modes=['native','whole','profile','static']
                    for name in modes:fixture.install_cache(block);call(name);torch.cuda.synchronize()
                    for repeat in range(6):
                        for name in modes[repeat%4:]+modes[:repeat%4]:
                            guard();fixture.install_cache(block);torch.cuda.synchronize()
                            beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            beg.record();t=time.perf_counter();out=call(name);end.record();end.synchronize()
                            result['timing'].append(dict(case=case,name=name,repeat=repeat,
                                stream_ms=beg.elapsed_time(end),host_ms=(time.perf_counter()-t)*1000))
                    # Three stage samples, audit off for the two extra calls; audit first excluded below.
                    for repeat in range(2):
                        fixture.install_cache(block);v=call('profile');torch.cuda.synchronize()
                        ticks=v['stage_ticks'].cpu().tolist()
                        result['stages'].append(dict(case=case,repeat=repeat,audit=False,
                            kernel_ns=ticks[40][1]-ticks[40][0],
                            stage_ns={label:ticks[i][1]-ticks[i][0] for i,label in enumerate(LABELS)}))
                    ticks=values['profile']['stage_ticks'].cpu().tolist()
                    row['audit_stage_ns']={label:ticks[i][1]-ticks[i][0] for i,label in enumerate(LABELS)}
                    # Full CUDA traces, disjoint from the event timing samples.
                    for name in ('native','whole'):
                        guard();fixture.install_cache(block)
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                            call(name);torch.cuda.synchronize()
                        ks=kernels(prof);row['traces'][name]=dict(kernels=ks,total_device_us=sum(k['us'] for k in ks))
                        prof.export_chrome_trace(str(a.output/f'{case}-{name}-TRACE.json'))
                    # Exact SAME intermediate input to Torch Linear and existing whole stage.
                    v=values['whole']
                    components=[('Q_proj',v['modulated1'],block.attn1.to_q,'self_q_linear'),
                        ('K_proj',v['modulated1'],block.attn1.to_k,'self_k_linear'),
                        ('V_proj',v['modulated1'],block.attn1.to_v,'self_v'),
                        ('self_output_proj',v['self_attention'],block.attn1.to_out[0],'self_out'),
                        ('cross_Q_proj',v['norm2'],block.attn2.to_q,'cross_q_linear'),
                        ('cross_K_proj',args[1],block.attn2.to_k,'cross_k_linear'),
                        ('cross_V_proj',args[1],block.attn2.to_v,'cross_v'),
                        ('cross_output_proj',v['cross_attention'],block.attn2.to_out[0],'cross_out'),
                        ('FFN1',v['modulated3'],block.ffn.net[0].proj,'ffn1'),
                        ('FFN2',v['gelu'],block.ffn.net[2],'ffn2')]
                    prim=[]
                    for label,x,layer,key in components:
                        x=x.reshape(-1,x.shape[-1]);y=layer(x);result['primitive_calls']+=1
                        prim.append(dict(label=label,M=x.shape[0],N=layer.weight.shape[0],K=x.shape[1],
                            shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),
                            same_input_output=compare(y,v[key].reshape_as(y))))
                        for repeat in range(3):
                            beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            beg.record();y=layer(x);result['primitive_calls']+=1;end.record();end.synchronize()
                            result['primitive_timing'].append(dict(case=case,label=label,repeat=repeat,stream_ms=beg.elapsed_time(end)))
                    guard()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                        for label,x,layer,key in components:
                            with torch.profiler.record_function('component::'+label):
                                y=layer(x.reshape(-1,x.shape[-1]));result['primitive_calls']+=1
                        torch.cuda.synchronize()
                    events={e.name:e for e in prof.events() if e.name.startswith('component::')}
                    for rec in prim:
                        e=events['component::'+rec['label']]
                        rec['GPU_us']=e.device_time_total;rec['CPU_us']=e.cpu_time_total
                    row['primitive_profile']=prim
                    row['primitive_kernel_names']=kernels(prof)
                    prof.export_chrome_trace(str(a.output/f'{case}-components-TRACE.json'))
                    save()
                    del values,states,fixture,args,kw,immutable,components,prim,v,out,y
        result['status']='COMPLETE_PROFILE_UNADMITTED_PROTOTYPE'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result.update(lifecycle_seconds=time.monotonic()-start,at_UTC=datetime.now(timezone.utc).isoformat());save()

if __name__=='__main__':main()

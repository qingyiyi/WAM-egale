"""Finite real-input gate, complete-block timing, and separate stage profiling."""
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
from tools.fixtures import BlockFixture,decode,tensors,visible_snapshot
from tools.metrics import byte_equal,compare,state_equal
from tools.replay import device_tree
from tools.resources import resource_for_gpu
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
from wam_ops.optimized.whole_layer_pipeline import PipelineBlock

HERE=Path(__file__).resolve().parent
WORK=Path(load_block.__code__.co_filename).resolve().parents[2]
VIDEO=WORK/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
LABELS=['table','norm1','modulate1','Q_proj','K_proj','V_proj','Q_RMS','K_RMS','Q_RoPE','K_RoPE',
 'private_KV','self_attention','self_output_proj','residual1','norm2','cross_Q_proj','cross_K_proj',
 'cross_V_proj','cross_Q_RMS','cross_K_RMS','cross_attention','cross_output_proj','residual2','norm3',
 'modulate3','FFN1','GELU','FFN2','output_residual']

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','timing','profile'),required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--process-index',type=int,default=0);a=p.parse_args()
    reg=json.loads(a.registration.read_text());a.output.mkdir(exist_ok=False)
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.version.git_version=='0fabc3ba44823f257e70ce397d989c8de5e362c1'
    assert not resource_for_gpu(reg['GPU_uuid'])['pids'],'GPU busy before context'
    start=time.monotonic();expected=None
    r=dict(status='RUNNING',mode=a.mode,process_index=a.process_index,rows=[],timing=[],profiles=[],resources=[],
        full_block_calls=0,robot_episodes=0,full30=0,original_Torch_admitted=False,formal_performance_permission=False)
    def save():
        q=a.output/'RESULT.pending';q.write_text(json.dumps(r,indent=2)+'\n');q.replace(a.output/'RESULT.json')
    def guard():
        nonlocal expected
        assert time.monotonic()-start<600
        assert time.time()<datetime.fromisoformat(reg['work_stop']).timestamp()-60
        assert r['full_block_calls']<=350
        obs=resource_for_gpu(reg['GPU_uuid']);r['resources'].append(obs);cur=set(obs['pids'].splitlines())
        if expected is None:assert len(cur)==1;expected=cur
        assert cur==expected,'foreign context; stop new work'
    try:
        block=load_block('cuda');guard();flags={n:(x.data_ptr(),x.requires_grad) for n,x in block.named_parameters()}
        impl={'Torch':block,'native':NativeBlock(block),'whole':WholeLayerBlock(block),
            'sync':PipelineBlock(block,HERE,'sync'),'async':PipelineBlock(block,HERE,'async'),
            'profile':PipelineBlock(block,HERE,'profile')}
        if a.mode=='profile':
            path=next(Path('/root/cxz/ops/whole-layer-warp-20261020/diagnostic').glob('wam_whole_diagnostic_profile*.so'))
            spec=importlib.util.spec_from_file_location('wam_whole_diagnostic_profile',path)
            old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        with torch.inference_mode():
            for kind,batch in [('action',2),('action',14),('video',2)]:
                for length in ('short','middle','long'):
                    guard();path=(FIXTURES if kind=='action' else VIDEO)/f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                    raw=decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))
                    f=BlockFixture(path,device_tree(raw));args,kw=f.arguments();case=f'{length}-B{batch}-{kind}'
                    row=dict(case=case,fixture=str(path),checks={},cache_sequences=[],traces={})
                    immutable={n:t.clone() for n,t in tensors((args,kw))}
                    cpu,gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state()
                    def call(name,audit=False,kwargs=None):
                        assert r['full_block_calls']<350;r['full_block_calls']+=1
                        if name=='Torch':return block(*args,**(kwargs or kw))
                        if name=='native':return impl[name](*args,**(kwargs or kw),debug=True)
                        if name=='old_profile':
                            cache=block.attn1.attn_caches[kw['cache_name']];binding=impl['whole']
                            return old.forward(binding.weights,*args,[cache[k] for k in ('k','v','id','mask','is_pred')],
                                kw.get('candidate_cache_rows'),kw['update_cache'],binding.eps,audit)
                        return impl[name](*args,**(kwargs or kw),debug=True,audit=audit)
                    if a.mode=='gate':
                        values={};states={}
                        for name in ('Torch','native','whole','sync','async','profile'):
                            f.install_cache(block);values[name]=call(name,True);torch.cuda.synchronize()
                            states[name]=visible_snapshot(block.attn1.attn_caches[kw['cache_name']])
                        row['original_native']=compare(values['Torch'],values['native']['output'])
                        row['original_async']=compare(values['Torch'],values['async']['output'])
                        for name in ('sync','async','profile'):
                            v=values[name];stats=v['tile_stats'].cpu().tolist()
                            row['checks'][name]=dict(intermediates={k:byte_equal(x,v[k]) for k,x in values['whole'].items()
                                if k not in ('tile_stats','resources')},
                                state=state_equal(states['whole'],states[name]),resources=v['resources'].tolist(),
                                exact_once=all(x[0]==x[1] and x[2]==0 for x in stats),cache_info=v['cache_info'].cpu().tolist())
                            row['checks'][name]['pass']=all(row['checks'][name]['intermediates'].values()) and row['checks'][name]['state']['pass_all'] and row['checks'][name]['exact_once']
                        f.install_cache(block);again=call('async',True);torch.cuda.synchronize()
                        row['repeat_async']=all(byte_equal(v,again[k]) for k,v in values['async'].items() if k!='resources')
                        row['inputs_unchanged']=all(byte_equal(immutable[n],t) for n,t in tensors((args,kw)))
                        row['RNG_unchanged']=byte_equal(cpu,torch.random.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state())
                        row['parameter_identity_unchanged']=flags=={n:(x.data_ptr(),x.requires_grad) for n,x in block.named_parameters()}
                        r['rows'].append(row);save()
                        assert row['original_native']['bitwise_pass'] and all(x['pass'] for x in row['checks'].values())
                        assert row['repeat_async'] and row['inputs_unchanged'] and row['RNG_unchanged'] and row['parameter_identity_unchanged']
                        if batch==2:
                            expected_states=[];f.install_cache(block)
                            for update in (1,2,0):
                                v=call('whole',True,dict(kw,update_cache=update));torch.cuda.synchronize()
                                expected_states.append((v,visible_snapshot(block.attn1.attn_caches[kw['cache_name']])))
                            f.install_cache(block)
                            for update,(v,state) in zip((1,2,0),expected_states):
                                new=call('async',True,dict(kw,update_cache=update));torch.cuda.synchronize()
                                check=dict(update=update,intermediates={k:byte_equal(x,new[k]) for k,x in v.items()
                                    if k not in ('tile_stats','resources')},state=state_equal(state,visible_snapshot(block.attn1.attn_caches[kw['cache_name']])))
                                check['pass']=all(check['intermediates'].values()) and check['state']['pass_all']
                                row['cache_sequences'].append(check);save();assert check['pass'],'legal cache transition changed'
                            del expected_states,v,new
                        del values,states,again
                    elif a.mode=='timing':
                        modes=['native','whole','sync','async']
                        for name in modes:f.install_cache(block);call(name);torch.cuda.synchronize()
                        for repeat in range(6):
                            offset=(repeat+a.process_index)%4;order=modes[offset:]+modes[:offset]
                            for name in order:
                                guard();f.install_cache(block);torch.cuda.synchronize()
                                beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                beg.record();t=time.perf_counter();v=call(name);end.record();end.synchronize()
                                r['timing'].append(dict(case=case,name=name,repeat=repeat,process=a.process_index,
                                    stream_ms=beg.elapsed_time(end),host_ms=(time.perf_counter()-t)*1000))
                        f.install_cache(block)
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                            v=call('async');torch.cuda.synchronize()
                        trace=a.output/f'{case}-TRACE.json';prof.export_chrome_trace(str(trace))
                        es=json.loads(trace.read_text())['traceEvents'];kernels=[e for e in es if e.get('cat')=='kernel']
                        row['traces']['async']=[dict(name=e['name'],us=e['dur'],metadata=e['args']) for e in kernels]
                        row['resources']=v['resources'].tolist()
                        assert len(kernels)==1 and 'pipeline_async' in kernels[0]['name']
                        r['rows'].append(row);save();del v
                    else:
                        for repeat in range(3):
                            for name in ('old_profile','profile'):
                                guard();f.install_cache(block);v=call(name);torch.cuda.synchronize()
                                ticks=v['stage_ticks'].cpu().tolist()
                                r['profiles'].append(dict(case=case,name=name,repeat=repeat,resources=v['resources'].tolist(),
                                    kernel_ns=ticks[40][1]-ticks[40][0],stage_ns={name:ticks[i][1]-ticks[i][0] for i,name in enumerate(LABELS)}))
                        save();del v
                    del raw,f,args,kw,immutable
        r['status']='COMPLETE_'+a.mode.upper()+'_UNADMITTED_PROTOTYPE'
    except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.update(lifecycle_seconds=time.monotonic()-start,at_UTC=datetime.now(timezone.utc).isoformat());save()
if __name__=='__main__':main()

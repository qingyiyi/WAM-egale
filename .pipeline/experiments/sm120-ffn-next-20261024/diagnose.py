"""当前A的有限完整block成本测量；不是闭环任务或正式机器人性能。"""
from pathlib import Path
import argparse
import gc
import json
import os
import statistics
import time
import traceback
import torch
from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture, decode, visible_snapshot, tensors
from tools.metrics import compare, state_equal, byte_equal
from tools.time_whole_layer import capture, device_tree, load_extension, resource
from wam_ops.optimized.whole_layer import WholeLayerBlock

P=Path(__file__).resolve().parent
T=P/os.environ.get('WAM_T1_DIR','T1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    start=time.time();out=dict(status='RUNNING',rows=[],calls=0,before=resource(),scope='B14 block0; eager/no Graph; cached CrossKV')
    def save():(a.output/'RESULT.json').write_text(json.dumps(out,indent=2))
    def tick():
        out['calls']+=1
        if out['calls']>4000 or time.time()-start>240:raise TimeoutError('finite complete-block diagnostic cap')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    try:
        if out['before']['processes']:raise RuntimeError('visible GPU compute owner already exists')
        A=load_extension(P/'A.so','wam_whole_qkv_tma_cuda')
        oldroot=P.parent
        old=load_extension(next((oldroot/'projection-warp-specialized/self-qkv-tma-migration/build/profile').glob('*.so')),'wam_whole_qkv_tma_profile_cuda')
        D=load_extension(next((T/'build/profile').glob('*.so')),'wam_sm120_ffn_cost_profile_cuda')
        out['storage_sizes']=dict(zip(['Shared','TileShared','FFN2_union','FFN2_mainloop','FP32_result','union_plus_BF16_tile'],D.ffn_storage_sizes()))
        out['modules']={k:v.__file__ for k,v in [('A',A),('old_profile',old),('diagnostic',D)]}
        block=load_block();bind=WholeLayerBlock(block);torch.cuda.synchronize()
        owner=resource()['processes'].splitlines();out['owner']=owner
        if len(owner)!=1:raise RuntimeError('expected one owned GPU context')
        with torch.inference_mode():
            for name in ['short','middle','long']:
                f=BlockFixture(FIXTURES/f'input-{name}-B14-action-layer0-first.pt',device_tree(decode(torch.load(FIXTURES/f'input-{name}-B14-action-layer0-first.pt',map_location='cpu',mmap=True,weights_only=False))))
                args,kw=f.arguments();cache=f.install_cache(block);initial=visible_snapshot(cache)
                inputs={k:v.clone() for k,v in tensors((args,kw))};rng=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
                with capture(block) as ref:ref['output']=block(*args,**kw);tick()
                bind.prepare_context(args[1],precompute=True)
                common=(bind.weights,*args,[cache[k] for k in ['k','v','id','mask','is_pred']],kw['candidate_cache_rows'],0,bind.eps)
                def call(lib,debug=False):tick();return lib.forward(*common,debug,bind.context_rows,bind.context_roots,bind.context_kv)
                row=dict(case=name,precision={},samples=[],cost_samples=[],stage_spans_samples={});out['rows'].append(row)
                for tag,lib in [('A',A),('old_profile',old),('diagnostic',D)]:
                    y=call(lib,True)
                    row['precision'][tag]={k:compare(v.reshape(-1),y[k].reshape(-1)) for k,v in ref.items()}
                    row.setdefault('resources',{})[tag]=y['resources'].tolist()
                    if not all(v['bitwise_pass'] for v in row['precision'][tag].values()):raise RuntimeError('finite precision '+tag)
                    if tag=='diagnostic':
                        trace=y['ffn_numeric'];running=None;checks=[]
                        for s in range(7):
                            v=trace[s,0]
                            if s:v=v+running.float()
                            if s==6:v=v+bind.weights[26].float().reshape(1,-1)
                            running=v.to(torch.bfloat16)
                            checks.append(compare(trace[s,1],running.float()))
                        row['seven_round_recurrence']=checks
                        if not all(v['bitwise_pass'] for v in checks):raise RuntimeError('seven-round recurrence differs')
                        row['numeric_file']='numeric-'+name+'.pt';torch.save(trace.cpu(),a.output/row['numeric_file'])
                labels=['A1','A2','old_profile','diagnostic'];libs={'A1':A,'A2':A,'old_profile':old,'diagnostic':D}
                for _ in range(10):
                    for label in labels:call(libs[label])
                for rep in range(24):
                    for label in (labels if rep%2==0 else list(reversed(labels))):
                        b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);b.record()
                        for _ in range(5):y=call(libs[label])
                        e.record();e.synchronize()
                        row['samples'].append(dict(rep=rep,arm=label,event_ms=b.elapsed_time(e)/5))
                        if label!='A1' and label!='A2':
                            ticks=y['stage_ticks'].cpu();spans=((ticks[:,:,1].max(1).values-ticks[:,:,0].min(1).values).double()/1e6).tolist()
                            row['stage_spans_samples'].setdefault(label,[]).append(spans)
                        if label=='diagnostic':
                            cost=y['ffn_cost'].cpu();row['cost_samples'].append(cost.tolist())
                            if int(cost[:,5].sum())!=672:raise RuntimeError('FFN2 job coverage changed')
                row['median_ms']={label:statistics.median(v['event_ms'] for v in row['samples'] if v['arm']==label) for label in labels}
                row['stage_medians_ms']={tag:[statistics.median(v[k] for v in values) for k in range(17)] for tag,values in row['stage_spans_samples'].items()}
                row['input_unchanged']=all(byte_equal(inputs[k],v) for k,v in tensors((args,kw)))
                row['KV']=state_equal(initial,visible_snapshot(cache));row['RNG_unchanged']=byte_equal(rng[0],torch.get_rng_state()) and byte_equal(rng[1],torch.cuda.get_rng_state())
                if not row['input_unchanged'] or not row['KV']['pass_all'] or not row['RNG_unchanged']:raise RuntimeError('state mutation')
                after=resource()['processes'].splitlines()
                if [x.rsplit(',',1)[0] for x in after]!=[x.rsplit(',',1)[0] for x in owner]:raise RuntimeError('GPU ownership changed')
                save();del f,args,kw,cache,initial,inputs,ref,y,trace,running
                block.attn1.attn_caches={};bind.context_kv=[];gc.collect();torch.cuda.empty_cache()
        out['status']='COMPLETE_FINITE_COST_DIAGNOSTIC'
    except BaseException as exc:out.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:out['lifecycle_seconds']=time.time()-start;save()
    print(out['status'],out.get('error',''),flush=True)
    if out['status']!='COMPLETE_FINITE_COST_DIAGNOSTIC':raise SystemExit(1)


if __name__=='__main__':main()

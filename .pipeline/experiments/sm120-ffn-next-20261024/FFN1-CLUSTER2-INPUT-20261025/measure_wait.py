"""软件等待位置诊断；完整真实block，不能用观察器代替正常计时。"""
from pathlib import Path
from datetime import datetime
import json,time,statistics as st,traceback
import torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot,tensors
from tools.time_whole_layer import device_tree,load_extension,capture,resource
from tools.metrics import compare,state_equal,byte_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock

P=Path(__file__).resolve().parent
torch.set_num_threads(4);torch.set_num_interop_threads(1)
job=json.loads((P/'JOB.json').read_text())
deadline=datetime.fromisoformat(job['absolute_cleanup_deadline']).timestamp()-job['cleanup_reserve_seconds']
result=dict(status='RUNNING',start=time.time(),calls=0,rows=[],before=resource(),names=['ready_wait','peer_credit_wait','input_empty_wait','result_wait','bulk_issue','role_loop'])
def save(): (P/'WAIT-RESULT.json').write_text(json.dumps(result,indent=2))
try:
    if result['before']['processes']:raise RuntimeError('GPU already occupied')
    normal=load_extension(next((P/'build/normal').glob('*.so')),'wam_sm120_ffn1_cluster_input_cuda')
    observer=load_extension(next((P/'build/wait').glob('*.so')),'wam_sm120_ffn1_cluster_input_wait_cuda')
    block=load_block();bind=WholeLayerBlock(block)
    with torch.inference_mode():
        for case in ('short','middle','long'):
            path=FIXTURES/f'input-{case}-B14-action-layer0-first.pt'
            f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
            args,kw=f.arguments();cache=f.install_cache(block)
            initial=visible_snapshot(cache);inputs={k:v.clone() for k,v in tensors((args,kw))}
            rng=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
            with capture(block) as target:target['output']=block(*args,**kw);result['calls']+=1
            bind.prepare_context(args[1],precompute=True)
            cl=[cache[k] for k in ('k','v','id','mask','is_pred')]
            common=(bind.weights,*args,cl,kw['candidate_cache_rows'],0,bind.eps)
            def call(lib,debug=False):
                if time.time()>deadline or result['calls']>=1000:raise TimeoutError('wait job budget')
                result['calls']+=1
                return lib.forward(*common,debug,bind.context_rows,bind.context_roots,bind.context_kv)
            observed=call(observer,True);baseline=call(normal,True)
            exact={k:compare(target[k].reshape(-1),observed[k].reshape(-1)) for k in target}
            prior=torch.load(P/'measure-1'/f'numeric-candidate-{case}.pt',map_location='cpu',weights_only=False)
            seven=compare(prior,observed['ffn_numeric'].cpu())
            row=dict(case=case,precision=exact,seven=seven,observer_resources=observed['resources'].tolist(),normal_resources=baseline['resources'].tolist(),samples=[],wait_samples=[],stage_samples=[])
            result['rows'].append(row);save()
            if not all(v['bitwise_pass'] for v in exact.values()) or not seven['bitwise_pass']:raise RuntimeError('observer precision changed')
            if not torch.equal(observed['ffn1_transfer_counts'],baseline['ffn1_transfer_counts']):raise RuntimeError('observer transfer count changed')
            for _ in range(10):call(normal);call(observer)
            for rep in range(16):
                seq=[('normal',normal),('observer',observer)]
                if rep%2:seq.reverse()
                for label,lib in seq:
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();a.record()
                    for _ in range(5):out=call(lib)
                    b.record();b.synchronize()
                    row['samples'].append(dict(rep=rep,arm=label,event_ms=a.elapsed_time(b)/5))
                    if label=='observer':
                        row['wait_samples'].append(out['ffn1_wait_trace'].cpu().tolist())
                        ticks=out['stage_ticks'].cpu()
                        row['stage_samples'].append(((ticks[:,:,1].max(1).values-ticks[:,:,0].min(1).values).double()/1e6).tolist())
            row['median_ms']={label:st.median(v['event_ms'] for v in row['samples'] if v['arm']==label) for label in ('normal','observer')}
            row['observer_overhead_pct']=100*(row['median_ms']['observer']/row['median_ms']['normal']-1)
            raw=torch.tensor(row.pop('wait_samples'),dtype=torch.int64)
            torch.save(raw,P/f'wait-raw-{case}.pt')
            per_cta=raw.double().median(0).values
            row['wait_by_CTA_us']=(per_cta/1000).tolist()
            row['rank_summary']={}
            for rank in (0,1):
                indices=torch.arange(rank,170,2)
                steps=torch.tensor([48*((224-int(i)//2+84)//85) for i in indices])
                prod=per_cta[indices,0]
                row['rank_summary'][str(rank)]=dict(
                    producer_median_us=(prod.median(0).values/1000).tolist(),
                    producer_per_K64_median_us=((prod/steps[:,None]).median(0).values/1000).tolist(),
                    MMA_ready_per_K64_median_us=(per_cta[indices,4:8,0]/steps[:,None]/1000).median().item(),
                    role_loop_max_us=(per_cta[indices,:,5].max()/1000).item())
            row['valid_KV']=state_equal(initial,visible_snapshot(cache))
            row['inputs_unchanged']=all(byte_equal(inputs[k],v) for k,v in tensors((args,kw)))
            row['RNG_unchanged']=byte_equal(rng[0],torch.get_rng_state()) and byte_equal(rng[1],torch.cuda.get_rng_state())
            if not row['valid_KV']['pass_all'] or not row['inputs_unchanged'] or not row['RNG_unchanged']:raise RuntimeError('observer state changed')
            save();block.attn1.attn_caches={};bind.context_kv=[]
    result['status']='COMPLETE_EXACT_WAIT_OBSERVER'
except BaseException as e:result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
finally:result.update(end=time.time(),after=resource());save();print(result['status'],result.get('error',''),flush=True)
if result['status']!='COMPLETE_EXACT_WAIT_OBSERVER':raise SystemExit(1)

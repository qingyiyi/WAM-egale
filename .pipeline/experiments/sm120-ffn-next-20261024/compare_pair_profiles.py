"""Same-round embedded-region diagnosis; no normal whole-layer claims from profiling."""
from pathlib import Path
from datetime import datetime
import argparse,ctypes,gc,json,os,statistics,time,traceback
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot,tensors
from tools.metrics import compare,state_equal,byte_equal
from tools.time_whole_layer import capture,device_tree,load_extension,resource
from wam_ops.optimized.whole_layer import WholeLayerBlock
P=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    start=time.time();deadline=start+180
    job=dict(goal='Interleaved current-A versus all numerically qualified FFN candidates, complete embedded-block regions',wall_cap_seconds=180,cpu_threads=4,storage_cap_bytes=33554432,call_cap=3000,cleanup_reserve_seconds=30,
        baseline='all original stage intervals in current A; profiling only, normal performance decisions remain separate',start=datetime.now().astimezone().isoformat(),authorization='existing unlimited 5090 research; original receipts preserved',flags='all scientific flags false')
    (args.output/'JOB.json').write_text(json.dumps(job,indent=2))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    result=dict(status='RUNNING',rows=[],calls=0,before=resource(),profiles_only=True)
    def save():(args.output/'RESULT.json').write_text(json.dumps(result,indent=2))
    def tick():
        result['calls']+=1
        if result['calls']>job['call_cap'] or time.time()>deadline-30:raise TimeoutError('registered finite diagnostic limit')
    try:
        if result['before']['processes']:raise RuntimeError('GPU already has visible compute contexts')
        identities={'A':('projection-warp-specialized/self-qkv-tma-migration','wam_whole_qkv_tma_profile_cuda'),
                    'PAIR':('sm120-ffn-next-20261024/T2-r1','wam_sm120_ffn_pair_profile_cuda')}
        modules={tag:load_extension(next((P.parent/folder/'build/profile').glob('*.so')),module) for tag,(folder,module) in identities.items()}
        result['modules']={k:v.__file__ for k,v in modules.items()}
        block=load_block();bind=WholeLayerBlock(block);torch.cuda.synchronize()
        owner=resource()['processes'].splitlines();result['owner']=owner
        driver=ctypes.CDLL('libcuda.so.1')
        result['device_attributes']={}
        for name,attribute in [('L2_bytes',38),('max_shared_optin_bytes',97)]:
            value=ctypes.c_int(-1);code=driver.cuDeviceGetAttribute(ctypes.byref(value),attribute,0)
            result['device_attributes'][name]={'status':int(code),'value':value.value}

        if len(owner)!=1:raise RuntimeError('expected single CUDA process')
        with torch.inference_mode():
            for case in ['short','middle','long']:
                path=FIXTURES/f'input-{case}-B14-action-layer0-first.pt';fixture=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
                a,kw=fixture.arguments();cache=fixture.install_cache(block);initial=visible_snapshot(cache)
                inputs={k:v.clone() for k,v in tensors((a,kw))};rng=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
                with capture(block) as ref:ref['output']=block(*a,**kw);tick()
                bind.prepare_context(a[1],precompute=True)
                cl=[cache[k] for k in ['k','v','id','mask','is_pred']]
                common=(bind.weights,*a,cl,kw['candidate_cache_rows'],0,bind.eps)
                def own(tag,debug=False):
                    tick();return modules[tag].forward(*common,debug,bind.context_rows,bind.context_roots,bind.context_kv)
                row=dict(case=case,precision={},samples=[]);result['rows'].append(row)
                for tag in modules:
                    y=own(tag,True);row['precision'][tag]={k:compare(ref[k].reshape(-1),y[k].reshape(-1)) for k in ref}
                    if not all(v['bitwise_pass'] for v in row['precision'][tag].values()):raise RuntimeError('profile finite precision failed: '+tag)
                labels=list(modules)
                for _ in range(10):
                    for tag in labels:own(tag)
                for rep in range(24):
                    order=labels if rep%2==0 else list(reversed(labels))
                    for tag in order:
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(5):y=own(tag)
                        end.record();end.synchronize()
                        ts=y['stage_ticks'].cpu();spans=((ts[:,:,1].max(1).values-ts[:,:,0].min(1).values).double()/1e6).tolist()
                        row['samples'].append(dict(rep=rep,arm=tag,event_ms=begin.elapsed_time(end)/5,regions_ms=spans))
                row['region_medians_ms']={tag:[statistics.median(x['regions_ms'][j] for x in row['samples'] if x['arm']==tag) for j in range(17)] for tag in labels}
                row['ffn_speed_A_over_candidate']={tag:row['region_medians_ms']['A'][15]/row['region_medians_ms'][tag][15] for tag in labels}
                # Re-measure the canonical eager mathematical regions in the same job.
                fullkv=[v.index_select(0,bind.context_rows) for v in bind.context_kv]
                table=block.scale_shift_table[None]+a[2].float()
                shift,scale,gate,cshift,cscale,cgate=[x.squeeze(1) for x in table.permute(0,2,1,3).chunk(6,1)]
                def rope(x):
                    x=x.unflatten(2,(24,128));z=torch.view_as_complex(x.double().reshape(*x.shape[:3],-1,2))
                    return torch.view_as_real(z*a[3]).flatten(3).to(x.dtype)
                def dense_self():
                    hk=cache['k'][:,initial['slots']].index_select(0,kw['candidate_cache_rows'])
                    hv=cache['v'][:,initial['slots']].index_select(0,kw['candidate_cache_rows'])
                    return block.attn1.attn_op(ref['self_q_rope'],torch.cat((hk,ref['self_k_rope']),1),torch.cat((hv,ref['self_v'].unflatten(2,(24,128))),1))
                operations={
                    1:lambda:(block.norm1(a[0].float())*(1+scale)+shift).to(a[0].dtype),
                    2:lambda:(block.attn1.to_q(ref['modulated1']),block.attn1.to_k(ref['modulated1']),block.attn1.to_v(ref['modulated1'])),
                    3:lambda:rope(block.attn1.norm_q(ref['self_q_linear'])),
                    4:lambda:rope(block.attn1.norm_k(ref['self_k_linear'])),5:dense_self,
                    6:lambda:block.attn1.to_out[0](ref['self_attention'].flatten(2)),
                    7:lambda:(a[0].float()+ref['self_out']*gate).to(a[0].dtype),
                    8:lambda:block.norm2(ref['residual1'].float()).to(a[0].dtype),
                    9:lambda:block.attn2.to_q(ref['norm2']),10:lambda:block.attn2.norm_q(ref['cross_q_linear']),
                    11:lambda:block.attn2.attn_op(ref['cross_q'].unflatten(2,(24,128)),fullkv[1].unflatten(2,(24,128)),fullkv[2].unflatten(2,(24,128))),
                    12:lambda:block.attn2.to_out[0](ref['cross_attention'].flatten(2)),
                    13:lambda:ref['residual1']+ref['cross_out'],
                    14:lambda:(block.norm3(ref['residual2'].float())*(1+cscale)+cshift).to(a[0].dtype),
                    15:lambda:(ref['residual2'].float()+block.ffn(ref['modulated3']).float()*cgate).to(a[0].dtype)}
                row['torch_region_ms']={}
                for region,op in operations.items():
                    for _ in range(10):op()
                    values=[]
                    for rep in range(12):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(20):z=op()
                        end.record();end.synchronize();values.append(begin.elapsed_time(end)/20)
                    row['torch_region_ms'][str(region)]=statistics.median(values)
                row['input_unchanged']=all(byte_equal(inputs[k],v) for k,v in tensors((a,kw)))
                row['KV_unchanged']=state_equal(initial,visible_snapshot(cache));row['RNG_unchanged']=byte_equal(rng[0],torch.get_rng_state()) and byte_equal(rng[1],torch.cuda.get_rng_state())
                if not row['input_unchanged'] or not row['KV_unchanged']['pass_all'] or not row['RNG_unchanged']:raise RuntimeError('state changed')
                after=resource()['processes'].splitlines()
                if [x.rsplit(',',1)[0] for x in after] != [x.rsplit(',',1)[0] for x in owner]:raise RuntimeError('GPU ownership changed')
                save();del fixture,a,kw,cache,initial,inputs,ref,y,fullkv
                block.attn1.attn_caches={};bind.context_kv=[];gc.collect();torch.cuda.empty_cache()
        result['status']='COMPLETE_FINITE_DIAGNOSTIC'
    except BaseException as e:result.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
    finally:result['lifecycle_seconds']=time.time()-start;save()
    print(result['status'],result.get('error',''),flush=True)
    if result['status']!='COMPLETE_FINITE_DIAGNOSTIC':raise SystemExit(1)
if __name__=='__main__':main()

"""Nine real signatures, direct original Torch state/output contract, no robot."""
import json,os,time,traceback
from pathlib import Path
import torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot,tensors
from tools.replay import device_tree
from tools.metrics import compare,byte_equal,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock, extension
B=Path(os.environ['WAM_RUN_DIR']);os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
t0=time.time();r=dict(status='RUNNING',full_block_calls=0,rows=[],production_admitted=False,formal_performance_permission=False)
def save():(B/'SIMPLE-GATE-RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 ext=extension();assert ext.context_tile_rows==int(os.environ['WAM_CONTEXT_TILE_M']) and ext.point_tile_elements==2048 and ext.cooperative_epilogue
 block=load_block('cuda');native=NativeBlock(block);new=WholeLayerBlock(block)
 work=Path(load_block.__code__.co_filename).resolve().parents[2];video=work/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
 with torch.inference_mode():
  for kind,batch in [('action',2),('action',14),('video',2)]:
   for length in ['short','middle','long']:
    assert time.time()-t0<500 and r['full_block_calls']+(10 if batch==2 else 4)<=72
    p=(FIXTURES if kind=='action' else video)/f'input-{length}-B{batch}-{kind}-layer0-first.pt';d=decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False));f=BlockFixture(p,device_tree(d));a,k=f.arguments()
    inputs={n:t.clone() for n,t in tensors((a,k))};cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state()
    def call(name,update=None):
     assert r['full_block_calls']<72
     r['full_block_calls']+=1;save();kw=dict(k) if update is None else dict(k,update_cache=update)
     return block(*a,**kw) if name=='Torch' else (native(*a,**kw,debug=True) if name=='native' else new(*a,**kw,debug=True))
    f.install_cache(block);ref=call('Torch');torch.cuda.synchronize();cache_ref=visible_snapshot(block.attn1.attn_caches[k['cache_name']])
    f.install_cache(block);n=call('native');f.install_cache(block);x=call('new');torch.cuda.synchronize();cache_new=visible_snapshot(block.attn1.attn_caches[k['cache_name']])
    row=dict(case=f'{length}-B{batch}-{kind}',fixture=str(p),native_vs_Torch=compare(ref,n['output']),new_vs_Torch=compare(ref,x['output']),cache_vs_Torch=state_equal(cache_ref,cache_new),stages={})
    for key in n:
     if key in x and key not in ('output','tile_stats','cache_info','resources'):
      row['stages'][key]=compare(n[key].reshape(-1),x[key].reshape(-1))
    repeat_output=x['output'].clone();f.install_cache(block);repeat=call('new');torch.cuda.synchronize()
    row['repeat_output']=compare(repeat_output,repeat['output']);row['repeat_cache']=state_equal(cache_new,visible_snapshot(block.attn1.attn_caches[k['cache_name']]));row['repeat_exact']=row['repeat_output']['bitwise_pass'] and row['repeat_cache']['pass_all'];del repeat,repeat_output
    row['tile_visit_audit']='NOT_COMPILED';row['resources']=x['resources'].tolist();
    row['compiled_tile_rows']=ext.context_tile_rows
    row['legal_state_sequences']=[]
    if batch==2:
     states=[];outputs=[];f.install_cache(block)
     for u in (1,2,0):
      z=call('Torch',u);torch.cuda.synchronize();states.append(visible_snapshot(block.attn1.attn_caches[k['cache_name']]));outputs.append(z.clone())
     f.install_cache(block)
     for u,state,out in zip((1,2,0),states,outputs):
      z=call('new',u);torch.cuda.synchronize();row['legal_state_sequences'].append(dict(update=u,output=compare(out,z['output']),cache=state_equal(state,visible_snapshot(block.attn1.attn_caches[k['cache_name']]))))
     del states,outputs,z
    row['inputs_unchanged']=all(byte_equal(inputs[n],t) for n,t in tensors((a,k)));row['RNG_unchanged']=byte_equal(cpu,torch.random.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state())
    r['rows'].append(row);save();del ref,n,x,cache_ref,cache_new,inputs,d,f,a,k
 r['all_pass']=len(r['rows'])==9 and all(x['native_vs_Torch']['bitwise_pass'] and x['new_vs_Torch']['bitwise_pass'] and x['cache_vs_Torch']['pass_all'] and all(z['bitwise_pass'] for z in x['stages'].values()) and x['repeat_exact'] and len(x['stages'])==23 and x['inputs_unchanged'] and x['RNG_unchanged'] and all(z['output']['bitwise_pass'] and z['cache']['pass_all'] for z in x['legal_state_sequences']) for x in r['rows']);r['output_pass_count']=sum(x['new_vs_Torch']['bitwise_pass'] for x in r['rows']);r['status']='COMPLETE_FINITE_GATE_NOT_PRODUCTION_ADMISSION' if r['all_pass'] else 'FAILED_FINITE_GATE'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-t0;save()

print(r["status"],r.get("error",""),flush=True)
if not r.get("all_pass"): raise SystemExit(1)

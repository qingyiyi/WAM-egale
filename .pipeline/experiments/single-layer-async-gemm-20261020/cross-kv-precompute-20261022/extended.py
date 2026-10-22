"""Finite structural controls derived from actual block inputs; not new robot data."""
from pathlib import Path
import os,json,time,traceback,importlib.util
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot,tensors
from tools.replay import device_tree
from tools.metrics import compare,byte_equal,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
import wam_ops.optimized.whole_layer as wl
B=Path(__file__).resolve().parent;reg=json.loads((B/'JOB.json').read_text());os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
sp=importlib.util.spec_from_file_location('candidate.wam_whole_fa2_cuda',reg['candidate_binary']);ext=importlib.util.module_from_spec(sp);sp.loader.exec_module(ext);wl.extension=lambda:ext
out=dict(status='RUNNING',full_block_calls=0,rows=[],scope='structural counterfactuals from serialized real inputs; no quality certificate',production_admitted=False);start=time.time()
def save():(B/'STRUCTURAL-RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
try:
 block=load_block('cuda');native=NativeBlock(block);owned=WholeLayerBlock(block)
 with torch.inference_mode():
  for length in ['short','middle','long']:
   path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
   for variant in reg['variants']:
    args,kw=f.arguments();kw=dict(kw)
    if variant=='cfg_alternating':kw['candidate_cache_rows']=torch.arange(14,device='cuda',dtype=kw['candidate_cache_rows'].dtype)%2
    elif variant=='cfg_swap':kw['candidate_cache_rows']=1-kw['candidate_cache_rows']
    def install():
     c=f.install_cache(block);slots=c['mask'].nonzero(as_tuple=False).flatten()
     if variant.startswith('drop_'):removed=slots[-int(variant.split('_')[1]):]
     elif variant=='holes16':removed=slots[3:3+16*4:4]
     else:removed=slots[:0]
     c['mask'][removed]=False;c['id'][removed]=-1;c['is_pred'][removed]=False
     return c
    if variant=='context_alternating':
     perm=torch.tensor([0,7,1,8,2,9,3,10,4,11,5,12,6,13],device='cuda');args=(args[0],args[1].index_select(0,perm),*args[2:])
    elif variant=='context_swap':args=(args[0],args[1].roll(7,0),*args[2:])
    owned.prepare_context(args[1], precompute=True)
    input_before={n:x.clone() for n,x in tensors((args,kw,owned.context_kv))};cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();c=install();before=visible_snapshot(c)
    record=dict(case=length,variant=variant,fixture=str(path),valid_history=int(c['mask'].sum().item()),cfg_rows=kw['candidate_cache_rows'].tolist(),context_rows=owned.context_rows.tolist(),context_roots=owned.context_roots.tolist())
    values={};states={}
    for name in ['Torch','native','owned','repeat']:
     c=install();out['full_block_calls']+=1;assert out['full_block_calls']<=130;save()
     z=block(*args,**kw) if name=='Torch' else (native(*args,**kw,debug=True) if name=='native' else owned(*args,**kw,debug=(name!='repeat')))
     torch.cuda.synchronize();z={'output':z} if name=='repeat' else z;values[name]=z;states[name]=state_equal(before,visible_snapshot(c));assert states[name]['pass_all']
    record['output']={n:compare(values['Torch'],values[n]['output']) for n in ['native','owned','repeat']}
    record['stages']={k:compare(values['native'][k].reshape(-1),values['owned'][k].reshape(-1)) for k in values['native'].keys()&values['owned'].keys() if k not in ['output','resources','tile_stats','cache_info']}
    record['state']=states;record['inputs_unchanged']=all(byte_equal(input_before[n],x) for n,x in tensors((args,kw,owned.context_kv)));record['rng_unchanged']=byte_equal(cpu,torch.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state())
    record['pass']=all(x['bitwise_pass'] for x in record['output'].values()) and len(record['stages'])==23 and all(x['bitwise_pass'] for x in record['stages'].values()) and record['inputs_unchanged'] and record['rng_unchanged'];out['rows'].append(record);save();del values,states,c,before,input_before
 out['all_pass']=len(out['rows'])==27 and all(x['pass'] for x in out['rows']);out['status']='COMPLETE_FINITE_STRUCTURAL_GATE' if out['all_pass'] else 'FAILED_FINITE_STRUCTURAL_GATE'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error',''),flush=True)
if not out.get('all_pass'):raise SystemExit(1)

"""Admit the same-context-work Torch control separately before timing."""
import os,json,time,traceback
from pathlib import Path
import torch
from baselines import grouped_cross
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
from wam_ops.optimized.native import NativeBlock

B=Path(__file__).resolve().parent
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
start=time.time();out={'status':'RUNNING','rows':[],'full_calls':0}
try:
    block=load_block('cuda');native=NativeBlock(block);binding=WholeLayerBlock(block)
    with torch.inference_mode():
        for length in ['short','middle','long']:
            f=BlockFixture.load(FIXTURES,length,14);f.data=device_tree(f.data);a,k=f.arguments()
            binding.prepare_context(a[1]);compact=a[1].index_select(0,binding.context_roots)
            c=f.install_cache(block);before=visible_snapshot(c)
            ref=native(*a,**k,debug=True);out['full_calls']+=1
            f.install_cache(block)
            with grouped_cross(block,compact,binding.context_rows):
                grouped=block(*a,**k);out['full_calls']+=1
            result=compare(ref['output'],grouped)
            state=state_equal(before,visible_snapshot(block.attn1.attn_caches[k['cache_name']]))
            out['rows'].append(dict(case=length,output=result,cache=state))
    out['grouped_baseline_admitted']=all(x['output']['bitwise_pass'] and x['cache']['pass_all'] for x in out['rows'])
    out['status']='COMPLETE_BASELINE_SCREEN'
except BaseException as e:
    out.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
finally:
    out['lifecycle_seconds']=time.time()-start
    (B/'BASELINE-GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['status'],out.get('grouped_baseline_admitted'),out.get('error',''))
if out['status']=='FAILED':raise SystemExit(1)

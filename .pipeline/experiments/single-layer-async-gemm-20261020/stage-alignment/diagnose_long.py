from pathlib import Path
import importlib.util,json,os,time,torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode
from tools.replay import device_tree
from tools.metrics import compare
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
spec=importlib.util.spec_from_file_location('wam_complex_before_cuda',next(B.glob('wam_complex_before_cuda*.so')));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
block=load_block('cuda');new=WholeLayerBlock(block);native=NativeBlock(block)
with torch.inference_mode():
 p=FIXTURES/'input-long-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments()
 f.install_cache(block);n=native(*a,**k,debug=True)
 c=f.install_cache(block);o=old.forward(new.weights,*a,[c[z] for z in ('k','v','id','mask','is_pred')],k.get('candidate_cache_rows'),k.get('update_cache',0),new.eps,False,True,False)
 f.install_cache(block);s=new(*a,**k,debug=True);torch.cuda.synchronize()
 r=dict(full_block_calls=3,status='COMPLETE_DIAGNOSTIC',matches={z:compare(o[z].reshape(-1),s[z].reshape(-1)) for z in o.keys()&s.keys()},native_simple={z:compare(n[z].reshape(-1),s[z].reshape(-1)) for z in n.keys()&s.keys()},cache_info=s['cache_info'].tolist())
 valid=int(s['cache_info'][0]);r['valid_slot_comparison']=compare(o['logical_slots'][:valid],s['logical_slots'][:valid]);r['slot_buffer_length']=s['logical_slots'].numel()
 (B/'LONG-DIAGNOSTIC.json').write_text(json.dumps(r,indent=2)+'\n')
 print('differences',[(z,v) for z,v in r['matches'].items() if not v['bitwise_pass']],flush=True)

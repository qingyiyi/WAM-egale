"""同一真实输入交错测 native / before / simple 完整单层；准备不计时。"""
from pathlib import Path
import importlib.util,json,os,sys,time,traceback,subprocess
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock,extension
B=Path(__file__).resolve().parent;process=int(sys.argv[1]);t0=time.time()
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
out=dict(status='RUNNING',process=process,full_block_calls=0,rows=[],samples=[],traces=[],formal_performance_permission=False,boundary='complete uncached single block eager, preparation excluded; profiler separate')
def save():(B/f'TIMING-{process}.json').write_text(json.dumps(out,indent=2)+'\n')
try:
 # Distinct module handles retain old binary without replacing the current wrapper.
 old_path=next(B.glob('wam_simple_before_cuda*.so'))
 spec=importlib.util.spec_from_file_location('wam_simple_before_cuda',old_path)
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 simple_ext=extension();block=load_block('cuda');native=NativeBlock(block);simple=WholeLayerBlock(block)
 out['before_DSO']=str(old_path);out['simple_DSO']=extension(True).__file__;out['generic_DSO']=simple_ext.__file__
 with torch.inference_mode():
  for length in ['short','middle','long']:
   p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt'
   out.setdefault('clock_samples',[]).append(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=timestamp,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw','--format=csv,noheader'],text=True))
   fixture=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))))
   args,kw=fixture.arguments()
   load=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
   assert os.environ.get('GPU_PRELAUNCH_EMPTY')=='1' and len([x for x in load.splitlines() if x.startswith('GPU-c3fa8d6f')])==1
   def call(name):
    assert out['full_block_calls']<165
    out['full_block_calls']+=1
    if name=='native':return native(*args,**kw,debug=True)
    if name=='simple':return simple(*args,**kw,debug=True)
    cache=block.attn1.attn_caches[kw['cache_name']]
    return old.forward(simple.weights,*args,[cache[x] for x in ('k','v','id','mask','is_pred')],kw.get('candidate_cache_rows'),kw.get('update_cache',0),simple.eps)
   values={};states={}
   for warm in range(6):
    for name in ['native','before','simple']:
     fixture.install_cache(block);values[name]=call(name);torch.cuda.synchronize();states[name]=visible_snapshot(block.attn1.attn_caches[kw['cache_name']])
   matches={name:compare(values['native']['output'],values[name]['output']) for name in ['before','simple']}
   stages={k:compare(values['before'][k].reshape(-1),values['simple'][k].reshape(-1)) for k in values['before'].keys()&values['simple'].keys() if k not in ['resources','cache_info','tile_stats','logical_slots','attention_meta']}
   old_meta,new_meta=values['before']['attention_meta'],values['simple']['attention_meta']
   stages['attention_logical_metadata']=compare(old_meta[:,:3],new_meta[:,:3])
   out.setdefault('attention_resource_metadata',[]).append(dict(case=length,before=old_meta.tolist(),simple=new_meta.tolist(),meaning='shared workspace bytes intentionally changed; logical length/split/Q still exact'))
   old_fa_bytes=max(128*128+2*64*128,64*128+2*128*128)*2
   new_fa_bytes=(64*128+2*64*128)*2
   assert bool((old_meta[:,3]==old_fa_bytes).all()) and bool((new_meta[:,3]==new_fa_bytes).all())
   valid=int(values['simple']['cache_info'][0]);stages['valid_logical_slots']=compare(values['before']['logical_slots','attention_meta'][:valid],values['simple']['logical_slots','attention_meta'][:valid])
   out['current_comparison']=dict(case=length,matches=matches,stages=stages);save()
   assert all(x['bitwise_pass'] for x in matches.values()) and all(x['bitwise_pass'] for x in stages.values())
   state_match={name:state_equal(states['native'],states[name]) for name in ['before','simple']}
   assert all(x['pass_all'] for x in state_match.values())
   out['rows'].append(dict(case=length,fixture=str(p),matches=matches,before_simple_intermediates=stages,valid_KV=state_match,resources={name:values[name]['resources'].tolist() for name in ['before','simple']},load=load))
   del values,states
   # Each order and reverse used; independent process shifts starting order.
   permutations=[('native','before','simple'),('simple','before','native'),('before','simple','native'),('native','simple','before'),('simple','native','before'),('before','native','simple')]
   for i in range(12):
    for name in permutations[(i+process-1)%6]:
     fixture.install_cache(block);torch.cuda.synchronize()
     a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     a.record();host=time.perf_counter();v=call(name);z.record();z.synchronize()
     out['samples'].append(dict(case=length,name=name,repeat=i,ms=a.elapsed_time(z),host_ms=(time.perf_counter()-host)*1000))
   fixture.install_cache(block)
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    v=call('simple');torch.cuda.synchronize()
   path=B/f'PROFILE-{process}-{length}.trace.json';prof.export_chrome_trace(str(path))
   kernels=[x['name'] for x in json.loads(path.read_text())['traceEvents'] if x.get('cat')=='kernel']
   assert len(kernels)==1,kernels
   out['traces'].append(dict(case=length,path=str(path),kernels=kernels));save()
   del fixture,args,kw,v
 out.pop('current_comparison',None);out['status']='COMPLETE_SIMPLE_BEFORE_NATIVE_COMPARISON'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out.update(lifecycle_seconds=time.time()-t0);save()
print(out['status'],out.get('error',''),flush=True)
if out['status']!='COMPLETE_SIMPLE_BEFORE_NATIVE_COMPARISON':raise SystemExit(1)

"""CUPTI launch metadata on a real middle B14 fixture; not latency evidence."""
from pathlib import Path
import json,os,time,traceback,subprocess,importlib.util
import torch
import torch.nn.functional as F
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode
from tools.replay import device_tree
from tools.metrics import compare
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
from wam_ops.optimized.prepared import extension,guarded_group
B=Path(__file__).resolve().parent;start=time.time();r=dict(status='RUNNING',calls=0,formal_performance_permission=False)
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
try:
 reg=json.loads((B/'JOB.json').read_text())
 import wam_ops.optimized.whole_layer as wl
 spec=importlib.util.spec_from_file_location('candidate.wam_whole_fa2_cuda',reg['candidate_binary']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);wl.extension=lambda:mod
 b=load_block('cuda');native=NativeBlock(b);bound=WholeLayerBlock(b)
 path=FIXTURES/'input-middle-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
 a,k=f.arguments();f.install_cache(b)
 with torch.inference_mode():
  v=native(*a,**k,debug=True);r['calls']+=1
  owned=bound(*a,**k,debug=True);r['calls']+=1
  assert all(torch.equal(v[key].reshape(-1),owned[key].reshape(-1)) for key in v.keys()&owned.keys())
  stages={3:('SelfQ',v['modulated1'].reshape(14,16,3072),3,'self_q_linear'),16:('CrossK',a[1],15,'cross_k_linear'),25:('FFN1',v['modulated3'].reshape(14,16,3072),23,'ffn1'),27:('FFN2',v['gelu'].reshape(14,16,14336),25,'ffn2')}
  calls={};refs={};shapes={}
  for sid,(name,x,wi,key) in stages.items():
   label='NATIVE_STAGE_'+str(sid)+'_'+name;calls[label]=lambda x=x,wi=wi:F.linear(x,bound.weights[wi],bound.weights[wi+1]);refs[label]=owned[key];shapes[label]=[list(x.shape),list(bound.weights[wi].shape)]
  cache=b.attn1.attn_caches[k['cache_name']];slots=cache['mask'].nonzero(as_tuple=False).squeeze(-1);valid=slots.numel();group=guarded_group(valid);candidate=extension('wam_candidate_cuda')
  p=Path('/root/cxz/WAM-egale/投机采样/experiments/shared_prefix_torch_v2/_shared_torch_v1.cpython-312-x86_64-linux-gnu.so');sp=importlib.util.spec_from_file_location('_shared_torch_v1',p);shared=importlib.util.module_from_spec(sp);sp.loader.exec_module(shared)
  q=v['self_q_rope'].reshape(14,16,24,128);ck=v['self_k_rope'].reshape(14,16,24,128);cv=v['self_v'].reshape(14,16,24,128)
  calls['SELF_GUARDED']=lambda:candidate.grouped(q,cache['k'],cache['v'],ck,cv,slots,group);refs['SELF_GUARDED']=v['self_attention']
  calls['SELF_SHARED_TORCH']=lambda:shared.shared(q,cache['k'],cache['v'],ck,cv,slots,k['candidate_cache_rows']);refs['SELF_SHARED_TORCH']=v['self_attention']
  cq=v['cross_q'].reshape(14,16,24,128);kk=v['cross_k'].reshape(14,512,24,128);vv=v['cross_v'].reshape(14,512,24,128)
  calls['CROSS_TORCH']=lambda:b.attn2.attn_op(cq,kk,vv);refs['CROSS_TORCH']=v['cross_attention']
  results={}
  for _ in range(30):
   for label,fn in calls.items():results[label]=fn();r['calls']+=1
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
   for label,fn in calls.items():
    with torch.profiler.record_function(label):results[label]=fn();r['calls']+=1
   torch.cuda.synchronize()
  prof.export_chrome_trace(str(B/'NATIVE-LAUNCH-TRACE.json'))
  trace=json.loads((B/'NATIVE-LAUNCH-TRACE.json').read_text());kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel'];r['kernel_events']=len(kernels);assert len(kernels)>=7,'CUDA launch metadata unavailable'
  r['precision']={label:compare(refs[label].reshape(-1),y.reshape(-1)) for label,y in results.items()};assert all(z['bitwise_pass'] for z in r['precision'].values())
 r.update(status='COMPLETE_LAUNCH_METADATA_NOT_PERFORMANCE',fixture=str(path),shapes=shapes,shared_prefix=valid,guarded_group=group)
 r['load']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-start;(B/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
print(r['status'],r.get('error',''))
if r['status']!='COMPLETE_LAUNCH_METADATA_NOT_PERFORMANCE':raise SystemExit(1)

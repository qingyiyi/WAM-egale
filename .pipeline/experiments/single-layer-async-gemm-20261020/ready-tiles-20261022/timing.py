"""Prior versus simplified host entry B14: complete layer on real B14 inputs."""
from pathlib import Path
import os,json,time,sys,traceback,importlib.util,subprocess,statistics
import torch
from contextlib import nullcontext
from wam_ops.optimized.prepared import PreparedBlock
import wam_ops.optimized.prepared as prepared_module
from functools import lru_cache
original_extension=prepared_module.extension
@lru_cache(None)
def prepared_extension(name):
 if name != "_shared_torch_v1":return original_extension(name)
 path=Path("/root/cxz/WAM-egale/投机采样/experiments/shared_prefix_torch_v2/_shared_torch_v1.cpython-312-x86_64-linux-gnu.so")
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
prepared_module.extension=prepared_extension
import ctypes
lib=ctypes.CDLL('libnvidia-ml.so.1');lib.nvmlInit_v2();handle=ctypes.c_void_p()
lib.nvmlDeviceGetHandleByUUID.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
assert lib.nvmlDeviceGetHandleByUUID(b'GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726',ctypes.byref(handle))==0
lib.nvmlDeviceGetClockInfo.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)]
def clocks():
 d={}
 for k,t in [('SM',1),('memory',2)]:
  v=ctypes.c_uint();rc=lib.nvmlDeviceGetClockInfo(handle,t,ctypes.byref(v));d[k]=v.value if rc==0 else 'UNKNOWN'
 return d
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock,extension
B=Path(__file__).resolve().parent;process=int(sys.argv[1]);start=time.time();GPU='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
r=dict(status='RUNNING',process=process,full_calls=0,rows=[],formal_performance_permission=False,torch_graph_boundary='Existing PreparedBlock dense uncached: static slots admitted before capture; original projections/norm/RoPE/attention/FFN, no context projection cache; not dynamic original forward')
def save():(B/f'TIMING-{process}.json').write_text(json.dumps(r,indent=2)+'\n')
def bump():r['full_calls']+=1;assert r['full_calls']<=14000
try:
 reg=json.loads((B/'JOB.json').read_text())
 def load_ext(path,name):
  spec=importlib.util.spec_from_file_location(name+'.wam_whole_fa2_cuda',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
 before=Path(reg['reference_binary']);candidate=next(B.glob('wam_whole_fa2_cuda*.so'))
 import hashlib
 assert hashlib.sha256(before.read_bytes()).hexdigest()==reg['reference_binary_SHA256']
 new=load_ext(candidate,'candidate');old=load_ext(before,'reference')
 r['binary_identity']={'before':str(before),'candidate':str(candidate),'before_sha256':hashlib.sha256(before.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}
 block=load_block('cuda');native=NativeBlock(block);bind=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   load=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([x for x in load.splitlines() if x.startswith(GPU)])==1
   p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments();f.install_cache(block);ref=native(*a,**k,debug=True);bump();torch.cuda.synchronize();f.install_cache(block);cache=block.attn1.attn_caches[k['cache_name']];snap=visible_snapshot(cache);cl=[cache[x] for x in ('k','v','id','mask','is_pred')];inputs=[x.clone() for x in a];cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();graphs={};vals={}
   prepared=PreparedBlock(block,a,k,cache)
   for label,ext in [('torch_static_uncached',None),('shared_torch_uncached',None),('guarded_uncached',None),('before',old),('staged',new),('candidate',new)]:
    g=torch.cuda.CUDAGraph()
    with (prepared.activate('native' if label=='shared_torch_uncached' else 'guarded' if label=='guarded_uncached' else 'dense',False) if ext is None else nullcontext()):
     for _ in range(2):
      warm=block(*a,**k) if ext is None else ext.forward(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps,False,**({'ready_tiles':True} if label=='candidate' else {}));bump()
     with torch.cuda.graph(g,stream=stream):
      y={'output':block(*a,**k)} if ext is None else ext.forward(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps,False,**({'ready_tiles':True} if label=='candidate' else {}))
    bump();graphs[label]=g;vals[label]=y
   for _ in range(600):
    for g in graphs.values():g.replay();bump()
   torch.cuda.synchronize();row=dict(case=length,fixture=str(p),load=load,samples=[],precision={},resources={},clock_before=clocks());assert row['clock_before']['SM']==1410,row['clock_before']
   for repeat in range(24):
    order=['torch_static_uncached','shared_torch_uncached','guarded_uncached','before','staged','candidate'];offset=(repeat+process)%6;order=order[offset:]+order[:offset]
    if repeat%2:order=order[::-1]
    for label in order:
     b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);b.record(stream);graphs[label].replay();bump();e.record(stream);e.synchronize();row['samples'].append(dict(repeat=repeat,arm=label,ms=b.elapsed_time(e)))
   for label,y in vals.items():
    checks={x:compare(ref[x].reshape(-1),y[x].reshape(-1)) for x in ref.keys()&y.keys()};assert all(z['bitwise_pass'] for z in checks.values());row['precision'][label]=checks;row['resources'][label]=y['resources'].tolist() if 'resources' in y else 'OMITTED_ON_NORMAL_PATH'
   row['clock_after']=clocks();assert row['clock_after']['SM']==1410,row['clock_after'];row['cache_unchanged']=state_equal(snap,visible_snapshot(cache));row['inputs_unchanged']=all(torch.equal(x,y) for x,y in zip(a,inputs));row['RNG_unchanged']=torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());assert row['cache_unchanged']['pass_all'] and row['inputs_unchanged'] and row['RNG_unchanged'];row['median_ms']={arm:statistics.median(x['ms'] for x in row['samples'] if x['arm']==arm) for arm in vals};r['rows'].append(row);save();del ref,graphs,vals,a,k,f,cl,snap,cache,inputs;torch.cuda.empty_cache()
 r['status']='COMPLETE_FINITE_WHOLE_COMPARISON'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-start;save()
print(r['status'],r.get('error',''),r['full_calls'],flush=True)
if r['status']!='COMPLETE_FINITE_WHOLE_COMPARISON':raise SystemExit(1)

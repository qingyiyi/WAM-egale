"""Same final K/V cache: original Torch vs four-group pipelined custom, with previous custom as regression control."""
from pathlib import Path
import os,json,time,sys,traceback,importlib.util,subprocess,statistics
import torch
from contextlib import nullcontext, ExitStack
from baselines import cached_cross
from wam_ops.optimized.prepared import PreparedBlock
import wam_ops.optimized.prepared as prepared_module
from functools import lru_cache
original_extension=prepared_module.extension
@lru_cache(None)
def prepared_extension(name):
 if name != '_shared_torch_v1': return original_extension(name)
 path=Path('/root/cxz/WAM-egale/投机采样/experiments/shared_prefix_torch_v2/_shared_torch_v1.cpython-312-x86_64-linux-gnu.so')
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
from wam_ops.optimized.whole_layer import WholeLayerBlock,extension
B=Path(__file__).resolve().parent;process=int(sys.argv[1]);start=time.time();GPU='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
r=dict(status='RUNNING',process=process,full_calls=0,rows=[],formal_performance_permission=False,comparison='Original Torch static dense history with cached final cross K/V; previous custom is a regression control only',torch_graph_boundary='Static slots admitted before capture; original Torch math and attention with same final cross K/V precomputed; warm full single block, not dynamic production forward')
def save():(B/f'TIMING-{process}.json').write_text(json.dumps(r,indent=2)+'\n')
def bump():r['full_calls']+=1;assert r['full_calls']<=18000
try:
 reg=json.loads((B/'JOB.json').read_text())
 def load_ext(path,name):
  spec=importlib.util.spec_from_file_location(name+'.wam_whole_fa2_cuda',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
 import hashlib
 binaries=reg['binaries']; loaded={name:load_ext(path,name) for name,path in binaries.items()}
 r['binary_identity']={name:dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for name,path in binaries.items()}
 arms=[(name,None if name in ['D','G'] else loaded[name]) for name in reg['arms']]
 block=load_block('cuda');bind=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   load=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([x for x in load.splitlines() if x.startswith(GPU)])==1
   p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments();f.install_cache(block);ref={'output':block(*a,**k)};bump();torch.cuda.synchronize();f.install_cache(block);cache=block.attn1.attn_caches[k['cache_name']];snap=visible_snapshot(cache);cl=[cache[x] for x in ('k','v','id','mask','is_pred')];inputs=[x.clone() for x in a];cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();graphs={};vals={}
   prepared=PreparedBlock(block,a,k,cache);prepared.fill()
   torch.cuda.synchronize();cold_start=time.perf_counter();bind.prepare_context(a[1],precompute=True);torch.cuda.synchronize();cold_ms=1000*(time.perf_counter()-cold_start)
   compact=a[1].index_select(0,bind.context_roots)
   torch.cuda.synchronize();expand_start=time.perf_counter();full_kv=[v.index_select(0,bind.context_rows) for v in bind.context_kv];torch.cuda.synchronize();expand_ms=1000*(time.perf_counter()-expand_start)
   cached_before=[v.clone() for v in bind.context_kv]
   grouped_ok=True
   for label,ext in arms:
    g=torch.cuda.CUDAGraph()
    with ExitStack() as stack:
     if ext is None:stack.enter_context(prepared.activate('guarded' if label=='G' else 'dense',False))
     if label in ['D','G']:stack.enter_context(cached_cross(block,full_kv))
     extra=(bind.context_rows,bind.context_roots,bind.context_kv) if ext is not None else ()
     if label in reg.get('modes',{}):extra=(*extra,reg['modes'][label])
     for _ in range(2):
      warm=block(*a,**k) if ext is None else ext.forward(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps,False,*extra);bump()
     with torch.cuda.graph(g,stream=stream):
      y={'output':block(*a,**k)} if ext is None else ext.forward(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps,False,*extra)
    bump();graphs[label]=g;vals[label]=y
   for _ in range(150):
    for g in graphs.values():g.replay();bump()
   torch.cuda.synchronize();row=dict(case=length,fixture=str(p),load=load,samples=[],precision={},resources={},clock_before=clocks());assert row['clock_before']['SM']>0,row['clock_before']
   for repeat in range(24):
    order=[arm for arm,_ in arms];offset=(repeat+process)%len(order);order=order[offset:]+order[:offset]
    if repeat%2:order=order[::-1]
    for label in order:
     b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);b.record(stream)
     for _ in range(10):graphs[label].replay();bump()
     e.record(stream);e.synchronize();row['samples'].append(dict(repeat=repeat,arm=label,ms=b.elapsed_time(e)/10))
   for label,y in vals.items():
    checks={x:compare(ref[x].reshape(-1),y[x].reshape(-1)) for x in ref.keys()&y.keys()};assert all(z['bitwise_pass'] for z in checks.values());row['precision'][label]=checks;row['resources'][label]=y['resources'].tolist() if 'resources' in y else 'OMITTED_ON_NORMAL_PATH'
   row['debug_resources']={}
   for label,ext in arms:
    if ext is not None:
     z=ext.forward(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps,True,bind.context_rows,bind.context_roots,bind.context_kv,*([reg['modes'][label]] if label in reg.get('modes',{}) else []));bump();torch.cuda.synchronize()
     assert compare(ref['output'],z['output'])['bitwise_pass']
     row['debug_resources'][label]=z['resources'].tolist()
     assert z['ffn_pipeline_state'].tolist()[:12]==[28,28,28,36]+[24]*8
     del z
   row['cold_prepare_wall_ms']=cold_ms;row['cold_baseline_expand_wall_ms']=expand_ms;row['cache_bytes']=dict(custom=sum(v.numel()*v.element_size() for v in bind.context_kv),baseline=sum(v.numel()*v.element_size() for v in full_kv));row['precomputed_cache_unchanged']=all(torch.equal(x,y) for x,y in zip(bind.context_kv,cached_before));assert row['precomputed_cache_unchanged'];row['context_rows']=bind.context_rows.tolist();row['context_roots']=bind.context_roots.tolist();row['clock_after']=clocks();assert row['clock_after']['SM']>0,row['clock_after'];row['cache_unchanged']=state_equal(snap,visible_snapshot(cache));row['inputs_unchanged']=all(torch.equal(x,y) for x,y in zip(a,inputs));row['RNG_unchanged']=torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());assert row['cache_unchanged']['pass_all'] and row['inputs_unchanged'] and row['RNG_unchanged'];row['median_ms']={arm:statistics.median(x['ms'] for x in row['samples'] if x['arm']==arm) for arm in vals};r['rows'].append(row);save();del ref,graphs,vals,a,k,f,cl,snap,cache,inputs;torch.cuda.empty_cache()
 r['status']='COMPLETE_FINITE_WHOLE_COMPARISON'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-start;save()
print(r['status'],r.get('error',''),r['full_calls'],flush=True)
if r['status']!='COMPLETE_FINITE_WHOLE_COMPARISON':raise SystemExit(1)

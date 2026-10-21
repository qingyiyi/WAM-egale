"""Prior versus simplified host entry B14: complete layer on real B14 inputs."""
from pathlib import Path
import os,json,time,sys,traceback,importlib.util,subprocess,statistics
import torch
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
r=dict(status='RUNNING',process=process,full_calls=0,rows=[],formal_performance_permission=False)
def save():(B/f'TIMING-{process}.json').write_text(json.dumps(r,indent=2)+'\n')
def bump():r['full_calls']+=1;assert r['full_calls']<=4000
try:
 s=importlib.util.spec_from_file_location('specialized.wam_whole_fa2_cuda',next(B.glob('wam_whole_fa2_cuda*.so')));new=importlib.util.module_from_spec(s);s.loader.exec_module(new);old=extension();
 import hashlib
 before=next(Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized').glob('wam_whole_fa2_cuda*.so'));candidate=next(B.glob('wam_whole_fa2_cuda*.so'))
 assert hashlib.sha256(before.read_bytes()).hexdigest()=='23238b3762a8596e4f832a7c8a49ef8c2e3e72407b62d8669fd325e3d99b0a6b'
 assert hashlib.sha256(before.read_bytes()).hexdigest()!=hashlib.sha256(candidate.read_bytes()).hexdigest()
 r['binary_identity']={'before':str(before),'candidate':str(candidate),'before_sha256':hashlib.sha256(before.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}
 block=load_block('cuda');native=NativeBlock(block);bind=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   load=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([x for x in load.splitlines() if x.startswith(GPU)])==1
   p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments();f.install_cache(block);ref=native(*a,**k,debug=True);bump();torch.cuda.synchronize();f.install_cache(block);cache=block.attn1.attn_caches[k['cache_name']];snap=visible_snapshot(cache);cl=[cache[x] for x in ('k','v','id','mask','is_pred')];inputs=[x.clone() for x in a];cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();graphs={};vals={}
   for label,ext in [('before',old),('specialized',new)]:
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):y=ext.forward(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps,*([False] if label=='specialized' else []))
    bump();graphs[label]=g;vals[label]=y
   for _ in range(600):
    for g in graphs.values():g.replay();bump()
   torch.cuda.synchronize();row=dict(case=length,fixture=str(p),load=load,samples=[],precision={},resources={},clock_before=clocks());assert row['clock_before']['SM']==1410,row['clock_before']
   for repeat in range(24):
    order=['before','specialized'] if (repeat+process)%2==0 else ['specialized','before']
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

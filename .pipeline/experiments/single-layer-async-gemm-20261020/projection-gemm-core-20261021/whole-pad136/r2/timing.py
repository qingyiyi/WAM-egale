"""Current4warp vs context-only padded4warp: complete layer on real B14 inputs."""
from pathlib import Path
import os,json,time,sys,traceback,importlib.util,subprocess,statistics
import torch
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
def bump():r['full_calls']+=1;assert r['full_calls']<=700
try:
 s=importlib.util.spec_from_file_location('pad.wam_whole_fa2_cuda',next(B.glob('wam_whole_fa2_cuda*.so')));new=importlib.util.module_from_spec(s);s.loader.exec_module(new);old=extension();block=load_block('cuda');native=NativeBlock(block);bind=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   load=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([x for x in load.splitlines() if x.startswith(GPU)])==1
   p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments();f.install_cache(block);ref=native(*a,**k,debug=True);bump();torch.cuda.synchronize();f.install_cache(block);cache=block.attn1.attn_caches[k['cache_name']];snap=visible_snapshot(cache);cl=[cache[x] for x in ('k','v','id','mask','is_pred')];inputs=[x.clone() for x in a];cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();graphs={};vals={}
   for label,ext in [('before',old),('pad136',new)]:
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):y=ext.forward(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps)
    bump();graphs[label]=g;vals[label]=y
   for _ in range(60):
    for g in graphs.values():g.replay();bump()
   torch.cuda.synchronize();row=dict(case=length,fixture=str(p),load=load,samples=[],precision={},resources={})
   for repeat in range(24):
    order=['before','pad136'] if (repeat+process)%2==0 else ['pad136','before']
    for label in order:
     b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);b.record(stream);graphs[label].replay();bump();e.record(stream);e.synchronize();row['samples'].append(dict(repeat=repeat,arm=label,ms=b.elapsed_time(e)))
   for label,y in vals.items():
    checks={x:compare(ref[x].reshape(-1),y[x].reshape(-1)) for x in ref.keys()&y.keys()};assert all(z['bitwise_pass'] for z in checks.values());row['precision'][label]=checks;row['resources'][label]=y['resources'].tolist()
   row['cache_unchanged']=state_equal(snap,visible_snapshot(cache));row['inputs_unchanged']=all(torch.equal(x,y) for x,y in zip(a,inputs));row['RNG_unchanged']=torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());assert row['cache_unchanged']['pass_all'] and row['inputs_unchanged'] and row['RNG_unchanged'];row['median_ms']={arm:statistics.median(x['ms'] for x in row['samples'] if x['arm']==arm) for arm in vals};r['rows'].append(row);save();del ref,graphs,vals,a,k,f,cl,snap,cache,inputs;torch.cuda.empty_cache()
 r['status']='COMPLETE_FINITE_WHOLE_COMPARISON'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-start;save()
print(r['status'],r.get('error',''),r['full_calls'],flush=True)
if r['status']!='COMPLETE_FINITE_WHOLE_COMPARISON':raise SystemExit(1)

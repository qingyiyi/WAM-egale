"""Same real intermediate inputs; globaltimer markers around captured stages."""
from pathlib import Path
import importlib.util
import json,os,sys,time,traceback,statistics
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock

B=Path(__file__).resolve().parent
process=int(sys.argv[1])
stages=json.loads((B/'STAGES.json').read_text())
names=['modulation','norm1','modulated1','self_q_linear','self_k_linear','self_v',
 'self_q_norm','self_k_norm','self_q_rope','self_k_rope','self_attention','self_out',
 'residual1','norm2','cross_q_linear','cross_k_linear','cross_v','cross_q','cross_k',
 'cross_attention','cross_out','residual2','norm3','modulated3','ffn1','gelu','ffn2','output']
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
t0=time.time()
out=dict(status='RUNNING',process=process,full_block_calls=0,stage_calls=0,rows=[],
 boundary='Globaltimer intervals; A/B CUDA graph marker launches, C in-kernel stage incl queue and synchronization',
 marker_overhead_subtracted=False,formal_performance_permission=False)
def save():
 p=B/f'simple-stage-process{process}-RESULT.json';p.with_suffix('.tmp').write_text(json.dumps(out,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
def extension(name):
 spec=importlib.util.spec_from_file_location(name,next(B.glob(name+'*.so')))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
try:
 own,original=extension('wam_stage_simple_cuda'),extension('wam_stage_native_cuda')
 block=load_block('cuda');native=NativeBlock(block);bound=WholeLayerBlock(block)
 stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  empty_ticks=torch.empty(2,dtype=torch.int64,device='cuda')
  empty_graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(empty_graph,stream=stream):own.marker(empty_ticks,0);own.marker(empty_ticks,1)
  empty=[]
  for _ in range(8):
   empty_graph.replay();torch.cuda.synchronize();v=empty_ticks.tolist();empty.append((v[1]-v[0])/1e6)
  out['marker_empty_ms']=empty
  for length in ('short','middle','long'):
   path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt'
   f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
   args,kw=f.arguments();cache=f.install_cache(block)
   ref=native(*args,**kw,debug=True);out['full_block_calls']+=1
   expected_cache=visible_snapshot(cache)
   f.install_cache(block);values=bound(*args,**kw,debug=True);out['full_block_calls']+=1
   torch.cuda.synchronize()
   for k in ref.keys()&values.keys():
    assert compare(ref[k].reshape(-1),values[k].reshape(-1))['bitwise_pass'],('full source',length,k)
   cache=block.attn1.attn_caches[kw['cache_name']]
   cache_list=[cache[k] for k in ('k','v','id','mask','is_pred')]
   common=(bound.weights,*args,cache_list,kw['candidate_cache_rows'],kw.get('update_cache',0),bound.eps)
   row=dict(case=length,fixture=str(path),input_shapes=[list(x.shape) for x in args],embedded=[],stages=[])
   for repeat in range(4):
    c=own.forward(*common,-1,[],[]);out['full_block_calls']+=1;torch.cuda.synchronize()
    matching={k:compare(ref[k].reshape(-1),c[k].reshape(-1)) for k in ref.keys()&c.keys()}
    assert all(v['bitwise_pass'] for v in matching.values()),('embedded output',length)
    row['embedded'].append(dict(repeat=repeat,ticks_ns=c['stage_ticks'].cpu().tolist(),
      resources=c['resources'].tolist(),cycles=c['stage_cycles'].cpu().tolist(),all_shared_intermediates_exact=True))
   a=[values[k] for k in names]
   meta=[values['logical_slots'],values['cache_info']]
   valid=int(meta[1][0].item())
   native_common=(bound.weights,*args,a,cache_list,kw['candidate_cache_rows'],meta[0],valid,bound.eps)
   for stage in stages:
    i,key=stage['id'],stage['output']
    if i==10:
     row['stages'].append(dict(**stage,status='NO_KV_WRITE_B14',A_ms=None,B_ms=None))
     continue
    expected=(a[names.index(key)].clone() if key else meta[0][:valid].clone())
    def A():return original.stage(i,*native_common)
    def C():return own.forward(*common,i,a,meta)
    av=A();bv=C();out['stage_calls']+=2;torch.cuda.synchronize()
    bvout=bv[key] if key else bv['logical_slots'][:valid]
    avout=av if key else av[:valid]
    eqA=compare(expected.reshape(-1),avout.reshape(-1))
    eqB=compare(expected.reshape(-1),bvout.reshape(-1))
    assert eqA['bitwise_pass'] and eqB['bitwise_pass'],('stage output',length,stage,eqA,eqB)
    rec=dict(**stage,match_A=eqA,match_B=eqB,standalone_resources=bv['resources'].tolist(),
             A_ms=[],B_ms=[],B_stage_ticks=[],B_stage_cycles=[],order=[],marker_overhead_subtracted=False)
    graphs=[];ticks=[];keep=[];outputs=[]
    for order in ('AB','BA'):
     captured={}
     t=torch.empty(4,dtype=torch.int64,device='cuda');g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):
      for which in order:
       offset=0 if which=='A' else 2
       own.marker(t,offset);y=A() if which=='A' else C();keep.append(y);captured[which]=y;own.marker(t,offset+1)
     out['stage_calls']+=2;graphs.append(g);ticks.append(t);outputs.append(captured)
    for warm in range(2):
     graphs[warm%2].replay();out['stage_calls']+=2
    for repeat in range(8):
     j=repeat%2;graphs[j].replay();out['stage_calls']+=2;torch.cuda.synchronize()
     t=ticks[j].tolist();rec['A_ms'].append((t[1]-t[0])/1e6);rec['B_ms'].append((t[3]-t[2])/1e6)
     rec['order'].append('AB' if j==0 else 'BA')
     rec['B_stage_ticks'].append(outputs[j]['B']['stage_ticks'][i].cpu().tolist())
     rec['B_stage_cycles'].append(outputs[j]['B']['stage_cycles'][i].cpu().tolist())
    row['stages'].append(rec)
    out['active_case']=length;out['active_stage']=stage['name'];out['partial_row']=row;save()
    del graphs,ticks,keep,outputs,captured,av,bv,expected
   final_cache=state_equal(expected_cache,visible_snapshot(cache))
   row['valid_KV_unchanged']=final_cache
   assert final_cache['pass_all'],('cache',length)
   out['rows'].append(row);out.pop('partial_row',None);save()
   del f,ref,values,args,kw,a,cache,cache_list,c
   torch.cuda.empty_cache()
 out['status']='COMPLETE_STAGE_ALIGNMENT'
except BaseException as e:
 out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:
 out['lifecycle_seconds']=time.time()-t0;save()
print(out['status'],out.get('error',''),out['full_block_calls'],out['stage_calls'],flush=True)
if out['status']!='COMPLETE_STAGE_ALIGNMENT':sys.exit(1)

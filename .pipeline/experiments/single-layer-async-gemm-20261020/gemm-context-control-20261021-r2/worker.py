"""Same native prefix for both implementations; full-layer stages measured separately."""
from pathlib import Path
import os,sys,json,time,traceback,statistics,importlib.util,ctypes,subprocess
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;S=B.parent/'minor-ops-20261021/stage-current';process=int(sys.argv[1])
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
GPU='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726';reg=json.loads((B/'JOB.json').read_text())
lib=ctypes.CDLL('libnvidia-ml.so.1');lib.nvmlInit_v2();handle=ctypes.c_void_p()
lib.nvmlDeviceGetHandleByUUID.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
assert lib.nvmlDeviceGetHandleByUUID(GPU.encode(),ctypes.byref(handle))==0
lib.nvmlDeviceGetClockInfo.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)]
def clocks():
 d={}
 for label,kind in [('SM',1),('memory',2)]:
  v=ctypes.c_uint();rc=lib.nvmlDeviceGetClockInfo(handle,kind,ctypes.byref(v));d[label]=v.value if rc==0 else 'UNKNOWN'
 return d
names=['modulation','norm1','modulated1','self_q_linear','self_k_linear','self_v','self_q_norm','self_k_norm','self_q_rope','self_k_rope','self_attention','self_out','residual1','norm2','cross_q_linear','cross_k_linear','cross_v','cross_q','cross_k','cross_attention','cross_out','residual2','norm3','modulated3','ffn1','gelu','ffn2','output']
stages=json.loads((S/'STAGES.json').read_text());stage_by_id={x['id']:x for x in stages}
selected=reg['matrix']['stages'];start=time.time()
out=dict(status='RUNNING',process=process,full_block_calls=0,stage_calls=0,rows=[],boundary='A/B graph marker intervals after matching native prefix; C in-kernel interval after fixed60 full warm; no marker subtraction',formal_performance_permission=False)
def save():
 p=B/f'RESULT-{process}.json';p.with_suffix('.tmp').write_text(json.dumps(out,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
def bump(full=0,stage=0):
 out['full_block_calls']+=full;out['stage_calls']+=stage
 assert out['full_block_calls']<=reg['full_block_calls_cap_per_process'] and out['stage_calls']<=reg['stage_calls_cap_per_process']
def ext(name):
 p=next(S.glob(name+'*.so'));sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
try:
 own,original=ext('wam_stage_simple_cuda'),ext('wam_stage_native_cuda')
 block=load_block('cuda');native=NativeBlock(block);bound=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   load=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   assert len([x for x in load.splitlines() if x.startswith(GPU)])==1,load
   f=BlockFixture(FIXTURES/f'input-{length}-B14-action-layer0-first.pt',device_tree(decode(torch.load(FIXTURES/f'input-{length}-B14-action-layer0-first.pt',map_location='cpu',mmap=True,weights_only=False))))
   args,kw=f.arguments();cache=f.install_cache(block);ref=native(*args,**kw,debug=True);bump(full=1)
   f.install_cache(block);v=bound(*args,**kw,debug=True);bump(full=1);torch.cuda.synchronize()
   assert all(torch.equal(ref[k].reshape(-1),v[k].reshape(-1)) for k in ref.keys()&v.keys())
   cache=block.attn1.attn_caches[kw['cache_name']];snapshot=visible_snapshot(cache)
   cache_list=[cache[k] for k in ('k','v','id','mask','is_pred')]
   common=(bound.weights,*args,cache_list,kw['candidate_cache_rows'],kw.get('update_cache',0),bound.eps)
   a=[v[k] for k in names];meta=[v['logical_slots'],v['cache_info']];valid=int(meta[1][0].item())
   initial_inputs=[x.clone() for x in args];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
   row=dict(case=length,input_shapes=[list(x.shape) for x in args],valid_history_KV=valid,embedded=[],stages=[])
   gfull=torch.cuda.CUDAGraph()
   with torch.cuda.graph(gfull,stream=stream):c=own.forward(*common,-1,[],[])
   bump(full=1)
   for _ in range(60):gfull.replay();bump(full=1)
   torch.cuda.synchronize();row['clock_after_fixed60']=clocks()
   for repeat in range(8):
    gfull.replay();bump(full=1);torch.cuda.synchronize()
    row['embedded'].append(dict(repeat=repeat,ticks_ns=c['stage_ticks'].cpu().tolist(),cycles=c['stage_cycles'].cpu().tolist(),resources=c['resources'].tolist()))
   assert all(torch.equal(ref[k].reshape(-1),c[k].reshape(-1)) for k in ref.keys()&c.keys())
   row['clock_after_embedded']=clocks()
   # Source layout metadata is already audited; B14 has no private-cache insertion.
   def native_stage(i,aa):
    bump(stage=1)
    return original.stage(i,bound.weights,*args,aa,cache_list,kw['candidate_cache_rows'],meta[0],valid,bound.eps)
   def make_prefix(i):
    aa=list(a)
    for pre in range(i):
     if pre==10:continue
     value=native_stage(pre,aa);aa[names.index(stage_by_id[pre]['output'])]=value
    return aa
   for sid in selected:
    key=stage_by_id[sid]['output'];expected=v[key].clone();record=dict(stage=sid,name=stage_by_id[sid]['name'],conditions=[])
    for condition in ['hot_single_stage','same_real_native_prefix']:
     graphs={};buffers={};objects={};costs={}
     for arm in ['A_native','B_owned']:
      ticks=torch.empty(2,dtype=torch.int64,device='cuda');g=torch.cuda.CUDAGraph();before=out['stage_calls']
      with torch.cuda.graph(g,stream=stream):
       aa=make_prefix(sid) if condition=='same_real_native_prefix' else list(a)
       own.marker(ticks,0)
       if arm=='A_native':y=native_stage(sid,aa)
       else:y=own.forward(*common,sid,aa,meta);bump(stage=1)
       own.marker(ticks,1)
      graphs[arm]=g;buffers[arm]=ticks;objects[arm]=(y,aa);costs[arm]=out['stage_calls']-before
     for _ in range(4):
      for arm,g in graphs.items():g.replay();bump(stage=costs[arm])
     torch.cuda.synchronize()
     rec=dict(condition=condition,clock_before=clocks(),samples=[],source_prefix=list(range(sid)) if condition!='hot_single_stage' else [],prefix_cache_insert_10_is_noop=True)
     for repeat in range(8):
      order=['A_native','B_owned'] if (repeat+process)%2==0 else ['B_owned','A_native']
      for arm in order:
       graphs[arm].replay();bump(stage=costs[arm]);torch.cuda.synchronize()
       t=buffers[arm].tolist();sample=dict(repeat=repeat,arm=arm,us=(t[1]-t[0])/1000)
       if arm=='B_owned':
        y=objects[arm][0];sample.update(internal_ns=y['stage_ticks'][sid].cpu().tolist(),internal_cycles=y['stage_cycles'][sid].cpu().tolist())
       rec['samples'].append(sample)
     rec['clock_after']=clocks();rec['precision']={};rec['prefix_equal']={}
     for arm,(y,aa) in objects.items():
      result=y if arm=='A_native' else y[key]
      rec['precision'][arm]=compare(expected.reshape(-1),result.reshape(-1));assert rec['precision'][arm]['bitwise_pass']
      if condition=='same_real_native_prefix':
       checks={stage_by_id[i]['output']:torch.equal(aa[names.index(stage_by_id[i]['output'])].reshape(-1),v[stage_by_id[i]['output']].reshape(-1)) for i in range(sid) if i!=10}
       assert all(checks.values()),(sid,arm,checks);rec['prefix_equal'][arm]=checks
     rec['owned_resources']=objects['B_owned'][0]['resources'].tolist();record['conditions'].append(rec)
     del graphs,buffers,objects,aa,y,g,ticks
    row['stages'].append(record);out['partial_row']=row;save()
   row['cache_check']=state_equal(snapshot,visible_snapshot(cache));assert row['cache_check']['pass_all']
   row['inputs_unchanged']=all(torch.equal(x,y) for x,y in zip(args,initial_inputs));assert row['inputs_unchanged']
   row['RNG_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());assert row['RNG_unchanged']
   row['load_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   assert len([x for x in row['load_after'].splitlines() if x.startswith(GPU)])==1
   out['rows'].append(row);out.pop('partial_row',None);save()
   del c,gfull,f,args,kw,v,ref,a,cache,cache_list,snapshot,initial_inputs,common
   torch.cuda.empty_cache()
 out['status']='COMPLETE_MATCHED_PREFIX_DIAGNOSTIC'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error',''),out['full_block_calls'],out['stage_calls'],flush=True)
if out['status']!='COMPLETE_MATCHED_PREFIX_DIAGNOSTIC':raise SystemExit(1)

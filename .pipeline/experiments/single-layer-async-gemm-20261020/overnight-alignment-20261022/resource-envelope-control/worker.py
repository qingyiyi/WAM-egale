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
B=Path(__file__).resolve().parent;S=B;OLD=Path('/root/cxz/ops/persistent-tile-20261019/evidence/async-gemm-20261020/minor-ops-20261021/stage-current');process=int(sys.argv[1])
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
stages=json.loads((OLD/'STAGES.json').read_text());stage_by_id={x['id']:x for x in stages}
selected=reg['matrix']['stages'];start=time.time()
import hashlib
for item in json.loads((B/'INPUT-SOURCE-MANIFEST.json').read_text())['source_files']:
 p=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')/item['relative']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['relative']
out=dict(status='RUNNING',process=process,full_block_calls=0,stage_calls=0,rows=[],boundary='A/B graph marker intervals after matching native prefix; C in-kernel interval after fixed600 full warm; no marker subtraction',formal_performance_permission=False)
def save():
 p=B/f'RESULT-{process}.json';p.with_suffix('.tmp').write_text(json.dumps(out,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
def bump(full=0,stage=0):
 out['full_block_calls']+=full;out['stage_calls']+=stage
 assert out['full_block_calls']<=reg['full_block_calls_cap_per_process'] and out['stage_calls']<=reg['stage_calls_cap_per_process']
def ext(name):
 p=next((OLD if name=='wam_stage_native_cuda' else S).glob(name+'*.so'));sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
try:
 own,original=ext('wam_stage_resource_cuda'),ext('wam_stage_native_cuda')
 from wam_ops.optimized.prepared import extension as prepared_extension,guarded_group
 grouped=prepared_extension('wam_candidate_cuda')
 shared_path=Path('/root/cxz/WAM-egale/投机采样/experiments/shared_prefix_torch_v2/_shared_torch_v1.cpython-312-x86_64-linux-gnu.so');sp=importlib.util.spec_from_file_location('_shared_torch_v1',shared_path);shared=importlib.util.module_from_spec(sp);sp.loader.exec_module(shared)
 import wam_ops.optimized.whole_layer as wl
 candidate=Path(reg['candidate_binary']);spec=importlib.util.spec_from_file_location('selected.wam_whole_fa2_cuda',candidate);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);wl.extension=lambda:mod
 block=load_block('cuda');native=NativeBlock(block);bound=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['middle']:
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
   for _ in range(600):gfull.replay();bump(full=1)
   torch.cuda.synchronize();row['clock_after_fixed600']=clocks();assert row['clock_after_fixed600']['SM']==1410,row['clock_after_fixed600']
   for repeat in range(16):
    gfull.replay();bump(full=1);torch.cuda.synchronize()
    row['embedded'].append(dict(repeat=repeat,ticks_ns={str(i):c['stage_ticks'][i].cpu().tolist() for i in selected},cycles={str(i):c['stage_cycles'][i].cpu().tolist() for i in selected},resources=c['resources'].tolist()))
   assert all(torch.equal(ref[k].reshape(-1),c[k].reshape(-1)) for k in ref.keys()&c.keys())
   row['clock_after_embedded']=clocks();assert row['clock_after_embedded']['SM']==1410,row['clock_after_embedded']
   gactive=torch.cuda.CUDAGraph()
   with torch.cuda.graph(gactive,stream=stream):active_value=bound(*args,**kw,debug=True)
   bump(full=1)
   for _ in range(600):gactive.replay();bump(full=1)
   torch.cuda.synchronize();row['instrumentation_control']=[]
   for repeat in range(16):
    pairs=[('active',gactive),('diagnostic',gfull)]
    for label,graph in (pairs if repeat%2==0 else pairs[::-1]):
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     begin.record(stream);graph.replay();end.record(stream);end.synchronize();bump(full=1)
     row['instrumentation_control'].append(dict(arm=label,repeat=repeat,ms=begin.elapsed_time(end)))
   assert all(torch.equal(ref[k].reshape(-1),active_value[k].reshape(-1)) for k in ref.keys()&active_value.keys())
   del active_value,gactive

   # Source layout metadata is already audited; B14 has no private-cache insertion.
   def native_stage(i,aa):
    bump(stage=1)
    return original.stage(i,bound.weights,*args,aa,cache_list,kw['candidate_cache_rows'],meta[0],valid,bound.eps)
   def strong_attention(arm,aa):
    bump(stage=1)
    query=aa[names.index('self_q_rope')].reshape(14,16,24,128)
    key=aa[names.index('self_k_rope')].reshape(14,16,24,128)
    value=aa[names.index('self_v')].reshape(14,16,24,128)
    if arm=='C_guarded':return grouped.grouped(query,cache_list[0],cache_list[1],key,value,meta[0][:valid],guarded_group(valid))
    return shared.shared(query,cache_list[0],cache_list[1],key,value,meta[0][:valid],kw['candidate_cache_rows'])
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
     for arm in ['A_native','B_owned','E_min_shared_108','F_min_shared_occupancy']:
      ticks=torch.empty(2,dtype=torch.int64,device='cuda');g=torch.cuda.CUDAGraph();before=out['stage_calls']
      with torch.cuda.graph(g,stream=stream):
       aa=make_prefix(sid) if condition=='same_real_native_prefix' else list(a)
       own.marker(ticks,0)
       if arm=='A_native':y=native_stage(sid,aa)
       else:
        mode=0 if arm=='B_owned' else 100 if arm=='E_min_shared_108' else 200
        y=own.forward(*common,sid+mode,aa,meta);bump(stage=1)
       own.marker(ticks,1)
      graphs[arm]=g;buffers[arm]=ticks;objects[arm]=(y,aa);costs[arm]=out['stage_calls']-before
     for _ in range(30):
      for arm,g in graphs.items():g.replay();bump(stage=costs[arm])
     torch.cuda.synchronize()
     rec=dict(condition=condition,clock_before=clocks(),samples=[],source_prefix=list(range(sid)) if condition!='hot_single_stage' else [],prefix_cache_insert_10_is_noop=True)
     for repeat in range(16):
      order=list(graphs);offset=(repeat+process)%len(order);order=order[offset:]+order[:offset]
      if repeat%2:order=order[::-1]
      for arm in order:
       graphs[arm].replay();bump(stage=costs[arm]);torch.cuda.synchronize()
       t=buffers[arm].tolist();sample=dict(repeat=repeat,arm=arm,us=(t[1]-t[0])/1000)
       if arm!='A_native':
        y=objects[arm][0];sample.update(internal_ns=y['stage_ticks'][sid].cpu().tolist(),internal_cycles=y['stage_cycles'][sid].cpu().tolist())
       rec['samples'].append(sample)
     rec['clock_after']=clocks();rec['precision']={};rec['prefix_equal']={}
     for arm,(y,aa) in objects.items():
      result=y[key] if arm!='A_native' else y
      rec['precision'][arm]=compare(expected.reshape(-1),result.reshape(-1));assert rec['precision'][arm]['bitwise_pass']
      if condition=='same_real_native_prefix':
       checks={stage_by_id[i]['output']:torch.equal(aa[names.index(stage_by_id[i]['output'])].reshape(-1),v[stage_by_id[i]['output']].reshape(-1)) for i in range(sid) if i!=10}
       assert all(checks.values()),(sid,arm,checks);rec['prefix_equal'][arm]=checks
     rec['owned_resources']={a:objects[a][0]['resources'].tolist() for a in objects if a!='A_native'};record['conditions'].append(rec)
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
 out['status']='COMPLETE_CURRENT_SOURCE_PREFIX_DIAGNOSTIC'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error',''),out['full_block_calls'],out['stage_calls'],flush=True)
if out['status']!='COMPLETE_CURRENT_SOURCE_PREFIX_DIAGNOSTIC':raise SystemExit(1)

"""同计算体调度对照：真实投影、相同前序、嵌入整层；两进程独立重复。"""
from pathlib import Path
import os,sys,json,time,traceback,importlib.util,subprocess,ctypes
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;OLD=B.parent/'minor-ops-20261021/stage-current'
process=int(sys.argv[1]);start=time.time();GPU='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
names=['modulation','norm1','modulated1','self_q_linear','self_k_linear','self_v','self_q_norm','self_k_norm','self_q_rope','self_k_rope','self_attention','self_out','residual1','norm2','cross_q_linear','cross_k_linear','cross_v','cross_q','cross_k','cross_attention','cross_out','residual2','norm3','modulated3','ffn1','gelu','ffn2','output']
byid={x['id']:x for x in json.loads((OLD/'STAGES.json').read_text())}
lib=ctypes.CDLL('libnvidia-ml.so.1');lib.nvmlInit_v2();handle=ctypes.c_void_p()
lib.nvmlDeviceGetHandleByUUID.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
assert lib.nvmlDeviceGetHandleByUUID(GPU.encode(),ctypes.byref(handle))==0
lib.nvmlDeviceGetClockInfo.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)]
def clocks():
 d={}
 for label,kind in [('SM',1),('memory',2)]:
  v=ctypes.c_uint();rc=lib.nvmlDeviceGetClockInfo(handle,kind,ctypes.byref(v));d[label]=v.value if rc==0 else 'UNKNOWN'
 return d
out=dict(status='RUNNING',process=process,full_calls=0,stage_calls=0,rows=[],formal_performance_permission=False)
def save():
 p=B/f'RESULT-{process}.json';p.with_suffix('.tmp').write_text(json.dumps(out,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
def bump(full=0,stage=0):
 out['full_calls']+=full;out['stage_calls']+=stage
 assert out['full_calls']<=1000 and out['stage_calls']<=100000

def ext(name,directory,alias=''):
 p=next(directory.glob(name+'*.so'));spec=importlib.util.spec_from_file_location(alias+name,p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def loadcheck():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 assert len([x for x in s.splitlines() if x.startswith(GPU)])==1,s
 return s
try:
 own=ext('wam_projection_schedule_cuda',B);out['kernel_resources']=own.setup()
 original=ext('wam_stage_native_cuda',OLD);trace=ext('wam_cost_trace_cuda',B)
 large=ext('wam_whole_fa2_cuda',B.parent/'projection-eight-warps-20261021/rejected-binary','large.')
 block=load_block('cuda');native=NativeBlock(block);active=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   load=loadcheck();p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt'
   f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))))
   args,kw=f.arguments();f.install_cache(block);ref=native(*args,**kw,debug=True);bump(full=1)
   f.install_cache(block);v=active(*args,**kw,debug=True);bump(full=1);torch.cuda.synchronize()
   assert all(torch.equal(ref[k].reshape(-1),v[k].reshape(-1)) for k in ref.keys()&v.keys())
   cache=block.attn1.attn_caches[kw['cache_name']];snap=visible_snapshot(cache)
   cl=[cache[k] for k in ('k','v','id','mask','is_pred')];a=[v[k] for k in names]
   meta=[v['logical_slots'],v['cache_info']];valid=int(meta[1][0]);common=(active.weights,*args,cl,kw['candidate_cache_rows'],kw.get('update_cache',0),active.eps)
   initial=[x.clone() for x in args];cr=torch.get_rng_state().clone();gr=torch.cuda.get_rng_state().clone()
   row=dict(case=length,fixture=str(p),valid_history=valid,context_shape=list(args[1].shape),context_stride=list(args[1].stride()),load_before=load,stages=[],embedded=[],full_samples=[])
   graphs={};vals={}
   for label,fn in [('active4',lambda:active(*args,**kw,debug=True)),('large8',lambda:large.forward(*common)),('trace8',lambda:trace.forward(*common))]:
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):y=fn()
    bump(full=1);graphs[label]=g;vals[label]=y
    for _ in range(60):g.replay();bump(full=1)
   torch.cuda.synchronize()
   for label,y in vals.items():
    checks={k:torch.equal(ref[k].reshape(-1),y[k].reshape(-1)) for k in ref.keys()&y.keys()}
    assert all(checks.values()),(length,label,'pre_timing_gate',checks)
   row['full_clock_before']=clocks()
   for repeat in range(12):
    order=list(graphs);shift=(repeat+process)%len(order);order=order[shift:]+order[:shift]
    for label in order:
     b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);b.record(stream);graphs[label].replay();e.record(stream);e.synchronize();bump(full=1)
     row['full_samples'].append(dict(repeat=repeat,arm=label,ms=b.elapsed_time(e)))
     if label=='trace8':
      ticks=vals[label]['diag_ticks'].cpu().tolist();row['embedded'].append(dict(repeat=repeat,stage_ticks={str(sid):ticks[sid] for sid in [16,17]},resources=vals[label]['resources'].tolist(),barriers={str(sid):vals[label]['diag_barriers'][sid].cpu().tolist() for sid in [16,17]}))
   row['full_clock_after']=clocks();row['whole_precision']={}
   for label,y in vals.items():
    checks={k:torch.equal(ref[k].reshape(-1),y[k].reshape(-1)) for k in ref.keys()&y.keys()}
    assert all(checks.values()),(length,label,checks);row['whole_precision'][label]=checks
   def nstage(i,aa):
    bump(stage=1);return original.stage(i,active.weights,*args,aa,cl,kw['candidate_cache_rows'],meta[0],valid,active.eps)
   def prefix(i):
    aa=list(a)
    for pre in range(i):
     if pre!=10:aa[names.index(byid[pre]['output'])]=nstage(pre,aa)
    return aa
   empty=torch.empty(0,device='cuda',dtype=torch.int32)
   for sid,wi in [(16,15),(17,17)]:
    key=byid[sid]['output'];shape=v[key].shape
    expected=nstage(sid,list(a)).reshape(-1)
    assert torch.equal(expected,v[key].reshape(-1)),(length,sid,'independent_native_stage')
    context=args[1];w,bias=active.weights[wi:wi+2];m=context.numel()//w.shape[1];n=w.shape[0]
    tasks=((m+255)//256)*((n+127)//128);assert tasks==672
    rec=dict(stage=sid,name=byid[sid]['name'],M=m,N=n,K=w.shape[1],tasks=tasks,audit=[],conditions=[])
    # 独立审计，visits/owners不会进入性能测量；NaN初始化避免漏写掩盖。
    for mode in [0,1,2]:
     yy=torch.full(shape,float('nan'),device='cuda',dtype=torch.bfloat16)
     q=torch.zeros(1,device='cuda',dtype=torch.int32);visits=torch.zeros(tasks,device='cuda',dtype=torch.int32);owners=torch.full_like(visits,-1)
     own.run(context,w,bias,yy,q,mode,visits,owners);bump(stage=1);torch.cuda.synchronize()
     check=compare(expected,yy.reshape(-1));assert check['bitwise_pass']
     vc=visits.tolist();oc=owners.tolist();assert all(x==1 for x in vc) and all(0<=x<(tasks if mode==0 else 108) for x in oc)
     if mode==0:assert oc==list(range(tasks))
     if mode==1:assert oc==[i%108 for i in range(tasks)]
     rec['audit'].append(dict(mode=mode,precision=check,visits=vc,owners=oc,queue_after=q.item()))
    for condition in ['hot_single_stage','same_real_native_prefix']:
     gs={};values={};ticks_by_arm={};costs={}
     for arm in ['native','ordinary','static108','atomic108']:
      ticks=torch.empty(3,device='cuda',dtype=torch.int64);yy=torch.empty(shape,device='cuda',dtype=torch.bfloat16);q=torch.zeros(1,device='cuda',dtype=torch.int32)
      g=torch.cuda.CUDAGraph();before=out['stage_calls']
      with torch.cuda.graph(g,stream=stream):
       aa=prefix(sid) if condition=='same_real_native_prefix' else list(a)
       own.marker(ticks,0)
       if arm=='atomic108':q.zero_()
       own.marker(ticks,1)
       if arm=='native':result=nstage(sid,aa)
       else:
        mode={'ordinary':0,'static108':1,'atomic108':2}[arm]
        own.run(context,w,bias,yy,q,mode,empty,empty);bump(stage=1);result=yy
       own.marker(ticks,2)
      gs[arm]=g;values[arm]=(result,aa,yy,q);ticks_by_arm[arm]=ticks;costs[arm]=out['stage_calls']-before
     for _ in range(60):
      for arm,g in gs.items():g.replay();bump(stage=costs[arm])
     torch.cuda.synchronize();r=dict(condition=condition,clock_before=clocks(),samples=[],prefix_stages=[i for i in range(sid) if i!=10] if condition=='same_real_native_prefix' else [])
     for repeat in range(12):
      order=list(gs);shift=(repeat+process)%len(order);order=order[shift:]+order[:shift]
      for arm in order:
       gs[arm].replay();bump(stage=costs[arm]);torch.cuda.synchronize();t=ticks_by_arm[arm].tolist()
       r['samples'].append(dict(repeat=repeat,arm=arm,kernel_interval_us=(t[2]-t[1])/1000,reset_interval_us=(t[1]-t[0])/1000,inclusive_interval_us=(t[2]-t[0])/1000))
     r['clock_after']=clocks();r['precision']={};r['prefix_exact']={}
     for arm,(result,aa,yy,q) in values.items():
      check=compare(expected,result.reshape(-1));assert check['bitwise_pass'];r['precision'][arm]=check
      if condition=='same_real_native_prefix':
       checks={byid[i]['output']:torch.equal(aa[names.index(byid[i]['output'])].reshape(-1),v[byid[i]['output']].reshape(-1)) for i in range(sid) if i!=10}
       assert all(checks.values());r['prefix_exact'][arm]=checks
     rec['conditions'].append(r);del gs,values,ticks_by_arm
    row['stages'].append(rec);out['partial_row']=row;save()
   row['cache_check']=state_equal(snap,visible_snapshot(cache));assert row['cache_check']['pass_all']
   row['inputs_unchanged']=all(torch.equal(x,y) for x,y in zip(args,initial));row['RNG_unchanged']=torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state());assert row['inputs_unchanged'] and row['RNG_unchanged']
   row['load_after']=loadcheck();out['rows'].append(row);out.pop('partial_row',None);save()
   del graphs,vals,ref,v,f,args,kw,a,cl,common,initial,snap,cache;torch.cuda.empty_cache()
 out['status']='COMPLETE_FINITE_SAME_BODY_DISPATCH_DIAGNOSTIC'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error'),out['full_calls'],out['stage_calls'],flush=True)
if out['status']!='COMPLETE_FINITE_SAME_BODY_DISPATCH_DIAGNOSTIC':raise SystemExit(1)

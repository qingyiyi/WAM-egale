"""Current full-layer stage/CTA timeline; same-prefix native stage comparison."""
from pathlib import Path
import os,json,sys,time,statistics,importlib.util,csv,subprocess,traceback
os.sched_setaffinity(0,{4,5})
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;OLD=B.parents[1]/'minor-ops-20261021/stage-current';GPU='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726';torch.set_num_threads(2);torch.set_num_interop_threads(1)
names=['modulation','norm1','modulated1','self_q_linear','self_k_linear','self_v','self_q_norm','self_k_norm','self_q_rope','self_k_rope','self_attention','self_out','residual1','norm2','cross_q_linear','cross_k_linear','cross_v','cross_q','cross_k','cross_attention','cross_out','residual2','norm3','modulated3','ffn1','gelu','ffn2','output']
stages=json.loads((OLD/'STAGES.json').read_text());byid={x['id']:x for x in stages}
def ext(n,p):
 s=importlib.util.spec_from_file_location(n,next(p.glob(n+'*.so')));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
out=dict(status='RUNNING',rows=[],full_calls=0,stage_calls=0,formal_performance_permission=False);start=time.time()
def save():(B/'TRACE-RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
def bump(full=0,stage=0):
 out['full_calls']+=full;out['stage_calls']+=stage
 assert out['full_calls']<=1000 and out['stage_calls']<=12000
try:
 diag=ext('wam_cost_trace_cuda',B.parent);original=ext('wam_stage_native_cuda',OLD);block=load_block('cuda');native=NativeBlock(block);owned=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ('short','middle','long'):
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([s for s in apps.splitlines() if s.startswith(GPU)])==1,apps
   path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
   args,kw=f.arguments();cache=f.install_cache(block);ref=native(*args,**kw,debug=True);bump(full=1);f.install_cache(block);v=owned(*args,**kw,debug=True);bump(full=1);torch.cuda.synchronize()
   assert all(torch.equal(ref[k].reshape(-1),v[k].reshape(-1)) for k in ref.keys()&v.keys())
   cache=block.attn1.attn_caches[kw['cache_name']];snap=visible_snapshot(cache);cl=[cache[k] for k in ('k','v','id','mask','is_pred')];a=[v[k] for k in names];meta=[v['logical_slots'],v['cache_info']];valid=int(meta[1][0]);common=(owned.weights,*args,cl,kw['candidate_cache_rows'],kw.get('update_cache',0),owned.eps)
   initial=[x.clone() for x in args];cr=torch.get_rng_state().clone();gr=torch.cuda.get_rng_state().clone();row=dict(case=length,fixture=str(path),active_source_resources=v['resources'].tolist(),embedded=[],native_stages=[],perturbation=[])
   graphs={};values={}
   for label,fn in [('active',lambda:owned(*args,**kw,debug=True)),('trace',lambda:diag.forward(*common))]:
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):y=fn()
    bump(full=1);graphs[label]=g;values[label]=y
    for _ in range(60):g.replay();bump(full=1)
   torch.cuda.synchronize()
   for repeat in range(8):
    for label in (['active','trace'] if repeat%2==0 else ['trace','active']):
     b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);b.record(stream);graphs[label].replay();e.record(stream);e.synchronize();bump(full=1);row['perturbation'].append(dict(repeat=repeat,arm=label,ms=b.elapsed_time(e)))
    y=values['trace'];ticks=y['diag_ticks'].cpu().tolist();tape=y['diag_barriers'].cpu().tolist();row['embedded'].append(dict(repeat=repeat,stage_ticks=ticks,resources=y['resources'].tolist()))
    with (B/f'CTA-{length}-{repeat}.csv').open('w') as fcsv:
     wr=csv.writer(fcsv,lineterminator='\n');wr.writerow(['stage','CTA','barrier','enter_ns','exit_ns'])
     for st,ctas in enumerate(tape):
      for cta,parts in enumerate(ctas):
       for part,(enter,leave) in enumerate(parts):
        if enter:wr.writerow([st,cta,part,enter,leave])
   for label,y in values.items():
    assert all(torch.equal(ref[k].reshape(-1),y[k].reshape(-1)) for k in ref.keys()&y.keys()),(length,label)
   row['all_shared_intermediates_exact']=True
   # Separate native stages: use same freshly executed original prefix.
   def nstage(i,aa):
    bump(stage=1);return original.stage(i,owned.weights,*args,aa,cl,kw['candidate_cache_rows'],meta[0],valid,owned.eps)
   for sid in range(29):
    if sid==10:continue
    ticks=torch.empty(2,dtype=torch.int64,device='cuda');g=torch.cuda.CUDAGraph();before=out['stage_calls']
    with torch.cuda.graph(g,stream=stream):
     aa=list(a)
     for pre in range(sid):
      if pre!=10:aa[names.index(byid[pre]['output'])]=nstage(pre,aa)
     diag.marker(ticks,0);nv=nstage(sid,aa);diag.marker(ticks,1)
    cost=out['stage_calls']-before
    for _ in range(4):g.replay();bump(stage=cost)
    times=[]
    for repeat in range(4):
     g.replay();bump(stage=cost);torch.cuda.synchronize();tt=ticks.tolist();times.append((tt[1]-tt[0])/1000)
    expected=v[byid[sid]['output']];assert torch.equal(nv.reshape(-1),expected.reshape(-1)),(length,sid)
    row['native_stages'].append(dict(stage=sid,name=byid[sid]['name'],us=times,finite_output_exact=True,prefix=list(range(sid)),boundary='external graph markers; includes native stage all kernels; no empty subtraction'))
    del g,ticks,nv,aa
   row['cache_check']=state_equal(snap,visible_snapshot(cache));assert row['cache_check']['pass_all']
   row['inputs_unchanged']=all(torch.equal(x,y) for x,y in zip(args,initial));row['RNG_unchanged']=torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state());assert row['inputs_unchanged'] and row['RNG_unchanged']
   row['load_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([s for s in row['load_after'].splitlines() if s.startswith(GPU)])==1
   out['rows'].append(row);save();del graphs,values,y,v,ref,f,args,kw,a,cache,cl,snap,initial,common;torch.cuda.empty_cache()
 out['status']='COMPLETE_CURRENT_SOURCE_TIMELINE'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error'),out['full_calls'],out['stage_calls'],flush=True)
if out['status']!='COMPLETE_CURRENT_SOURCE_TIMELINE':raise SystemExit(1)

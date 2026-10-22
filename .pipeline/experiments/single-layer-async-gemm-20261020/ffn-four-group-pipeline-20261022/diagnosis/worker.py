"""Real-input FFN diagnostic. Instrumented timeline and normal elapsed are distinct."""
from pathlib import Path
import os,sys,time,json,traceback,importlib.util,statistics,subprocess
import torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock,extension
B=Path(__file__).resolve().parent
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
process=int(sys.argv[1]);start=time.time();r={'status':'RUNNING','process':process,'rows':[],'formal_performance_permission':False,'full_calls':0,'subgraph_calls':0}
def save():(B/f'PROFILE-{process}.json').write_text(json.dumps(r,indent=2)+'\n')
def gpu_apps():return subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
def clocks():return subprocess.check_output(['nvidia-smi','--query-gpu=uuid,clocks.sm,clocks.mem','--format=csv,noheader'],text=True)
try:
 spec=importlib.util.spec_from_file_location('wam_ffn_diagnostic_cuda',next(B.glob('wam_ffn_diagnostic_cuda*.so')));diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
 original=extension();block=load_block('cuda');bind=WholeLayerBlock(block);stream=torch.cuda.Stream()
 r['activation_repr']=repr(block.ffn.net[0]);r['GPU']=torch.cuda.get_device_name();r['source_scope']='Frozen production R2; generated instrumented copy, standalone/full routes share same kernel and resources; no optimized entry replacement'
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   load=gpu_apps();assert [line.split(',')[1].strip() for line in load.splitlines() if line.startswith('GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726')]==[str(os.getpid())],load
   p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments()
   f.install_cache(block);reference=block(*a,**k);r['full_calls']+=1
   f.install_cache(block);cache=block.attn1.attn_caches[k['cache_name']];snap=visible_snapshot(cache);cl=[cache[x] for x in ('k','v','id','mask','is_pred')]
   bind.prepare_context(a[1],precompute=True);extra=(bind.context_rows,bind.context_roots,bind.context_kv)
   args=(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps)
   orig=original.forward(*args,True,*extra);r['full_calls']+=1
   mod=orig['modulated3'].view_as(a[0]);residual=orig['residual2'].view_as(a[0]);ffn_inputs=[mod,residual]
   gate=(block.scale_shift_table[None]+a[2].float()).select(2,5)
   def torch_ffn():return (residual.float()+block.ffn(mod).float()*gate).to(mod.dtype)
   act=orig['gelu'].view(14,16,-1);f1=orig['ffn1'].view_as(act);f2=orig['ffn2'].view_as(mod)
   def torch_ffn1_gelu():return block.ffn.net[0](mod)
   def torch_ffn2_residual():return (residual.float()+block.ffn.net[2](act).float()*gate).to(mod.dtype)
   arms={
    'production_full':lambda:original.forward(*args,False,*extra),
    'diagnostic_full_off':lambda:diag.forward(*args,False,*extra,False,[]),
    'diagnostic_full_on':lambda:diag.forward(*args,False,*extra,True,[]),
    'custom_ffn_off':lambda:diag.forward(*args,False,*extra,False,ffn_inputs),
    'custom_ffn_on':lambda:diag.forward(*args,False,*extra,True,ffn_inputs),
    'torch_ffn':torch_ffn,
    'torch_ffn1_gelu':torch_ffn1_gelu,
    'torch_ffn2_residual':torch_ffn2_residual,
    'torch_ffn1':lambda:block.ffn.net[0].proj(mod),
    'torch_gelu':lambda:block.ffn.net[0].gelu(f1),
    'torch_ffn2':lambda:block.ffn.net[2](act),
    'torch_residual':lambda:(residual.float()+f2.float()*gate).to(mod.dtype)}
   vals={};graphs={};precision={}
   for name,fn in arms.items():
    for _ in range(2):fn()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):vals[name]=fn()
    graphs[name]=g
    if 'full' in name:r['full_calls']+=3
    else:r['subgraph_calls']+=3
   torch.cuda.synchronize()
   for name in ['production_full','diagnostic_full_off','diagnostic_full_on','custom_ffn_off','custom_ffn_on','torch_ffn','torch_ffn2_residual']:
    v=vals[name];y=v['output'] if isinstance(v,dict) else v
    precision[name]=compare(reference,y);assert precision[name]['bitwise_pass'],(name,precision[name])
   precision['ffn1_gelu']=compare(act,vals['torch_ffn1_gelu']);precision['ffn1']=compare(f1,vals['torch_ffn1']);precision['gelu']=compare(act,vals['torch_gelu']);precision['ffn2']=compare(f2,vals['torch_ffn2'])
   assert all(v['bitwise_pass'] for v in precision.values()),precision
   row={'case':length,'fixture':str(p),'precision':precision,'load':load,'clock_before':clocks(),'samples':[],'full_timelines':[],'isolated_timelines':[]}
   # Warm each arm equally; interleave order and measure 10 replays/event for sub-ms precision.
   for _ in range(150):
    for name,g in graphs.items():g.replay()
   for repeat in range(24):
    names=list(graphs);offset=(repeat+process)%len(names);names=names[offset:]+names[:offset]
    if repeat%2:names.reverse()
    for name in names:
     b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     b.record(stream)
     for _ in range(10):graphs[name].replay()
     e.record(stream);e.synchronize();row['samples'].append({'repeat':repeat,'arm':name,'ms':b.elapsed_time(e)/10})
   r['full_calls']+=3*(150+240);r['subgraph_calls']+=9*(150+240)
   for _ in range(12):
    for name,key in [('diagnostic_full_on','full_timelines'),('custom_ffn_on','isolated_timelines')]:
     graphs[name].replay();torch.cuda.synchronize();row[key].append(vals[name]['ffn_diag'].cpu().tolist())
   r['full_calls']+=12;r['subgraph_calls']+=12
   profile_debug=diag.forward(*args,True,*extra,True,[]);r['full_calls']+=1;torch.cuda.synchronize();row['diagnostic_resources']=profile_debug['resources'].tolist();row['production_resources']=orig['resources'].tolist()
   row['debug_vs_reference']=compare(reference,profile_debug['output']);assert row['debug_vs_reference']['bitwise_pass']
   row['cache_unchanged']=state_equal(snap,visible_snapshot(cache));assert row['cache_unchanged']['pass_all']
   row['clock_after']=clocks();row['median_ms']={name:statistics.median(s['ms'] for s in row['samples'] if s['arm']==name) for name in graphs}
   row['load_after']=gpu_apps();r['rows'].append(row);save()
   del vals,graphs,orig,profile_debug,reference,f,a,k,mod,residual,act,f1,f2,ffn_inputs,cache,cl,snap,gate
   torch.cuda.empty_cache()
 r['status']='COMPLETE_FINITE_FFN_DIAGNOSIS'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-start;save()
print(r['status'],r.get('error',''),flush=True)
if r['status']!='COMPLETE_FINITE_FFN_DIAGNOSIS':raise SystemExit(1)

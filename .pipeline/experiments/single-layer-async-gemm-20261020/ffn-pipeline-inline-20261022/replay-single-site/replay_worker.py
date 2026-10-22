"""Selected-source independent FFN and full-copy perturbation, real inputs only."""
from pathlib import Path
import os,sys,time,json,traceback,importlib.util,statistics,subprocess
import torch
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,visible_snapshot,tensors
from tools.replay import device_tree
from tools.metrics import compare,state_equal,byte_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;reg=json.loads((B/'JOB.json').read_text());process=int(sys.argv[1]);start=time.time()
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
r=dict(status='RUNNING',process=process,rows=[],production_admitted=False,formal_performance_permission=False)
def save():(B/f'REPLAY-{process}.json').write_text(json.dumps(r,indent=2)+'\n')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def apps():return subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
def clocks():return subprocess.check_output(['nvidia-smi','--query-gpu=uuid,clocks.sm,clocks.mem','--format=csv,noheader'],text=True)
try:
 previous=load('/root/cxz/ops/persistent-tile-20261019/evidence/async-gemm-20261020/ffn-epilogue-abi-20261022/replay/wam_ffn_replay_cuda.cpython-312-x86_64-linux-gnu.so','previous.wam_ffn_replay_cuda')
 original=load(reg['selected_binary'],'selected.wam_whole_fa2_cuda');diag=load(next(B.glob('wam_ffn_replay_cuda*.so')),'wam_ffn_replay_cuda')
 block=load_block('cuda');bind=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   gpu_load=apps();assert len([x for x in gpu_load.splitlines() if x.startswith(reg['GPU'])])==1,gpu_load
   path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))));a,k=f.arguments()
   f.install_cache(block);reference=block(*a,**k);f.install_cache(block);cache=block.attn1.attn_caches[k['cache_name']];snap=visible_snapshot(cache);cl=[cache[x] for x in ('k','v','id','mask','is_pred')]
   bind.prepare_context(a[1],precompute=True);extra=(bind.context_rows,bind.context_roots,bind.context_kv);args=(bind.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),bind.eps)
   orig=original.forward(*args,True,*extra);torch.cuda.synchronize();assert compare(reference,orig['output'])['bitwise_pass']
   mod=orig['modulated3'].view_as(a[0]);residual=orig['residual2'].view_as(a[0]);ffn_inputs=[mod,residual];act=orig['gelu'].view(14,16,-1)
   snapshots={n:t.clone() for n,t in tensors((a,k,bind.context_kv,ffn_inputs))};cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
   gate=(block.scale_shift_table[None]+a[2].float()).select(2,5)
   arms={
    'selected_full':lambda:original.forward(*args,False,*extra),
    'replay_copy_full':lambda:diag.forward(*args,False,*extra,[]),
    'custom_ffn':lambda:diag.forward(*args,False,*extra,ffn_inputs),
    'previous_ffn':lambda:previous.forward(*args,False,*extra,ffn_inputs),
    'torch_ffn':lambda:(residual.float()+block.ffn(mod).float()*gate).to(mod.dtype),
    'torch_ffn1_gelu':lambda:block.ffn.net[0](mod),
    'torch_ffn2_residual':lambda:(residual.float()+block.ffn.net[2](act).float()*gate).to(mod.dtype)}
   vals={};graphs={}
   for name,fn in arms.items():
    for _ in range(2):fn()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):vals[name]=fn()
    graphs[name]=g
   for g in graphs.values():g.replay()
   torch.cuda.synchronize();checks={}
   for name,v in vals.items():
    out=v['output'] if isinstance(v,dict) else v
    checks[name]=compare(act if name=='torch_ffn1_gelu' else reference,out);assert checks[name]['bitwise_pass'],(name,checks[name])
   row=dict(case=length,fixture=str(path),precision=checks,load_before=gpu_load,clock_before=clocks(),samples=[])
   for _ in range(150):
    for g in graphs.values():g.replay()
   for repeat in range(24):
    order=list(graphs);offset=(process+repeat)%len(order);order=order[offset:]+order[:offset]
    if repeat%2:order.reverse()
    for name in order:
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record(stream)
     for _ in range(10):graphs[name].replay()
     end.record(stream);end.synchronize();row['samples'].append(dict(repeat=repeat,arm=name,ms=begin.elapsed_time(end)/10))
   debug=diag.forward(*args,True,*extra,ffn_inputs);full_debug=diag.forward(*args,True,*extra,[]);torch.cuda.synchronize()
   row['replay_stages']={n:compare(orig[n].reshape(-1),debug[n].reshape(-1)) for n in ['ffn1','gelu','ffn2','output']};assert all(v['bitwise_pass'] for v in row['replay_stages'].values())
   row['full_debug_output']=compare(reference,full_debug['output']);assert row['full_debug_output']['bitwise_pass']
   row['selected_resources']=orig['resources'].tolist();row['replay_resources']=debug['resources'].tolist();row['replay_full_resources']=full_debug['resources'].tolist()
   row['ffn_state']=debug['ffn_pipeline_state'].tolist();assert row['ffn_state'][:12]==[28,28,28,36]+[24]*8
   row['inputs_unchanged']=all(byte_equal(snapshots[n],t) for n,t in tensors((a,k,bind.context_kv,ffn_inputs)));row['cache_unchanged']=state_equal(snap,visible_snapshot(cache));row['RNG_unchanged']=byte_equal(cpu,torch.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state());assert row['inputs_unchanged'] and row['cache_unchanged']['pass_all'] and row['RNG_unchanged']
   row['clock_after']=clocks();row['load_after']=apps();assert len([x for x in row['load_after'].splitlines() if x.startswith(reg['GPU'])])==1
   row['median_ms']={n:statistics.median(x['ms'] for x in row['samples'] if x['arm']==n) for n in arms};row['full_copy_perturbation_percent']=100*(row['median_ms']['replay_copy_full']/row['median_ms']['selected_full']-1);row['speedup_ffn_vs_Torch']=row['median_ms']['torch_ffn']/row['median_ms']['custom_ffn']
   r['rows'].append(row);save();del graphs,vals,orig,debug,full_debug,reference,mod,residual,ffn_inputs,act,gate,snapshots,a,k,cache,cl,snap,f;torch.cuda.empty_cache()
 r['status']='COMPLETE_FINITE_FFN_REPLAY'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-start;save()
print(r['status'],r.get('error',''),flush=True)
if r['status']!='COMPLETE_FINITE_FFN_REPLAY':raise SystemExit(1)

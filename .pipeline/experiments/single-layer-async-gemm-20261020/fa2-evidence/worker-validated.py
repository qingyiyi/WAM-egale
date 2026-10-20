"""Real whole-block FA2 checks; same existing M39 evidence location, no robot work."""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,tensors,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,byte_equal,state_equal
from tools.resources import resource_for_gpu
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
from wam_ops.optimized.whole_layer_pipeline import PipelineBlock
B=Path(__file__).resolve().parent
PARENT=B.parent
WORK=Path(load_block.__code__.co_filename).resolve().parents[2]
VIDEO=WORK/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'

def run():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(exist_ok=False);start=json.loads((B/'START.json').read_text());t0=time.time();os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
 r=dict(status='RUNNING',mode=a.mode,rows=[],samples=[],full_block_calls=0,controlled_SDPA_calls=0,controlled_Linear_calls=0,robot_episodes=0,production_admitted=False,formal_performance_permission=False)
 def save():(a.out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 expected=None
 def guard():
  nonlocal expected
  assert time.time()<start['deadline']-60 and time.time()-t0<540
  assert r['full_block_calls']<200
  cur=set(resource_for_gpu(start['GPU'])['pids'].splitlines())
  if expected is None:assert len(cur)==1;expected=cur
  assert cur==expected,'foreign GPU context'
 try:
  assert not resource_for_gpu(start['GPU'])['pids']
  block=load_block('cuda');guard()
  impl={'native':NativeBlock(block),'old':PipelineBlock(block,PARENT,'async'),'fa2':WholeLayerBlock(block)}
  parameters={n:(x.data_ptr(),x.requires_grad) for n,x in block.named_parameters()}
  with torch.inference_mode():
   for kind,batch in [('action',2),('action',14),('video',2)]:
    for length in ('short','middle','long'):
     guard();path=(FIXTURES if kind=='action' else VIDEO)/f'input-{length}-B{batch}-{kind}-layer0-first.pt'
     data=decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False));f=BlockFixture(path,device_tree(data));args,kw=f.arguments();case=f'{length}-B{batch}-{kind}'
     row=dict(case=case,fixture=str(path));q=args[0].shape[1]
     frozen={n:t.clone() for n,t in tensors((args,kw))};cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state()
     def call(name,update=None,audit=False):
      assert time.time()<start['deadline']-60 and r['full_block_calls']<200
      r['full_block_calls']+=1;k=kw if update is None else dict(kw,update_cache=update)
      if name=='Torch':return block(*args,**k)
      if name=='native':return impl[name](*args,**k,debug=True)
      return impl[name](*args,**k,debug=True,audit=audit)
     if a.mode=='validate':
      v={};states={}
      for name in ('Torch','native','old','fa2'):
       f.install_cache(block);v[name]=call(name,audit=True);torch.cuda.synchronize();states[name]=visible_snapshot(block.attn1.attn_caches[kw['cache_name']])
      row['native_vs_Torch']=compare(v['Torch'],v['native']['output'])
      row['old_vs_Torch']=compare(v['Torch'],v['old']['output'])
      row['fa2_vs_Torch']=compare(v['Torch'],v['fa2']['output'])
      row['fa2_vs_native']=compare(v['native']['output'],v['fa2']['output'])
      row['DAG_native']={k:compare(v['native'][k],v['fa2'][k]) for k in v['native'] if k in v['fa2'] and k not in ('tile_stats','resources','cache_info')}
      x=v['fa2'];row['resources']=x['resources'].tolist();row['attention_meta']=x['attention_meta'].cpu().tolist()
      stats=x['tile_stats'].cpu().tolist();row['exact_once']=all(z[0]==z[1] and z[2]==0 for z in stats);row['tile_stats']=stats
      row['cache_vs_old']=state_equal(states['old'],states['fa2'])
      # Same-input attention controls isolate attention math from upstream GEMM.
      cache=block.attn1.attn_caches[kw['cache_name']];n=int(x['cache_info'][0]);slots=x['logical_slots'][:n]
      if batch==14:
       owners=kw['candidate_cache_rows'];ks=cache['k'][owners][:,slots];vs=cache['v'][owners][:,slots]
       ks=torch.cat((ks,x['self_k_rope'].view(batch,q,24,128)),1)
       vs=torch.cat((vs,x['self_v'].view(batch,q,24,128)),1)
      else:ks=cache['k'][:,slots];vs=cache['v'][:,slots]
      row['same_input_attention']={}
      for label,qv,kv,vv in [('self',x['self_q_rope'].view(batch,q,24,128),ks,vs),('cross',x['cross_q'].view(batch,q,24,128),x['cross_k'].view(batch,512,24,128),x['cross_v'].view(batch,512,24,128))]:
       with sdpa_kernel(SDPBackend.FLASH_ATTENTION):ref=torch.nn.functional.scaled_dot_product_attention(qv.transpose(1,2),kv.transpose(1,2),vv.transpose(1,2)).transpose(1,2).reshape(batch*q,3072)
       r['controlled_SDPA_calls']+=1;row['same_input_attention'][label]=compare(ref,x[label+'_attention'])
      row['same_input_Q']=compare(block.attn1.to_q(x['modulated1']),x['self_q_linear']);r['controlled_Linear_calls']+=1
      row['same_input_FFN2']=compare(block.ffn.net[2](x['gelu']),x['ffn2']);r['controlled_Linear_calls']+=1
      aa=v['Torch'].reshape(-1).float();bb=x['output'].reshape(-1).float();delta=(aa-bb).abs();i=int(delta.argmax());row['max_error_location']=dict(flat_index=i,reference=float(aa[i]),candidate=float(bb[i]),abs=float(delta[i]))
      f.install_cache(block);repeat=call('fa2',audit=True);torch.cuda.synchronize();row['repeat_output']=byte_equal(x['output'],repeat['output'])
      row['inputs_unchanged']=all(byte_equal(frozen[n],t) for n,t in tensors((args,kw)))
      row['RNG_unchanged']=byte_equal(cpu,torch.random.get_rng_state()) and byte_equal(gpu,torch.cuda.get_rng_state())
      row['parameters_unchanged']=parameters=={n:(x.data_ptr(),x.requires_grad) for n,x in block.named_parameters()}
      row['legal_sequences']=[]
      if batch==2:
       expect=[];f.install_cache(block)
       for u in (1,2,0):z=call('old',u,True);torch.cuda.synchronize();expect.append(visible_snapshot(block.attn1.attn_caches[kw['cache_name']]))
       f.install_cache(block)
       for u,state in zip((1,2,0),expect):
        z=call('fa2',u,True);torch.cuda.synchronize();row['legal_sequences'].append(dict(update=u,cache=state_equal(state,visible_snapshot(block.attn1.attn_caches[kw['cache_name']])),finite=bool(torch.isfinite(z['output']).all())))
       del expect,z
      r['rows'].append(row);save()
      del v,x,repeat,states,ks,vs,ref,aa,bb,delta
     else:
      modes=['native','old','fa2'];idx=int(a.mode[-1])
      for name in modes:f.install_cache(block);call(name);torch.cuda.synchronize()
      for repeat in range(4):
       shift=(repeat+idx)%3
       for name in modes[shift:]+modes[:shift]:
        guard()
        f.install_cache(block);torch.cuda.synchronize();beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        beg.record();h=time.perf_counter();z=call(name);end.record();end.synchronize()
        r['samples'].append(dict(case=case,process=idx,repeat=repeat,name=name,ms=beg.elapsed_time(end),host_ms=(time.perf_counter()-h)*1000))
      f.install_cache(block)
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:z=call('fa2');torch.cuda.synchronize()
      trace=a.out/f'{case}-TRACE.json';prof.export_chrome_trace(str(trace));events=json.loads(trace.read_text())['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel'];row['kernel_count']=len(kernels);row['kernel_names']=[e['name'] for e in kernels];row['resources']=z['resources'].tolist();assert len(kernels)==1
      r['rows'].append(row);save();del z
     del data,f,args,kw,frozen
  if a.mode=='validate':
   r['attention_gate']=all(all(c['bitwise_pass'] for c in x['same_input_attention'].values()) for x in r['rows'])
   r['state_gate']=all(x['exact_once'] and x['cache_vs_old']['pass_all'] and x['repeat_output'] and x['inputs_unchanged'] and x['RNG_unchanged'] and x['parameters_unchanged'] and all(v['cache']['pass_all'] and v['finite'] for v in x['legal_sequences']) for x in r['rows'])
   r['Torch_gate']=all(x['fa2_vs_Torch']['bitwise_pass'] for x in r['rows'])
  r['status']='COMPLETE_UNADMITTED'
 except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
 finally:r['lifecycle_seconds']=time.time()-t0;save()
if __name__=='__main__':run()

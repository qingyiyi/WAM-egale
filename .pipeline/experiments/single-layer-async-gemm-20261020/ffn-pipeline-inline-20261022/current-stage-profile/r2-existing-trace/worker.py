"""Current full-kernel phase stamps vs original Torch stage graphs, same real tensors.
Stage ratios are diagnostic: custom includes embedded sync/queue, Torch is hot standalone.
Production whole-layer timings are kept separate. No hardware counters required.
"""
from pathlib import Path
import os,json,time,traceback,importlib.util,statistics,subprocess
import torch
from contextlib import ExitStack
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
from wam_ops.optimized.prepared import PreparedBlock
from baselines import cached_cross
B=Path(__file__).resolve().parent
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
normal=load(B.parent.parent/'tail-k64/wam_whole_fa2_cuda.cpython-312-x86_64-linux-gnu.so','selected.wam_whole_fa2_cuda')
diag=load(B/'wam_current_stage_cuda.cpython-312-x86_64-linux-gnu.so','wam_current_stage_cuda')
start=time.time();out=dict(status='RUNNING',rows=[],full_calls=0,stage_calls=0,formal_performance_permission=False,boundary='custom CTA0 intervals include existing stage synchronization and queue costs; Torch original stage expressions timed in isolated hot CUDA Graphs on matching real intermediate inputs; not additive full-layer attribution')
def bump(n=1):out['full_calls']+=n;assert out['full_calls']<=5000
def save():(B/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
try:
 block=load_block('cuda');binding=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ['short','middle','long']:
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert len([x for x in apps.splitlines() if x.startswith('GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726')])==1
   path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
   a,k=f.arguments();f.install_cache(block);orig=block(*a,**k);bump();f.install_cache(block)
   cache=block.attn1.attn_caches[k['cache_name']];before=visible_snapshot(cache);cl=[cache[x] for x in ['k','v','id','mask','is_pred']]
   binding.prepare_context(a[1],precompute=True);saved_kv=[x.clone() for x in binding.context_kv];saved_inputs=[x.clone() for x in a];rng=[torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()]
   common=(binding.weights,*a,cl,k['candidate_cache_rows'],k.get('update_cache',0),binding.eps)
   extra=(binding.context_rows,binding.context_roots,binding.context_kv)
   z=normal.forward(*common,True,*extra);bump();d=diag.forward(*common,True,*extra,True);bump()
   checks={key:compare(z[key].reshape(-1),d[key].reshape(-1)) for key in z.keys()&d.keys() if key not in ['resources','cache_info','logical_slots','attention_meta','ffn_pipeline_state','ffn_pipeline_timestamps']}
   assert all(v['bitwise_pass'] for v in checks.values());assert compare(orig,z['output'])['bitwise_pass']
   prepared=PreparedBlock(block,a,k,cache);prepared.fill();full_kv=[v.index_select(0,binding.context_rows) for v in binding.context_kv]
   graphs={};values={}
   for label,ext,enabled in [('selected',normal,False),('copy_off',diag,False),('copy_on',diag,True),('torch_whole',None,False)]:
    with ExitStack() as stack:
     if ext is None:stack.enter_context(prepared.activate('dense',False));stack.enter_context(cached_cross(block,full_kv))
     def call():
      if ext is None:return {'output':block(*a,**k)}
      return ext.forward(*common,False,*extra,*([enabled] if ext is diag else []))
     for _ in range(2):call();bump()
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):v=call()
     bump();graphs[label]=g;values[label]=v
   for _ in range(100):
    for g in graphs.values():g.replay();bump()
   samples=[];tapes=[]
   for repeat in range(24):
    names=list(graphs);offset=repeat%4;names=names[offset:]+names[:offset]
    if repeat%2:names=names[::-1]
    for label in names:
     e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record(stream)
     for _ in range(10):graphs[label].replay();bump()
     e1.record(stream);e1.synchronize();samples.append(dict(repeat=repeat,arm=label,us=e0.elapsed_time(e1)*100))
    tapes.append(values['copy_on']['phase_ticks'].cpu().tolist())
   for y in values.values():assert compare(orig,y['output'])['bitwise_pass']
   # Original module expressions, using the same actual full-layer intermediates.
   shape=a[0].shape;head_shape=(*shape[:2],24,128)
   def v(name):return z[name].reshape(shape)
   table=block.scale_shift_table[None]+a[2].float()
   shift,scale,gate,shift3,scale3,gate3=[x.squeeze(1) for x in table.permute(0,2,1,3).chunk(6,dim=1)]
   def rotate(t):
    x=t.reshape(head_shape);q=torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:3],-1,2));return torch.view_as_real(q*a[3]).flatten(3).to(x.dtype)
   def self_attention():
    hk=cache['k'][:,prepared.slots].index_select(0,prepared.rows);hv=cache['v'][:,prepared.slots].index_select(0,prepared.rows)
    return block.attn1.attn_op(v('self_q_rope').reshape(head_shape),torch.cat((hk,v('self_k_rope').reshape(head_shape)),1),torch.cat((hv,v('self_v').reshape(head_shape)),1))
   funcs={
    'table':(lambda:block.scale_shift_table[None]+a[2].float(),'modulation'),
    '1':(lambda:block.norm1(a[0].float()),'norm1'),
    '2':(lambda:(v('norm1')*(1.+scale)+shift).type_as(a[0]),'modulated1'),
    '3':(lambda:block.attn1.to_q(v('modulated1')),'self_q_linear'),
    '4':(lambda:block.attn1.to_k(v('modulated1')),'self_k_linear'),
    '5':(lambda:block.attn1.to_v(v('modulated1')),'self_v'),
    '6':(lambda:rotate(block.attn1.norm_q(v('self_q_linear'))),'self_q_rope'),
    '7':(lambda:rotate(block.attn1.norm_k(v('self_k_linear'))),'self_k_rope'),
    '9':(self_attention,'self_attention'),
    '10':(lambda:block.attn1.to_out[0](v('self_attention')),'self_out'),
    '11':(lambda:(a[0].float()+v('self_out')*gate).type_as(a[0]),'residual1'),
    '12':(lambda:block.norm2(v('residual1').float()).type_as(a[0]),'norm2'),
    '13':(lambda:block.attn2.to_q(v('norm2')),'cross_q_linear'),
    '14':(lambda:block.attn2.norm_q(v('cross_q_linear')),'cross_q'),
    '15':(lambda:block.attn2.attn_op(v('cross_q').reshape(head_shape),full_kv[1].reshape(14,512,24,128),full_kv[2].reshape(14,512,24,128)),'cross_attention'),
    '16':(lambda:block.attn2.to_out[0](v('cross_attention')),'cross_out'),
    '17':(lambda:v('residual1')+v('cross_out'),'residual2'),
    '18':(lambda:block.norm3(v('residual2').float()),'norm3'),
    '19':(lambda:(v('norm3')*(1.+scale3)+shift3).type_as(a[0]),'modulated3'),
    '20':(lambda:(v('residual2').float()+block.ffn(v('modulated3')).float()*gate3).type_as(a[0]),'output')}
   stages={}
   for sid,(fn,name) in funcs.items():
    for _ in range(2):test=fn()
    check=compare(z[name].reshape(-1),test.reshape(-1));assert check['bitwise_pass'],(sid,name,check)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):test=fn()
    for _ in range(50):g.replay()
    times=[]
    for repeat in range(24):
     e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record(stream)
     for _ in range(10):g.replay()
     e1.record(stream);e1.synchronize();times.append(e0.elapsed_time(e1)*100)
    out['stage_calls']+=293
    custom=[(t[int(sid)][1]-t[int(sid)][0])/1000 for t in tapes] if sid!='table' else []
    if custom:assert all(x>0 for x in custom),(sid,custom)
    stages[sid]=dict(name=name,Torch_us=times,custom_embedded_us=custom,precision=check,median_torch_us=statistics.median(times),median_custom_us=statistics.median(custom) if custom else None,torch_over_custom=statistics.median(times)/statistics.median(custom) if custom else None)
   assert state_equal(before,visible_snapshot(cache))['pass_all'];assert all(torch.equal(x,y) for x,y in zip(saved_inputs,a));assert all(torch.equal(x,y) for x,y in zip(saved_kv,binding.context_kv));assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
   row=dict(case=length,fixture=str(path),full_precision=checks,selected_resources=z['resources'].tolist(),copy_resources=d['resources'].tolist(),samples=samples,full_median_us={arm:statistics.median(x['us'] for x in samples if x['arm']==arm) for arm in graphs},phase_tapes=tapes,stages=stages,finite_state_pass=True)
   out['rows'].append(row);save();del graphs,values,z,d,orig,f,a,k,cache,cl,before;torch.cuda.empty_cache()
 out['status']='COMPLETE_CURRENT_STAGE_DIAGNOSTIC'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error',''),out['full_calls'],flush=True)
if out['status']!='COMPLETE_CURRENT_STAGE_DIAGNOSTIC':raise SystemExit(1)

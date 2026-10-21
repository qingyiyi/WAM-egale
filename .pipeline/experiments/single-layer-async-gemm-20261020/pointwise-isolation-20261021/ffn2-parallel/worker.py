"""Real B14 intermediates: literal Torch / native / bare math / persistent stage.
Hot CUDA-graph stage latency only; separate profiler is never timing evidence.
"""
from pathlib import Path
import os,sys,json,time,traceback,importlib.util,statistics
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent
S=B.parents[1]/'stage-alignment/overnight/stage_selected'
process=int(sys.argv[1]);os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
start=time.time();out=dict(status='RUNNING',process=process,full_block_calls=0,stage_calls=0,rows=[],boundary='hot fixed-input CUDA Graph isolated stage; excludes host allocation, not full layer or E2E',torch_version=torch.__version__,torch_commit=torch.version.git_version)
def save():
 p=B/f'process{process}-RESULT.json';p.write_text(json.dumps(out,indent=2)+'\n')
def ext(root,name):
 s=importlib.util.spec_from_file_location(name,next(root.glob(name+'*.so')));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
names=['modulation','norm1','modulated1','self_q_linear','self_k_linear','self_v','self_q_norm','self_k_norm','self_q_rope','self_k_rope','self_attention','self_out','residual1','norm2','cross_q_linear','cross_k_linear','cross_v','cross_q','cross_k','cross_attention','cross_out','residual2','norm3','modulated3','ffn1','gelu','ffn2','output']
try:
 bare=ext(B,'wam_isolate_cuda');own=ext(S,'wam_stage_simple_cuda');native_stage=ext(S,'wam_stage_native_cuda')
 block=load_block('cuda');native=NativeBlock(block);bound=WholeLayerBlock(block)
 out['literal_activation']=repr(block.ffn.net[0]);out['device']=torch.cuda.get_device_name();out['bare_resources']={f'{k}-{m}':bare.resources(k,m) for k in (0,1) for m in range(4 if k else 3)}
 stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ('short','middle','long'):
   fpath=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(fpath,device_tree(decode(torch.load(fpath,map_location='cpu',mmap=True,weights_only=False))))
   args,kw=f.arguments();cache=f.install_cache(block);ref=native(*args,**kw,debug=True);out['full_block_calls']+=1
   f.install_cache(block);values=bound(*args,**kw,debug=True);out['full_block_calls']+=1;torch.cuda.synchronize()
   assert all(compare(ref[k].reshape(-1),values[k].reshape(-1))['bitwise_pass'] for k in ref.keys()&values.keys())
   expected_cache=visible_snapshot(cache);cache=block.attn1.attn_caches[kw['cache_name']]
   caches=[cache[k] for k in ('k','v','id','mask','is_pred')]
   a=[values[k] for k in names];meta=[values['logical_slots'],values['cache_info']];valid=int(meta[1][0].item())
   common=(bound.weights,*args,caches,kw['candidate_cache_rows'],kw.get('update_cache',0),bound.eps)
   nc=(bound.weights,*args,a,caches,kw['candidate_cache_rows'],meta[0],valid,bound.eps)
   assert args[2].dtype==torch.bfloat16 and args[2].is_contiguous() and args[2].shape[1]==args[0].shape[1]
   for stage,key in ((27,'ffn2'),):
    expected=values[key].clone();row=dict(case=length,stage=stage,key=key,fixture=str(fpath),shape=list(expected.shape),arms={})
    funcs={'native':lambda:native_stage.stage(stage,*nc),'persistent':lambda:own.forward(*common,stage,a,meta)[key]}
    if stage==0:
     funcs['torch']=lambda:block.scale_shift_table[None]+args[2].float()
     funcs.update({label:(lambda mode=mode:bare.point(args[2],bound.weights[0],True,mode)) for mode,label in enumerate(('bare_scalar','bare_108','bare_vec4'))})
    elif stage==26:
     funcs['torch']=lambda:torch.nn.functional.gelu(values['ffn1'],approximate='tanh')
     funcs.update({label:(lambda mode=mode:bare.point(values['ffn1'],bound.weights[0],False,mode)) for mode,label in enumerate(('bare_scalar','bare_108','bare_vec4'))})
    else:
     funcs['torch']=lambda:torch.nn.functional.linear(values['gelu'],bound.weights[25],bound.weights[26])
     funcs.update({label:(lambda mode=mode:bare.ffn(values['gelu'],bound.weights[25],bound.weights[26],mode)) for mode,label in enumerate(('bare_M64','bare_M128','bare_shared_partial','bare_M128_parallel_K'))})
    graphs={};keep=[]
    for label,fn in funcs.items():
     result=fn();out['stage_calls']+=1;torch.cuda.synchronize()
     row['arms'][label]=dict(precision=compare(expected.reshape(-1),result.reshape(-1)),samples_ms=[])
     # A failing mathematical arm is reported, never integrated or used as an aligned speedup.
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):
      for _ in range(5):keep.append(fn())
     out['stage_calls']+=5;graphs[label]=g
    for g in graphs.values():g.replay();out['stage_calls']+=5
    torch.cuda.synchronize()
    labels=list(funcs)
    for repeat in range(8):
     order=labels if (repeat+process)%2==0 else labels[::-1]
     for label in order:
      begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
      begin.record(stream);graphs[label].replay();end.record(stream);end.synchronize();out['stage_calls']+=5
      row['arms'][label]['samples_ms'].append(begin.elapsed_time(end)/5)
    for arm in row['arms'].values():arm['median_ms']=statistics.median(arm['samples_ms'])
    if process==1 and length=='middle':
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
      for label in ('torch','native'):
       with torch.profiler.record_function(label+'_'+key):keep.append(funcs[label]());out['stage_calls']+=1
      torch.cuda.synchronize()
     prof.export_chrome_trace(str(B/f'profile-{key}.json'))
     row['profile_gpu']=[dict(name=e.name,device_time_total=e.device_time_total) for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']
    out['rows'].append(row);save();del graphs,keep,funcs,expected
   row['valid_KV_unchanged']=state_equal(expected_cache,visible_snapshot(cache));assert row['valid_KV_unchanged']['pass_all']
   del f,args,kw,ref,values,a,cache,caches,meta;torch.cuda.empty_cache()
 out['status']='COMPLETE_ISOLATION_MEASUREMENT'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error',''),flush=True)
if out['status']!='COMPLETE_ISOLATION_MEASUREMENT':raise SystemExit(1)

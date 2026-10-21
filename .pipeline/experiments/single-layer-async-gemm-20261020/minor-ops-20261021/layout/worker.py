"""Literal Torch, native, prior executor, own best-layout and fixed-grid control."""
from pathlib import Path
import os,sys,json,time,traceback,importlib.util,statistics
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent;S=B.parents[1]/'stage-alignment/overnight/stage_selected'
process=int(sys.argv[1]);os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
start=time.time();out=dict(status='RUNNING',process=process,full_block_calls=0,stage_calls=0,rows=[],boundary='hot fixed-input CUDA Graph isolated stage, excludes host allocation, not E2E',prior_executor_version='stage_selected pre_vec4 RMS-vector sequential',torch_version=torch.__version__,torch_commit=torch.version.git_version)
def save():(B/f'process{process}-RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
def ext(root,name):
 s=importlib.util.spec_from_file_location(name,next(root.glob(name+'*.so')));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
names=['modulation','norm1','modulated1','self_q_linear','self_k_linear','self_v','self_q_norm','self_k_norm','self_q_rope','self_k_rope','self_attention','self_out','residual1','norm2','cross_q_linear','cross_k_linear','cross_v','cross_q','cross_k','cross_attention','cross_out','residual2','norm3','modulated3','ffn1','gelu','ffn2','output']
stages=[(2,'modulated1'),(8,'self_q_rope'),(13,'residual1'),(22,'residual2')]
try:
 bare=ext(B,'wam_minor_cuda');own=ext(S,'wam_stage_simple_cuda');original=ext(S,'wam_stage_native_cuda');block=load_block('cuda');native=NativeBlock(block);bound=WholeLayerBlock(block);stream=torch.cuda.Stream()
 with torch.inference_mode(),torch.cuda.stream(stream):
  for length in ('short','middle','long'):
   path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt';f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
   args,kw=f.arguments();f.install_cache(block);ref=native(*args,**kw,debug=True);f.install_cache(block);v=bound(*args,**kw,debug=True);out['full_block_calls']+=2;torch.cuda.synchronize()
   assert all(compare(ref[k].reshape(-1),v[k].reshape(-1))['bitwise_pass'] for k in ref.keys()&v.keys())
   cache=block.attn1.attn_caches[kw['cache_name']];expected_cache=visible_snapshot(cache);cl=[cache[k] for k in ('k','v','id','mask','is_pred')];a=[v[k] for k in names];meta=[v['logical_slots'],v['cache_info']];valid=int(meta[1][0].item());batch,q=args[0].shape[:2];D=3072
   common=(bound.weights,*args,cl,kw['candidate_cache_rows'],kw.get('update_cache',0),bound.eps);nc=(bound.weights,*args,a,cl,kw['candidate_cache_rows'],meta[0],valid,bound.eps);empty=args[0].new_empty(0);mod=v['modulation'].reshape(batch,q,6,D)
   for stage,key in stages:
    expected=v[key].clone();row=dict(case=length,stage=stage,key=key,shape=list(expected.shape),arms={})
    funcs={'native':lambda:original.stage(stage,*nc),'prior_executor':lambda:own.forward(*common,stage,a,meta)[key]}
    if stage in (6,7,18,19):
     src,wi,ei,model={6:('self_q_linear',11,3,block.attn1.norm_q),7:('self_k_linear',12,4,block.attn1.norm_k),18:('cross_q_linear',21,5,block.attn2.norm_q),19:('cross_k_linear',22,6,block.attn2.norm_k)}[stage]
     x=v[src];weight=bound.weights[wi];eps=bound.eps[ei];funcs['torch']=lambda:model(x)
     funcs['own_native_tree']=lambda:bare.rms(x,weight,eps,0);funcs['own_108_control']=lambda:bare.rms(x,weight,eps,1)
    elif stage in (8,9):
     x=v['self_q_norm' if stage==8 else 'self_k_norm']
     def torch_rope():
      z=torch.view_as_complex(x.double().reshape(batch,q,24,64,2));return torch.view_as_real(z*args[3]).flatten(3).to(x.dtype)
     funcs['torch']=torch_rope
     funcs.update({label:(lambda mode=mode:bare.rope(x,args[3],q,mode)) for mode,label in enumerate(('own_pair','own_108_control','own_vec4','own_vec4_128'))})
    else:
     if stage==0:
      x=args[2];y=empty;w=bound.weights[0];op=0;param=0;funcs['torch']=lambda:block.scale_shift_table[None]+args[2].float()
     elif stage in (2,24):
      src,offset=('norm1',0) if stage==2 else ('norm3',3);x=v[src];y=empty;w=v['modulation'];op=1;param=offset
      funcs['torch']=lambda:(x.reshape(batch,q,D)*(1.+mod[:,:,offset+1,:])+mod[:,:,offset,:]).to(args[0].dtype)
     else:
      op=2;w=v['modulation']
      if stage==13:x=args[0];y=v['self_out'];param=2
      elif stage==22:x=v['residual1'];y=v['cross_out'];param=-1
      else:x=v['residual2'];y=v['ffn2'];param=5
      funcs['torch']=(lambda:x.reshape(batch,q,D)+y.reshape(batch,q,D)) if param<0 else (lambda:(x.reshape(batch,q,D).float()+y.reshape(batch,q,D).float()*mod[:,:,param,:]).to(args[0].dtype))
     funcs.update({label:(lambda mode=mode:bare.point(x,y,w,op,param,mode)) for mode,label in enumerate(('own_scalar','own_108_control','own_vec4','own_vec4_128'))})
    graphs={};keep=[]
    for label,fn in funcs.items():
     result=fn();out['stage_calls']+=1;torch.cuda.synchronize();row['arms'][label]=dict(precision=compare(expected.reshape(-1),result.reshape(-1)),samples_ms=[])
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):
      for _ in range(5):keep.append(fn())
     out['stage_calls']+=5;graphs[label]=g
    for g in graphs.values():g.replay();out['stage_calls']+=5
    torch.cuda.synchronize();labels=list(funcs)
    for repeat in range(8):
     for label in (labels if (repeat+process)%2==0 else labels[::-1]):
      aa=torch.cuda.Event(enable_timing=True);zz=torch.cuda.Event(enable_timing=True);aa.record(stream);graphs[label].replay();zz.record(stream);zz.synchronize();out['stage_calls']+=5;row['arms'][label]['samples_ms'].append(aa.elapsed_time(zz)/5)
    for t in row['arms'].values():t['median_ms']=statistics.median(t['samples_ms'])
    out['rows'].append(row);save();del graphs,keep,funcs,expected
   audit=state_equal(expected_cache,visible_snapshot(cache));assert audit['pass_all'];out.setdefault('KV_checks',[]).append(dict(case=length,check=audit));del f,args,kw,ref,v,a,cl,cache;torch.cuda.empty_cache()
 out['status']='COMPLETE_MINOR_OPS_ISOLATION'
except BaseException as e:out.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:out['lifecycle_seconds']=time.time()-start;save()
print(out['status'],out.get('error',''),flush=True)
if out['status']!='COMPLETE_MINOR_OPS_ISOLATION':raise SystemExit(1)

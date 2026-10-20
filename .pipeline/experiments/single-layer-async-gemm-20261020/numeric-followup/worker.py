"""Same-input diagnostic of reduced-precision BLAS and fused-bias boundaries.
No global model/baseline settings are changed; process-local controls restored.
"""
import json,os,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from tools.block_loader import load_block,FIXTURES
from tools.fixtures import BlockFixture,decode,tensors
from tools.replay import device_tree
from tools.metrics import compare,byte_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(__file__).resolve().parent
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
t0=time.time();r=dict(status='RUNNING',full_block_calls=0,primitive_calls=0,rows=[],episodes=0,production_admitted=False)
def save():(B/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 block=load_block('cuda');native=NativeBlock(block);whole=WholeLayerBlock(block)
 WORK=Path(load_block.__code__.co_filename).resolve().parents[2]
 video=WORK/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
 original=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
 with torch.inference_mode():
  for kind,batch,length in [('video',2,'long'),('action',2,'short'),('action',14,'short')]:
   assert time.time()-t0<540
   path=(FIXTURES if kind=='action' else video)/f'input-{length}-B{batch}-{kind}-layer0-first.pt'
   d=decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False));fix=BlockFixture(path,device_tree(d));a,k=fix.arguments();fix.install_cache(block)
   ref=native(*a,**k,debug=True);r['full_block_calls']+=1;fix.install_cache(block)
   new=whole(*a,**k,debug=True,audit=True);r['full_block_calls']+=1;torch.cuda.synchronize()
   for name,key,layer,outkey in [('FFN2','gelu',block.ffn.net[2],'ffn2'),('Q','modulated1',block.attn1.to_q,'self_q_linear')]:
    x=new[key];w,b=layer.weight,layer.bias
    row=dict(case=f'{length}-B{batch}-{kind}',name=name,shape=[x.numel()//x.shape[-1],w.shape[0],w.shape[1]])
    default=F.linear(x,w,b);r['primitive_calls']+=1
    row['whole_vs_default_same_input']=compare(default,new[outkey])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    no_reduce=F.linear(x,w,b);r['primitive_calls']+=1
    row['default_vs_no_reduced_precision']=compare(default,no_reduce)
    row['whole_vs_no_reduced_precision']=compare(no_reduce,new[outkey])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
    no_bias=F.linear(x,w,None);r['primitive_calls']+=1
    epilogue=(no_bias.float()+b.float()).bfloat16()
    row['default_vs_separate_BF16_round_then_bias']=compare(default,epilogue)
    # FP32 control is a diagnostic, not a proposed replacement/gold standard.
    prev=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    fp=F.linear(x.float(),w.float(),b.float()).bfloat16();r['primitive_calls']+=1
    torch.backends.cuda.matmul.allow_tf32=prev
    row['default_vs_FP32_control']=compare(default,fp)
    row['whole_vs_FP32_control']=compare(fp,new[outkey])
    row['GPU_parameters_dtype']=str(w.dtype)
    r['rows'].append(row);save();del default,no_reduce,no_bias,epilogue,fp
   r.setdefault('whole_results',[]).append(dict(case=f'{length}-B{batch}-{kind}',comparison=compare(ref['output'],new['output'])))
   del ref,new,fix,a,k,d
 torch.cuda.synchronize();r['status']='COMPLETE_SAME_INPUT_COUNTERFACTUAL_NOT_ADMISSION'
except BaseException as e:r.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:
 if 'original' in globals():torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
 r['lifecycle_seconds']=time.time()-t0;save()

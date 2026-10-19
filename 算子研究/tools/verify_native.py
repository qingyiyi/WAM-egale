"""真实完整block的native逐位/全部有效状态与原语定位；不采机器人数据。"""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import time
import traceback

import torch

from .block_loader import FIXTURES, load_block
from .fixtures import BlockFixture, tensors, visible_snapshot
from .metrics import byte_equal, compare, state_equal
from .replay import device_tree, resource, write
from wam_ops.baseline.native import NativeBlock, extension


def capture(block, arguments, kwargs):
    """原模型实际中间量；reference只调用原模型，native不调用这些hooks。"""
    saved, handles = {}, []
    def hook(name, module, pre=False, cast=False):
        def record(m, inputs, output=None):
            value = inputs[0] if pre else output
            saved[name] = (value.to(torch.bfloat16) if cast else value).detach().clone()
        handles.append((module.register_forward_pre_hook(record) if pre else module.register_forward_hook(record)))
    for label, module in [('norm1',block.norm1),('norm3',block.norm3)]:hook(label,module)
    hook('residual1',block.norm2,pre=True,cast=True)
    hook('norm2',block.norm2,cast=True)
    hook('residual2',block.norm3,pre=True,cast=True)
    for prefix,a in [('self',block.attn1),('cross',block.attn2)]:
        hook(prefix+'_q_raw',a.to_q);hook(prefix+'_k_raw',a.to_k)
        hook(prefix+'_q_norm' if prefix=='self' else 'cross_q',a.norm_q)
        hook(prefix+'_k_norm' if prefix=='self' else 'cross_k',a.norm_k)
        hook(prefix+'_v',a.to_v);hook(prefix+'_out',a.to_out[0])
    hook('modulated1',block.attn1.to_q,pre=True)
    hook('modulated3',block.ffn.net[0].proj,pre=True)
    hook('ffn1',block.ffn.net[0].proj);hook('gelu',block.ffn.net[0]);hook('ffn2',block.ffn.net[2])
    originals = (block.attn1.attn_op,block.attn2.attn_op)
    def attention(prefix, operation):
        def call(q,k,v):
            if prefix=='self':
                saved['self_q_rope'],saved['self_k_rope']=q.clone(), k[:,:q.shape[1]].clone()
            out=operation(q,k,v);saved[prefix+'_attention']=out.clone();return out
        return call
    block.attn1.attn_op=attention('self',originals[0])
    block.attn2.attn_op=attention('cross',originals[1])
    try:
        out=block(*arguments,**kwargs)
        saved['output']=out.clone()
        saved['modulation']=(block.scale_shift_table[None]+arguments[2].float()).reshape(-1,6,3072)
    finally:
        for h in handles:h.remove()
        block.attn1.attn_op,block.attn2.attn_op=originals
    # K passed to original SDPA includes history; use stored pre-RoPE key for
    # primitive checks, and leave self_k_rope out until a correct capture exists.
    saved.pop('self_k_rope',None)
    return saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-idle-seconds',type=int,default=300)
    p.add_argument('--allow-shared-correctness',action='store_true')
    p.add_argument('--optimized',action='store_true')
    args=p.parse_args();reg=json.loads(args.registration.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
    started=time.monotonic();block=load_block(device='cpu')
    if torch.cuda.is_initialized():raise RuntimeError('CPU preload initialized CUDA')
    if args.allow_shared_correctness and not reg.get('shared_correctness_authorized'):
        raise PermissionError('shared correctness flag requires registration')
    obs=resource();deadline=started+args.wait_idle_seconds
    fields=[s.strip() for s in obs['gpu'].split(',')]
    if fields[0]!=reg['physical_gpu_uuid'] or float(fields[2])<4000:
        raise RuntimeError('GPU identity/free memory guard')
    result=dict(status='WAITING_NO_CONTEXT',rows=[],resources=[obs],block_forward_intents=0,
        sharing='CORRECTNESS_ONLY_NO_PERFORMANCE' if args.allow_shared_correctness else 'natural_idle',
        primitive_forward_intents=0,formal_performance_permission=False)
    def save():write(args.output/'RESULT.json',result)
    save()
    while obs['pids'] and not args.allow_shared_correctness and time.monotonic()<deadline:
        time.sleep(min(10,max(0,deadline-time.monotonic())));obs=resource();result['resources'].append(obs);save()
    if obs['pids'] and not args.allow_shared_correctness:
        result['status']='WAIT_EXPIRED_NO_CONTEXT';save();raise RuntimeError(result['status'])
    result.update(status='RUNNING_NATIVE_GATES',idle_wait_seconds=time.monotonic()-started)
    gpu_started=time.monotonic()
    def check():
        if time.monotonic()-gpu_started>480 or time.time()>datetime.fromisoformat(reg['absolute_work_stop']).timestamp()-60:
            raise TimeoutError('native worker lifecycle reserve')
        obs=resource();result['resources'].append(obs)
        if float(obs['gpu'].split(',')[2])<1500:raise RuntimeError('free memory reserve')
        if not args.allow_shared_correctness and set(obs['pids'].splitlines())-{str(os.getpid())}:
            raise RuntimeError('FOREIGN_GPU_PID_STOP')
    try:
        torch.cuda.set_device(0);block=block.to('cuda');native=NativeBlock(block,fused_ffn1=args.optimized);ext=extension()
        immutable_weights=[x.detach().clone() for x in native.weights]
        with torch.inference_mode():
            for length in ('short','middle','long'):
                for batch in (14,2):
                    check();f=BlockFixture.load(FIXTURES,length,batch);f=BlockFixture(f.path,device_tree(f.data))
                    arguments,kwargs=f.arguments();cache=f.install_cache(block);initial=visible_snapshot(cache)
                    inputs={k:x.clone() for k,x in tensors((arguments,kwargs))}
                    rng_cpu,rng_gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state()
                    result['block_forward_intents']+=1;ref=capture(block,arguments,kwargs);gold_state=visible_snapshot(cache)
                    row=dict(case=f.case_id,reference_saved_gold=compare(f.data['block_gold'],ref['output']),intermediates={},primitives={})
                    result['rows'].append(row)
                    for k in ('id','mask','is_pred'):cache[k].copy_(initial[k])
                    for k in ('k','v'):cache[k].index_copy_(1,initial['slots'],initial[k])
                    result['block_forward_intents']+=1;out=native(*arguments,**kwargs,debug=True)
                    for k,v in out.items():
                        if k not in ref:continue
                        if v.numel()!=ref[k].numel():raise RuntimeError('intermediate element count '+k)
                        row['intermediates'][k]=compare(ref[k],v.reshape_as(ref[k]))
                    # Teacher-query primitives isolate upstream recursive error.
                    ops={
                        'norm1':lambda:ext.norm(arguments[0].float(),None,None,native.eps[0],False),
                        'self_q_norm':lambda:ext.norm(ref['self_q_raw'],block.attn1.norm_q.weight,None,native.eps[3],True),
                        'cross_k':lambda:ext.norm(ref['cross_k_raw'],block.attn2.norm_k.weight,None,native.eps[6],True),
                        'ffn1':lambda:ext.linear(ref['modulated3'],native.weights[23],native.weights[24]),
                        'gelu':lambda:ext.gelu(ref['ffn1']),
                        'ffn2':lambda:ext.linear(ref['gelu'],native.weights[25],native.weights[26]),
                    }
                    for k,op in ops.items():
                        result['primitive_forward_intents']+=1;value=op();row['primitives'][k]=compare(ref[k],value.reshape_as(ref[k]))
                    row['state']=state_equal(gold_state,visible_snapshot(cache))
                    row['inputs_unchanged']=all(byte_equal(inputs[k],x) for k,x in tensors((arguments,kwargs)))
                    row['RNG_unchanged']=byte_equal(rng_cpu,torch.random.get_rng_state()) and byte_equal(rng_gpu,torch.cuda.get_rng_state())
                    row['passed']=all(x['bitwise_pass'] for x in row['intermediates'].values()) and all(x['bitwise_pass'] for x in row['primitives'].values()) and row['reference_saved_gold']['bitwise_pass'] and row['state']['pass_all'] and row['inputs_unchanged'] and row['RNG_unchanged']
                    save();del f,ref,out,inputs;block.attn1.attn_caches={};gc.collect()
            result['weights_unchanged']=all(byte_equal(a,b) for a,b in zip(immutable_weights,native.weights))
        result['status']='STRICT_SIX_BLOCK_PASS' if all(r['passed'] for r in result['rows']) and result['weights_unchanged'] else 'NUMERIC_OR_STATE_REJECTED'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(finished_at=datetime.now().astimezone().isoformat(),lifecycle_seconds=time.monotonic()-started,GPU_stage_wall_seconds=time.monotonic()-gpu_started)
        save()


if __name__=='__main__':main()

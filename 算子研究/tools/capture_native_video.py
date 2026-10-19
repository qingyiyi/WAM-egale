"""从已有完整canonical policy快照重放，补存真实Q128 block输入。

不采机器人轨迹、不改服务源；仅原target，完整video20/action50请求。
只编译原snapshot.py的纯恢复函数，不执行其digest辅助函数。
"""
import argparse
import ast
import copy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from .fixtures import decode
from .metrics import compare
from .replay import resource, write

ROOT = Path(__file__).resolve().parents[2]


def run(server, source, args, reg, start):
    model = server.transformer
    if len(model.blocks) != 30 or server._cache_mode != 'original' or server._p0_action_draft_mode != 'off':
        raise RuntimeError('canonical frozen original target required')
    plan = json.loads(args.sources.read_text())
    result = dict(status='RUNNING', target_forward_attempts=0, captured=[],
                  complete_policy_calls=[], episodes=0, performance=False,
                  scope='complete canonical target frozen policy replay, no robot episode')
    control = dict(label=None, armed=False, current=None)
    block = model.blocks[0]
    handles = []
    original_op = block.attn1.attn_op

    def save():
        write(args.output/'RESULT.json', result)

    def check():
        if time.monotonic()-start > 1000:
            raise TimeoutError('capture worker lifecycle')
        if result['target_forward_attempts'] >= 240:
            raise RuntimeError('capture-specific target forward cap240')

    def count(module, inputs, kwargs):
        check()
        result['target_forward_attempts'] += 1

    def pre(module, inputs, kwargs):
        if not control['armed'] or inputs[0].shape[1] != 128 or kwargs.get('update_cache', 0) != 0:
            return
        if control['current'] is not None:
            raise RuntimeError('overlapping block capture')
        cache = module.attn1.attn_caches[kwargs.get('cache_name', 'pos')]
        metadata = {k: v.detach().cpu().clone() for k, v in cache.items() if k in ('id','mask','is_pred')}
        control['current'] = dict(
            schema='real-attention-replay-compact/v1',
            identity=(control['label'], 2, 'video', 'first', 0),
            source='canonical_target_full50_video20_frozen_policy_replay',
            block_args=cpu_tree(inputs), block_kwargs=cpu_tree(kwargs),
            cache_metadata=metadata, history_slots=metadata['mask'].nonzero().flatten(),
            pool_shape=list(cache['k'].shape))

    def attention(q, k, v):
        y = original_op(q, k, v)
        point = control['current']
        if point is not None:
            point.update(self_q=cpu_tree(q), self_k=cpu_tree(k), self_v=cpu_tree(v), self_gold=cpu_tree(y))
        return y

    def post(module, inputs, kwargs, output):
        point = control['current']
        if point is None:
            return
        point['block_gold'] = cpu_tree(output)
        label = control['label']
        historical_path = (ROOT/'投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r3/process1'
                           /f'input-{label}-B2-video-layer0-first.pt')
        historical = decode(torch.load(historical_path, map_location='cpu', mmap=True, weights_only=False))
        audit = {name: compare(historical[name], point[name])
                 for name in ('self_q','self_k','self_v','self_gold','block_gold')}
        if not all(row['bitwise_pass'] for row in audit.values()):
            raise RuntimeError(f'{label} complete-policy capture differs from historical source')
        path = args.output/f'input-{label}-B2-video-layer0-first.pt'
        torch.save(point, path)
        result['captured'].append(dict(label=label, path=str(path), bytes=path.stat().st_size,
            historical_numeric_audit=audit, history_tokens=len(point['history_slots']),
            args_shapes=[list(x.shape) if isinstance(x,torch.Tensor) else None for x in point['block_args']]))
        control.update(armed=False,current=None)
        save()

    snapfile = ROOT/'投机采样/experiments/fixed_load_cost/snapshot.py'
    tree = ast.parse(snapfile.read_text())
    fields = next(n for n in tree.body if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id=='FIELDS' for t in n.targets))
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'cpu','restore_cache','restore','seed'}]
    ns = dict(torch=torch,np=np,random=random,copy=copy)
    exec(compile(ast.Module(body=[fields,*nodes],type_ignores=[]),str(snapfile)+':pure-restoration','exec'),ns)

    def restore_state(state):
        for layer,saved in zip(model.blocks,state['caches']):
            cache = {key:saved[key].to(server.device).clone() for key in ('id','mask','is_pred')}
            cache.update(k=torch.empty(saved['pool_shape'],device=server.device,dtype=torch.bfloat16),
                         v=torch.empty(saved['pool_shape'],device=server.device,dtype=torch.bfloat16))
            layer.attn1.attn_caches={server.cache_name:cache}
        ns['restore'](server,state,cuda=True)
        server._p0_action_draft_mode='off'

    try:
        # No prompt encoding or forward; exact saved RNG is restored afterwards.
        server.infer(dict(reset=True,prompt=None))
        handles = [model.register_forward_pre_hook(count,with_kwargs=True),
                   block.register_forward_pre_hook(pre,with_kwargs=True),
                   block.register_forward_hook(post,with_kwargs=True)]
        block.attn1.attn_op=attention
        short=torch.load(plan['short']['path'],map_location='cpu',mmap=True,weights_only=False)
        for label in ('short','middle','long'):
            check()
            if label=='middle':
                restore_state(short['state'])
                artifacts={(x['frame'],x['kind']):x['path'] for x in plan['middle_prefix_artifacts']}
                server.exp_save_root=str(args.output/'middle-prefix-replay')
                Path(server.exp_save_root).mkdir()
                for frame in (4,8,12,16,20,24):
                    if server.frame_st_id!=frame:
                        raise RuntimeError('middle prefix frame mismatch')
                    saved={kind:torch.load(artifacts[frame,kind],map_location='cpu',weights_only=False)
                           for kind in ('latents','actions','obs_data')}
                    server.infer(dict(p0_counterfactual_prime_video_cache=True,frame_st_id=frame,
                                      video_terminal_state=saved['latents']))
                    server.infer(dict(compute_kv_cache=True,obs=saved['obs_data'],
                                      state=server.postprocess_action(saved['actions'])))
                if server.frame_st_id!=28:
                    raise RuntimeError('middle frame28 expected')
                ns['seed'](20260922,True)
                context=copy.deepcopy(short['request']['p0_trace_context'])
                context['fixed_fixture']='T-init3-chunk7'
                request=dict(obs=saved['obs_data'][-1],p0_trace_context=context)
            else:
                payload=short if label=='short' else torch.load(plan['long']['path'],map_location='cpu',mmap=True,weights_only=False)
                restore_state(payload['state'])
                request=copy.deepcopy(payload['request'])
            server.exp_save_root=str(args.output/f'canonical-{label}')
            Path(server.exp_save_root).mkdir()
            control.update(label=label,armed=True,current=None)
            before=result['target_forward_attempts']
            returned=server.infer(request)
            control.update(label=None,armed=False,current=None)
            result['complete_policy_calls'].append(dict(label=label,
                target_forwards=result['target_forward_attempts']-before,
                returned_type=type(returned).__name__,terminated=True))
            save()
            print('COMPLETE_CAPTURE_POLICY',label,flush=True)
            gc.collect();torch.cuda.empty_cache()
        if len(result['captured'])!=3:
            raise RuntimeError('missing complete video block fixture')
        result['status']='THREE_COMPLETE_CANONICAL_REPLAYS_AND_VIDEO_FIXTURES_CAPTURED'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        block.attn1.attn_op=original_op
        source.save_async.__globals__['executor'].shutdown(wait=True)
        result.update(lifecycle_seconds=time.monotonic()-start,
                      allocated_peak_bytes=torch.cuda.max_memory_allocated())
        save()


def cpu_tree(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();reg=json.loads(args.registration.read_text())
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();obs=resource();fields=obs['gpu'].split(',')
    write(args.output/'ADMISSION.json',obs)
    if fields[0].strip()!=reg['physical_gpu_uuid'] or float(fields[2])<28672:
        raise RuntimeError('full canonical replay needs GPU identity and >=28GiB free; no CUDA created')
    if not reg.get('shared_correctness_authorized'):
        raise PermissionError('shared correctness not registered')
    for key in list(os.environ):
        if key.startswith(('WAM_P0_','P0_','LINGBOT_')):
            os.environ.pop(key)
    frozen=json.loads(args.sources.read_text())['frozen_reference']
    settings=dict(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',LINGBOT_ATTN_MODE='torch',
        LINGBOT_NSYS_PROFILE='off',LINGBOT_SEED='20260922',LINGBOT_VA_LIBERO_SUITE='libero_10',
        LINGBOT_VA_LIBERO_CHECKPOINT=frozen['target_checkpoint_path'],
        LINGBOT_VA_LIBERO_CHECKPOINT_ID=frozen['target_checkpoint_id'],
        LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=frozen['registry'],WAM_VAE_DEVICE='cuda',
        WAM_P0_PHASE_PROFILE='0',WAM_P0_ACTION_DRAFT_MODE='off',
        MASTER_ADDR='127.0.0.1',MASTER_PORT='29742',RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1')
    os.environ.update(settings)
    torch.cuda.set_per_process_memory_fraction(.70)
    sys.path.insert(0,str(ROOT/'投机采样/wam-p0-worktree'))
    from wan_va import wan_va_server as source
    def execute(server,rank,host,port):
        if rank!=0:
            raise RuntimeError('single worker only')
        with torch.no_grad():
            run(server,source,args,reg,start)
    source.run_async_server_mode=execute
    sys.argv=[sys.argv[0],'--config-name','libero','--port','29743','--save_root',str(args.output/'server')]
    source.init_logger()
    source.main()


if __name__=='__main__':
    main()

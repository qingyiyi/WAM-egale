"""Audit every original asynchronous action, latent and observation save on CPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_tree(value, name):
    if isinstance(value, torch.Tensor):
        require(value.numel()>0 and (not value.is_floating_point() or bool(torch.isfinite(value).all())),
                'empty/nonfinite saved tensor: '+name)
        return 1
    if isinstance(value, np.ndarray):
        require(value.size>0 and (value.dtype.kind not in 'fc' or bool(np.isfinite(value).all())),
                'empty/nonfinite saved array: '+name)
        return 1
    if isinstance(value, dict):
        return sum(finite_tree(v,name+'/'+str(k)) for k,v in value.items())
    if isinstance(value, (tuple,list)):
        return sum(finite_tree(v,name+'/'+str(i)) for i,v in enumerate(value))
    require(value is None or isinstance(value,(str,int,float,bool)), 'unknown saved value: '+name)
    return 0


def coverage(directory: Path, record: dict, counter: dict):
    directory=Path(directory)
    actual={}
    for path in directory.rglob('*.pt'):
        require(path.name not in actual and path.name.startswith(('actions_','latents_','obs_data_')),
                'unexpected/duplicate saved output: '+str(path))
        actual[path.name]=path
    require(bool(actual),'original saved outputs absent')
    expected=[]; policy=[]; cache=[]
    for request in counter['requests']:
        kind=request['kind'];frame=request['server_frame_st_id_before']
        if kind=='reset':
            require(request['expected_saved_outputs']==[] and request['server_frame_st_id_after']==0,
                    'reset saved/frame accounting differs')
            continue
        chunk=request['trace_context']['rollout_chunk']
        require(type(chunk) is int and frame==4*chunk,'original frame/chunk sequence differs')
        if kind=='policy':
            policy.append(chunk);names=[f'actions_{frame}.pt',f'latents_{frame}.pt']
            require(request['server_frame_st_id_after']==frame,'policy advanced original video frame')
        elif kind=='cache':
            cache.append(chunk);names=[f'obs_data_{frame}.pt']
            require(request['server_frame_st_id_after']==frame+4,'cache did not advance four video frames')
        else:
            raise ValueError('unknown complete request kind')
        require(request['expected_saved_outputs']==names,'request-to-save filenames differ')
        expected+=names
    require(policy==list(range(record['infer_chunks'])) and cache==list(range(record['cache_infer_calls'])),
            'policy/cache chunk sequences have gaps or duplicates')
    require(len(expected)==len(set(expected)) and set(expected)==set(actual),
            'actual saved output inventory differs from all complete requests')
    tensors=0
    for name,path in sorted(actual.items()):
        tensors+=finite_tree(torch.load(path,map_location='cpu',weights_only=False),name)
    require(tensors>0,'saved original output tensors absent')
    return dict(action_files=len(policy),latent_files=len(policy),observation_files=len(cache),
                files=len(actual),tensor_arrays_read=tensors,coverage_exact=True,
                all_saved_values_finite=True)

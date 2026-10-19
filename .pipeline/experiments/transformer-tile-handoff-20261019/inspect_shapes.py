"""CPU-only: inspect saved logical inputs and safetensors headers, never weights."""
from datetime import datetime
from pathlib import Path
import gc
import json
import os
import struct
import time

os.sched_setaffinity(0, {4, 5})
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
START = time.monotonic()


def signature(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    serialized_stride=list(value.stride()), device=str(value.device))
    if isinstance(value, dict) and 'encoding' in value:
        return dict(shape=value['shape'], encoding=value['encoding'],
                    stored_root=signature(value['value']),
                    dtype=str(value['value'].dtype),
                    stride_scope='decoded contiguous logical fixture, not original production CUDA stride')
    if isinstance(value, dict):
        return {k: signature(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [signature(v) for v in value]
    return value


def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def small_tensor(value):
    if isinstance(value,torch.Tensor):
        result=value
    elif isinstance(value,dict) and 'encoding' in value:
        result=value['value']
        if value['encoding']=='two_CFG_groups_repeated7':
            result=result.repeat_interleave(7,dim=0)
        elif value['encoding']!='tensor':
            raise ValueError('unsupported small metadata encoding')
    else:
        raise TypeError('not saved tensor metadata')
    if result.numel()>10000:
        raise ValueError('metadata-only adapter refuses large tensor')
    return result


config_dir = ROOT/'models/lingbot-va-posttrain-libero-long-modelscope-20260922/transformer'
config = json.loads((config_dir/'config.json').read_text())
index = json.loads((config_dir/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
headers = {}
params = []
for key,shard in sorted(index.items()):
    if not key.startswith('blocks.0.'):
        continue
    if shard not in headers:
        with (config_dir/shard).open('rb') as stream:
            size = struct.unpack('<Q',stream.read(8))[0]
            if size > 32*2**20:
                raise ValueError('unexpected large safetensors header')
            headers[shard] = json.loads(stream.read(size))
    entry=headers[shard][key]
    params.append(dict(name=key,shape=entry['shape'],checkpoint_dtype=entry['dtype'],
        runtime_dtype=('torch.float32' if key=='blocks.0.scale_shift_table' or key.startswith('blocks.0.norm2.')
                       else 'torch.bfloat16')))
write('PARAMETERS.json',dict(config=config,parameters=params,
    scope='safetensors header metadata only, no tensor payload/hash',header_files_read=len(headers)))

action = ROOT/'投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r3/process1'
video = ROOT/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
rows=[]
for length in ('short','middle','long'):
    for batch,kind,base in ((2,'action',action),(14,'action',action),(2,'video',video)):
        path=base/f'input-{length}-B{batch}-{kind}-layer0-first.pt'
        raw=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        args=raw['block_args'];kwargs=raw['block_kwargs'];meta=raw['cache_metadata']
        fields={k:signature(raw[k]) for k in ('self_q','self_k','self_v','cross_q','cross_k','cross_v','block_gold','history_k','history_v') if k in raw}
        row=dict(case=f'{length}-B{batch}-{kind}',input_path=str(path.relative_to(ROOT)),
            file_bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
            schema=raw['schema'],identity=raw['identity'],
            block_argument_names=['hidden_states','encoder_hidden_states','temb','rotary_emb'],
            block_args=signature(args),block_kwargs=signature(kwargs),
            cache_pool_shape=raw['pool_shape'],cache_metadata=signature(meta),
            pre_call_visible_slots=int(small_tensor(meta['mask']).sum()),
            saved_history_slot_count=int(small_tensor(raw['history_slots']).numel()),fields=fields)
        candidate=kwargs.get('candidate_cache_rows')
        if candidate is not None:
            row['candidate_cache_rows_values']=small_tensor(candidate).tolist()
        assert row['pre_call_visible_slots']==row['saved_history_slot_count']
        assert signature(args[0])['shape']==[batch,16 if kind=='action' else 128,3072]
        rows.append(row)
        del raw,args,kwargs,meta
        gc.collect()
assert not torch.cuda.is_initialized()
write('SHAPES.json',dict(at=datetime.now().astimezone().isoformat(),rows=rows,
    scope='existing serialized logical CPU fixtures; not original CUDA storage-offset/alias reconstruction',
    GPU_context_created=torch.cuda.is_initialized(),new_episodes=0,new_tensors_saved=False,
    CPU_wall_seconds=time.monotonic()-START))
print(json.dumps(dict(rows=len(rows),parameters=len(params),GPU_used=False,
    CPU_wall_seconds=time.monotonic()-START)))

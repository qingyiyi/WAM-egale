"""只检验trace支持的FFN2分块/中间舍入假设；不改Torch设置或精度门。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .block_loader import load_block
from .metrics import compare
from .replay import resource, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--serial',action='store_true',help='在原两种并行workspace假设后，检验有bias/无bias的串行BF16累加')
    args=p.parse_args()
    reg=json.loads(args.registration.read_text())
    before=resource()
    if before['pids'] or before['gpu'].split(',')[0].strip()!=reg['physical_gpu_uuid']:
        raise RuntimeError('GPU_BUSY_NO_CONTEXT_CREATED')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start=time.monotonic()
    block=load_block()
    result=dict(status='RUNNING',rows=[],started_at=datetime.now().astimezone().isoformat(),
        basis='actual original FFN2 gridZ: B2=7, B14=2; two workspace dtype hypotheses only',
        scope='two short real FFN inputs; fixed split counts; no new kernel configurations',
        original_linear_calls=0, full_block_forwards=0, primitive_calls=0)
    try:
        with torch.inference_mode():
            w1,b1=block.ffn.net[0].proj.weight,block.ffn.net[0].proj.bias
            w2,b2=block.ffn.net[2].weight,block.ffn.net[2].bias
            for batch,split in ((2,7),(14,2)):
                data=torch.load(args.inputs/f'short-B{batch}-action-layer0-first-ffn.pt',
                    mmap=True,map_location='cpu',weights_only=False)
                x=data['input'].to('cuda').flatten(0,1)
                a=F.gelu(F.linear(x,w1,b1),approximate='tanh')
                gold=F.linear(a,w2,b2)
                result['original_linear_calls']+=2
                width=a.shape[1]//split
                partial=torch.stack([F.linear(a[:,s*width:(s+1)*width].float(),
                    w2[:,s*width:(s+1)*width].float()) for s in range(split)])
                result['original_linear_calls']+=split
                row=dict(batch=batch,partitions=split,K_per_partition=width,
                    FP32_workspace=compare(gold,(partial.sum(0)+b2.float()).bfloat16()),
                    BF16_workspace=compare(gold,(partial.bfloat16().float().sum(0)+b2.float()).bfloat16()))
                if args.serial:
                    for start_with_bias in (True,False):
                        acc=b2.expand_as(gold).clone() if start_with_bias else torch.zeros_like(gold)
                        for part in partial:
                            acc=(acc.float()+part).bfloat16()
                        if not start_with_bias:
                            acc=(acc.float()+b2.float()).bfloat16()
                        row['serial_bias_first' if start_with_bias else 'serial_bias_last']=compare(gold,acc)
                    torch.save(dict(partial_FP32=partial.cpu(),gold=gold.cpu(),bias=b2.cpu(),
                        partitions=split,source=str(args.inputs/f'short-B{batch}-action-layer0-first-ffn.pt')),
                        args.output.with_name(f'short-B{batch}-serial-inputs.pt'))
                result['rows'].append(row)
        result['status']='COMPLETE_TWO_FIXED_REDUCTION_DIAGNOSTICS'
    finally:
        result.update(lifecycle_seconds=time.monotonic()-start,finished_at=datetime.now().astimezone().isoformat())
        write(args.output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()

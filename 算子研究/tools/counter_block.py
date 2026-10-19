"""NCU只包围一个已校验的真实完整block；profile不进入性能表。"""
import argparse
import json
import os
from pathlib import Path

import torch

from .block_loader import FIXTURES, load_block
from .fixtures import BlockFixture, visible_snapshot
from .metrics import byte_equal, state_equal
from .replay import device_tree, resource, write
from wam_ops.optimized.prepared import PreparedBlock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--profile-result', type=Path, required=True)
    p.add_argument('--length', choices=['short','middle','long'], required=True)
    p.add_argument('--batch', type=int, choices=[2,14], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    before = resource()
    if before['pids'] or before['gpu'].split(',')[0].strip() != reg['physical_gpu_uuid']:
        raise RuntimeError('GPU_BUSY_NO_CONTEXT_CREATED')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {4,5})
    row = next(r for r in json.loads(args.profile_result.read_text())['rows']
        if r['case'] == f'{args.length}-B{args.batch}-action-layer0-first')
    engine = row['strongest_static_baseline']['engine']
    block = load_block()
    with torch.inference_mode():
        fixture = BlockFixture.load(FIXTURES,args.length,args.batch)
        fixture = BlockFixture(fixture.path, device_tree(fixture.data))
        arguments, kwargs = fixture.arguments()
        cache = fixture.install_cache(block)
        oracle = block(*arguments, **kwargs)
        initial = visible_snapshot(cache)
        prepared = PreparedBlock(block,arguments,kwargs,cache)
        prepared.fill()
        with prepared.activate(engine,True):
            admitted = block(*arguments,**kwargs)
            if not byte_equal(oracle,admitted) or not state_equal(initial,visible_snapshot(cache))['pass_all']:
                raise RuntimeError('counter path not admitted')
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(row['case'])
            torch.cuda.profiler.start()
            output = block(*arguments,**kwargs)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            torch.cuda.nvtx.range_pop()
        result = dict(case=row['case'],engine=engine,cached=True, full_block_forwards=3,
            output_bitwise=byte_equal(output,oracle), cache=state_equal(initial,visible_snapshot(cache)),
            profiled_invocations=1, scope='eager strongly cached complete block; replay/cache policy in supervisor command',
            formal_performance_permission=False)
        write(args.output,result)
        print(json.dumps(result))


if __name__=='__main__':
    main()

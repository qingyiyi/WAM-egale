"""单层完整block交错计时；GPU0独占，另一张卡负载明确保留。"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import statistics
import time

import torch

from .block_loader import FIXTURES, load_block
from .fixtures import BlockFixture, decode, visible_snapshot
from .metrics import byte_equal, state_equal
from .replay import device_tree, write
from .resources import resource_for_gpu
from wam_ops.install import install_block

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / '.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
ENGINES = [('original_A', None), ('sequential', {}), ('tiled', {'execution_mode': 'tiled'}),
           ('tiled_ffn', {'execution_mode': 'tiled', 'fused_ffn1': True}),
           ('persistent', {'execution_mode': 'persistent'}),
           ('persistent_ffn', {'execution_mode': 'persistent', 'fused_ffn1': True}),
           ('original_B', None)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    args.output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, set(reg['CPU_allowed']))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    result = dict(status='RUNNING', rows=[], resources=[], block_forward_intents=0,
                  scope='single_complete_dynamic_block_including_KV_pack_and_sync',
                  protocol='3 rotated rounds; 1 call/sample; 1 checked warmup/engine; A/A',
                  host_sharing='other_GPU_evaluation_may_run',
                  performance_class='EXPLORATORY_SAME_GPU_BLOCK_ONLY_NOT_FINAL_E2E',
                  formal_performance_permission=False)

    def save():
        write(args.output / 'RESULT.json', result)

    def guard():
        if time.monotonic() - started > 420 or time.time() >= datetime.fromisoformat(reg['absolute_work_stop']).timestamp() - 60:
            raise TimeoutError('timing lifecycle reserve')
        obs = resource_for_gpu(reg['physical_gpu_uuid'])
        result['resources'].append(obs)
        observed = set(obs['pids'].splitlines())
        expected = set(result.get('nvml_process_set', []))
        if observed != expected:
            raise RuntimeError('registered GPU process set changed: stop timing')

    try:
        guard()
        block = load_block()
        # NVML reports host PIDs, which need not exist in this container's /proc.
        # The GPU was empty before our sole CUDA worker initialized. Bind the
        # single newly observed context, then require that exact set throughout.
        # This is an observed context association, not a proven host PID mapping.
        torch.cuda.synchronize()
        own = resource_for_gpu(reg['physical_gpu_uuid'])
        ids = own['pids'].splitlines()
        if len(ids) != 1 or not torch.cuda.is_initialized() or torch.cuda.memory_allocated() == 0:
            raise RuntimeError('cannot isolate sole newly initialized GPU context')
        result['nvml_process_set'] = ids
        result['container_pid'] = os.getpid()
        result['pid_association'] = 'empty_GPU_then_single_local_CUDA_worker; host_PID_mapping_not_proven'
        result['resources'].append(own)
        with torch.inference_mode():
            for kind, batch in [('action', 2), ('action', 14), ('video', 2)]:
                for length in ('short', 'middle', 'long'):
                    guard()
                    path = (FIXTURES if kind == 'action' else VIDEO) / f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                    f = BlockFixture(path, device_tree(decode(torch.load(path, map_location='cpu', mmap=True, weights_only=False))))
                    arguments, kwargs = f.arguments()
                    cache = f.install_cache(block)
                    initial = visible_snapshot(cache)

                    def restore():
                        for k in ('id', 'mask', 'is_pred'):
                            cache[k].copy_(initial[k])
                        for k in ('k', 'v'):
                            cache[k].index_copy_(1, initial['slots'], initial[k])

                    def install(options):
                        return nullcontext() if options is None else install_block(block, 'optimized', **options)

                    row = dict(case=f'{length}-B{batch}-{kind}', input_path=str(path), raw=[])
                    result['rows'].append(row)
                    gold = gold_state = None
                    for name, options in ENGINES:
                        restore()
                        with install(options):
                            result['block_forward_intents'] += 1
                            out = block(*arguments, **kwargs)
                        torch.cuda.synchronize()
                        if gold is None:
                            gold, gold_state = out.clone(), visible_snapshot(cache)
                        if not byte_equal(gold, out) or not byte_equal(f.data['block_gold'], out) or not state_equal(gold_state, visible_snapshot(cache))['pass_all']:
                            raise RuntimeError('checked timing warmup failed: ' + name)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    for rnd in range(3):
                        order = ENGINES[rnd:] + ENGINES[:rnd]
                        for name, options in order:
                            restore()
                            with install(options):
                                torch.cuda.synchronize()
                                wall = time.perf_counter()
                                begin.record()
                                result['block_forward_intents'] += 1
                                out = block(*arguments, **kwargs)
                                end.record()
                                end.synchronize()
                                wall_ms = (time.perf_counter() - wall) * 1000
                            if not byte_equal(gold, out):
                                raise RuntimeError('timed output failed: ' + name)
                            row['raw'].append(dict(round=rnd, engine=name, event_ms=begin.elapsed_time(end), wall_ms=wall_ms))
                    guard()
                    row['median_event_ms'] = {name: statistics.median(x['event_ms'] for x in row['raw'] if x['engine'] == name) for name, _ in ENGINES}
                    row['median_wall_ms'] = {name: statistics.median(x['wall_ms'] for x in row['raw'] if x['engine'] == name) for name, _ in ENGINES}
                    row['post_state_pass'] = state_equal(gold_state, visible_snapshot(cache))['pass_all']
                    if not row['post_state_pass']:
                        raise RuntimeError('post timing state failed')
                    save()
                    del f, arguments, kwargs, cache, initial, gold, gold_state, out
                    block.attn1.attn_caches = {}
                    gc.collect()
        result['status'] = 'NINE_COMPLETE_BLOCK_EXPLORATORY_TIMING_COMPLETE'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL', error=repr(exc))
        raise
    finally:
        result['lifecycle_seconds'] = time.monotonic() - started
        save()


if __name__ == '__main__':
    main()

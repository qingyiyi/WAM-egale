"""真实block中捕获Q/K/V与FFN输入，核验常驻CTA的逐tile覆盖及输出。"""
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
from .fixtures import BlockFixture, decode
from .metrics import compare
from .replay import device_tree, write
from .resources import resource_for_gpu
from wam_ops.install import install_block
from wam_ops.optimized.native import extension

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / '.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'


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
    start = time.monotonic()
    result = dict(status='RUNNING', rows=[], block_forward_intents=0, primitive_forwards=0,
                  timing='DISPATCH_AND_COVERAGE_ONLY_NO_PERFORMANCE')

    def save():
        write(args.output / 'RESULT.json', result)

    def coverage(values, meta):
        count, workers, splits, shared = meta.tolist()
        counts = values.cpu().tolist()
        seen, done = counts[:count], counts[count:]
        expected = [len(range(i, count, workers)) for i in range(workers)]
        return dict(tasks=count, workers=workers, splits=splits, dynamic_smem_bytes=shared,
                    task_visits=seen, worker_tiles=done,
                    passed=len(counts) == count + workers and all(x == 1 for x in seen) and done == expected,
                    at_least_one_worker_reused=max(done) > 1)

    try:
        obs = resource_for_gpu(reg['physical_gpu_uuid'])
        if obs['pids'] or float(obs['gpu'].split(',')[2]) < 4000:
            raise RuntimeError('registered GPU not available')
        block = load_block()
        ext = extension()
        with torch.inference_mode():
            for kind, batch in [('action', 2), ('action', 14), ('video', 2)]:
                for length in ('short', 'middle', 'long'):
                    if time.monotonic() - start > 360 or time.time() >= datetime.fromisoformat(reg['absolute_work_stop']).timestamp() - 60:
                        raise TimeoutError('audit lifecycle reserve')
                    path = (FIXTURES if kind == 'action' else VIDEO) / f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                    f = BlockFixture(path, device_tree(decode(torch.load(path, weights_only=False, mmap=True, map_location='cpu'))))
                    arguments, kwargs = f.arguments()
                    f.install_cache(block)
                    actual = {}
                    operations = (block.attn1.attn_op, block.attn2.attn_op)

                    def capture(name, op):
                        def call(q, k, v):
                            out = op(q, k, v)
                            actual[name] = (q.clone(), k.clone(), v.clone(), out.clone())
                            return out
                        return call

                    try:
                        block.attn1.attn_op = capture('self', operations[0])
                        block.attn2.attn_op = capture('cross', operations[1])
                        result['block_forward_intents'] += 1
                        gold = block(*arguments, **kwargs)
                    finally:
                        block.attn1.attn_op, block.attn2.attn_op = operations
                    row = dict(case=f'{length}-B{batch}-{kind}', attention=[], ffn=[], dispatch=None)
                    for name, (q, k, v, out) in actual.items():
                        # Default worker count and a forced two-worker stress replay
                        # of the same complete operator, not a different robot case.
                        for workers in (0, 2):
                            result['primitive_forwards'] += 1
                            value, visits, meta = ext.audit_sdpa(q, k, v, workers)
                            check = dict(stage=name, worker_limit=workers, q_shape=list(q.shape),
                                         k_shape=list(k.shape), output=compare(out, value), coverage=coverage(visits, meta))
                            check['passed'] = check['output']['bitwise_pass'] and check['coverage']['passed']
                            row['attention'].append(check)
                    f.install_cache(block)
                    with install_block(block, 'optimized', execution_mode='sequential'):
                        result['block_forward_intents'] += 1
                        debug = block.__dict__['_native_block'](*arguments, **kwargs, debug=True)
                    if kind == 'action':
                        for workers in (0, 2):
                            result['primitive_forwards'] += 1
                            value, visits, meta = ext.audit_ffn1(debug['modulated3'].reshape(-1, 3072),
                                block.ffn.net[0].proj.weight.detach(), block.ffn.net[0].proj.bias.detach(), workers)
                            check = dict(worker_limit=workers, output=compare(debug['gelu'], value.reshape_as(debug['gelu'])),
                                         coverage=coverage(visits, meta))
                            check['passed'] = check['output']['bitwise_pass'] and check['coverage']['passed']
                            row['ffn'].append(check)
                    if kind == 'action' and length == 'short':
                        f.install_cache(block)
                        with install_block(block, 'optimized', execution_mode='persistent', fused_ffn1=True):
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                                result['block_forward_intents'] += 1
                                value = block(*arguments, **kwargs)
                                torch.cuda.synchronize()
                        names = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
                        row['dispatch'] = dict(names=names, output=compare(gold, value),
                            attention_observed=any('persistent_attention' in n for n in names),
                            ffn_observed=any('persistent_linear' in n for n in names), timing_used=False)
                        row['dispatch']['passed'] = (row['dispatch']['output']['bitwise_pass'] and
                            row['dispatch']['attention_observed'] and row['dispatch']['ffn_observed'])
                    row['passed'] = all(x['passed'] for x in row['attention'] + row['ffn']) and (
                        row['dispatch'] is None or row['dispatch']['passed'])
                    result['rows'].append(row)
                    save()
                    if not row['passed']:
                        raise RuntimeError('persistent coverage/output/dispatch failed: ' + row['case'])
                    del f, arguments, actual, debug, gold, value, visits, meta
                    block.attn1.attn_caches = {}
                    gc.collect()
        result['status'] = 'NINE_REAL_CASES_EXACT_COVERAGE_AND_OUTPUT_PASS'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['lifecycle_seconds'] = time.monotonic() - start
        save()


if __name__ == '__main__':
    main()

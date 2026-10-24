"""Three actual complete blocks through the unmodified public extension loader.

The frozen public source remains unchanged; inject one independent candidate module. This script
uses explicit loader injection and the real public cached-context implementation.
"""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import time
import traceback

import torch
from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture, decode, visible_snapshot, tensors
from tools.metrics import compare, state_equal, byte_equal
from tools.time_whole_layer import capture, device_tree, resource, load_extension
from wam_ops.optimized.whole_layer import WholeLayerBlock
import wam_ops.optimized.whole_layer as whole_api

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--module-name', required=True)
    parser.add_argument('--expect-tma', choices=('true','false'), default='false')
    parser.add_argument('--stream', choices=('default', 'custom'), default='default')
    parser.add_argument('--expect-splitk', choices=('true', 'false'), default='true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    job = json.loads((ROOT / 'JOB.json').read_text())
    deadline = datetime.fromisoformat(job['absolute_cleanup_deadline']).timestamp() - 90
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.time()
    result = dict(status='RUNNING', rows=[], calls=0, before=resource(),
                  expected_splitk=args.expect_splitk == 'true', task_schema='PAIR2048x2-384-owner-tasks-v1', stream=args.stream,
                  public_source=whole_api.__file__, loader='independent candidate DSO through same WholeLayerBlock API; explicit loader injection',
                  no_hash='USER_NO_SHA', registered_job=job)

    def save():
        (args.output / 'RESULT.json').write_text(json.dumps(result, indent=2))

    try:
        if result['before']['processes']:
            raise RuntimeError('visible GPU context already exists')
        extension = load_extension(args.binary, args.module_name)
        whole_api.extension = lambda: extension
        if extension.__name__ != args.module_name:
            raise RuntimeError(f'public module identity mismatch: {extension.__name__}')
        if Path(extension.__file__).resolve() != args.binary.resolve():
            raise RuntimeError(f'public binary path mismatch: {extension.__file__}')
        result['loaded_module'] = dict(name=extension.__name__, path=extension.__file__)
        block = load_block()
        bind = WholeLayerBlock(block)
        torch.cuda.synchronize()
        owner = resource()['processes'].splitlines()
        if len(owner) != 1:
            raise RuntimeError('expected one owned visible CUDA context')
        stream = torch.cuda.Stream() if args.stream == 'custom' else torch.cuda.default_stream()
        stream.wait_stream(torch.cuda.current_stream())
        result['stream_id'] = int(stream.cuda_stream)
        with torch.cuda.stream(stream), torch.inference_mode():
            for length in ('short', 'middle', 'long'):
                if time.time() > deadline:
                    raise TimeoutError('registered cleanup deadline')
                path = FIXTURES / f'input-{length}-B14-action-layer0-first.pt'
                fixture = BlockFixture(path, device_tree(decode(torch.load(
                    path, map_location='cpu', mmap=True, weights_only=False))))
                a, kw = fixture.arguments()
                cache = fixture.install_cache(block)
                initial = visible_snapshot(cache)
                inputs = {key: value.clone() for key, value in tensors((a, kw))}
                rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
                with capture(block) as refs:
                    refs['output'] = block(*a, **kw)
                    result['calls'] += 1
                bind.prepare_context(a[1], precompute=True)
                debug = bind(*a, **kw, debug=True)
                result['calls'] += 1
                normal = bind(*a, **kw, debug=False)
                result['calls'] += 1
                required = set(refs) | {'resources', 'ffn_pipeline_state'}
                if result['expected_splitk']:
                    required.add('ffn_splitk_commit')
                missing = sorted(required - set(debug))
                if missing:
                    result['rows'].append(dict(case=length, missing_fields=missing, pass_all=False))
                    save()
                    raise RuntimeError(f'missing required public debug fields: {missing}')
                numeric = {key: compare(refs[key].reshape(-1), debug[key].reshape(-1)) for key in refs}
                state = debug['ffn_pipeline_state'].tolist()
                row = dict(case=length, precision=numeric,
                           all_exact=bool(numeric) and all(value['bitwise_pass'] for value in numeric.values()),
                           normal_debug=compare(debug['output'], normal),
                           normal_reference=compare(refs['output'], normal),
                           valid_KV=state_equal(initial, visible_snapshot(cache)),
                           input_unchanged=all(byte_equal(inputs[key], value) for key, value in tensors((a, kw))),
                           RNG_unchanged=byte_equal(rng[0], torch.get_rng_state()) and byte_equal(rng[1], torch.cuda.get_rng_state()),
                           resources=debug['resources'].tolist(), ffn_pipeline_state=state)
                if args.expect_tma == 'true':
                    row['projection_tma_counts'] = debug['projection_tma_counts'].tolist()
                    row['tma_count_pass'] = row['projection_tma_counts'] == [[[48,48,10,48] for _ in range(96)] for _ in range(6)]
                    if not row['tma_count_pass']: raise RuntimeError('public TMA count mismatch')
                    row['role_count_pass'] = debug['projection_role_counts'].tolist() == [[([[48,48,96]]+[[10,10,10]]*3+[[48,48,10]]*4) for _ in range(96)] for _ in range(6)]
                    if not row['role_count_pass']: raise RuntimeError('public warp role count mismatch')
                if result['expected_splitk']:
                    commits = debug['ffn_splitk_commit'].tolist()
                    row['ffn_splitk_commit'] = commits
                    row['ffn_pair_schema']=debug['ffn_pair_schema'].tolist()
                    if row['ffn_pair_schema']!=[4,7,96,384]:raise RuntimeError('new pair schema mismatch')
                    row['splitk_count_pass'] = (len(commits) == 96 and commits == [7] * 96 and
                                                state[4:8] == [96] * 4 and state[8:12] == [24] * 4)
                else:
                    row['splitk_count_pass'] = None
                    row['splitk_count_check'] = 'NOT_APPLICABLE: explicitly declared retained non-splitK implementation'
                row['pass_all'] = (row['all_exact'] and row['normal_debug']['bitwise_pass'] and
                                   row['normal_reference']['bitwise_pass'] and row['valid_KV']['pass_all'] and
                                   row['input_unchanged'] and row['RNG_unchanged'] and
                                   (row['splitk_count_pass'] if result['expected_splitk'] else True))
                result['rows'].append(row)
                save()
                if not row['pass_all']:
                    raise RuntimeError('public finite numerical/state/count gate failed')
                current = resource()['processes'].splitlines()
                if [v.rsplit(',', 1)[0] for v in current] != [v.rsplit(',', 1)[0] for v in owner]:
                    raise RuntimeError('visible CUDA context ownership changed')
                del fixture, a, kw, cache, refs, debug, normal, inputs, initial
                block.attn1.attn_caches = {}
                bind.context_kv = []
                gc.collect()
                torch.cuda.empty_cache()
        stream.synchronize()
        if len(result['rows']) != 3 or not all(row['pass_all'] for row in result['rows']):
            raise RuntimeError('incomplete public fixture coverage')
        result['status'] = 'FINITE_EXACT_PASS'
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
    finally:
        result['lifecycle_seconds'] = time.time() - start
        save()
    print(result['status'], result.get('error', ''), flush=True)
    if result['status'] != 'FINITE_EXACT_PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

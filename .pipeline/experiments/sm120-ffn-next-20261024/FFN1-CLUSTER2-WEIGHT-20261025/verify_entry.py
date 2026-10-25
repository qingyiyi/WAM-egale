"""采用后直接核验公开Python入口；同三份真实输入，不计入性能表。"""
from pathlib import Path
import json
import torch
from tools.block_loader import load_block, FIXTURES
from tools.fixtures import BlockFixture, decode, visible_snapshot, tensors
from tools.time_whole_layer import device_tree, resource
from tools.metrics import compare, state_equal, byte_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock, extension

P = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
result = dict(status='RUNNING', before=resource(), rows=[], calls=0)
try:
    if result['before']['processes']:
        raise RuntimeError('GPU context present before admission')
    block = load_block()
    bind = WholeLayerBlock(block)
    with torch.inference_mode():
        for case in ['short', 'middle', 'long']:
            fixture = FIXTURES/f'input-{case}-B14-action-layer0-first.pt'
            f = BlockFixture(fixture, device_tree(decode(torch.load(
                fixture, map_location='cpu', mmap=True, weights_only=False))))
            args, kwargs = f.arguments()
            cache = f.install_cache(block)
            snapshot = visible_snapshot(cache)
            inputs = {k:v.clone() for k,v in tensors((args,kwargs))}
            rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            target = block(*args, **kwargs)
            bind.prepare_context(args[1], precompute=True)
            normal = bind(*args, **kwargs)
            debug = bind(*args, **kwargs, debug=True)
            result['calls'] += 3
            row = dict(case=case, normal=compare(target,normal),
                       debug=compare(target,debug['output']),
                       valid_KV=state_equal(snapshot,visible_snapshot(cache)),
                       inputs_unchanged=all(byte_equal(inputs[k],v) for k,v in tensors((args,kwargs))),
                       RNG_unchanged=byte_equal(rng[0],torch.get_rng_state()) and byte_equal(rng[1],torch.cuda.get_rng_state()),
                       binary=extension().__file__)
            result['rows'].append(row)
            if not (row['normal']['bitwise_pass'] and row['debug']['bitwise_pass']
                    and row['valid_KV']['pass_all'] and row['inputs_unchanged'] and row['RNG_unchanged']):
                raise RuntimeError('public entry differs from tested path')
            block.attn1.attn_caches = {}
            bind.context_kv = []
    result['status'] = 'COMPLETE_PUBLIC_ENTRY_EXACT'
except BaseException as error:
    result.update(status='FAILED', error=repr(error))
finally:
    result['after'] = resource()
    (P/'PUBLIC-ENTRY-RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], result.get('error',''), flush=True)
if result['status'] != 'COMPLETE_PUBLIC_ENTRY_EXACT':
    raise SystemExit(1)

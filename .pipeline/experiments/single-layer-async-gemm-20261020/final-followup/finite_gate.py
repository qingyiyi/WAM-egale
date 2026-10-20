"""Final public-output, saved DAG, and effective-state gate for both routes."""
import json, os, time, traceback
from pathlib import Path
import torch
from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture, decode, visible_snapshot, tensors
from tools.replay import device_tree
from tools.metrics import compare, byte_equal, state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock

B = Path(__file__).resolve().parent
os.sched_setaffinity(0, {4, 5})
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
t0 = time.time()
r = dict(status='RUNNING', rows=[], full_block_calls=0,
         production_admitted=False, formal_performance_permission=False)

def save():
    (B/'FINITE-GATE-RESULT.json').write_text(json.dumps(r, indent=2)+'\n')

try:
    block = load_block('cuda')
    native = NativeBlock(block)
    routes = {name: WholeLayerBlock(block, tile_pipeline=flag)
              for name, flag in [('static', False), ('ready', True)]}
    work = Path(load_block.__code__.co_filename).resolve().parents[2]
    video = work/'.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
    with torch.inference_mode():
        for kind, batch in [('action', 2), ('action', 14), ('video', 2)]:
            for length in ['short', 'middle', 'long']:
                p = (FIXTURES if kind=='action' else video)/f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                d = decode(torch.load(p, map_location='cpu', mmap=True, weights_only=False))
                f = BlockFixture(p, device_tree(d))
                args, kw = f.arguments()
                inputs = {n: v.clone() for n, v in tensors((args, kw))}
                cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()

                def call(name, update=None):
                    assert r['full_block_calls'] < 108 and time.time()-t0 < 550
                    r['full_block_calls'] += 1
                    k = dict(kw) if update is None else dict(kw, update_cache=update)
                    if name=='Torch':
                        return block(*args, **k)
                    if name=='native':
                        return native(*args, **k, debug=True)
                    return routes[name](*args, **k, debug=True, audit=True)

                f.install_cache(block)
                ref = call('Torch')
                torch.cuda.synchronize()
                cache_ref = visible_snapshot(block.attn1.attn_caches[kw['cache_name']])
                f.install_cache(block)
                n = call('native')
                snapshots, outputs = [], []
                if batch==2:
                    f.install_cache(block)
                    for update in [1, 2, 0]:
                        value = call('Torch', update)
                        torch.cuda.synchronize()
                        snapshots.append(visible_snapshot(block.attn1.attn_caches[kw['cache_name']]))
                        outputs.append(value.clone())

                for route in routes:
                    f.install_cache(block)
                    x = call(route)
                    torch.cuda.synchronize()
                    row = dict(case=f'{length}-B{batch}-{kind}', route=route, fixture=str(p),
                        native_vs_Torch=compare(ref, n['output']),
                        new_vs_Torch=compare(ref, x['output']),
                        cache_vs_Torch=state_equal(cache_ref, visible_snapshot(block.attn1.attn_caches[kw['cache_name']])),
                        stages={}, resources=x['resources'].tolist(), legal_state_sequences=[])
                    for key in n:
                        if key in x and key not in ['output', 'tile_stats', 'cache_info', 'resources']:
                            row['stages'][key] = compare(n[key].reshape(-1), x[key].reshape(-1))
                    stats = x['tile_stats'].cpu().tolist()
                    row['exact_once'] = all(v[0]==v[1] and v[2]==0 for v in stats)
                    if route=='ready':
                        row['ffn_schedule'] = x['ffn_schedule'].cpu().tolist()
                    f.install_cache(block)
                    repeat = call(route)
                    torch.cuda.synchronize()
                    row['repeat_exact'] = byte_equal(x['output'], repeat['output'])
                    if batch==2:
                        f.install_cache(block)
                        for update, state, output in zip([1, 2, 0], snapshots, outputs):
                            value = call(route, update)
                            torch.cuda.synchronize()
                            row['legal_state_sequences'].append(dict(update=update,
                                output=compare(output, value['output']),
                                cache=state_equal(state, visible_snapshot(block.attn1.attn_caches[kw['cache_name']]))))
                    row['inputs_unchanged'] = all(byte_equal(inputs[n], t) for n, t in tensors((args, kw)))
                    row['RNG_unchanged'] = byte_equal(cpu, torch.random.get_rng_state()) and byte_equal(gpu, torch.cuda.get_rng_state())
                    r['rows'].append(row)
                    save()
                    del x, repeat
                del ref, n, cache_ref, snapshots, outputs, inputs, d, f, args, kw
    r['output_pass_count'] = sum(row['new_vs_Torch']['bitwise_pass'] for row in r['rows'])
    r['status'] = 'COMPLETE_TWO_ROUTE_FINITE_GATE_NOT_PRODUCTION_ADMISSION'
except BaseException as error:
    r.update(status='FAILED_OR_PARTIAL', error=repr(error), traceback=traceback.format_exc())
finally:
    r.update(end=time.time(), lifecycle_seconds=time.time()-t0)
    save()

"""Same-boundary phase timestamps plus paired instrumentation controls, B14 only."""
from pathlib import Path
import importlib.util
import json
import os
import time
import traceback
import subprocess
import torch
from tools.block_loader import load_block, FIXTURES
from tools.fixtures import BlockFixture, decode, visible_snapshot
from tools.replay import device_tree
from tools.metrics import state_equal
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock, extension

B = Path(__file__).resolve().parent
os.sched_setaffinity(0, {4, 5})
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
started = time.time()
result = dict(status='RUNNING', full_block_calls=0, rows=[],
              boundary='internal GEMM phase timestamps; separate paired instrumentation control',
              formal_performance_permission=False, process_count=1)


def save():
    (B / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    modules = {'before_plain': load('old.wam_whole_fa2_cuda', next((B.parent / 'cross/before').glob('*.so'))),
               'after_plain': extension()}
    for label in ('before', 'after'):
        name = 'wam_gemm_' + label + '_profile_cuda'
        modules[label + '_marked'] = load(name, next(B.glob(name + '*.so')))
    block = load_block('cuda')
    binding, native = WholeLayerBlock(block), NativeBlock(block)
    with torch.inference_mode():
        for length in ('short', 'middle', 'long'):
            path = FIXTURES / f'input-{length}-B14-action-layer0-first.pt'
            fixture = BlockFixture(path, device_tree(decode(torch.load(path, map_location='cpu', mmap=True, weights_only=False))))
            args, kw = fixture.arguments()
            fixture.install_cache(block)
            ref = native(*args, **kw, debug=True)
            result['full_block_calls'] += 1
            cache = block.attn1.attn_caches[kw['cache_name']]
            initial = visible_snapshot(cache)
            inputs = [x.clone() for x in args]
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            common = (binding.weights, *args, [cache[n] for n in ('k','v','id','mask','is_pred')],
                      kw['candidate_cache_rows'], kw.get('update_cache', 0), binding.eps)

            def call(name):
                assert result['full_block_calls'] < 1000
                result['full_block_calls'] += 1
                return modules[name].forward(*common)

            for _ in range(60):
                for name in modules:
                    temp = call(name)
            torch.cuda.synchronize()
            row = dict(case=length, samples=[], finite_checks={}, resources={}, stage_samples=[])
            row['GPU_state_before'] = subprocess.check_output(['nvidia-smi', '-i', '0',
                '--query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw',
                '--format=csv,noheader'], text=True)
            order = list(modules)
            for repeat in range(8):
                for name in (order if repeat % 2 == 0 else order[::-1]):
                    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    value = call(name)
                    z.record()
                    z.synchronize()
                    row['samples'].append(dict(repeat=repeat, arm=name, ms=a.elapsed_time(z)))
                    checks = {k: torch.equal(ref[k].reshape(-1), value[k].reshape(-1)) for k in ref.keys() & value.keys()}
                    assert all(checks.values()), (length, name, checks)
                    row['finite_checks'][name] = checks
                    row['resources'][name] = value['resources'].tolist()
                    if name.endswith('marked'):
                        ticks = value['profile_ticks'].cpu().tolist()
                        for sid, stage in ((3,'self_Q'),(16,'cross_K'),(17,'cross_V'),(25,'FFN1'),(27,'FFN2')):
                            row['stage_samples'].append(dict(arm=name, repeat=repeat, stage=stage,
                                us=(ticks[sid][1] - ticks[sid][0]) / 1000))
            row['GPU_state_after'] = subprocess.check_output(['nvidia-smi', '-i', '0',
                '--query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw',
                '--format=csv,noheader'], text=True)
            row['valid_KV'] = state_equal(initial, visible_snapshot(cache))
            row['inputs_unchanged'] = all(torch.equal(x,y) for x,y in zip(inputs,args))
            row['RNG_unchanged'] = torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            assert row['valid_KV']['pass_all'] and row['inputs_unchanged'] and row['RNG_unchanged']
            result['rows'].append(row)
            save()
    result['status'] = 'COMPLETE_PHASE_DIAGNOSTIC_WITH_INSTRUMENTATION_CONTROL'
except BaseException as error:
    result.update(status='FAILED_OR_PARTIAL', error=repr(error), traceback=traceback.format_exc())
finally:
    result['lifecycle_seconds'] = time.time() - started
    save()
if result['status'] != 'COMPLETE_PHASE_DIAGNOSTIC_WITH_INSTRUMENTATION_CONTROL':
    raise SystemExit(1)

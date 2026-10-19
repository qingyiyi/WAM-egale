"""完整真实block/KV状态重放：原Torch、冻结基线、独立顺序和tile子图。"""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from .block_loader import FIXTURES, load_block
from .fixtures import BlockFixture, decode, tensors, visible_snapshot
from .metrics import byte_equal, compare, state_equal
from .replay import device_tree, write
from .resources import resource_for_gpu
from wam_ops.install import install_block

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / '.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
VARIANTS = (('baseline', {}), ('sequential', {'execution_mode': 'sequential'}),
            ('tiled', {'execution_mode': 'tiled'}),
            ('tiled_ffn', {'execution_mode': 'tiled', 'fused_ffn1': True}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--standalone', action='store_true', help='Only optimized; assert baseline was never imported/loaded')
    p.add_argument('--include-persistent', action='store_true')
    p.add_argument('--audit-dispatch', action='store_true', help='Record kernel names, never performance, for two real action cases')
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    resource = lambda: resource_for_gpu(reg['physical_gpu_uuid'])
    if not reg.get('shared_correctness_authorized'):
        raise PermissionError('shared correctness requires registration')
    args.output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {4, 5})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    result = dict(status='PREPARING', rows=[], resources=[], block_forward_intents=0,
                  mode='isolated_only' if args.standalone else 'original_baseline_seq_tiled_ffn',
                  formal_performance_permission=False, timing='CORRECTNESS_ONLY_SHARED_GPU')

    def save():
        write(args.output / 'RESULT.json', result)

    def guard():
        if time.monotonic() - start > 480:
            raise TimeoutError('worker lifecycle limit')
        if time.time() > datetime.fromisoformat(reg['absolute_work_stop']).timestamp() - 60:
            raise TimeoutError('absolute work stop reserve')
        if result['block_forward_intents'] >= reg['full_block_forward_cap']:
            raise RuntimeError('block forward budget')
        obs = resource()
        result['resources'].append(obs)
        fields = [x.strip() for x in obs['gpu'].split(',')]
        minimum = 4000 if not torch.cuda.is_initialized() else 800
        if fields[0] != reg['physical_gpu_uuid'] or float(fields[2]) < minimum:
            raise RuntimeError('GPU identity/free memory guard')

    def call(fn):
        result['block_forward_intents'] += 1
        return fn()

    try:
        block = load_block(device='cpu')
        guard()
        block = block.to('cuda')
        # Match the canonical inference parameter flags as well as the values.
        block.requires_grad_(True)
        original_class = type(block)
        addresses = {k: (v.data_ptr(), v.requires_grad) for k, v in block.named_parameters()}
        rejected = []
        invalid_options = [('optimized', {'execution_mode': 'invalid'})]
        if not args.standalone:
            invalid_options.append(('baseline', {'execution_mode': 'tiled'}))
        for implementation, options in invalid_options:
            try:
                with install_block(block, implementation, **options):
                    raise AssertionError('invalid configuration was accepted')
            except ValueError:
                rejected.append(implementation)
            assert type(block) is original_class and '_native_block' not in block.__dict__
        result['invalid_options_rejected_and_restored'] = rejected
        variants = VARIANTS[1:] if args.standalone else VARIANTS
        if args.include_persistent:
            variants = variants + (('persistent', {'execution_mode': 'persistent'}),
                ('persistent_ffn', {'execution_mode': 'persistent', 'fused_ffn1': True}))
        result['status'] = 'RUNNING'
        with torch.inference_mode():
            for kind, batch in [('action', 2), ('action', 14), ('video', 2)]:
                for length in ('short', 'middle', 'long'):
                    guard()
                    root = FIXTURES if kind == 'action' else VIDEO
                    path = root / f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                    raw = decode(torch.load(path, map_location='cpu', mmap=True, weights_only=False))
                    if tuple(raw['identity']) != (length, batch, kind, 'first', 0):
                        raise ValueError('saved fixture identity mismatch')
                    case = f'{length}-B{batch}-{kind}-layer0-first'
                    fixture = BlockFixture(path, device_tree(raw))
                    arguments, kwargs = fixture.arguments()
                    inputs = {k: v.clone() for k, v in tensors((arguments, kwargs))}
                    cpu_rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
                    fixture.install_cache(block)
                    reference = call(lambda: block(*arguments, **kwargs))
                    reference_state = visible_snapshot(block.attn1.attn_caches[kwargs['cache_name']])
                    row = dict(case=case, input_path=str(path), identity=list(raw['identity']),
                               original_saved_gold=compare(fixture.data['block_gold'], reference), variants={})
                    base_debug = None
                    for name, options in variants:
                        fixture.install_cache(block)
                        implementation = 'baseline' if name == 'baseline' else 'optimized'
                        with install_block(block, implementation, **options):
                            native = block.__dict__['_native_block']
                            debug = call(lambda: native(*arguments, **kwargs, debug=True))
                            public = call(lambda: block(*arguments, **kwargs))
                            checks = dict(output=compare(reference, debug['output']),
                                          public_forward=compare(reference, public),
                                          state=state_equal(reference_state, visible_snapshot(
                                              block.attn1.attn_caches[kwargs['cache_name']])),
                                          native_module=type(native).__module__, intermediates={})
                            if name == 'baseline':
                                base_debug = {k: v.clone() for k, v in debug.items()}
                            elif base_debug is not None:
                                checks['intermediates'] = {k: compare(base_debug[k], v.reshape_as(base_debug[k]))
                                    for k, v in debug.items() if k in base_debug and k != 'execution_flags'}
                            if name != 'baseline':
                                checks['execution_flags'] = debug['execution_flags'].tolist()
                                checks['flags_match'] = checks['execution_flags'] == [
                                    {'sequential': 0, 'tiled': 1, 'persistent': 2}[options['execution_mode']], int(options.get('fused_ffn1', False))]
                            checks['passed'] = (checks['output']['bitwise_pass'] and
                                checks['public_forward']['bitwise_pass'] and checks['state']['pass_all'] and
                                checks.get('flags_match', True) and
                                all(v['bitwise_pass'] for v in checks['intermediates'].values()))
                            row['variants'][name] = checks
                            if base_debug is None and name == 'sequential':
                                base_debug = {k: v.clone() for k, v in debug.items()}
                        assert type(block) is original_class and '_native_block' not in block.__dict__
                        del debug, public
                    row['immutable'] = (all(byte_equal(inputs[k], v) for k, v in tensors((arguments, kwargs)))
                        and byte_equal(cpu_rng, torch.random.get_rng_state())
                        and byte_equal(gpu_rng, torch.cuda.get_rng_state())
                        and addresses == {k: (v.data_ptr(), v.requires_grad) for k, v in block.named_parameters()})
                    row['passed'] = row['original_saved_gold']['bitwise_pass'] and row['immutable'] and all(
                        v['passed'] for v in row['variants'].values())
                    result['rows'].append(row)
                    save()
                    if args.audit_dispatch and kind == 'action' and length == 'short':
                        fixture.install_cache(block)
                        with install_block(block, 'optimized', execution_mode='tiled', fused_ffn1=True):
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                  torch.profiler.ProfilerActivity.CUDA]) as prof:
                                audited = call(lambda: block(*arguments, **kwargs))
                                torch.cuda.synchronize()
                        names = sorted({event.name for event in prof.events()
                                        if event.device_type == torch.autograd.DeviceType.CUDA})
                        audit = dict(case=case, GPU_kernel_names=names,
                                     output=compare(reference, audited),
                                     attention_tile_observed=any('standard_tiles' in n or 'split_tiles' in n for n in names),
                                     # Kineto can return either demangled C++ or its exact Itanium ABI symbol.
                                     FFN_tile_observed=any('wam_optimized_ffn::linear_tile' in n or
                                         n.startswith('_ZN17wam_optimized_ffn11linear_tileI') for n in names),
                                     timings_used=False)
                        audit['passed'] = (audit['attention_tile_observed'] and audit['FFN_tile_observed']
                                           and audit['output']['bitwise_pass'])
                        result.setdefault('dispatch_rows', []).append(audit)
                        del audited, prof
                        save()
                    # Cache evolution is a legal counterfactual API replay, not a new robot trajectory.
                    if batch == 2:
                        fixture.install_cache(block)
                        gold = []
                        for update in (1, 2, 0):
                            kw = dict(kwargs, update_cache=update)
                            value = call(lambda: block(*arguments, **kw))
                            gold.append((value.clone(), visible_snapshot(block.attn1.attn_caches[kw['cache_name']])))
                        for name, options in variants:
                            fixture.install_cache(block)
                            with install_block(block, 'baseline' if name == 'baseline' else 'optimized', **options):
                                for index, update in enumerate((1, 2, 0)):
                                    kw = dict(kwargs, update_cache=update)
                                    value = call(lambda: block(*arguments, **kw))
                                    check = dict(case=case, variant=name, update=update, sequence_index=index,
                                                 output=compare(gold[index][0], value),
                                                 state=state_equal(gold[index][1], visible_snapshot(
                                                     block.attn1.attn_caches[kw['cache_name']])))
                                    check['passed'] = check['output']['bitwise_pass'] and check['state']['pass_all']
                                    result.setdefault('state_rows', []).append(check)
                            del value
                        del gold
                        save()
                    post_state_immutable = (
                        all(byte_equal(inputs[k], v) for k, v in tensors((arguments, kwargs)))
                        and byte_equal(cpu_rng, torch.random.get_rng_state())
                        and byte_equal(gpu_rng, torch.cuda.get_rng_state())
                        and addresses == {k: (v.data_ptr(), v.requires_grad) for k, v in block.named_parameters()}
                        and type(block) is original_class and '_native_block' not in block.__dict__)
                    row['post_state_inputs_RNG_parameters_and_restore'] = post_state_immutable
                    row['passed'] = row['passed'] and post_state_immutable
                    save()
                    del fixture, raw, arguments, inputs, reference, reference_state, base_debug
                    block.attn1.attn_caches = {}
                    gc.collect()
                    torch.cuda.empty_cache()
            if args.standalone:
                maps = Path('/proc/self/maps').read_text()
                result['baseline_module_imported'] = any(k.startswith('wam_ops.baseline') for k in sys.modules)
                result['old_DSO_loaded'] = any(n in maps for n in ('wam_block_cuda.', 'wam_attention_cuda.', 'wam_ffn_cuda.'))
                if result['baseline_module_imported'] or result['old_DSO_loaded']:
                    raise RuntimeError('optimized isolation violated')
            result['status'] = 'FINITE_BLOCK_AND_STATE_PARITY_PASS' if all(
                x['passed'] for x in result['rows'] + result.get('state_rows', [])
                + result.get('dispatch_rows', [])) else 'NUMERIC_OR_STATE_REJECTED'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(lifecycle_seconds=time.monotonic()-start,
                      finished_at=datetime.now().astimezone().isoformat())
        save()
    if result['status'] != 'FINITE_BLOCK_AND_STATE_PARITY_PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

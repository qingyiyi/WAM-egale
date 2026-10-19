"""六个真实完整block的强基线门检、探索性计时与独立profiler。

仅测静态已admit的fixture。冷填充/Graph构建与replay分列；不表示policy性能。
"""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from .block_loader import FIXTURES, load_block
from .fixtures import BlockFixture, tensors, visible_snapshot
from .metrics import byte_equal, compare, state_equal
from .replay import device_tree, resource, write
from wam_ops.optimized.prepared import PreparedBlock, guarded_group


@contextmanager
def annotations(block):
    scopes, handles = {}, []
    for name, module in block.named_modules():
        if not name:
            continue
        def before(module, inputs, label=name):
            ctx = torch.profiler.record_function('block/' + label)
            scopes.setdefault(id(module), []).append(ctx)
            ctx.__enter__()
        def after(module, inputs, output):
            scopes[id(module)].pop().__exit__(None, None, None)
        handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def profile(block, operation, destination):
    torch.cuda.synchronize()
    with annotations(block), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True) as prof:
        with torch.profiler.record_function('complete_block'):
            operation()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(destination))
    scopes, operators, gemms, kernels = [], [], [], {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and not (
                event.name.startswith('block/') or event.name == 'complete_block'):
            row = kernels.setdefault(event.name, dict(name=event.name, calls=0, duration_us=0))
            row['calls'] += 1
            row['duration_us'] += event.device_time_total
        if event.name == 'aten::addmm':
            shapes = event.input_shapes
            gemms.append(dict(name=event.name, shapes=shapes,
                M=shapes[1][0], K=shapes[1][1], N=shapes[2][1]))
    for event in prof.key_averages(group_by_input_shape=True):
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        row = dict(name=event.key, calls=event.count, shapes=event.input_shapes,
            device_type=str(event.device_type),
            self_cpu_us=event.self_cpu_time_total, cpu_us=event.cpu_time_total,
            self_device_us=event.self_device_time_total, device_us=event.device_time_total,
            self_device_memory_bytes=event.self_device_memory_usage)
        (scopes if event.key.startswith('block/') or event.key == 'complete_block' else operators).append(row)
    return dict(scopes=scopes, gemms=gemms,
        operators=sorted(operators, key=lambda x: -x['self_device_us']),
        kernels=sorted(kernels.values(), key=lambda x: -x['duration_us']),
        trace=str(destination), profiler_timings_excluded_from_benchmark=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    before = resource()
    if before['pids'] or before['gpu'].split(',')[0].strip() != reg['physical_gpu_uuid']:
        raise RuntimeError('GPU_BUSY_NO_CONTEXT_CREATED')
    args.output.mkdir(exist_ok=False, parents=True)
    os.sched_setaffinity(0, {4, 5})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    deadline = min(start + reg['profile_GPU_lifecycle_seconds_cap'] - 60,
        start + datetime.fromisoformat(reg['absolute_work_stop']).timestamp() - time.time())
    result = dict(status='RUNNING', started_at=datetime.now().astimezone().isoformat(),
        rows=[], block_forwards=0, resources=[before],
        scope='six saved logical complete block0 inputs; static admission/replay',
        formal_performance_permission=False, unexplained_idle_NVML_utilization=before['gpu'])

    def save():
        write(args.output / 'RESULT.json', result)

    def check():
        if time.monotonic() >= deadline or result['block_forwards'] >= reg['full_block_forward_cap']:
            raise TimeoutError('profile lifecycle or forward cap')
        sample = resource()
        result['resources'].append(sample)
        if set(sample['pids'].splitlines()) - {str(os.getpid())}:
            raise RuntimeError('FOREIGN_GPU_PID_STOP_MEASUREMENT')
        size = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
        if size > reg['new_storage_cap_bytes'] * .8:
            raise RuntimeError('profile storage reserve')

    def counted(operation):
        if time.monotonic() >= deadline or result['block_forwards'] >= reg['full_block_forward_cap']:
            raise TimeoutError('profile lifecycle or forward cap')
        result['block_forwards'] += 1
        return operation()

    try:
        torch.cuda.set_device(0)
        block = load_block()
        result['block_parameters'] = sum(p.numel() for p in block.parameters())
        with torch.inference_mode():
            for length in ('short', 'middle', 'long'):
                for batch in (2, 14):
                    check()
                    fixture = BlockFixture.load(FIXTURES, length, batch)
                    fixture = BlockFixture(fixture.path, device_tree(fixture.data))
                    arguments, kwargs = fixture.arguments()
                    cache = fixture.install_cache(block)
                    initial = visible_snapshot(cache)
                    immutable = {name: tensor.clone() for name, tensor in tensors((arguments, kwargs))}
                    cpu_rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()

                    def restore():
                        for name in ('id', 'mask', 'is_pred'):
                            cache[name].copy_(initial[name])
                        for name in ('k', 'v'):
                            cache[name].index_copy_(1, initial['slots'], initial[name])

                    operation = lambda: block(*arguments, **kwargs)
                    oracle = counted(operation).clone()
                    expected_state = visible_snapshot(cache)
                    historical = compare(fixture.data['block_gold'], oracle)
                    if not historical['bitwise_pass']:
                        raise RuntimeError('lazy single-block loader differs from saved original gold')
                    restore()
                    prepared = PreparedBlock(block, arguments, kwargs, cache)
                    torch.cuda.synchronize()
                    fill_begin = time.perf_counter()
                    fill_bytes = prepared.fill()
                    torch.cuda.synchronize()
                    row = dict(case=fixture.case_id, total_KV=initial['slots'].numel() + 16,
                        query_tokens_per_candidate=16, candidate_count=batch // 2,
                        guarded_group=guarded_group(initial['slots'].numel()),
                        historical_gold=historical, cold_fill_wall_ms=(time.perf_counter()-fill_begin)*1000,
                        cold_fill_bytes=fill_bytes, arms=[], admission_and_graph_build_excluded_from_replay=True)
                    result['rows'].append(row)
                    save()
                    engines = ('dense',) if batch == 2 else ('dense', 'native', 'guarded')

                    def validate(output):
                        numeric = compare(oracle, output)
                        state = state_equal(expected_state, visible_snapshot(cache))
                        inputs = all(byte_equal(immutable[name], tensor) for name, tensor in tensors((arguments, kwargs)))
                        rng = byte_equal(cpu_rng, torch.random.get_rng_state()) and byte_equal(gpu_rng, torch.cuda.get_rng_state())
                        return dict(output=numeric, cache_state=state, inputs_unchanged=inputs,
                            RNG_unchanged=rng, passed=numeric['bitwise_pass'] and state['pass_all'] and inputs and rng)

                    candidates = [('original', False)] + [(e, c) for e in engines for c in (False, True)]
                    for engine, cached in candidates:
                        check()
                        restore()
                        context = nullcontext() if engine == 'original' else prepared.activate(engine, cached)
                        with context:
                            eager = counted(operation)
                            admission = validate(eager)
                            arm = dict(engine=engine, cached=cached, eager_admission=admission)
                            row['arms'].append(arm)
                            save()
                            if not admission['passed']:
                                arm['status'] = 'REJECTED_NUMERIC_OR_STATE'
                                continue
                            graph = None
                            if engine != 'original':
                                stream = torch.cuda.Stream()
                                stream.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(stream):
                                    counted(operation)
                                    counted(operation)
                                torch.cuda.current_stream().wait_stream(stream)
                                torch.cuda.synchronize()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    graph_output = counted(operation)
                                graph.replay()
                                result['block_forwards'] += 1
                                torch.cuda.synchronize()
                                arm['graph_admission'] = validate(graph_output)
                                if not arm['graph_admission']['passed']:
                                    arm['status'] = 'REJECTED_GRAPH_NUMERIC_OR_STATE'
                                    continue
                            samples = []
                            for repeat in range(7):
                                for execution in (('eager', 'graph') if graph is not None else ('eager',)):
                                    torch.cuda.synchronize()
                                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    wall = time.perf_counter()
                                    begin.record()
                                    if execution == 'graph':
                                        counted(graph.replay)
                                        output = graph_output
                                    else:
                                        output = counted(operation)
                                    end.record()
                                    end.synchronize()
                                    samples.append(dict(execution=execution, repeat=repeat,
                                        cuda_ms=begin.elapsed_time(end), wall_ms=1000*(time.perf_counter()-wall)))
                                    if not byte_equal(output, oracle):
                                        raise RuntimeError('repeated output changed')
                            arm.update(status='ADMITTED', samples=samples,
                                median_ms={x: statistics.median(s['cuda_ms'] for s in samples if s['execution']==x)
                                    for x in {s['execution'] for s in samples}})
                            arm['post_state'] = validate(output)
                            if not arm['post_state']['passed']:
                                raise RuntimeError('post-replay state changed')
                        del graph
                        gc.collect()
                        save()
                    admitted = [a for a in row['arms'] if a.get('status') == 'ADMITTED' and a['cached']]
                    if not admitted:
                        raise RuntimeError('no legal cached strong baseline')
                    strongest = min(admitted, key=lambda a: a['median_ms']['graph'])
                    row['strongest_static_baseline'] = dict(engine=strongest['engine'], cached=True,
                        execution='graph', cuda_ms=strongest['median_ms']['graph'])
                    for label, engine, cached in [('original', 'original', False),
                            ('strong_cached', strongest['engine'], True)]:
                        restore()
                        context = nullcontext() if engine == 'original' else prepared.activate(engine, cached)
                        with context:
                            captured_ffn = {}
                            def save_ffn(module, inp):
                                captured_ffn['input'] = inp[0].detach().clone()
                            handle = block.ffn.register_forward_pre_hook(save_ffn)
                            counted(operation)
                            handle.remove()
                            torch.save(dict(input=captured_ffn['input'].cpu(), source=str(fixture.path),
                                checkpoint_block=0, cached=bool(cached)), args.output / (fixture.case_id + '-ffn.pt'))
                            row[label + '_profile'] = profile(block, lambda: counted(operation),
                                args.output / (fixture.case_id + '-' + label + '.trace.json'))
                    print(json.dumps(dict(case=row['case'], strongest=row['strongest_static_baseline'],
                        block_forwards=result['block_forwards']), ensure_ascii=False), flush=True)
                    block.attn1.attn_caches = {}
                    del prepared, fixture, cache, oracle, expected_state, immutable
                    gc.collect()
                    torch.cuda.empty_cache()
                    save()
        result['status'] = 'COMPLETE_SIX_REAL_BLOCK_STRONG_BASELINES_AND_PROFILES'
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except BaseException as error:
        result.update(status='FAILED_OR_PARTIAL', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_at=datetime.now().astimezone().isoformat(),
            lifecycle_seconds=time.monotonic()-start)
        save()


if __name__ == '__main__':
    main()

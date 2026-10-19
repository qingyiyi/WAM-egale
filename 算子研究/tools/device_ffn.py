"""在六个已保存真实FFN输入上验证独立device原语，不改变生产模型。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from torch.nn import functional as F

from .block_loader import load_block
from .metrics import byte_equal, compare
from .replay import resource, write
from wam_ops.optimized.prepared import extension


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--metadata-only', action='store_true', help='一个真实输入核对SMEM opt-in后的资源属性，不做timing')
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    observation = resource()
    if observation['pids'] or observation['gpu'].split(',')[0].strip() != reg['physical_gpu_uuid']:
        raise RuntimeError('GPU_BUSY_NO_CONTEXT_CREATED')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {4, 5})
    start = time.monotonic()
    deadline = min(start + reg['device_GPU_lifecycle_seconds_cap'] - 60,
        start + datetime.fromisoformat(reg['absolute_work_stop']).timestamp() - time.time())
    result = dict(status='RUNNING', rows=[], primitive_calls=0, original_ffn_calls=0,
        started_at=datetime.now().astimezone().isoformat(), resources=[observation],
        production_admitted=False, formal_performance_permission=False,
        strict_gate='all finite bytes equal for each original BF16 intermediate and output',
        scope='real block0 FFN subgraph only; no persistent scheduler, no closed-loop quality')

    def save():
        write(args.output / 'RESULT.json', result)

    def check():
        if time.monotonic() >= deadline or result['primitive_calls'] >= reg['device_primitive_calls_cap']:
            raise TimeoutError('device lifecycle/call cap')
        sample = resource()
        result['resources'].append(sample)
        if set(sample['pids'].splitlines()) - {str(os.getpid())}:
            raise RuntimeError('FOREIGN_GPU_PID_STOP_MEASUREMENT')

    try:
        torch.cuda.set_device(0)
        block = load_block()
        native = extension('wam_ffn_cuda')
        result['kernel_metadata'] = native.metadata()
        result['torch_flags'] = dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        parameters = [p for p in block.ffn.parameters()]
        versions = [p._version for p in parameters]
        parameter_copies = [p.detach().clone() for p in parameters]
        w1, b1 = block.ffn.net[0].proj.weight, block.ffn.net[0].proj.bias
        w2, b2 = block.ffn.net[2].weight, block.ffn.net[2].bias
        with torch.inference_mode():
            if args.metadata_only:
                path=args.inputs/'short-B2-action-layer0-first-ffn.pt'
                x=torch.load(path,mmap=True,map_location='cpu',weights_only=False)['input'].to('cuda').flatten(0,1)
                gold1=F.linear(x,w1,b1)
                golda=F.gelu(gold1,approximate='tanh')
                projected,activated=torch.empty_like(gold1),torch.empty_like(golda)
                native.linear_out(x,w1,b1,projected)
                native.ffn1_gelu_out(x,w1,b1,activated)
                torch.cuda.synchronize()
                result.update(status='COMPLETE_REAL_INPUT_POST_OPTIN_METADATA_CHECK',primitive_calls=2,
                    no_timing=True,linear1=compare(gold1,projected),fused_ffn1=compare(golda,activated),
                    post_optin_metadata=native.metadata(),
                    note='initial active-CTA query0 preceded dynamic-SMEM opt-in and is invalid as physical occupancy; post-opt-in query is authoritative only for this kernel configuration')
                if not result['linear1']['bitwise_pass'] or not result['fused_ffn1']['bitwise_pass']:
                    raise RuntimeError('metadata real-input gate failed')
                return
            for length in ('short', 'middle', 'long'):
                for batch in (2, 14):
                    check()
                    path = args.inputs / f'{length}-B{batch}-action-layer0-first-ffn.pt'
                    saved = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
                    shaped = saved['input'].to('cuda')
                    x = shaped.flatten(0, 1)
                    immutable = x.clone()
                    cpu_rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
                    gold1 = F.linear(x, w1, b1)
                    golda = F.gelu(gold1, approximate='tanh')
                    gold2 = F.linear(golda, w2, b2)
                    original = block.ffn(shaped).flatten(0, 1)
                    result['original_ffn_calls'] += 1
                    if not byte_equal(gold2, original):
                        raise RuntimeError('explicit original FFN does not match module')
                    projected, activated, output = torch.empty_like(gold1), torch.empty_like(golda), torch.empty_like(gold2)
                    native.linear_out(x, w1, b1, projected)
                    native.gelu_out(gold1, activated)
                    native.linear_out(golda, w2, b2, output)
                    result['primitive_calls'] += 3
                    independent = dict(linear1=compare(gold1, projected),
                        gelu=compare(golda, activated), linear2=compare(gold2, output))
                    native.ffn_out(x, w1, b1, w2, b2, projected, activated, output)
                    result['primitive_calls'] += 3
                    chain = dict(linear1=compare(gold1, projected), gelu=compare(golda, activated),
                        output=compare(gold2, output))
                    row = dict(case=f'{length}-B{batch}', source=str(path), M=x.shape[0],
                        original_module_matches_explicit=True, independent=independent, chain=chain,
                        admitted=all(v['bitwise_pass'] for v in (*independent.values(), *chain.values())))
                    result['rows'].append(row)
                    # Even a rejected primitive gets bounded exploratory resource/timing data.
                    # These numbers cannot be called a qualified optimization/speedup.
                    operations = dict(torch=lambda: block.ffn(shaped),
                        device=lambda: native.ffn_out(x, w1, b1, w2, b2, projected, activated, output))
                    graphs, outputs = {}, {}
                    for name, operation in operations.items():
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            operation()
                            operation()
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            outputs[name] = operation()
                        graphs[name] = graph
                        if name == 'device':
                            result['primitive_calls'] += 9
                        else:
                            result['original_ffn_calls'] += 3
                    raw = []
                    for repeat in range(15):
                        for name in (('torch', 'device') if repeat % 2 == 0 else ('device', 'torch')):
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()
                            wall = time.perf_counter()
                            begin.record()
                            graphs[name].replay()
                            end.record()
                            end.synchronize()
                            raw.append(dict(repeat=repeat, engine=name, cuda_ms=begin.elapsed_time(end),
                                wall_ms=1000*(time.perf_counter()-wall)))
                            if name == 'device':
                                result['primitive_calls'] += 3
                            else:
                                result['original_ffn_calls'] += 1
                    row['exploratory_graph_timing'] = dict(samples=raw,
                        median_ms={name: statistics.median(x['cuda_ms'] for x in raw if x['engine']==name)
                            for name in graphs}, eligible_as_qualified_speedup=row['admitted'])
                    row['post_output'] = compare(gold2, output)
                    row['inputs_unchanged'] = byte_equal(x, immutable)
                    row['parameter_versions_unchanged'] = all(p._version == v for p,v in zip(parameters,versions))
                    row['parameter_bytes_unchanged'] = all(byte_equal(p, copy) for p,copy in zip(parameters,parameter_copies))
                    row['RNG_unchanged'] = byte_equal(cpu_rng, torch.random.get_rng_state()) and byte_equal(gpu_rng, torch.cuda.get_rng_state())
                    if not all(row[key] for key in ('inputs_unchanged', 'RNG_unchanged',
                            'parameter_versions_unchanged', 'parameter_bytes_unchanged')):
                        raise RuntimeError('input/parameter/RNG state changed')
                    save()
                    print(json.dumps(dict(case=row['case'], admitted=row['admitted'],
                        different={name:v['different_elements'] for name,v in independent.items()},
                        median=row['exploratory_graph_timing']['median_ms'])), flush=True)
                    del graph, graphs, outputs, projected, activated, output, shaped, x, immutable
        all_pass = all(row['admitted'] and row['post_output']['bitwise_pass'] for row in result['rows'])
        result.update(status='COMPLETE_DEVICE_PRIMITIVE_MATRIX',
            strict_numerical_gate_passed=all_pass,
            next='LOCAL_FULL_BLOCK_INTEGRATION_ALLOWED_FOR_NUMERICAL_TEST_ONLY' if all_pass else 'STOP_NO_INTEGRATION_NUMERICAL_GATE_FAILED',
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='FAILED_OR_PARTIAL', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_at=datetime.now().astimezone().isoformat(), lifecycle_seconds=time.monotonic()-start)
        save()


if __name__ == '__main__':
    main()

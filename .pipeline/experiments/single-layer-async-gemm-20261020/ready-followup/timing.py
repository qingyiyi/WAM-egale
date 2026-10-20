"""Current DSO and native in fresh processes; no archived same-name extension."""
import json, os, sys, time, traceback, subprocess
from pathlib import Path
import torch
from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture, decode
from tools.replay import device_tree
from tools.metrics import compare
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock

B = Path(__file__).resolve().parent
repeat_process = int(sys.argv[1])
out = B / f'timing-{repeat_process}-RESULT.json'
os.sched_setaffinity(0, {4, 5})
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
t0 = time.time()
result = dict(status='RUNNING', process_repeat=repeat_process,
              full_block_calls=0, rows=[], samples=[], traces=[],
              production_admitted=False, formal_performance_permission=False,
              boundary='uncached complete single block, eager; preparation excluded')

def save():
    out.write_text(json.dumps(result, indent=2) + '\n')

try:
    block = load_block('cuda')
    native, static, new = NativeBlock(block), WholeLayerBlock(block,tile_pipeline=False), WholeLayerBlock(block,tile_pipeline=True)
    result['DSO_path'] = str(next(Path(sys.modules[WholeLayerBlock.__module__].__file__)
                                .parent.glob('wam_whole_fa2_cuda*.so')))
    work = Path(load_block.__code__.co_filename).resolve().parents[2]
    video = work / '.pipeline/experiments/block-native-baseline-20261019/native-video-capture-attempt1/results'
    with torch.inference_mode():
        for kind, batch in [('action', 2), ('action', 14), ('video', 2)]:
            for length in ['short', 'middle', 'long']:
                assert time.time() - t0 < 500
                p = (FIXTURES if kind == 'action' else video) / f'input-{length}-B{batch}-{kind}-layer0-first.pt'
                data = decode(torch.load(p, map_location='cpu', mmap=True, weights_only=False))
                fixture = BlockFixture(p, device_tree(data))
                args, kw = fixture.arguments()
                case = f'{length}-B{batch}-{kind}'
                load = subprocess.check_output([
                    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                    '--format=csv,noheader'], text=True)
                gpu_rows = [line for line in load.splitlines() if line.startswith('GPU-c3fa8d6f')]
                if len(gpu_rows) != 1:
                    raise RuntimeError('expected one compute context after empty-card precheck')
                # NVML can expose host PIDs outside the container PID namespace.
                # Preserve raw identity and report that mapping limitation.
                result['pid_namespace'] = [line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('NSpid:', 'Pid:'))]
                result['raw_NVML_processes'] = load
                result['GPU_prelaunch_empty'] = os.environ.get('GPU_PRELAUNCH_EMPTY') == '1'
                if not result['GPU_prelaunch_empty']:
                    raise RuntimeError('missing launcher empty-card precheck')

                def call(name):
                    result['full_block_calls'] += 1
                    if name == 'native':
                        return native(*args, **kw, debug=True)
                    return (static if name=='static' else new)(*args, **kw, debug=True, audit=False)

                warm = {}
                for name in ['native', 'static', 'new']:
                    fixture.install_cache(block)
                    warm[name] = call(name)
                torch.cuda.synchronize()
                comparison = compare(warm['native']['output'], warm['new']['output'])
                resources = warm['new']['resources'].tolist()
                static_comparison = compare(warm['static']['output'],warm['new']['output'])
                schedule = warm['new']['ffn_schedule'].cpu().tolist()
                if not comparison['bitwise_pass'] or not static_comparison['bitwise_pass'] or resources[2] != 121376:
                    raise RuntimeError('same-input output or current DSO identity failed')
                result['rows'].append(dict(case=case, fixture=str(p),
                    output=comparison, static_vs_pipeline=static_comparison, ffn_schedule=schedule, resources=resources, external_load=load))
                del warm
                for i in range(4):
                    order = ('native','static','new')
                    offset=(i+repeat_process)%3
                    names=order[offset:]+order[:offset]
                    if i%2:names=tuple(reversed(names))
                    for name in names:
                        fixture.install_cache(block)
                        torch.cuda.synchronize()
                        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        host = time.perf_counter()
                        value = call(name)
                        z.record()
                        z.synchronize()
                        result['samples'].append(dict(case=case, name=name, repeat=i,
                            ms=a.elapsed_time(z), host_ms=(time.perf_counter()-host)*1000))
                fixture.install_cache(block)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    value = call('new')
                    torch.cuda.synchronize()
                trace = B / f'timing-{repeat_process}-{case}.trace.json'
                prof.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())['traceEvents']
                kernels = [x['name'] for x in events if x.get('cat') == 'kernel']
                result['traces'].append(dict(case=case, path=str(trace), kernels=kernels))
                if len(kernels) != 1:
                    raise RuntimeError('single computation launch contract failed')
                save()
                del value, data, fixture, args, kw
    result['status'] = 'COMPLETE_SINGLE_LAYER_ENGINEERING_TIMING'
except BaseException as error:
    result.update(status='FAILED_OR_PARTIAL', error=repr(error), traceback=traceback.format_exc())
finally:
    result.update(end=time.time(), lifecycle_seconds=time.time()-t0)
    save()

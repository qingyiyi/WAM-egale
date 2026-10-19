"""从Chrome原trace按CPU launch归属汇总GPU kernel，不混入GPU注解时段。"""
import argparse
import hashlib
import json
from pathlib import Path

from .replay import write

STAGES = ('block/norm1','block/attn1','block/norm2','block/attn2','block/norm3','block/ffn')


def trace_summary(path):
    events = json.loads(path.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    scopes = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name') in STAGES]
    launches = {e.get('args',{}).get('correlation'):e for e in events if e.get('cat') == 'cuda_runtime'}
    stages, rows = {}, []
    for kernel in kernels:
        launch = launches.get(kernel.get('args',{}).get('correlation'))
        owners = [s for s in scopes if launch and s['tid'] == launch['tid']
            and s['ts'] <= launch['ts'] <= s['ts']+s['dur']]
        stage = owners[0]['name'] if owners else 'residual_modulation_cast_or_other'
        stages[stage] = stages.get(stage,0) + kernel['dur']
        rows.append(dict(name=kernel['name'],duration_us=kernel['dur'],stage=stage,
            args=kernel.get('args',{})))
    total = sum(e['duration_us'] for e in rows)
    ordered = sorted(kernels,key=lambda e:e['ts'])
    span = ordered[-1]['ts'] + ordered[-1]['dur']-ordered[0]['ts']
    return dict(trace=str(path),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        kernel_count=len(rows),sum_kernel_us=total,stage_us=stages,
        stage_fraction={k:v/total for k,v in stages.items()},kernels=rows,
        profiled_kernel_span_us=span, profiled_span_minus_kernel_us=span-total,
        launch_gap_boundary='instrumented eager only; includes hook/profiler CPU gaps; not Graph replay',
        occupancy_boundary='trace estimate is not measured achieved occupancy')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    original=json.loads(args.source.read_text())
    result=dict(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        method='Chrome cat=kernel duration; CUDA-runtime correlation -> containing CPU module annotation',
        excludes='gpu_user_annotation ranges; never add overlapping profiler inclusive scopes',rows=[])
    for row in original['rows']:
        result['rows'].append(dict(case=row['case'],
            strongest=row['strongest_static_baseline'],
            original=trace_summary(Path(row['original_profile']['trace'])),
            strong_cached=trace_summary(Path(row['strong_cached_profile']['trace']))))
    write(args.output,result)


if __name__=='__main__':
    main()

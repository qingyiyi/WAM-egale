"""CPU-only six-block full-verifier and post-OFF diagnostic analysis."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics as S


def off_analysis(rows):
    groups=defaultdict(dict)
    for v in rows:
        if (v['status']!='PASS' or not v['profile_OFF'] or v['query_count']!=29
            or not v['cold_setup_staging_keys_release_included'] or not v['bitwise_all_outputs']
            or not math.isfinite(v['wall_sec']) or v['wall_sec']<=0):raise ValueError('invalid OFF row')
        arm=v['arm'];block=v['block']
        if block in groups[arm]:raise ValueError('duplicate OFF block')
        groups[arm][block]=v
    if set(groups)!= {'R','C','HC'}:raise ValueError('missing OFF arm')
    metrics={}
    for arm,g in groups.items():
        if sorted(g)!=list(range(6)):raise ValueError('six blocks required')
        raw=[g[b]['wall_sec'] for b in range(6)];med=S.median(raw)
        mad=S.median(abs(x-med) for x in raw)/med
        metrics[arm]=dict(raw_sec=raw,median_sec=med,MAD_over_median=mad,
            CUDA_event_ms=[g[b]['CUDA_event_ms'] for b in range(6)],
            max_extra_peak_bytes=max(g[b]['extra_peak_bytes'] for b in range(6)))
    valid=all(v['MAD_over_median']<=.07 for v in metrics.values())
    effects=[]
    for a,b in [('C','R'),('HC','R'),('HC','C')]:
        ratio=metrics[a]['median_sec']/metrics[b]['median_sec']
        paired=[x/y for x,y in zip(metrics[a]['raw_sec'],metrics[b]['raw_sec'])]
        signal='full_verifier_positive_pilot' if valid and ratio<=.95 and all(x<1 for x in paired) else 'full_verifier_negative_pilot' if valid and ratio>=1.05 and all(x>1 for x in paired) else 'flat_or_mixed'
        effects.append(dict(comparison=a+'/'+b,ratio=ratio,improvement_percent=100*(1-ratio),paired_block_ratios=paired,signal=signal))
    return dict(status='PASS' if valid else 'FAIL',scope='single-worker ALL30 changing29 full-verifier replay, NOT whole policy/episode',
        metrics=metrics,effects=effects,stability_gate_MAD_max=.07,setup_staging_keys_release_included=True,
        success_environment_steps_E2E_proven=False,no_slow_samples_removed=True)


def trace_analysis(path):
    events=json.loads(Path(path).read_text())['traceEvents']
    complete=[v for v in events if v.get('ph')=='X' and v.get('dur',0)>=0]
    ann=[v for v in complete if v.get('cat')=='user_annotation' and v.get('name','').startswith('CHC.')]
    roots=[v for v in ann if v['name'].startswith('CHC.VERIFIER.')]
    if Counter(v['name'] for v in roots)!=Counter({'CHC.VERIFIER.R':1,'CHC.VERIFIER.HC':1}):raise ValueError('R/HC diagnostic ranges')
    coverage={}
    for root in roots:
        contained=[v for v in ann if v.get('tid')==root.get('tid') and root['ts']<=v['ts'] and v['ts']+v['dur']<=root['ts']+root['dur']+1]
        counts=Counter(v['name'] for v in contained)
        if any(counts['CHC.MODEL.blocks.'+str(i)]!=29 for i in range(30)):raise ValueError('diagnostic full30 changing29 block coverage')
        coverage[root['name']]=dict(layers=30,queries=29,blocks=870,
            text_KV_scope_count=sum(counts[f'CHC.MODEL.blocks.{i}.attn2.to_{kv}'] for i in range(30) for kv in ['k','v']))
    kernels=[v for v in complete if v.get('cat')=='kernel']
    if not kernels:raise ValueError('no CUDA kernel activity')
    # Associate GPU external identity to CPU launch op, not temporal
    # containment in asynchronous GPU execution.
    ext={};corr={}
    for v in complete:
        a=v.get('args',{});eid=a.get('External id',a.get('External Id'))
        if v.get('cat')=='cpu_op' and eid is not None:ext[int(eid)]=v['name']
        if v.get('cat') in ('cuda_runtime','cuda_driver') and eid is not None and a.get('correlation') is not None:corr[int(a['correlation'])]=int(eid)
    bins=defaultdict(lambda:dict(count=0,kernel_work_ms=0.))
    for v in kernels:
        a=v.get('args',{});eid=a.get('External id',a.get('External Id'))
        if eid is None:eid=corr.get(a.get('correlation'))
        op=ext.get(eid,'UNASSIGNED');bins[op]['count']+=1;bins[op]['kernel_work_ms']+=v['dur']/1000
    sync=[v for v in complete if v.get('cat') in ('cuda_runtime','cuda_driver') and 'synchroniz' in v.get('name','').lower()]
    return dict(status='PASS',trace=str(path),coverage=coverage,kernel_count=len(kernels),
        GPU_work_by_CPU_launch_op=dict(bins),CPU_sync_count=len(sync),
        CPU_sync_work_ms=sum(v['dur'] for v in sync)/1000,profile_ON_not_performance_primary=True,
        kernel_work_not_additive_wall_time=True)

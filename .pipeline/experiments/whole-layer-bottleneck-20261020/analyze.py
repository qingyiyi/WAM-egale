"""Summarize measured latency; profiler and event samples remain disjoint."""
import csv
import json
from pathlib import Path
import statistics as st

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'remote-evidence/profile-1/RESULT.json').read_text())
s=json.loads((HERE/'remote-evidence/STATUS.json').read_text())
GEMM=['Q_proj','K_proj','V_proj','self_output_proj','cross_Q_proj','cross_K_proj','cross_V_proj','cross_output_proj','FFN1','FFN2']
summary=[];components=[];trace_details=[]
def trace_events(case,kind):
    return json.loads((HERE/f'remote-evidence/profile-1/{case}-{kind}-TRACE.json').read_text())['traceEvents']
def component_events(case):
    events=trace_events(case,'components')
    cpu={e['args']['External id']:e for e in events if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})}
    scopes=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('component::')]
    mapping={e['name'].split('::')[1]:dict(kernels=[],memory=[]) for e in scopes}
    for e in events:
        if e.get('cat') not in ('kernel','gpu_memset','gpu_memcpy'):continue
        op=cpu.get(e.get('args',{}).get('External id'));assert op is not None
        matched=[x for x in scopes if x['ts']<=op['ts'] and op['ts']+op['dur']<=x['ts']+x['dur']+1]
        assert len(matched)==1
        label=matched[0]['name'].split('::')[1]
        mapping[label]['kernels' if e['cat']=='kernel' else 'memory'].append(e)
    assert all(len(v['kernels'])==1 for v in mapping.values())
    return mapping
for row in r['rows']:
    case=row['case'];times={name:st.median(x['stream_ms'] for x in r['timing'] if x['case']==case and x['name']==name)
        for name in ('native','whole','profile','static')}
    stage=[x for x in r['stages'] if x['case']==case and not x['audit']]
    ns={name:st.mean(x['stage_ns'][name] for x in stage) for name in stage[0]['stage_ns']}
    total=st.mean(x['kernel_ns'] for x in stage)
    mapping=component_events(case)
    native_events=trace_events(case,'native');whole_events=trace_events(case,'whole')
    native_kernels=[e for e in native_events if e.get('cat')=='kernel']
    whole_kernels=[e for e in whole_events if e.get('cat')=='kernel']
    pair_ratios=[next(t['stream_ms'] for t in r['timing'] if t['case']==case and t['name']=='static' and t['repeat']==i)/
                 next(t['stream_ms'] for t in r['timing'] if t['case']==case and t['name']=='profile' and t['repeat']==i) for i in range(6)]
    flash=[e for e in native_kernels if 'flash_fwd' in e['name']]
    assert len(flash) in (2,4)
    half=len(flash)//2
    trace_details.append(dict(case=case,components=mapping,native_kernels=native_kernels,whole_kernels=whole_kernels))
    q=dict(case=case,**times,whole_over_native=times['whole']/times['native'],
        static_latency_reduction=1-times['static']/times['profile'],
        paired_static_over_profile=pair_ratios,paired_static_increase_median=st.median(pair_ratios)-1,
        GEMM_fraction=sum(ns[k] for k in GEMM)/total,
        cross_KV_fraction=(ns['cross_K_proj']+ns['cross_V_proj'])/total,
        self_attention_fraction=ns['self_attention']/total,
        cross_attention_fraction=ns['cross_attention']/total,
        other_fraction=1-sum(ns[k] for k in GEMM+['self_attention','cross_attention'])/total,
        measured_stage_kernel_ms=total/1e6,
        whole_GEMM_ms=sum(ns[k] for k in GEMM)/1e6,
        Torch_same_input_GEMM_kernel_ms=sum(e['dur'] for v in mapping.values() for e in v['kernels'])/1000,
        native_trace_kernel_count=len(native_kernels),
        whole_trace_kernel_count=len(whole_kernels),
        native_trace_compute_kernel_ms=sum(e['dur'] for e in native_kernels)/1000,
        native_self_attention_kernel_ms=sum(e['dur'] for e in flash[:half])/1000,
        native_cross_attention_kernel_ms=sum(e['dur'] for e in flash[half:])/1000,
        prototype_self_attention_stage_ms=ns['self_attention']/1e6,
        prototype_cross_attention_stage_ms=ns['cross_attention']/1e6,
        native_trace_device_ms=row['traces']['native']['total_device_us']/1000,
        whole_trace_device_ms=row['traces']['whole']['total_device_us']/1000,
        original_vs_native=row['native_vs_Torch'],original_vs_whole=row['whole_vs_Torch'],
        ablation_checks=row['checks'],whole_resources=row['whole_resources'])
    summary.append(q)
    for p in row['primitive_profile']:
        elapsed=ns[p['label']];flops=2*p['M']*p['N']*p['K']
        matched=mapping[p['label']];kernel_us=sum(e['dur'] for e in matched['kernels'])
        components.append(dict(case=case,label=p['label'],M=p['M'],N=p['N'],K=p['K'],
            whole_stage_ms=elapsed/1e6,Torch_kernel_ms=kernel_us/1000,
            Torch_memory_ms=sum(e['dur'] for e in matched['memory'])/1000,
            raw_profiler_scope_device_range_ms=p['GPU_us']/1000,
            stage_over_Torch_kernel=elapsed/(kernel_us*1000),
            useful_matmul_FLOPs=flops,whole_stage_useful_TFLOPs=flops/elapsed/1000,
            Torch_kernel_useful_TFLOPs=flops/kernel_us/1e6,
            Torch_stream_ms=st.median(x['stream_ms'] for x in r['primitive_timing'] if x['case']==case and x['label']==p['label']),
            same_input_bitwise=p['same_input_output']['bitwise_pass']))
out=dict(status=r['status'],summary=summary,components=components,resource=s,
    blocks=r['full_block_calls'],primitives=r['primitive_calls'],
    restrictions=['prototype numerical FAIL remains','uncached eager reference, not strongest cached Graph',
        'single worker/process; six interleaved event samples per cell',
        'stage globaltimer includes barriers; isolated Torch primitive profile is not same full-block timeline',
        'static queue recompilation can change register allocation; not pure atomic instruction cycle isolation',
        'DRAM/L2/stalls/achieved occupancy UNKNOWN'])
(HERE/'RESULT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
(HERE/'TRACE-ATTRIBUTION.json').write_text(json.dumps(trace_details,indent=2)+'\n')
for name,data in [('COMPONENTS.csv',components),('TIMINGS.csv',r['timing']),('PRIMITIVE-TIMINGS.csv',r['primitive_timing'])]:
    with (HERE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
print(json.dumps([{k:v for k,v in x.items() if k not in ('original_vs_native','original_vs_whole','ablation_checks')} for x in summary],indent=2))

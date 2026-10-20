"""Aggregate every registered sample without selecting a best process/repeat."""
import csv
import json
from pathlib import Path
import statistics as st

HERE=Path(__file__).resolve().parent
E=HERE/'remote-evidence'
terminal=json.loads((E/'STATUS.json').read_text())
assert terminal['status']=='COMPLETE_ASYNC_GEMM_RESEARCH_UNADMITTED',terminal['status']
gate=json.loads((E/'gate-1/RESULT.json').read_text())
timing=[json.loads((E/f'timing-{i}/RESULT.json').read_text()) for i in (1,2)]
profile=json.loads((E/'stages-1/RESULT.json').read_text())
samples=[x for r in timing for x in r['timing']]
GEMM=['Q_proj','K_proj','V_proj','self_output_proj','cross_Q_proj','cross_K_proj','cross_V_proj','cross_output_proj','FFN1','FFN2']
rows=[];stages=[]
for g in gate['rows']:
    case=g['case'];medians={name:st.median(x['stream_ms'] for x in samples if x['case']==case and x['name']==name)
        for name in ('native','whole','sync','async')}
    by_process=[]
    for process in (1,2):
        values={name:st.median(x['stream_ms'] for x in samples if x['case']==case and x['name']==name and x['process']==process)
            for name in medians}
        by_process.append(dict(process=process,**values,whole_to_async_speedup=values['whole']/values['async'],
                              sync_to_async_speedup=values['sync']/values['async']))
    paired={}
    for reference in ('whole','sync','native'):
        ratios=[]
        for process in (1,2):
            for repeat in range(6):
                vs={x['name']:x['stream_ms'] for x in samples if x['case']==case and x['process']==process and x['repeat']==repeat}
                ratios.append(vs[reference]/vs['async'])
        paired[reference]=dict(values=ratios,median=st.median(ratios),min=min(ratios),max=max(ratios))
    ps={name:[x for x in profile['profiles'] if x['case']==case and x['name']==name] for name in ('old_profile','profile')}
    pm={name:st.mean(x['kernel_ns'] for x in val)/1e6 for name,val in ps.items()}
    pg={name:sum(st.mean(x['stage_ns'][k] for x in val) for k in GEMM)/1e6 for name,val in ps.items()}
    for label in profile['profiles'][0]['stage_ns']:
        stages.append(dict(case=case,stage=label,old_ms=st.mean(x['stage_ns'][label] for x in ps['old_profile'])/1e6,
                          async_ms=st.mean(x['stage_ns'][label] for x in ps['profile'])/1e6))
    rows.append(dict(case=case,**medians,whole_to_async=medians['whole']/medians['async'],
        async_latency_reduction=1-medians['async']/medians['whole'],sync_to_async=medians['sync']/medians['async'],
        async_over_native=medians['async']/medians['native'],by_process=by_process,paired=paired,
        profile_kernel_ms=pm,profile_GEMM_ms=pg,profile_GEMM_speedup=pg['old_profile']/pg['profile'],
        native_bitwise=g['original_native']['bitwise_pass'],Torch_bitwise=g['original_async']['bitwise_pass'],
        matches_failed_whole=all(v['pass'] for v in g['checks'].values()),
        cache_transitions_pass=all(x['pass'] for x in g['cache_sequences']),
        resources={k:v['resources'] for k,v in g['checks'].items()}))
result=dict(status=terminal['status'],rows=rows,resource=terminal,
    blocks=terminal['full_block_calls'],timing_samples=len(samples),gate_cases=len(gate['rows']),
    legal_cache_transitions=sum(len(g['cache_sequences']) for g in gate['rows']),
    finite_Torch_admitted=all(g['original_async']['bitwise_pass'] for g in gate['rows']),
    production_admitted=False,formal_performance_permission=False)
(HERE/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for n,values in [('TIMINGS.csv',samples),('STAGES.csv',stages)]:
    with (HERE/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values)
print(json.dumps([{k:v for k,v in x.items() if k not in ('by_process','paired','resources')} for x in rows],indent=2))

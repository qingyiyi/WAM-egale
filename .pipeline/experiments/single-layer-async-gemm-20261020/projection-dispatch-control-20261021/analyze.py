from pathlib import Path
import json,csv,statistics
B=Path(__file__).resolve().parent
rows=[];full=[];embedded=[];all_clocks=[];audit=[]
for proc in [1,2]:
 d=json.loads((B/f'RESULT-{proc}.json').read_text());assert d['status']=='COMPLETE_FINITE_SAME_BODY_DISPATCH_DIAGNOSTIC'
 for r in d['rows']:
  all_clocks.extend([r['full_clock_before'],r['full_clock_after']])
  for arm in ['active4','large8','trace8']:
   v=[x['ms'] for x in r['full_samples'] if x['arm']==arm]
   full.append(dict(process=proc,case=r['case'],arm=arm,median_ms=statistics.median(v),min_ms=min(v),max_ms=max(v)))
  for sid in [16,17]:
   vals=[(x['stage_ticks'][str(sid)][1]-x['stage_ticks'][str(sid)][0])/1000 for x in r['embedded']]
   # Max per-CTA span includes arrival skew and waiting, not a separable wall fraction.
   final_fence=[max((cta[1][1]-cta[1][0])/1000 for cta in x['barriers'][str(sid)]) for x in r['embedded']]
   embedded.append(dict(process=proc,case=r['case'],stage=sid,median_us=statistics.median(vals),max_CTA_final_fence_median_us=statistics.median(final_fence),resources=r['embedded'][0]['resources']))
  for st in r['stages']:
   for a in st['audit']:audit.append(dict(process=proc,case=r['case'],stage=st['stage'],mode=a['mode'],bitwise=a['precision']['bitwise_pass'],max_abs=a['precision'].get('max_abs'),tiles=len(a['visits']),exact_once=all(x==1 for x in a['visits'])))
   for c in st['conditions']:
    all_clocks.extend([c['clock_before'],c['clock_after']])
    for arm in ['native','ordinary','static108','atomic108']:
     s=[x for x in c['samples'] if x['arm']==arm]
     row=dict(process=proc,case=r['case'],stage=st['stage'],name=st['name'],condition=c['condition'],arm=arm)
     for metric in ['kernel_interval_us','inclusive_interval_us','reset_interval_us']:
      v=[x[metric] for x in s];row[metric]=statistics.median(v);row[metric+'_min']=min(v);row[metric+'_max']=max(v)
     rows.append(row)
for name,data in [('TIMING-SUMMARY.csv',rows),('FULL-SUMMARY.csv',full),('EMBEDDED-SUMMARY.csv',embedded),('FINITE-GATE.csv',audit)]:
 with (B/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
ratios=[]
for proc in [1,2]:
 for case in ['short','middle','long']:
  for sid in [16,17]:
   for condition in ['hot_single_stage','same_real_native_prefix']:
    a={x['arm']:x for x in rows if x['process']==proc and x['case']==case and x['stage']==sid and x['condition']==condition}
    ratios.append(dict(process=proc,case=case,stage=sid,condition=condition,static_vs_ordinary=a['static108']['kernel_interval_us']/a['ordinary']['kernel_interval_us'],atomic_vs_ordinary=a['atomic108']['kernel_interval_us']/a['ordinary']['kernel_interval_us'],atomic_inclusive_vs_ordinary=a['atomic108']['inclusive_interval_us']/a['ordinary']['inclusive_interval_us'],ordinary_vs_native=a['ordinary']['kernel_interval_us']/a['native']['kernel_interval_us'],static_vs_native=a['static108']['kernel_interval_us']/a['native']['kernel_interval_us']))
summary=dict(rows=rows,full=full,embedded=embedded,ratios=ratios,kernel_resources=json.loads((B/'RESULT-1.json').read_text())['kernel_resources'],clock_configs=[dict(t) for t in sorted(set(tuple(sorted(x.items())) for x in all_clocks))],audit_count=len(audit),finite_gate_pass=all(x['bitwise'] and x['exact_once'] for x in audit))
(B/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(kernel_resources=summary['kernel_resources'],clocks=summary['clock_configs'],audit_count=len(audit),pass_all=summary['finite_gate_pass'])))
for case in ['short','middle','long']:
 print(case)
 for sid in [16,17]:
  a={arm:statistics.median(x['kernel_interval_us'] for x in rows if x['case']==case and x['stage']==sid and x['condition']=='same_real_native_prefix' and x['arm']==arm) for arm in ['native','ordinary','static108','atomic108']}
  e=statistics.median(x['median_us'] for x in embedded if x['case']==case and x['stage']==sid)
  print(sid,a,'embedded8',e)

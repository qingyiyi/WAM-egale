from pathlib import Path
import json,statistics,csv
B=Path('/rjs/cxz/论文探索/.pipeline/experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021');rows=[];controls=[]
for process in [1,2]:
 d=json.loads((B/f'RESULT-{process}.json').read_text());
 short=json.loads((B/'short-steady-r2'/f'RESULT-{process}.json').read_text());
 d['rows']=[short['rows'][0]]+[r for r in d['rows'] if r['case']!='short'];assert d['status']=='COMPLETE_CURRENT_SOURCE_PREFIX_DIAGNOSTIC'
 for r in d['rows']:
  ctrl={a:statistics.median(x['ms'] for x in r['instrumentation_control'] if x['arm']==a) for a in ['active','diagnostic']}
  controls.append(dict(process=process,case=r['case'],**ctrl,perturbation_percent=100*(ctrl['diagnostic']/ctrl['active']-1),clock_full=r.get('clock_after_fixed60',r.get('clock_after_fixed600')),clock_embedded=r['clock_after_embedded']))
  for s in r['stages']:
   embedded=statistics.median((e['ticks_ns'][str(s['stage'])][1]-e['ticks_ns'][str(s['stage'])][0])/1000 for e in r['embedded'])
   for c in s['conditions']:
    native=statistics.median(x['us'] for x in c['samples'] if x['arm']=='A_native');own=statistics.median(x['us'] for x in c['samples'] if x['arm']=='B_owned')
    inside=statistics.median((x['internal_ns'][1]-x['internal_ns'][0])/1000 for x in c['samples'] if x['arm']=='B_owned')
    rows.append(dict(process=process,case=r['case'],stage=s['stage'],name=s['name'],condition=c['condition'],native_us=native,own_us=own,own_vs_native_percent=100*(own/native-1),inside_us=inside,embedded_us=embedded,embedded_vs_inside_percent=100*(embedded/inside-1),owned_resources=c['owned_resources'],clock_before=c['clock_before'],clock_after=c['clock_after']))
with (B/'STEADY-STAGE-COMPARISON.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(B/'STEADY-ANALYSIS.json').write_text(json.dumps(dict(rows=rows,controls=controls),indent=2)+'\n')
print('control',controls)
for name in dict.fromkeys(x['name'] for x in rows):
 a=[x for x in rows if x['name']==name and x['condition']=='same_real_native_prefix']
 print(name,'native',round(min(x['native_us'] for x in a),3),round(max(x['native_us'] for x in a),3),'owned',round(min(x['own_us'] for x in a),3),round(max(x['own_us'] for x in a),3),'gap%',round(min(x['own_vs_native_percent'] for x in a),2),round(max(x['own_vs_native_percent'] for x in a),2),'embedded',round(min(x['embedded_us'] for x in a),3),round(max(x['embedded_us'] for x in a),3),'embeddingchange%',round(min(x['embedded_vs_inside_percent'] for x in a),2),round(max(x['embedded_vs_inside_percent'] for x in a),2),'resources',a[0]['owned_resources'])
print('clocks',dict((k,sorted({x[p][k] for x in rows for p in ['clock_before','clock_after']})) for k in ['SM','memory']))

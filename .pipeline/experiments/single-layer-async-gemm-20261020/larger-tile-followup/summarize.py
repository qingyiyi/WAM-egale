"""Summarize persisted finite gates/timing; profiler samples are excluded."""
from pathlib import Path
import json,statistics
B=Path(__file__).resolve().parent
out={'status':'PARTIAL','variants':{},'formal_performance_permission':False,'production_admitted':False}
for v in ['cooperative64','context128']:
 V=B/v;row={};p=V/'CUTLASS-GATE-RESULT.json'
 if p.exists():
  g=json.loads(p.read_text());row['gate']={k:g.get(k) for k in ['status','all_pass','full_block_calls','lifecycle_seconds']};row['gate']['rows']=[{'case':x['case'],'output_exact':x['new_vs_Torch']['bitwise_pass'],'stages':len(x['stages']),'repeat_exact':x['repeat_exact'],'resources':x['resources']} for x in g.get('rows',[])]
 ts=[]
 for p in sorted(V.glob('cutlass-timing-*-RESULT.json')):
  d=json.loads(p.read_text());ts.append(d)
 row['processes']=[{'repeat':x['process_repeat'],'status':x['status'],'calls':x['full_block_calls'],'lifecycle_seconds':x['lifecycle_seconds']} for x in ts];row['timing']={}
 for case in ['short-B14-action','middle-B14-action','long-B14-action']:
  vals={n:[s['ms'] for d in ts for s in d['samples'] if s['case']==case and s['name']==n] for n in ['native','new']}
  if not all(vals.values()):continue
  meds={n:statistics.median(x) for n,x in vals.items()};per=[]
  for d in ts:
   pv={n:[s['ms'] for s in d['samples'] if s['case']==case and s['name']==n] for n in vals}
   if all(pv.values()):per.append(statistics.median(pv['new'])/statistics.median(pv['native']))
  row['timing'][case]={'n_per_engine':{n:len(x) for n,x in vals.items()},'median_ms':meds,'ratio':meds['new']/meds['native'],'per_process_ratio':per,'single_kernel_counts':[len(t['kernels']) for d in ts for t in d['traces'] if t['case']==case]}
 row['meets_2x_all_three']=len(row['timing'])==3 and all(max(x['per_process_ratio'])<=2 for x in row['timing'].values())
 out['variants'][v]=row
out['status']='COMPLETE' if all(x.get('gate',{}).get('all_pass') and len(x['processes'])==2 and all(p['status']=='COMPLETE_SINGLE_LAYER_ENGINEERING_TIMING' for p in x['processes']) for x in out['variants'].values()) else 'PARTIAL'
(B/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

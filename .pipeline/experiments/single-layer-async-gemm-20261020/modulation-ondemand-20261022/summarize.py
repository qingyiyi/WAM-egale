from pathlib import Path
import csv,json,statistics
b=Path(__file__).resolve().parent
processes=[json.loads((b/f'TIMING-{i}.json').read_text()) for i in [1,2]]
assert all(x['status']=='COMPLETE_FINITE_WHOLE_COMPARISON' for x in processes)
rows=[]
for p in processes:
 for v in p['rows']:
  ms=v['median_ms'];rows.append(dict(process=p['process'],kv=v['case'],**ms,change_vs_before_percent=100*(ms['candidate']/ms['before']-1),change_vs_guarded_percent=100*(ms['candidate']/ms['guarded_uncached']-1),cache_pass=v['cache_unchanged']['pass_all'],inputs_pass=v['inputs_unchanged'],RNG_pass=v['RNG_unchanged']))
with (b/'WHOLE-LAYER-TIMING.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
a=[]
for kv in ['short','middle','long']:
 x=[z for z in rows if z['kv']==kv];m={k:statistics.mean(z[k] for z in x) for k in ['torch_static_uncached','shared_torch_uncached','guarded_uncached','before','candidate']}
 a.append(dict(kv=kv,**m,latency_change_percent=100*(m['candidate']/m['before']-1),speedup_vs_before=m['before']/m['candidate'],latency_change_vs_guarded_percent=100*(m['candidate']/m['guarded_uncached']-1),both_processes_faster=all(z['candidate']<z['before'] for z in x)))
r=dict(status='COMPLETE_FINITE_WHOLE_COMPARISON',aggregation='mean of two independent process medians; raw values retained',scope='normal debug=false complete layer Graph; uncached context, no robot or full-model',rows=a,all_rows_faster=all(z['both_processes_faster'] for z in a),formal_performance_permission=False)
(b/'GPU-SUMMARY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

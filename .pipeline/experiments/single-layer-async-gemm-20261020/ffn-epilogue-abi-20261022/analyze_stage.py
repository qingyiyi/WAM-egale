from pathlib import Path
import json,statistics,hashlib,sys
b=Path(sys.argv[1]);g=json.loads((b/'SIMPLE-GATE-RESULT.json').read_text());s=json.loads((b/'STRUCTURAL-RESULT.json').read_text());ts=[json.loads((b/f'TIMING-{p}.json').read_text()) for p in [1,2]]
assert g['all_pass'] and s['all_pass'] and all(t['status']=='COMPLETE_FINITE_WHOLE_COMPARISON' for t in ts)
rows=[]
for i,kv in enumerate([288,1152,2160]):
 arms=['torch_static_cached_final','previous_cached','custom_cached'];mm={a:statistics.mean(t['rows'][i]['median_ms'][a] for t in ts) for a in arms}
 row=dict(kv=kv,median_ms_mean_of_two=mm,speedup_vs_Torch=mm[arms[0]]/mm[arms[2]],latency_reduction_percent_vs_previous=100*(1-mm[arms[2]]/mm[arms[1]]),per_process_reduction=[100*(1-t['rows'][i]['median_ms'][arms[2]]/t['rows'][i]['median_ms'][arms[1]]) for t in ts],debug={})
 for a in arms[1:]:
  obs=[z for t in ts for z in t['rows'][i]['debug_group_timestamps'][a]]
  row['debug'][a]=dict(n=len(obs),last_group_ready_after_first_us=statistics.median((max(z['timestamps'][:4])-min(z['timestamps'][:4]))/1000 for z in obs),last_done_after_first_ready_us=statistics.median((max(z['timestamps'][8:12])-min(z['timestamps'][:4]))/1000 for z in obs),last_done_after_last_ready_us=statistics.median((max(z['timestamps'][8:12])-max(z['timestamps'][:4]))/1000 for z in obs),resources=obs[0]['resources'])
 for t in ts:assert next(x for x in g['rows'] if x['case']==t['rows'][i]['case']+'-B14-action')['fixture']==t['rows'][i]['fixture']
 rows.append(row)
r=dict(status='COMPLETE_FINITE_STAGE',rows=rows,real_signatures=len(g['rows']),structural_controls=len(s['rows']),finite_calls=g['full_block_calls']+s['full_block_calls'],timing_worker_accounted_calls=sum(t['full_calls'] for t in ts),call_accounting_note='includes graph capture registrations, warm/replay and debug observations, not teacher forwards',candidate_binary_SHA256=hashlib.sha256(next(b.glob('wam_whole*.so')).read_bytes()).hexdigest(),fixture_identity_gate_vs_timing='MATCH_6_OF_6',four_scientific_permissions=[False]*4,production_admitted=False)
(b/'SUMMARY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

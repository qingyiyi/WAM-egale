from pathlib import Path
import json,csv,statistics
B=Path(__file__).resolve().parent
rows=[]
for process in [1,2]:
 p=B/f'TIMING-{process}.json'
 if not p.exists():continue
 d=json.loads(p.read_text())
 for row in d.get('rows',[]):
  m=row['median_ms'];ready=m['candidate']
  z=dict(process=process,kv=row['case'],**m)
  for baseline in ['before','staged','guarded_uncached','torch_static_uncached']:
   z['ready_latency_change_percent_vs_'+baseline]=100*(ready/m[baseline]-1)
  z['cache_pass']=row['cache_unchanged']['pass_all']
  z['RNG_pass']=row['RNG_unchanged'];z['inputs_pass']=row['inputs_unchanged'];rows.append(z)
if rows:
 with (B/'WHOLE-LAYER-TIMING.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
r=dict(rows=rows,complete_two_processes=len(rows)==6,scope='one real B14 layer uncached Graph; not robot/chunk/E2E',production_admitted=False)
(B/'GPU-SUMMARY.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))

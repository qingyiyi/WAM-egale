from pathlib import Path
import json,statistics,sys
E=Path(__file__).resolve().parent;prefix=sys.argv[1];paths=[E/f'{prefix}{i}'/f'TIMING-{i}.json' for i in [1,2]]
raw=[json.loads(p.read_text()) for p in paths];assert all(x['status']=='COMPLETE_FINITE_WHOLE_COMPARISON' for x in raw)
result=[]
for case in ['short','middle','long']:
 rows=[next(r for r in x['rows'] if r['case']==case) for x in raw];arms=list(rows[0]['median_ms']);pairs=[]
 for process,row in enumerate(rows,1):
  rs={a:[x['ms'] for x in row['samples'] if x['arm']==a] for a in arms}
  stats={a:dict(median_ms=statistics.median(v),min_ms=min(v),max_ms=max(v),q1_ms=statistics.quantiles(v,n=4)[0],q3_ms=statistics.quantiles(v,n=4)[2]) for a,v in rs.items()}
  for a in arms:
   if a=='A':continue
   diffs=[100*(x/y-1) for x,y in zip(rs[a],rs['A'])]
   stats[a]['paired_vs_A_pct']=dict(median=statistics.median(diffs),q1=statistics.quantiles(diffs,n=4)[0],q3=statistics.quantiles(diffs,n=4)[2])
  pairs.append(dict(process=process,stats=stats))
 med={a:statistics.mean(row['median_ms'][a] for row in rows) for a in arms}
 r=dict(case=case,median_ms=med,per_process=pairs,normal_resources=[row['debug_resources'] for row in rows])
 if 'B' in med and 'C' in med:
  r['C_over_A']=med['C']/med['A'];r['C_over_B']=med['C']/med['B'];r['B_over_A']=med['B']/med['A']
 result.append(r)
out=E/f'{prefix.rstrip("-")}-SUMMARY.json';out.write_text(json.dumps(result,indent=2)+'\n')
for r in result:
 print(r['case'],r['median_ms'])
 for p in r['per_process']:print(' process',p['process'],{a:round(z['paired_vs_A_pct']['median'],4) for a,z in p['stats'].items() if a!='A'})
 if 'C_over_A' in r:print(' C/A',r['C_over_A'],'C/B',r['C_over_B'],'B/A',r['B_over_A'])

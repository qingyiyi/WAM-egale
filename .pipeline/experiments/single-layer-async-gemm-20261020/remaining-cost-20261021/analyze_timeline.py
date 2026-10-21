from pathlib import Path
import json,csv,statistics as st,collections
b=Path('/rjs/cxz/论文探索/.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021')
d=json.loads((b/'trace-r2/TRACE-RESULT.json').read_text()); rows=[]; barriers=[]; result=[]
for r in d['rows']:
 c=r['case']; by={x['stage']:x for x in r['native_stages']}
 per=[]
 for i in range(30):
  vals=[(e['stage_ticks'][i][1]-e['stage_ticks'][i][0])/1000 for e in r['embedded']]
  native=st.median(by[i]['us']) if i in by else None
  v=dict(case=c,stage=i,name=by.get(i,{}).get('name','layout' if i==29 else 'cache_insert_noop'),own_callsite_us=st.median(vals),native_external_us=native,descriptive_difference_us=None if native is None else st.median(vals)-native)
  rows.append(v);per.append(v)
 for p in sorted((b/'trace-r2').glob(f'CTA-{c}-*.csv')):
  raw=list(csv.DictReader(p.open())); groups=collections.defaultdict(list)
  for x in raw:groups[int(x['stage']),int(x['barrier'])].append(x)
  for (stage,part),g in groups.items():
   enter=[int(x['enter_ns']) for x in g];ex=[int(x['exit_ns']) for x in g]
   wait=[(e-a)/1000 for a,e in zip(enter,ex)]
   barriers.append(dict(case=c,repeat=int(p.stem.split('-')[-1]),stage=stage,barrier=part,CTA_count=len(g),arrival_spread_us=(max(enter)-min(enter))/1000,post_last_arrival_us=(max(ex)-max(enter))/1000,mean_CTA_wait_us=st.mean(wait),max_CTA_wait_us=max(wait),mean_arrival_deficit_us=st.mean([(max(enter)-a)/1000 for a in enter])))
 perf={arm:st.median([x['ms'] for x in r['perturbation'] if x['arm']==arm]) for arm in ('active','trace')}
 result.append(dict(case=c,all_intermediates_exact=r['all_shared_intermediates_exact'],valid_KV=r['cache_check']['pass_all'],inputs_unchanged=r['inputs_unchanged'],RNG_unchanged=r['RNG_unchanged'],active_resources=r['active_source_resources'],trace_resources=r['embedded'][0]['resources'],**perf,perturbation_pct=100*(perf['trace']/perf['active']-1),sum_callsite_median_us=sum(x['own_callsite_us'] for x in per),sum_native_external_us=sum(x['native_external_us'] for x in per if x['native_external_us'] is not None)))
for name,data in [('STAGE-SUMMARY.csv',rows),('BARRIER-SUMMARY.csv',barriers)]:
 with (b/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
(b/'TIMELINE-SUMMARY.json').write_text(json.dumps(dict(cases=result,boundary='Own callsite vs native external device-marker, diagnostic only. No empty-marker correction. Sum not an additive E2E attribution; barriers overlap across CTAs.',full_calls=d['full_calls'],stage_calls=d['stage_calls']),indent=2)+'\n')
print(json.dumps(result,indent=2))
for c in ['short','middle','long']:
 print('\n',c,'barriers stage/part/arrival spread/postlast/meanwait')
 for i in [0,3,4,11,16,17,19,20,25,26,27]:
  for part in [0,1,2]:
   z=[x for x in barriers if x['case']==c and x['stage']==i and x['barrier']==part]
   if z:print(i,part,*[round(st.median([x[k] for x in z]),2) for k in ['arrival_spread_us','post_last_arrival_us','mean_CTA_wait_us']])

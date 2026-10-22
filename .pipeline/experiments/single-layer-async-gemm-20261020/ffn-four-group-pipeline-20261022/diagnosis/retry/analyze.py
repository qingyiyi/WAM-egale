from pathlib import Path
import json,statistics,csv
B=Path(__file__).resolve().parent
med=statistics.median

def timeline(raw):
 f1=[raw[i*8:(i+1)*8] for i in range(112)]
 f2=[raw[896+i*12:896+(i+1)*12] for i in range(96)]
 c=[raw[2048+i*4:2048+(i+1)*4] for i in range(108)]
 assert all(0<x[0]<=x[1]<=x[2]<=x[3]<=x[4] for x in f1)
 assert all(0<x[0]<=x[1]<=x[2]<=x[3]<=x[4]<=x[5] and x[8]<=x[9] for x in f2)
 assert all(0<x[0]<=x[1]<=x[2]<=x[3] for x in c)
 t0=min(x[0] for x in c);tend=max(x[3] for x in c);span=tend-t0
 last=max(range(96),key=lambda i:f2[i][5]);z=f2[last]
 lastproducer=max(range(112),key=lambda i:f1[i][4]);p=f1[lastproducer]
 def ns(x):return x/1000
 out={'span_us':ns(span),'producer_span_us':ns(max(x[4] for x in f1)-t0),'last_producer_tile':lastproducer,'last_producer_group':lastproducer//28,'last_consumer_task':last,'last_consumer_group':last//24,
 'consumer_after_last_producer_us':ns(z[5]-p[4]),'consumer_begin_vs_last_producer_us':ns(z[0]-p[4]),
 'last_consumer_partition_us':ns(z[2]-z[1]),'last_consumer_mma_warp_us':ns(z[9]-z[8]),'last_consumer_commit_wait_us':ns(z[3]-z[2]),'last_consumer_epilogue_us':ns(z[4]-z[3]),'last_consumer_publish_us':ns(z[5]-z[4]),'exit_after_last_consumer_us':ns(tend-z[5])}
 groups=[]
 for g in range(4):
  a=f1[g*28:(g+1)*28];b=f2[g*24:(g+1)*24]
  groups.append({'group':g,'producer_first_us':ns(min(x[0] for x in a)-t0),'producer_last_us':ns(max(x[4] for x in a)-t0),'consumer_first_us':ns(min(x[0] for x in b)-t0),'consumer_last_us':ns(max(x[5] for x in b)-t0),
 'producer_mma_median_us':ns(med(x[2]-x[1] for x in a)),'producer_epilogue_median_us':ns(med(x[3]-x[2] for x in a)),'producer_publish_median_us':ns(med(x[4]-x[3] for x in a)),
 'consumer_partition_median_us':ns(med(x[2]-x[1] for x in b)),'consumer_mma_warp_median_us':ns(med(x[9]-x[8] for x in b)),'consumer_wait_median_us':ns(med(x[3]-x[2] for x in b)),'consumer_wait_max_us':ns(max(x[3]-x[2] for x in b)),'consumer_commit_median_us':ns(med(x[4]-x[3] for x in b)),'consumer_publish_median_us':ns(med(x[5]-x[4] for x in b))})
 # These are accumulated CTA occupancy periods, not additive layer latency or SM utilization counters.
 work={
 'ffn1_setup':sum(x[1]-x[0] for x in f1),
 'ffn1_mma_and_barrier':sum(x[2]-x[1] for x in f1),
 'ffn1_epilogue':sum(x[3]-x[2] for x in f1),
 'ffn1_publish':sum(x[4]-x[3] for x in f1),
 'ffn2_partition':sum(x[2]-x[1] for x in f2),
 'ffn2_commit_wait':sum(x[3]-x[2] for x in f2),
 'ffn2_epilogue':sum(x[4]-x[3] for x in f2),
 'ffn2_publish':sum(x[5]-x[4] for x in f2),
 'exit_barrier':sum(x[3]-x[2] for x in c)}
 total=sum(x[3]-x[0] for x in c)
 work['queue_and_unattributed']=total-sum(work.values());out['cta_time_fraction']={k:v/total for k,v in work.items()};out['groups']=groups
 # How many CTAs are actually executing each useful phase across the entire FFN interval.
 active=sum(x[3]-x[1] for x in f1)+sum((x[2]-x[1])+(x[4]-x[3]) for x in f2)
 out['average_compute_epilogue_ctas']=active/span
 out['last4_producer_tiles']=[{'tile':i,'start_us':ns(f1[i][0]-t0),'end_us':ns(f1[i][4]-t0),'cta':f1[i][5]} for i in range(108,112)]
 return out
rows=[]
for process in [1,2]:
 d=json.loads((B/f'PROFILE-{process}.json').read_text());assert d['status']=='COMPLETE_FINITE_FFN_DIAGNOSIS'
 for row in d['rows']:
  out={'process':process,'case':row['case'],'median_ms':row['median_ms'],'production_resources':row['production_resources'],'diagnostic_resources':row['diagnostic_resources']}
  for label in ['full','isolated']:
   traces=[timeline(t) for t in row[label+'_timelines']]
   scalar={k:med(x[k] for x in traces) for k,v in traces[0].items() if isinstance(v,(float,int))}
   scalar['cta_time_fraction']={k:med(x['cta_time_fraction'][k] for x in traces) for k in traces[0]['cta_time_fraction']}
   scalar['groups']=[{k:med(x['groups'][g][k] for x in traces) for k in traces[0]['groups'][g]} for g in range(4)]
   out[label]=scalar;out[label+'_individual']=traces
  m=row['median_ms'];out['diagnostic_off_vs_production_ratio']=m['diagnostic_full_off']/m['production_full'];out['instrumentation_vs_off_ratio']=m['diagnostic_full_on']/m['diagnostic_full_off'];out['isolated_ffn_speedup_vs_Torch']=m['torch_ffn']/m['custom_ffn_off']
  rows.append(out)
(B/'ANALYSIS.json').write_text(json.dumps({'rows':rows,'scope':'Instrumented CTA milestones, not hardware counter attribution; task times overlap. Independent graph FFN is not in-layer stage latency.'},indent=2)+'\n')
for x in rows:
 print(x['process'],x['case'],'speedup',round(x['isolated_ffn_speedup_vs_Torch'],4),'perturb',round(x['diagnostic_off_vs_production_ratio'],4),round(x['instrumentation_vs_off_ratio'],4))
 print('time',x['median_ms'])
 print('full', {k:v for k,v in x['full'].items() if k not in ['groups']})
 print('groups',x['full']['groups'])

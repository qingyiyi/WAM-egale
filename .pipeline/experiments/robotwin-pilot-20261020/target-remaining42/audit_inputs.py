"""CPU audit of already-complete first-six supervision; no replay or teacher call."""
from pathlib import Path
import json,os,time,torch
R=Path('/root/cxz/robotwin');B=Path(__file__).resolve().parent
os.sched_setaffinity(0,{18,19});torch.set_num_threads(2)
out={'status':'RUNNING','start':time.time(),'episodes':[],'new_teacher_calls':0}
manifest=json.loads((R/'runs/prefilter-4/STATUS.json').read_text())
old=R/'runs/target-first6-3';terminal=json.loads((old/'STATUS.json').read_text())
assert terminal['status']=='COMPLETE_FIRST6_TARGET_ONLY' and not terminal['remaining_owned']
cases=[]
for t in manifest['tasks']:
 assert len(t['valid'])==19
 for i,c in enumerate(t['valid']):
  assert c['valid_index']==i and c['task']==t['task'] and Path(c['initial']).is_file()
  if i<8:assert c['role']=={'stage':'TARGET','split':'FIT' if i<6 else 'CAL'}
 c=t['valid'][0];name=c['task']+'-valid00';e=json.loads((old/name/'STATUS.json').read_text());assert e['status'].startswith('COMPLETE_') and all(e['initial_checks'].values())
 chunks=sorted((old/'server'/name).glob('chunk-*.pt'));assert len(chunks)==e['policy_chunks'];rows=0
 for i,p in enumerate(chunks):
  d=torch.load(p,map_location='cpu',weights_only=False);assert d['case']==name and d['chunk']==i and d['source_kind']=='actual_target_visited_query' and d['full_conditions_for_teacher_requery'] is False
  assert len(d['rows'])==50
  for row in d['rows']:
   assert row['state_full'].shape==(1,30,2,16,1) and row['teacher_direction_full'].shape==(1,30,2,16,1)
   assert len(row['effective_channel_ids'])==16 and len(set(row['effective_channel_ids']))==16
   assert torch.equal(row['state_full'],row['sample_before'])
   assert torch.isfinite(row['teacher_direction_full']).all()
   rows+=1
 out['episodes'].append({'case':name,'status':e['status'],'chunks':len(chunks),'rows':rows,'initial_checks':e['initial_checks']})
for i in range(1,8):
 for t in manifest['tasks']:cases.append(t['valid'][i])
assert len(cases)==42 and sum(c['role']['split']=='FIT' for c in cases)==30
bank=torch.load(old/'server/text-bank.pt',map_location='cpu',weights_only=False)
assert bank['source']=='7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb' and bank['target_revision']=='8c9dea8abbc5c91cc9e18bc3264b8915083bbe70'
assert set(bank['embeddings'])=={c['instruction'] for t in manifest['tasks'] for c in t['valid']}|{''}
assert all(x.shape==(1,512,4096) and x.dtype==torch.bfloat16 for x in bank['embeddings'].values())
out.update(status='COMPLETE_CPU_INPUT_AUDIT',bank_path=str(old/'server/text-bank.pt'),bank_prompts=len(bank['embeddings']),remaining=42,FIT=30,CAL=12,lifecycle_seconds=time.time()-out['start'])
(B/'INPUT-MANIFEST.json').write_text(json.dumps(cases,indent=2)+'\n');(B/'INPUT-AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

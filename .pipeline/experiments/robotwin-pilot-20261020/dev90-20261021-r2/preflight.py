from pathlib import Path
import ast,hashlib,json,os,socket,datetime
os.sched_setaffinity(0,{18,19})
import torch
torch.set_num_threads(2)
R=Path('/root/cxz/robotwin');J=R/'runs/dev90-20261021-r2';B=R/'runs/tf-migration-20261021'
plan=json.loads((J/'PLAN.json').read_text());queue=json.loads((J/'INPUT-MANIFEST.json').read_text());original=json.loads((R/'runs/prefilter-4/STATUS.json').read_text())
expected=[t['valid'][i] for i in range(14,19) for t in original['tasks']]
assert len(queue)==90 and len(expected)==30
for i,c in enumerate(expected):
 for j,arm in enumerate([('T','C','P'),('C','P','T'),('P','T','C')][i%3]):
  row=queue[3*i+j];assert all(row[k]==v for k,v in c.items()) and row['arm']==arm and row['pair_id']==i
 assert Path(c['initial']).is_file() and c['role']['stage']=='DEV'
meta=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
assert meta['metadata']['condition']=='ON3000' and meta['metadata']['updates']==3000
cal=json.loads((R/'runs/current18-followup-20261021/CAL/RESULT.json').read_text());assert cal['checkpoint']==plan['checkpoint'] and cal['decision_flips']==0 and cal['rows']==11195 and cal['fixed_parent_tau']==plan['tau']
bank=torch.load(plan['reset_bank'],map_location='cpu',weights_only=False);entries=bank['embeddings'];assert all(c['instruction'] in entries for c in queue);assert all(x.shape==(1,512,4096) and x.dtype==torch.bfloat16 for x in entries.values())
assert not (J/'STATUS.json').exists()
for port in [34971,34976]:
 sock=socket.socket();sock.settimeout(1);assert sock.connect_ex(('127.0.0.1',port))!=0,('occupied',port);sock.close()
files=[J/n for n in ['route.py','server.py','eval_runtime.py','client.py','guard.py','PLAN.json','INPUT-MANIFEST.json']]+[B/n for n in ['runtime.py','common.py','server_math.py','buffers.py','tree_api.py','prefix_selector.py','draft.py']]
for p in files:
 if p.suffix=='.py':ast.parse(p.read_text())
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest[plan['checkpoint']]=hashlib.sha256(Path(plan['checkpoint']).read_bytes()).hexdigest()
(J/'SOURCE-AND-CHECKPOINT-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
size=sum(p.stat().st_size for p in (R/'runs').rglob('*') if p.is_file())
assert size+plan['storage_cap_bytes']<=plan['total_evidence_cap_bytes']
row=dict(status='PASS_FIXED_INPUT_AND_DEPENDENCY_REVIEW',observed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),pairs=30,episodes=90,checkpoint_metadata=meta['metadata'],checkpoint_bytes=Path(plan['checkpoint']).stat().st_size,bank_entries=len(entries),all_prompts_present=True,existing_evidence_bytes=size,added_cap_bytes=plan['storage_cap_bytes'],CUDA_started=False,complete_episode_started=False)
(J/'CPU-READY.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))

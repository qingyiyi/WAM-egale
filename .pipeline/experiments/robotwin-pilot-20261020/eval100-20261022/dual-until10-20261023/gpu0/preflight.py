from pathlib import Path
import ast,json,hashlib,os,socket
import torch
J=Path(__file__).resolve().parent;R=Path('/root/cxz/robotwin');plan=json.loads((J/'PLAN.json').read_text());worker=json.loads((J/'WORKER.json').read_text());os.sched_setaffinity(0,set(worker['CPU']['guard']));torch.set_num_threads(2)
queue=json.loads((J/'INPUT-MANIFEST.json').read_text());assert len(queue)==1800 and len({x['pair_id'] for x in queue})==600
assert len({(x['task'],x['seed'],x['arm']) for x in queue})==1800
for i in range(600):
 rows=queue[i*3:(i+1)*3];assert {x['arm'] for x in rows}=={'T','C','P'}
 assert len({(x['task'],x['seed'],x['initial'],x['instruction']) for x in rows})==1
 assert Path(rows[0]['initial']).is_file()
meta=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
assert meta['metadata']['condition']=='ON3000' and meta['metadata']['updates']==3000
cal=json.loads((R/'runs/current18-followup-20261021/CAL/RESULT.json').read_text());assert cal['checkpoint']==plan['checkpoint'] and cal['decision_flips']==0 and cal['fixed_parent_tau']==plan['tau']
bank=torch.load(plan['reset_bank'],map_location='cpu',weights_only=False)
assert all(c['instruction'] in bank['embeddings'] for c in queue) and '' in bank['embeddings']
assert all(x.shape==(1,512,4096) and x.dtype==torch.bfloat16 for x in bank['embeddings'].values())
for port in [worker['port'],worker['master_port']]:
 s=socket.socket();s.settimeout(1);assert s.connect_ex(('127.0.0.1',port))!=0;s.close()
files=list(J.glob('*.py'))+[J/'PLAN.json',J/'INPUT-MANIFEST.json',Path(plan['checkpoint'])]
for f in files:
 if f.suffix=='.py':ast.parse(f.read_text())
manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(J/'SOURCE-AND-CHECKPOINT-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(J/'CPU-READY.json').write_text(json.dumps(dict(status='PASS_FIXED_INPUT_AND_DEPENDENCY_REVIEW',episode_count=len(queue),CUDA_started=False,checkpoint_metadata=meta['metadata'],bank_policy='existing original full512 BF16 entries reused; missing fixed prompts encoded by original GPU T5 at server setup'),indent=2)+'\n')

"""Seed claims from preserved completed legacy pairs; no target/model invocation."""
from pathlib import Path
import json,shutil,datetime,hashlib
R=Path('/root/cxz/robotwin/runs/eval100-20261022');D=R/'dual-until10-20261023';old=R/'gpu1';drain=json.loads((D/'DRAIN-STATUS.json').read_text());assert drain['status']=='COMPLETE_OLD_WORKER_DRAINED_AFTER_FULL_PAIR'
s=json.loads((old/'STATUS.json').read_text());assert 'end' in s and not s['remaining_owned'];assert len(s['episodes'])%3==0 and len(s['pair_audits'])*3==len(s['episodes'])
queue=json.loads((old/'INPUT-MANIFEST.json').read_text());assert len(queue)==1800
assert not (D/'PAIR-CLAIMS.json').exists(),'Shared campaign must not be reinitialized'
claims=dict(status='SHARED_FIXED_PAIR_QUEUE',created_at=datetime.datetime.now().astimezone().isoformat(),pairs={},halt=None,soft_stops={},legacy_namespace=str(old))
for audit in s['pair_audits']:
 assert audit['pass_all'];pair=audit['pair_id'];episodes=[dict(e,namespace=str(old)) for e in s['episodes'] if e['pair_id']==pair];assert len(episodes)==3
 claims['pairs'][str(pair)]=dict(status='COMPLETE_PAIR',owner='legacy-gpu1',GPU_uuid='GPU-4acf84ff-31ce-dead-c2a4-2a122b123b8c',namespace=str(old),episodes=episodes,prefix_audit=audit)
(D/'PAIR-CLAIMS.json').write_text(json.dumps(claims,indent=2)+'\n')
for i in [0,1]:
 J=D/f'gpu{i}';shutil.copy2(old/'INPUT-MANIFEST.json',J/'INPUT-MANIFEST.json')
 q=json.loads((J/'PLAN.json').read_text());assert q['checkpoint']==json.loads((old/'PLAN.json').read_text())['checkpoint']
 for f in J.glob('*.py'):
  import ast;ast.parse(f.read_text())
report=dict(status='READY_SHARED_FIXED_PAIRS_NO_GPU_STARTED',legacy_complete=len(s['episodes']),legacy_pairs=len(s['pair_audits']),remaining_episodes=1800-len(s['episodes']),input_manifest_sha256=hashlib.sha256((old/'INPUT-MANIFEST.json').read_bytes()).hexdigest(),legacy_GPU_consumed_seconds=json.loads((R/'GPU-LEDGER.json').read_text())['consumed_seconds'],pair_claims=len(claims['pairs']),scientific_config_unchanged=True,bank_shared_reference=True,CPU_queue_check=json.loads((D/'CPU-QUEUE-RESULT.json').read_text()))
(D/'INITIALIZATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

from pathlib import Path
from datetime import datetime
import json,subprocess
P=Path('/root/autodl-tmp/cxz/operator-sm120-20261023/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025')
def alive(d):
 try:return Path(f'/proc/{d["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]==str(d['birth'])
 except FileNotFoundError:return False
rows=[]
for f in sorted(P.glob('*-STATUS.json')):
 d=json.loads(f.read_text())
 if 'guard' in d:rows.append(dict(name=f.name,status=d['status'],returncode=d.get('returncode'),lifecycle_seconds=d.get('lifecycle_seconds'),owned=d['owned'],guard=d['guard'],owned_alive=alive(d['owned']),guard_alive=alive(d['guard'])))
assert all(x['status']=='EXITED_OK' and not x['owned_alive'] and not x['guard_alive'] for x in rows)
assert not alive(json.loads((P/'RUN-OWNED.json').read_text()))
assert (P/'IO-RESUME.json').exists()
S=P.parent/'FFN-STAGE-SPLIT-20261025';current=P.parent.parent/'workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu'
assert (S/'whole_layer.cu').read_bytes()==current.read_bytes(),'current S source changed'
job=json.loads((P/'JOB.json').read_text());calls=sum(json.loads((P/f'measure-{n}/RESULT.json').read_text())['calls'] for n in (1,2));now=datetime.now().astimezone()
size=sum(f.stat().st_size for f in P.rglob('*') if f.is_file());assert calls<=job['calls_cap'] and size<job['storage_cap_bytes']
receipt=dict(status='COMPLETE_OWNED_EXIT_VERIFIED',at=now.isoformat(),phases=rows,parent_exited=True,migration_IO_resumed=True,retained_S_source_byte_equal=True,action='all owned children already exited; no kill; no instance shutdown',gpu_contexts=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout)
(P/'OWNED-CLEANUP.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
job.update(status='COMPLETE_NEGATIVE_RETAIN_S',closed_at=now.isoformat(),actual_lifecycle_seconds=(now-datetime.fromisoformat(job['T0'])).total_seconds(),actual_calls=calls,cleanup_complete=True,retained='S');(P/'JOB.json').write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n')
report=dict(status=job['status'],GPU_lifecycle_seconds=job['actual_lifecycle_seconds'],CUDA_active_seconds='UNKNOWN; no aggregate GPU active counter',block_forward_calls=calls,block_call_cap=job['calls_cap'],phase_seconds=sum(x['lifecycle_seconds'] for x in rows),storage_bytes=size,storage_cap_bytes=job['storage_cap_bytes'],failed_GPU_phases=[],timeouts=[],stop='A2 is slower than old sharing-A and S; no further candidate in this job',cleanup=receipt['status']);(P/'RESOURCE-AND-STOP-REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

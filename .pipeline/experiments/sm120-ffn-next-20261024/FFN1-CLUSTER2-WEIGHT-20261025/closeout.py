"""只核验本作业已退出进程；不终止其他任务。"""
from pathlib import Path
from datetime import datetime
import json,subprocess
P=Path(__file__).resolve().parent
def alive(identity):
    try:return Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]==str(identity['birth'])
    except FileNotFoundError:return False
phases=[]
for f in sorted(P.glob('*-STATUS.json')):
    d=json.loads(f.read_text())
    if 'guard' not in d:continue
    phases.append(dict(file=f.name,status=d['status'],returncode=d.get('returncode'),lifecycle_seconds=d['lifecycle_seconds'],guard=d['guard'],guard_alive=alive(d['guard']),owned=d.get('owned'),owned_alive=alive(d['owned']) if d.get('owned') else None))
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout
assert all(not x['guard_alive'] and not x['owned_alive'] and x['returncode']==0 for x in phases)
now=datetime.now().astimezone()
receipt=dict(status='COMPLETE_OWNED_EXIT_VERIFIED',checked_at=now.isoformat(),phases=phases,GPU_contexts=gpu,action='all phase guards and owned children already exited; no kill performed; no instance shutdown')
(P/'OWNED-CLEANUP.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
job=json.loads((P/'JOB.json').read_text())
calls=sum(json.loads((P/f'measure-{n}/RESULT.json').read_text())['calls'] for n in (1,2))
calls+=json.loads((P/'WAIT-RESULT.json').read_text())['calls']+json.loads((P/'PUBLIC-ENTRY-RESULT.json').read_text())['calls']
size=sum(f.stat().st_size for f in P.rglob('*') if f.is_file())
assert calls<=job['calls_cap'] and size<=job['storage_cap_bytes']
job.update(status='COMPLETE_NEGATIVE_RETAIN_S',retained='S',actual_calls=calls,closed_at=now.isoformat(),actual_lifecycle_seconds=(now-datetime.fromisoformat(job['T0'])).total_seconds(),cleanup_complete=True)
(P/'JOB.json').write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n')
resource=dict(status=job['status'],actual_calls=calls,cap_calls=job['calls_cap'],lifecycle_seconds=job['actual_lifecycle_seconds'],lifecycle_cap_seconds=job['wall_seconds_cap'],phase_lifecycle_sum_seconds=sum(p['lifecycle_seconds'] for p in phases),CUDA_active_seconds='UNKNOWN: no aggregate GPU-active instrumentation',storage_bytes=size,storage_cap_bytes=job['storage_cap_bytes'],failed_gpu_phases=[],timeouts=[],cpu_preparation_failure='one source generation string match; no GPU work; retained in CPU-REVIEW',remaining_call_budget=job['calls_cap']-calls,cleanup=receipt['status'],stop='D no-go; no further candidate, S retained. L is a control, not automatically adopted.')
(P/'RESOURCE-AND-STOP-REPORT.json').write_text(json.dumps(resource,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(resource,ensure_ascii=False))

"""Bounded complete-real-block experiment under renewed RTX5090 authorization."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import os
import signal
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
def identity(pid):
    return {'pid': pid, 'birth': Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]}
def alive(who):
    try: return identity(who['pid']) == who
    except FileNotFoundError: return False
start = time.time()
deadline = start + 1200
tz = timezone(timedelta(hours=8))
job = dict(goal='Migrate current W0 TMA/W1-3 epilogue/W4-7 MMA to Self Q/K/V only',
           status='RUNNING', input_manifest='parent INPUT-MANIFEST.json; short/middle/long actual B14 block0 fixtures',
           start=datetime.fromtimestamp(start,tz).isoformat(),
           absolute_cleanup_deadline=datetime.fromtimestamp(deadline,tz).isoformat(),
           gpu='existing RTX5090 GPU-e5e7821f-9633-6cc8-d06c-d0d0df4c9256',
           prior_ledger_seconds=35926.41251785547,
           authorization='User explicitly added unrestricted existing RTX5090 use on 2026-10-24; old receipts unchanged',
           total_cap_seconds=None, wall_cap_seconds=1200, cleanup_reserve_seconds=90,
           cpu_threads=4, storage_cap_bytes=1073741824, call_cap=12000,
           source='frozen selected WARP-SPECIALIZED-4MMA plus CHANGE.patch',
           baseline='Torch/Guarded/frozen selected A; same cached context; eager/no Graph',
           acceptance='all captured tensors/output exact; input/KV/RNG/counts; repeated whole-block benefit; public two streams',
           cleanup='only owned PID+birth; return source/DSO/results; keep failed artifacts; instance stays on',
           engineering_quality_screen=False, independent_test_complete=False,
           noninferiority_established=False, formal_performance_permission=False, production_admitted=False)
(P/'JOB.json').write_text(json.dumps(job,indent=2))
record=dict(status='RUNNING', guard=identity(os.getpid()), phases=[])
def save(): (P/'STATUS.json').write_text(json.dumps(record,indent=2))
def disk_bytes(): return sum(x.stat().st_size for x in P.rglob('*') if x.is_file())
def phase(name,cmd,cap):
    if time.time() > deadline-120: raise TimeoutError('insufficient lifecycle reserve')
    q=dict(name=name, command=cmd, start=time.time(), status='RUNNING')
    record['phases'].append(q)
    print('PHASE_START', name, flush=True)
    with (P/f'{name}.log').open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        q['identity']=identity(child.pid); save()
        try:
            q['returncode']=child.wait(timeout=max(1,min(cap,deadline-time.time()-90)))
            q['status']='EXITED_OK' if q['returncode']==0 else 'FAILED'
        except subprocess.TimeoutExpired:
            q['status']='TIMEOUT'
            if alive(q['identity']): os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if alive(q['identity']): os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)
        finally:
            q.update(ended=time.time(), owned_child_exited=not alive(q['identity']))
            q['lifecycle_seconds']=q['ended']-q['start']; save()
    if q['status']!='EXITED_OK': raise RuntimeError(name+' '+q['status'])
    if disk_bytes()>job['storage_cap_bytes']: raise RuntimeError('new namespace storage cap exceeded')
    print('PHASE_DONE',name,round(q['lifecycle_seconds'],3),flush=True)
try:
    save()
    for label,script in [('normal','build_candidate.py'),('profile','build_whole_profile.py')]:
        phase('build-'+label,[sys.executable,str(P/script),'build_ext','--build-temp',str(P/f'build-temp/{label}'),
                            '--build-lib',str(P/f'build/{label}')],240)
    binary=next((P/'build/normal').glob('*.so'))
    prof=next((P/'build/profile').glob('*.so'))
    phase('precision',[sys.executable,str(P/'precision_candidate.py'),'--binary',str(binary),
                       '--module-name','wam_whole_qkv_tma_cuda','--expect-tma','true',
                       '--output',str(P/'results/precision')],120)
    for process in (1,2):
        cmd=[sys.executable,str(P/'measure.py'),'--process',str(process),'--output',str(P/f'results/measure{process}-data'),
             '--candidate',str(binary),'--label','SELF-QKV-TMA-4MMA','--module-name','wam_whole_qkv_tma_cuda',
             '--reference',str(P/'A.so'),'--reference-module-name','wam_whole_warp_specialized_cuda']
        if process==1: cmd+=['--stages','--profile',str(prof),'--profile-module-name','wam_whole_qkv_tma_profile_cuda']
        phase('measure'+str(process),cmd,200)
    record['status']='MEASUREMENTS_COMPLETE_PENDING_RETENTION'
except BaseException as exc:
    record.update(status='FAILED_OR_PARTIAL',error=repr(exc))
finally:
    record.update(ended=time.time(), lifecycle_seconds=time.time()-start, storage_bytes=disk_bytes())
    job.update(status=record['status'],ended=datetime.now(tz).isoformat())
    (P/'JOB.json').write_text(json.dumps(job,indent=2)); save()
    print('TERMINAL',record['status'],record.get('error',''),flush=True)
if record['status']=='FAILED_OR_PARTIAL': raise SystemExit(1)

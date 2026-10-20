"""Bounded M37 parent; CPU compile and serial GPU workers, no automatic retry."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

BASE=Path('/root/cxz/ops/device-contract-20261020')
WORK=Path('/root/cxz/ops/persistent-tile-20261019/workspace')
OPS=WORK/'算子研究'
OPT=OPS/'wam_ops/optimized'
UUID='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'

def write(name,obj):
    q=BASE/(name+'.pending');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(BASE/name)

def identity(pid):
    return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),
                pgid=os.getpgid(pid),uid=Path(f'/proc/{pid}').stat().st_uid)

def matches(record):
    try:return identity(record['pid'])==record
    except (FileNotFoundError,ProcessLookupError):return False

def contexts():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    return [s.split(',')[1].strip() for s in raw.splitlines() if s.split(',')[0].strip()==UUID]

def main():
    os.sched_setaffinity(0,{4,5})
    if contexts():raise RuntimeError('GPU_BUSY_NO_WORKER_STARTED')
    if (BASE/'REGISTRATION.json').exists():raise RuntimeError('no namespace reuse or automatic retry')
    start=time.monotonic();now=datetime.now(timezone.utc)
    for name in ('tmp','cache','build'): (BASE/name).mkdir(exist_ok=True)
    source_files=[OPT/'csrc/contract_linear.cu',OPT/'build_contract.py',
                  BASE/'validate_contract.py',BASE/'inspect_gemm.py']
    reg=dict(goal='four real same-input Q/FFN2 device contracts; one geometry-informed candidate',
        user_authorization='latest user explicitly agreed to continue the numerical-contract then asynchronous-tile work',
        T0=now.isoformat(),work_stop=(now+timedelta(minutes=35)).isoformat(),
        cleanup_deadline=(now+timedelta(minutes=45)).isoformat(),GPU_uuid=UUID,
        lifecycle_cap_seconds=2700,primitive_call_cap=120,full_block_cap=8,robot_episodes=0,full30=0,
        CPU=[4,5],storage_cap_bytes=512*1024**2,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        engineering_quality_screen=False,independent_test_complete=False,
        noninferiority_established=False,formal_performance_permission=False)
    write('REGISTRATION.json',reg)
    frozen=[OPT/'csrc/whole_layer.cu',OPT/'whole_layer.py',OPS/'wam_ops/baseline/native.py',
            WORK/'投机采样/wam-p0-worktree/wan_va/modules/model.py']
    frozen_hash={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    write('FROZEN-BEFORE.json',frozen_hash)
    state=dict(status='RUNNING',parent=identity(os.getpid()),jobs=[])
    env=dict(os.environ,PYTHONPATH=str(OPS),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',
        CUDA_VISIBLE_DEVICES=UUID,TMPDIR=str(BASE/'tmp'),XDG_CACHE_HOME=str(BASE/'cache'),
        CUDA_CACHE_PATH=str(BASE/'cache/cuda'),TORCH_EXTENSIONS_DIR=str(BASE/'cache/torch'),
        CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.0')
    for name in ('cuda','torch'):(BASE/'cache'/name).mkdir(exist_ok=True)
    def run(name,cmd,cap,cpu=False):
        if datetime.now(timezone.utc)+timedelta(seconds=cap+60)>datetime.fromisoformat(reg['work_stop']):
            raise TimeoutError('whole next phase plus cleanup reserve unavailable')
        if not cpu and contexts():raise RuntimeError('foreign context before phase')
        before=time.monotonic()
        with (BASE/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(cmd,cwd=OPS,env=dict(env,CUDA_VISIBLE_DEVICES='' if cpu else UUID),
                                   stdout=log,stderr=log,start_new_session=True)
            owned=identity(child.pid);write(name+'-OWNED.json',owned)
            timeout=False
            try:rc=child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timeout=True
                if matches(owned):os.killpg(owned['pgid'],signal.SIGTERM)
                try:rc=child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if matches(owned):os.killpg(owned['pgid'],signal.SIGKILL)
                    rc=child.wait(timeout=20)
        rec=dict(name=name,returncode=rc,timed_out=timeout,root_reaped=True,
                 lifecycle_seconds=time.monotonic()-before,remaining_NVML=contexts(),owned=owned)
        state['jobs'].append(rec);write(name+'-CLEANUP.json',rec);write('STATUS.json',state)
        if rc or timeout or rec['remaining_NVML']:raise RuntimeError(name+' did not close cleanly')
    try:
        py='/root/cxz/envs/va/bin/python'
        run('metadata-1',[py,'-B',str(BASE/'inspect_gemm.py'),'--output',str(BASE/'metadata-1')],300)
        actual=json.loads((BASE/'metadata-1/RESULT.json').read_text())
        if actual['status']!='COMPLETE_PUBLIC_API_METADATA_READ':raise RuntimeError('metadata failed')
        expected={(2,'Q'):(15,1,17),(2,'FFN2'):(21,7,15),
                  (14,'Q'):(20,2,15),(14,'FFN2'):(20,2,15)}
        gate=[]
        for row in actual['rows']:
            key=(row['batch'],row['name'])
            if key in expected:
                c=row['configuration'];seen=(c['tile_id'],c['splitK'],c['stages_id'])
                gate.append(dict(batch=key[0],name=key[1],expected=expected[key],observed=seen,match=seen==expected[key]))
        write('GEOMETRY-GATE.json',gate)
        if len(gate)!=4 or not all(r['match'] for r in gate):raise RuntimeError('remote public geometry differs; no blind adaptation')
        run('build-1',[py,str(OPT/'build_contract.py'),'build_ext','--build-lib',str(OPT),
                       '--build-temp',str(BASE/'build')],900,cpu=True)
        run('contract-1',[py,'-B',str(BASE/'validate_contract.py'),'--registration',str(BASE/'REGISTRATION.json'),
                           '--output',str(BASE/'contract-1')],600)
        result=json.loads((BASE/'contract-1/RESULT.json').read_text())
        state['status']=result['status']
        state['integration_started']=False
        state['numerical_gate_does_not_cover_attention_or_whole_layer']=True
    except BaseException as exc:
        state.update(status='STOPPED_FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        state.update(at_UTC=datetime.now(timezone.utc).isoformat(),
            lifecycle_seconds=time.monotonic()-start,remaining_NVML=contexts(),foreign_processes_signalled=False,
            frozen_items_unchanged={str(p):hashlib.sha256(p.read_bytes()).hexdigest()==frozen_hash[str(p)] for p in frozen})
        write('STATUS.json',state)

if __name__=='__main__':main()

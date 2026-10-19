"""One finite profile and conditionally one padding challenger, no retries."""
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

BASE=Path('/root/cxz/ops/whole-layer-warp-20261020')
TIMING=Path('/root/cxz/performance/libero-four-suite-20261019/isolated')
WORK=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究')
UUID='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'


def read(p):return json.loads(p.read_text())
def write(name,d):
    p=BASE/'diagnostic'/name;q=p.with_suffix('.pending');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def pin(pid):
    r=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,birth_ticks=int(r[19]),pgid=os.getpgid(pid),uid=Path(f'/proc/{pid}').stat().st_uid)
def matches(d):
    try:return pin(d['pid'])==d
    except (FileNotFoundError,ProcessLookupError):return False
def contexts():
    s=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    return [x for x in s.splitlines() if x.strip().isdigit()]


def main():
    os.sched_setaffinity(0,{4,5});state=dict(status='CPU_WAITING_OBJECT_TIMING',jobs=[],owned=pin(os.getpid()),
        original_numerical_gate=False,original_primary_source_and_DSO_unchanged=True)
    write('QUEUE-STATUS.json',state)
    try:
        while True:
            assert datetime.now(timezone.utc)<datetime.fromisoformat('2026-10-20T00:00:00+00:00'),'08:00 CPU wait stop'
            p=TIMING/'GPU0-UNSTARTED-CONTINUATION.json';done=False
            if p.exists():
                t=read(p);done=t['status'] in ('UNSTARTED_OBJECT_COMPLETE_ORIGINAL_PANEL_PARTIAL',
                    'STOPPED_OR_PARTIAL_UNSTARTED_OBJECT')
                if done and t.get('active'):
                    active=Path(t['active']['job']);done=(active/'GUARD-FINAL.json').exists() and read(active/'GUARD-FINAL.json').get('status')=='CLEAN'
            built=BASE/'diagnostic/BUILD-PROFILE-1-RESULT.json'
            if built.exists():assert read(built)['returncode']==0,'CPU profile build failed; no GPU'
            if done and built.exists() and not contexts():break
            time.sleep(20)
        now=datetime.now(timezone.utc);work=now+timedelta(seconds=780);cleanup=now+timedelta(seconds=900)
        reg=dict(goal='device-stage profile; at most one conditional padding challenger',
            physical_gpu_uuid=UUID,actual_T0=now.isoformat(),absolute_work_stop=work.isoformat(),
            absolute_cleanup_deadline=cleanup.isoformat(),full_block_cap=83,GPU_lifecycle_cap_seconds=900,
            parent_registration=str(BASE/'DIAGNOSTIC-REGISTRATION.json'),full30=0,episodes=0)
        write('GPU-REGISTRATION.json',reg);state.update(status='RUNNING_FINITE_DEVICE_PROFILE',actual_T0=now.isoformat())
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,PYTHONPATH=str(WORK),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            TMPDIR=str(BASE/'tmp'),XDG_CACHE_HOME=str(BASE/'cache'),CUDA_CACHE_PATH=str(BASE/'cache/cuda'),
            PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        def run(name,cmd,environment,cap):
            assert time.time()+cap+30<cleanup.timestamp(),'no full job/cleanup reserve'
            begin=time.monotonic()
            with (BASE/'diagnostic'/(name+'.log')).open('xb') as f:
                child=subprocess.Popen(cmd,env=environment,cwd=WORK,stdout=f,stderr=f,start_new_session=True)
                owned=pin(child.pid);write(name+'-OWNED.json',dict(**owned,argv=cmd))
                timeout=False
                try:rc=child.wait(timeout=cap)
                except subprocess.TimeoutExpired:
                    timeout=True
                    if matches(owned):os.killpg(owned['pgid'],signal.SIGTERM)
                    try:rc=child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if matches(owned):os.killpg(owned['pgid'],signal.SIGKILL)
                        rc=child.wait(timeout=10)
            receipt=dict(name=name,returncode=rc,timeout=timeout,owned=owned,root_reaped=True,
                lifecycle_seconds=time.monotonic()-begin,remaining_NVML=contexts())
            state['jobs'].append(receipt);write(name+'-CLEANUP.json',receipt);write('QUEUE-STATUS.json',state)
            assert rc==0 and not timeout and not contexts(),'job failed or card acquired; no dependent job'
        def worker(mode,name,cap=300):
            run(name,['/root/cxz/envs/va/bin/python','-B',str(BASE/'profile_worker.py'),
                '--mode',mode,'--output',str(BASE/'diagnostic'/name),
                '--registration',str(BASE/'diagnostic/GPU-REGISTRATION.json')],env,cap)
            r=read(BASE/'diagnostic'/name/'RESULT.json')
            assert r['status']=='COMPLETE_DIAGNOSTIC_MATCHES_UNADMITTED_V1','profile isolation failed; no tuning'
            return r
        result=worker('profile','profile-1')
        if result['padding_trigger']:
            assert time.time()+180+300+60<work.timestamp(),'no complete conditional build/worker reserve'
            e=dict(env,CUDA_VISIBLE_DEVICES='',CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',
                TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='2',DIAG_KINDS='pad40')
            run('BUILD-PAD40-1',['/root/cxz/envs/va/bin/python',str(BASE/'build_profile.py'),
                'build_ext','--build-lib',str(BASE/'diagnostic'),'--build-temp',str(BASE/'diagnostic/build-pad40')],e,180)
            second=worker('pad40','pad40-1')
            assert result['complete_block_calls']+second['complete_block_calls']<=83
            state['status']='COMPLETE_PROFILE_AND_ONE_CHALLENGER_NO_PRODUCTION_ADMISSION'
        else:
            state['status']='COMPLETE_PROFILE_CHALLENGER_NOT_TRIGGERED'
        state['B14_GEMM_kernel_fraction']=result['B14_GEMM_kernel_fraction']
    except BaseException as exc:
        state.update(status='STOPPED_FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        state.update(finished_at=datetime.now(timezone.utc).isoformat(),remaining_NVML=contexts(),
            original_numerical_gate=False,foreign_processes_signalled=False)
        write('QUEUE-STATUS.json',state)


if __name__=='__main__':main()

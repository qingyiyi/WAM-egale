"""One owned process, idle guard, no retries, terminal accounting."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN.parent/'single_worker_chc_gpu_v4'))
from resource_guard import NVML,Idle
def main():
 out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
 nv=NVML();idle=Idle();rows=[];start=time.monotonic();child=None;reason=None
 try:
  for _ in range(30):
   sample=nv.sample();rows.append(sample)
   if idle.add(sample,time.monotonic()):break
   time.sleep(1)
  if len(idle.samples)<15:raise RuntimeError('no clean GPU window')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  with (out/'worker.log').open('x') as log:
   child=subprocess.Popen(['/rjs/cxz/miniconda3/envs/va/bin/python','-B',str(OWN/'replay.py'),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   gpu_start=time.monotonic()
   while child.poll() is None:
    sample=nv.sample();rows.append(sample)
    foreign=[p for kind in ('compute','graphics') for p in sample[kind] if p['pid']!=child.pid]
    if foreign:raise RuntimeError('foreign GPU process appeared: '+repr(foreign))
    if time.monotonic()-gpu_start>570:raise TimeoutError('600s GPU limit incl 30s cleanup')
    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>2**30:raise RuntimeError('output limit')
    time.sleep(1)
  if child.returncode:raise RuntimeError('worker exit '+str(child.returncode))
 except BaseException as e:reason=repr(e)
 finally:
  if child is not None and child.poll() is None:
   (out/'STOP_REQUESTED').write_text(reason or 'stop')
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  final=nv.sample();nv.close()
  result=dict(status='COMPLETE_MECHANISM_REPLAY' if reason is None else 'FAILED_OR_BLOCKED',reason=reason,elapsed_sec=time.monotonic()-start,GPU_wall_sec=0 if child is None else time.monotonic()-gpu_start,owned_pid=child.pid if child else None,own_process_reaped=child is None or child.poll() is not None,final_resources=final,samples=rows,formal_robot_performance=False)
  (out/'SUPERVISOR-RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
if __name__=='__main__':main()

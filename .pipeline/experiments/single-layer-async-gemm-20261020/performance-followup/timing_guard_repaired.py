import subprocess,os,json,time,signal
from pathlib import Path
B=Path(__file__).resolve().parent;R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized');T=time.time();jobs=[]
N=B;B=N.parent/'performance-followup'
os.sched_setaffinity(0,{4,5})
for sub in ('norm-build','tmp','cache'):(B/sub).mkdir(exist_ok=True)
env=dict(os.environ,CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',WAM_FA2_BUILD_ROOT=str(B/'norm-build'),TORCH_CUDA_ARCH_LIST='8.0',TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'),MAX_JOBS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R.parents[1]))
def identity(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
for name,cmd,cap,cpu in [('o2-timing-3',['/root/cxz/envs/va/bin/python','-B',str(B/'timing.py'),'3'],600,False),('o2-timing-4',['/root/cxz/envs/va/bin/python','-B',str(B/'timing.py'),'4'],600,False)]:
 precheck=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 if any(line.startswith('GPU-c3fa8d6f') for line in precheck.splitlines()):raise RuntimeError('GPU0 not empty before worker')
 env['GPU_PRELAUNCH_EMPTY']='1'
 with (B/(name+'.log')).open('x') as out:
  p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES='' if cpu else 'GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'),stdout=out,stderr=out,start_new_session=True);rec=dict(name=name,owned=identity(p.pid),start=time.time());jobs.append(rec)
  try:rc=p.wait(timeout=cap)
  except subprocess.TimeoutExpired:
   rec['timeout']=True
   if identity(p.pid)==rec['owned']:os.killpg(p.pid,signal.SIGTERM)
   try:rc=p.wait(timeout=15)
   except subprocess.TimeoutExpired:
    if identity(p.pid)==rec['owned']:os.killpg(p.pid,signal.SIGKILL)
    rc=p.wait(timeout=15)
  rec.update(rc=rc,reaped=True,end=time.time());B.joinpath('O2-TIMING-REPAIRED-JOBS.json').write_text(json.dumps(dict(T0=T,jobs=jobs,lifecycle_seconds=time.time()-T),indent=2))
  if rc:break


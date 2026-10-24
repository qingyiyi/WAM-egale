from pathlib import Path
import subprocess,os,time,json,signal
b=Path(__file__).resolve().parent
os.sched_setaffinity(0,{18,19});start=time.time()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(b/'tmp'),XDG_CACHE_HOME=str(b/'cache'),PYTHONDONTWRITEBYTECODE='1')
for n in ['tmp','cache']:(b/n).mkdir(exist_ok=True)
with (b/'Q32-BUILD.log').open('x') as f:p=subprocess.Popen(['/root/cxz/envs/va/bin/python',str(b/'build_q32.py'),'build_ext','--build-lib',str(b),'--build-temp',str(b/'q32-build')],cwd=b,stdout=f,stderr=f,env=env,start_new_session=True)
owned=dict(pid=p.pid,birth_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(p.pid));r=dict(status='BUILDING_CPU',start=start,owned=owned,cap_seconds=900,GPU_seconds=0)
(b/'Q32-BUILD-STATUS.json').write_text(json.dumps(r,indent=2))
try:rc=p.wait(timeout=900)
except subprocess.TimeoutExpired:
 r['timeout']=True
 q=Path(f'/proc/{p.pid}/stat')
 if q.exists() and int(q.read_text().rsplit(')',1)[1].split()[19])==owned['birth_ticks']:os.killpg(owned['pgid'],signal.SIGTERM)
 try:rc=p.wait(timeout=15)
 except subprocess.TimeoutExpired:
  if q.exists() and int(q.read_text().rsplit(')',1)[1].split()[19])==owned['birth_ticks']:os.killpg(owned['pgid'],signal.SIGKILL)
  rc=p.wait(timeout=15)
r.update(status='BUILT_Q32_NOT_VALIDATED' if rc==0 else 'FAILED_BUILD',returncode=rc,end=time.time(),wall_seconds=time.time()-start,root_reaped=True)
(b/'Q32-BUILD-STATUS.json').write_text(json.dumps(r,indent=2))

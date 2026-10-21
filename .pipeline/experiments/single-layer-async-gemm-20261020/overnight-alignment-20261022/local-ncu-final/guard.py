from pathlib import Path
import os,json,time,datetime,subprocess,signal,traceback
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});j=json.loads((B/'JOB.json').read_text());s=dict(status='REGISTERED',start=time.time());p=None

def ident(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
def alive(x):
 try:return ident(x['pid'])==x
 except (FileNotFoundError,ProcessLookupError):return False
def save():(B/'STATUS.json').write_text(json.dumps(s,indent=2)+'\n')
try:
 assert not (B/'STATUS.json').exists()
 s['owned_guard']=ident(os.getpid());s['status']='WAITING_NATURAL_IDLE';save()
 cutoff=datetime.datetime.fromisoformat(j['latest_worker_start']).timestamp();empty_streak=0
 while time.time()<cutoff:
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  util=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
  s.update(last_check=time.time(),admission=apps,gpu_snapshot=util);save()
  values=[int(x.strip()) for x in util.split(',')];idle=not apps.strip() and values[0]<128 and values[1]<=5
  empty_streak=empty_streak+1 if idle else 0
  if empty_streak>=2:break
  time.sleep(5)
 else:
  s.update(status='EXPIRED_WITHOUT_GPU_WINDOW',GPU_lifecycle_seconds=0,worker_GPU_seconds=0,job_lifecycle_charged_seconds=time.time()-s['start']);save();raise SystemExit(0)
 assert time.time()+750<datetime.datetime.fromisoformat(j['deadline']).timestamp()
 for n in ['cache','tmp']:(B/n).mkdir(exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=j['GPU'],PYTHONPATH='/rjs/cxz/论文探索/算子研究',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'),CUDA_CACHE_PATH=str(B/'cache/cuda'))
 with (B/'worker.log').open('x') as log:p=subprocess.Popen(j['command'],cwd=B,env=env,stdout=log,stderr=log,start_new_session=True)
 s.update(status='RUNNING_NCU',owned=ident(p.pid),GPU_start=time.time());save()
 try:
  limit=time.time()+420
  while p.poll() is None and time.time()<limit:
   time.sleep(1)
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
   foreign=[]
   for x in apps.splitlines():
    if x.strip().isdigit():
     try:
      if os.getpgid(int(x))!=s['owned']['pgid']:foreign.append(int(x))
     except ProcessLookupError:pass
   if foreign:
    s['foreign_processes']=foreign
    if alive(s['owned']):os.killpg(s['owned']['pgid'],signal.SIGTERM)
    break
  rc=p.wait(timeout=max(1,limit-time.time()))
 except subprocess.TimeoutExpired:
  s['timeout']=True
  if alive(s['owned']):os.killpg(s['owned']['pgid'],signal.SIGTERM)
  try:rc=p.wait(timeout=15)
  except subprocess.TimeoutExpired:
   if alive(s['owned']):os.killpg(s['owned']['pgid'],signal.SIGKILL)
   rc=p.wait(timeout=15)
 s.update(status='COMPLETE' if rc==0 and not s.get('timeout') and not s.get('foreign_processes') else 'FAILED_OR_PARTIAL',returncode=rc,root_reaped=True,identity_alive=alive(s['owned']),GPU_lifecycle_seconds=time.time()-s['GPU_start'])
except Exception as e:s.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:s['end']=time.time();s['job_lifecycle_charged_seconds']=time.time()-s['start'];s['worker_GPU_seconds']=s.get('GPU_lifecycle_seconds',0);save()

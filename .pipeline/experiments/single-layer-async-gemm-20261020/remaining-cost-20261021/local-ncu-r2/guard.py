from pathlib import Path
import json,os,time,subprocess,datetime,signal,traceback
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});j=json.loads((B/'JOB.json').read_text());assert not (B/'STATUS.json').exists();start=time.time();s={'status':'PREPARING','start':start};p=None

def save():(B/'STATUS.json').write_text(json.dumps(s,indent=2)+'\n')
def ident(pid):return {'pid':pid,'birth_ticks':int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),'pgid':os.getpgid(pid)}
def alive(x):
 try:return ident(x['pid'])==x
 except (FileNotFoundError,ProcessLookupError):return False
try:
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);s['load_before']=apps;assert not apps.strip(),'local GPU compute context appeared'
 assert time.time()+660<datetime.datetime.fromisoformat(j['deadline']).timestamp()
 for n in ['cache','tmp']:(B/n).mkdir(exist_ok=True)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':j['GPU'],'PYTHONPATH':'/rjs/cxz/论文探索/算子研究','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(B/'tmp'),'XDG_CACHE_HOME':str(B/'cache'),'CUDA_CACHE_PATH':str(B/'cache/cuda')}
 s['status']='RUNNING_NCU';save()
 with (B/'worker.log').open('x') as f:p=subprocess.Popen(j['steps'][0][1],cwd=B,env=env,stdout=f,stderr=f,start_new_session=True)
 s.update(owned=ident(p.pid),GPU_start=time.time());save()
 try:p.wait(timeout=600)
 except subprocess.TimeoutExpired:
  s['timeout']=True
  if alive(s['owned']):os.killpg(s['owned']['pgid'],signal.SIGTERM)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:
   if alive(s['owned']):os.killpg(s['owned']['pgid'],signal.SIGKILL)
   p.wait(timeout=15)
 s.update(status='COMPLETE' if p.returncode==0 and not s.get('timeout') else 'FAILED_OR_PARTIAL',returncode=p.returncode,root_reaped=True,identity_alive=alive(s['owned']),GPU_lifecycle_seconds=time.time()-s['GPU_start'])
except BaseException as e:s.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:s.update(end=time.time(),wall_seconds=time.time()-start);save()

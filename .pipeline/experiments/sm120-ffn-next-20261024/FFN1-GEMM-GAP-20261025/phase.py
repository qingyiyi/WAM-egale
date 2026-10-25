from pathlib import Path
from datetime import datetime
import os,sys,json,time,subprocess,signal
P=Path(__file__).resolve().parent
job=json.loads((P/'JOB.json').read_text());deadline=datetime.fromisoformat(job['absolute_cleanup_deadline']).timestamp()
name,cap=sys.argv[1],int(sys.argv[2]);cmd=sys.argv[3:]
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
R=P.parent.parent; env=os.environ.copy(); V=Path('/root/autodl-tmp/cxz/envs/wam-sm120'); cu=V/'lib/python3.12/site-packages/nvidia/cu13'
(P/'tmp').mkdir(exist_ok=True)
env.update(TMPDIR=str(P/'tmp'),TMP=str(P/'tmp'),TEMP=str(P/'tmp'),CUDA_HOME=str(cu),PATH=str(cu/'bin')+':'+str(V/'bin')+':'+env['PATH'],PYTHONPATH=str(R/'workspace/算子研究')+':'+str(R/'workspace/算子研究/wam_ops/optimized'),WAM_FA2_BUILD_ROOT=str(R/'build/generated-normal'),WAM_REUSE_GENERATED='1',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='4',OMP_NUM_THREADS='4')
def identity(pid):
 return dict(pid=pid,birth=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
def same(o):
 try:return identity(o['pid'])==o
 except FileNotFoundError:return False
q=dict(name=name,start=time.time(),command=cmd,guard=identity(os.getpid()),status='RUNNING')
def save():(P/(name+'-STATUS.json')).write_text(json.dumps(q,indent=2))
try:
 if time.time()>deadline-job.get('cleanup_reserve_seconds',300):raise TimeoutError('cleanup reserve')
 with (P/(name+'.log')).open('w') as f:
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);q['owned']=identity(p.pid);save()
  try:q['returncode']=p.wait(timeout=min(cap,deadline-time.time()-job.get('cleanup_reserve_seconds',300)));q['status']='EXITED_OK' if p.returncode==0 else 'FAILED'
  except subprocess.TimeoutExpired:
   q['status']='TIMEOUT'
   if same(q['owned']):os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:
    if same(q['owned']):os.killpg(p.pid,signal.SIGKILL)
    p.wait(timeout=8)
  q['child_exited']=not same(q['owned'])
except BaseException as e:q.update(status='FAILED',error=repr(e))
finally:q.update(end=time.time(),lifecycle_seconds=time.time()-q['start']);save();print(json.dumps(q),flush=True)
if q['status']!='EXITED_OK':sys.exit(1)

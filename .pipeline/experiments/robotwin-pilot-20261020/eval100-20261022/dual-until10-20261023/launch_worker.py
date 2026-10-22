"""Launch a new worker once; GPU0 waits for the explicit completed-operator handoff."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;index=int(sys.argv[1]);J=D/f'gpu{index}';os.sched_setaffinity(0,{18 if index==0 else 26});start=time.time();deadline=datetime.datetime.fromisoformat('2026-10-23T03:00:00+08:00').timestamp()
state=dict(status='WAITING_OPERATOR_HANDOFF' if index==0 else 'CHECKING_LEGACY_DRAIN',start=start,deadline=deadline,GPU_used_by_launcher=False,pid=os.getpid(),birth_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]))
def save():
 p=J/'LAUNCHER-STATUS.json';p.with_suffix('.tmp').write_text(json.dumps(state,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
def identity(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
try:
 assert not (J/'LAUNCHER-STATUS.json').exists();assert not (J/'STATUS.json').exists();save()
 assert json.loads((D/'INITIALIZATION.json').read_text())['status']=='READY_SHARED_FIXED_PAIRS_NO_GPU_STARTED'
 if index==0:
  while not (D/'OPERATOR-HANDOFF.json').exists():
   if json.loads((D/'PAIR-CLAIMS.json').read_text()).get('halt'):raise RuntimeError('Global campaign halt before GPU0 launch')
   if time.time()>=deadline:raise TimeoutError('No operator handoff within finite launcher window')
   time.sleep(5)
  h=json.loads((D/'OPERATOR-HANDOFF.json').read_text());assert h['operator_owned_cleaned'] and h['GPU0_empty']
 else:
  old=json.loads((D.parent/'gpu1/STATUS.json').read_text());old_id=old['owned_guard']
  until=time.time()+60
  while True:
   p=Path(f"/proc/{old_id['pid']}/stat")
   if not p.exists():break
   fields=p.read_text().rsplit(')',1)[1].split()
   if int(fields[19])!=old_id['birth_ticks'] or fields[0]=='Z':break
   if time.time()>=until:raise TimeoutError('Legacy guard identity not fully exited')
   time.sleep(1)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
 with (J/'guard.log').open('x') as log:guard=subprocess.Popen(['/root/cxz/envs/va/bin/python','-B',str(J/'guard.py')],cwd=D,env=env,stdout=log,stderr=log,start_new_session=True)
 state.update(status='GUARD_LAUNCHED',guard=identity(guard.pid));save()
 until=time.time()+30
 while not (J/'STATUS.json').exists():
  if guard.poll() is not None:raise RuntimeError('Guard exited before status')
  if time.time()>=until:raise TimeoutError('Guard did not create status')
  time.sleep(1)
 with (J/'archiver.log').open('x') as log:archive=subprocess.Popen(['/root/cxz/envs/va/bin/python','-B',str(J/'archive_completed.py')],cwd=D,env=env,stdout=log,stderr=log,start_new_session=True)
 state.update(status='COMPLETE_WORKER_AND_ARCHIVER_LAUNCHED',archiver=identity(archive.pid))
except BaseException as e:state.update(status='FAILED_LAUNCH_NO_AUTOMATIC_RETRY',error=repr(e),traceback=traceback.format_exc())
finally:state['end']=time.time();save()

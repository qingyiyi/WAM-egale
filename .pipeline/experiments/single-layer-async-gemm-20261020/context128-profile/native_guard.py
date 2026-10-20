from pathlib import Path
import os,subprocess,json,time,datetime,signal,traceback
B=Path(__file__).resolve().parent;reg=json.loads((B/'REGISTRATION.json').read_text());R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究');uuid=reg['GPU'];os.sched_setaffinity(0,{4,5})
s={'status':'RUNNING','start':time.time(),'full_block_cap':12};p=None
try:
 assert json.loads((B/'RESULT.json').read_text())['full_block_calls']==12
 ctx=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(x.startswith(uuid) for x in ctx.splitlines())
 cap=min(180,int(datetime.datetime.fromisoformat(reg['deadline']).timestamp()-time.time()-60));assert cap>0
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R),TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'))
 with (B/'native-profile.log').open('x') as log:p=subprocess.Popen(['/root/cxz/envs/va/bin/python','-B',str(B/'native_profile.py')],cwd=B,env=env,stdout=log,stderr=log,start_new_session=True)
 birth=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]);s.update(pid=p.pid,birth_ticks=birth,cap_seconds=cap)
 (B/'NATIVE-PROFILE-STATUS.json').write_text(json.dumps(s,indent=2)+'\n')
 try:rc=p.wait(timeout=cap)
 except subprocess.TimeoutExpired:
  s['timeout']=True
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  try:rc=p.wait(timeout=15)
  except subprocess.TimeoutExpired:
   if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
   rc=p.wait(timeout=15)
 s.update(returncode=rc,root_reaped=True);assert rc==0 and not s.get('timeout')
 d=json.loads((B/'NATIVE-PROFILE-RESULT.json').read_text());assert d['status']=='COMPLETE_NATIVE_PROFILE_FINITE_OUTPUT_CHECK' and d['full_block_calls']==12;s['status']='COMPLETE'
except BaseException as e:s.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:s.update(end=time.time(),lifecycle_seconds=time.time()-s['start']);(B/'NATIVE-PROFILE-STATUS.json').write_text(json.dumps(s,indent=2)+'\n')

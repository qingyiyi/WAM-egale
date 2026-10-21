from pathlib import Path
import json,time,subprocess,datetime,os
B=Path(__file__).resolve().parent;os.sched_setaffinity(0,{4,5});j=json.loads((B/'JOB.json').read_text());start=time.time();latest=datetime.datetime.fromisoformat(j['deadline']).timestamp()-540
out={'status':'WAITING_FOR_NATURAL_IDLE','start':start,'latest_admission_epoch':latest,'owned':{'pid':os.getpid(),'birth_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])}};quiet=0
while time.time()<latest:
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).strip()
 util=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
 out.update(observed=time.time(),compute_apps=apps,utilization=util);(B/'WATCH-STATUS.json').write_text(json.dumps(out,indent=2)+'\n')
 quiet=quiet+1 if not apps and util<=5 else 0
 if quiet>=2:
  admitted=time.time();(B/'WINDOW-ADMISSION.json').write_text(json.dumps({'authorization':'User explicitly says natural idle start','admitted_epoch':admitted,'window_end_epoch':min(admitted+600,datetime.datetime.fromisoformat(j['deadline']).timestamp()),'GPU_apps_empty':True,'utilization':util},indent=2)+'\n')
  out.update(status='LAUNCHING_ON_NATURAL_IDLE',wait_seconds=admitted-start);(B/'WATCH-STATUS.json').write_text(json.dumps(out,indent=2)+'\n')
  with (B/'GUARD.log').open('x') as f:p=subprocess.run(['python',str(B/'guard.py')],stdout=f,stderr=f)
  out.update(status='GUARD_RETURNED',guard_returncode=p.returncode,end=time.time());break
 time.sleep(5)
else:out.update(status='NO_IDLE_WINDOW_BEFORE_ADMISSION_DEADLINE',end=time.time(),wait_seconds=time.time()-start)
(B/'WATCH-STATUS.json').write_text(json.dumps(out,indent=2)+'\n')

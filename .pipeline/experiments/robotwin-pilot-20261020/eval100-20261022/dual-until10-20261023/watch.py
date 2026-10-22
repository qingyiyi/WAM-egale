"""Read-only five-minute health/merged progress; no retries and no GPU work."""
from pathlib import Path
import datetime,json,os,time,traceback,subprocess
D=Path(__file__).resolve().parent;os.sched_setaffinity(0,{18});deadline=datetime.datetime.fromisoformat('2026-11-01T22:18:22.728607+08:00').timestamp();start=time.time();state=dict(status='RUNNING_CPU_ONLY_WATCH',start=start,pid=os.getpid(),birth_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),GPU_used=False,polls=0)
def save(name,d):
 p=D/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def live(o):
 if not o:return None
 try:
  f=Path(f"/proc/{o['pid']}/stat").read_text().rsplit(')',1)[1].split();return int(f[19])==o['birth_ticks'] and f[0]!='Z'
 except FileNotFoundError:return False
try:
 assert not (D/'WATCH-STATUS.json').exists();save('WATCH-STATUS.json',state)
 while time.time()<deadline:
  claims=json.loads((D/'PAIR-CLAIMS.json').read_text());episodes=[e for c in claims['pairs'].values() for e in c.get('episodes',[])];workers={};issues=[]
  for i in [0,1]:
   J=D/f'gpu{i}';p=J/'STATUS.json';l=json.loads((J/'LAUNCHER-STATUS.json').read_text()) if (J/'LAUNCHER-STATUS.json').exists() else {}
   if p.exists():
    d=json.loads(p.read_text());workers[str(i)]=dict(status=d['status'],active_case=d.get('active_case'),completed=len(d['episodes']),teacher_attempted=d.get('teacher_attempted'),guard_alive=live(d.get('owned_guard')),heartbeat_age_seconds=time.time()-p.stat().st_mtime,error=d.get('error'),end=d.get('end'))
    if 'end' not in d and (not workers[str(i)]['guard_alive'] or workers[str(i)]['heartbeat_age_seconds']>600):issues.append(f'GPU{i} guard/heartbeat requires review')
   else:workers[str(i)]=dict(status=l.get('status','NOT_STARTED'),launcher_alive=live(l))
  ledger=json.loads((D.parent/'GPU-LEDGER.json').read_text());spent=ledger['consumed_seconds']+sum(time.time()-x['start'] for x in ledger.get('active_jobs',{}).values())
  counts={a:dict(complete=sum(e['arm']==a for e in episodes),success=sum(e['arm']==a and bool(e['success']) for e in episodes)) for a in ['T','C','P']}
  result=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),completed=len(episodes),total=1800,complete_pairs=sum(p['status']=='COMPLETE_PAIR' for p in claims['pairs'].values()),counts=counts,workers=workers,halt=claims.get('halt'),issues=issues,GPU_lifecycle_seconds_including_active=spent,GPU_cap_seconds=576000,soft_stops=claims.get('soft_stops',{}),not_independent_TEST=True,formal_performance_permission=False,engineering_quality_screen=False,noninferiority_established=False,independent_test_complete=False)
  save('LATEST.json',result)
  with (D/'HISTORY.jsonl').open('a') as f:f.write(json.dumps({k:result[k] for k in ['checked_at','completed','complete_pairs','counts','GPU_lifecycle_seconds_including_active','issues']})+'\n')
  state.update(polls=state['polls']+1,last_check=result['checked_at'],completed=len(episodes));save('WATCH-STATUS.json',state)
  all_terminal=all(w.get('end') for w in workers.values())
  if all_terminal:state['status']='COMPLETE_WATCH_ALL_WORKERS_TERMINAL';break
  time.sleep(300)
 else:state['status']='WATCH_ABSOLUTE_DEADLINE'
except BaseException as e:state.update(status='FAILED_CPU_WATCH_NO_RESTART',error=repr(e),traceback=traceback.format_exc())
finally:state['end']=time.time();save('WATCH-STATUS.json',state)

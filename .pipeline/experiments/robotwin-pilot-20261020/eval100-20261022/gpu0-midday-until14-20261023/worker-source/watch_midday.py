"""Finite CPU-only watch for the newly authorized midday GPU0 namespace."""
from pathlib import Path
import datetime,json,os,time,traceback
J=Path(__file__).resolve().parent;D=J.parent;R=D.parent;os.sched_setaffinity(0,{18})
end=datetime.datetime.fromisoformat('2026-10-23T15:50:00+08:00').timestamp()
def save(n,d):
 f=J/n;t=f.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(f)
def live(o):
 if not o:return None
 try:
  f=Path(f"/proc/{o['pid']}/stat").read_text().rsplit(')',1)[1].split();return int(f[19])==o['birth_ticks'] and f[0]!='Z'
 except FileNotFoundError:return False
s=dict(status='RUNNING_CPU_ONLY_MIDDAY_WATCH',start=time.time(),pid=os.getpid(),birth_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),GPU_used=False,polls=0)
try:
 assert not (J/'WATCH-STATUS.json').exists();save('WATCH-STATUS.json',s)
 while time.time()<end:
  q=json.loads((D/'PAIR-CLAIMS.json').read_text());es=[e for p in q['pairs'].values() for e in p.get('episodes',[])];workers={};issues=[]
  for name in ['gpu0','gpu0-until14-20261023','gpu1']:
   f=D/name/'STATUS.json'
   if not f.exists():continue
   a=json.loads(f.read_text());workers[name]={k:a.get(k) for k in ['status','active_case','active_pair','end','error']};workers[name].update(completed=len(a['episodes']),guard_alive=live(a.get('owned_guard')),heartbeat_age_seconds=time.time()-f.stat().st_mtime)
   if not a.get('end') and (not workers[name]['guard_alive'] or workers[name]['heartbeat_age_seconds']>600):issues.append(name+' needs review')
  l=json.loads((R/'GPU-LEDGER.json').read_text());v=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),completed=len(es),planned=1800,complete_pairs=sum(p['status']=='COMPLETE_PAIR' for p in q['pairs'].values()),counts={a:dict(completed=sum(e['arm']==a for e in es),success=sum(e['arm']==a and bool(e['success']) for e in es)) for a in ['T','C','P']},workers=workers,halt=q.get('halt'),issues=issues,GPU_lifecycle_seconds_including_active=l['consumed_seconds']+sum(time.time()-x['start'] for x in l.get('active_jobs',{}).values()),GPU_cap_seconds=576000,formal_performance_permission=False,independent_test_complete=False)
  save('LATEST.json',v)
  with (J/'WATCH-HISTORY.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
  s.update(polls=s['polls']+1,last_check=v['checked_at'],completed=len(es));save('WATCH-STATUS.json',s)
  if workers.get(J.name,{}).get('end'):s['status']='COMPLETE_MIDDAY_WORKER_TERMINAL';break
  time.sleep(300)
 else:s['status']='WATCH_ABSOLUTE_DEADLINE'
except BaseException as e:s.update(status='FAILED_CPU_WATCH_NO_RESTART',error=repr(e),traceback=traceback.format_exc())
finally:s['end']=time.time();save('WATCH-STATUS.json',s)

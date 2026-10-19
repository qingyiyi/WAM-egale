"""Bounded low-duty CPU SSH watch; never restarts or signals remote experiments."""
from pathlib import Path
import json,os,sys,subprocess,time,signal
from datetime import datetime,timezone
JOB=Path(sys.argv[1]).resolve();PLAN=json.loads((JOB/'REGISTRATION.json').read_text());ROOT=Path(__file__).resolve().parents[3]
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
def write(name,d):
 p=JOB/name;t=p.with_suffix('.pending');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
p=Path('/proc')/str(os.getpid());f=(p/'stat').read_text().rsplit(')',1)[1].split();write('OWNED-PROCESS.json',dict(pid=os.getpid(),birth_ticks=int(f[19]),uid=p.stat().st_uid,pgid=int(f[2]),role='CPU_only_readonly_health_watch'))
c=json.loads(Path(PLAN['existing_connection_config']).read_text());argv=['ssh','-T','-p',str(c['port']),'-i',c['identity_file'],'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile='+c['known_hosts'],c['host'],'python3 -']
code=(Path(__file__).with_name('remote_probe.py')).read_text();started=time.time();polls=0;errors=0;terminal='ABSOLUTE_DEADLINE_REACHED'
def stop(*_):raise InterruptedError('user stops owned CPU watch')
signal.signal(signal.SIGTERM,stop)
try:
 while time.time()<PLAN['deadline_epoch'] and not (JOB/'STOP_REQUESTED').exists():
  tick=time.time();polls+=1
  try:
   v=subprocess.run(argv,input=code,capture_output=True,text=True,timeout=60);assert v.returncode==0,v.stderr[-1000:]
   d=json.loads(v.stdout);d.update(polled_at=datetime.now(timezone.utc).isoformat(),poll_number=polls);write('LATEST.json',d)
   summary=dict(at=d['polled_at'],groups={k:x['complete_common_case_groups'] for k,x in d['suites'].items()},episodes={k:x['unique_complete_episodes'] for k,x in d['suites'].items()},object_status=d['object']['status'],controllers={**{k:x['controller']['match'] for k,x in d['suites'].items()},'object':d['object']['controller']['match']})
   with (JOB/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(summary,ensure_ascii=False)+'\n')
   write('HEARTBEAT.json',dict(status='RUNNING_CPU_READONLY_WATCH',polls=polls,errors=errors,last_success_at=d['polled_at'],next_poll_epoch=tick+PLAN['interval_seconds']))
   if all(x['complete_common_case_groups']==500 for x in d['suites'].values()) and d['object']['terminal']=='COMPLETE_DESCRIPTIVE_REMOTE_OBJECT500':terminal='ALL_THREE_REMOTE_QUEUES_COMPLETE';break
  except Exception as e:
   errors+=1;write('LAST-ERROR.json',dict(at=datetime.now(timezone.utc).isoformat(),error=repr(e),poll=polls));write('HEARTBEAT.json',dict(status='REMOTE_CHECK_FAILED_WILL_RECHECK_NO_RESTART',polls=polls,errors=errors))
  assert sum(p.stat().st_size for p in JOB.rglob('*') if p.is_file())<PLAN['storage_cap_bytes']
  while time.time()<min(tick+PLAN['interval_seconds'],PLAN['deadline_epoch']) and not (JOB/'STOP_REQUESTED').exists():time.sleep(min(5,max(.1,tick+PLAN['interval_seconds']-time.time())))
 if (JOB/'STOP_REQUESTED').exists():terminal='STOPPED_AT_USER_REQUEST'
except BaseException as e:terminal='WATCH_STOPPED_WITH_ERROR';write('WATCH-ERROR.json',dict(error=repr(e)))
finally:write('TERMINAL.json',dict(status=terminal,polls=polls,errors=errors,wall_seconds=time.time()-started,remote_mutations=False,remote_signals_sent=False,GPU_workers=0))

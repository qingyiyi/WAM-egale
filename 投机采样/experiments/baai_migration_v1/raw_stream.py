"""Bounded uncompressed transfer to ~/cxz only; no remote hashes, tests, models or root aliases."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os
from pathlib import Path
import shlex,signal,subprocess,threading,time,traceback

REMOTE=r"""
from pathlib import Path
import json,os,shutil,signal,subprocess,sys,time
p=json.loads(sys.argv[1]);root=(Path.home()/'cxz').resolve();dest=Path(p['destination']);job=root/'jobs/baai-migration-20261017-r2';job.mkdir(parents=True,exist_ok=True)
assert Path.cwd().resolve()==root and dest.is_absolute() and dest.is_relative_to(root)
assert time.time()<p['deadline_epoch'];os.sched_setaffinity(0,{p['remote_CPU']})
def wr(n,v):
 with (job/(p['name']+'-'+n+'.json')).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def pin(pid):
 b=Path('/proc')/str(pid);v=(b/'stat').read_text().rpartition(') ')[2].split();return dict(pid=pid,birth_ticks=int(v[19]),pgid=int(v[2]),uid=b.stat().st_uid)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original six-hour migration cap')));signal.alarm(max(1,int(p['deadline_epoch']-time.time())))
assert shutil.disk_usage(root).free>20*2**30;dest.mkdir(parents=True,exist_ok=True)
wr('STARTED',dict(own=pin(os.getpid()),at=time.time(),destination=str(dest),CPU=p['remote_CPU'],GPU=0,deadline_epoch=p['deadline_epoch']))
child=None;result=None
try:
 child=subprocess.Popen(['/bin/tar','-xpf','-','-C',str(dest),'--no-same-owner'],stdin=sys.stdin.buffer,env=dict(os.environ,TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
 cp=pin(child.pid);wr('OWNED-EXTRACTOR',cp);rc=child.wait(timeout=max(1,p['deadline_epoch']-time.time()-10));assert rc==0,'actual tar extraction failed'
 total=0
 for entry in p['entries']:
  target=dest/entry;assert target.exists() or target.is_symlink(),entry
  if target.is_file():total+=target.stat().st_size
 result=dict(status='UNCOMPRESSED_STREAM_EXTRACTED',tar_exit=rc,entries=p['entries'],destination=str(dest),verified_top_entries=True,independent_byte_identity='UNKNOWN_NO_SHA',no_GPU=True,no_external_workspace_write=True)
 wr('RESULT',result);print(json.dumps(result),flush=True)
except BaseException as exc:wr('FAILED',dict(error=repr(exc),no_retry=True));raise
finally:
 signal.alarm(0)
 if child is not None and child.poll() is None:
  try:
   now=pin(child.pid)
   if all(now[k]==cp[k] for k in ('birth_ticks','pgid','uid')):os.killpg(cp['pgid'],signal.SIGTERM)
   child.wait(timeout=10)
  except (ProcessLookupError,subprocess.TimeoutExpired):pass
 wr('CLEANUP',dict(child_exit=child.poll() if child else None,owned_extractor_reaped=child is None or child.poll() is not None,source_deleted=False,no_foreign_signal=True))
"""

def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)

def main():
 a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--manifest',type=Path,required=True);x=a.parse_args();job=x.job.resolve();p=json.loads(x.manifest.read_text());assert not (job/'STREAM-STARTED.json').exists()
 assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2
 write(job/'STREAM-STARTED.json',dict(pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),at=time.time(),deadline_epoch=p['deadline_epoch'],GPU=0))
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original six hour staging deadline')));signal.alarm(max(1,int(p['deadline_epoch']-time.time())))
 abort=threading.Event();lock=threading.Lock();live=[];results=[]
 def one(spec):
  name=spec['name'];start=time.time();sent=0;last=0;tar=None;ssh=None
  assert spec['destination'].startswith('/root/cxz/') and time.time()<p['deadline_epoch']
  rp=dict(spec,deadline_epoch=p['deadline_epoch']);remote='cd "$HOME/cxz" && '+shlex.join(['/usr/bin/python3','-B','-c',REMOTE,json.dumps(rp,ensure_ascii=False)])
  sshargs=['ssh','-T','-x','-a','-i',p['private_key_local_only'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ConnectionAttempts=1','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+p['known_hosts'],'-p','2222',p['user_host'],remote]
  excludes=['--exclude=__pycache__','--exclude=*.pyc','--exclude=.git']+['--exclude='+v for v in spec.get('excludes',[])]
  args=['/bin/tar','--format=posix',*excludes,'-C',spec['base'],'-cf','-',*spec['entries']]
  with (job/(name+'-tar.log')).open('xb') as tl,(job/(name+'-ssh.log')).open('xb') as sl:
   try:
    tar=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=tl,start_new_session=True)
    ssh=subprocess.Popen(sshargs,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sl,start_new_session=True)
    with lock:
     live.extend([tar,ssh]);write(job/(name+'-LOCAL-OWNED.json'),dict(processes=[dict(pid=c.pid,birth_ticks=int(Path(f'/proc/{c.pid}/stat').read_text().rpartition(') ')[2].split()[19]),pgid=os.getpgid(c.pid),uid=os.getuid()) for c in (tar,ssh)]))
    while True:
     if abort.is_set():raise RuntimeError('another declared transfer failed; preserve partial, no retry')
     if time.time()>=p['deadline_epoch']:raise TimeoutError('original migration deadline')
     chunk=tar.stdout.read(2**20)
     if not chunk:break
     ssh.stdin.write(chunk);sent+=len(chunk)
     if time.time()-last>=20:
      write(job/(name+'-STATUS.json'),dict(status='STREAMING',transmitted_bytes=sent,wall_sec=time.time()-start));print(json.dumps(dict(stage=name,transmitted_MiB=round(sent/2**20,1),wall_sec=round(time.time()-start,1))),flush=True);last=time.time()
    ssh.stdin.close();trc=tar.wait(timeout=30);raw=ssh.stdout.read(65537);src=ssh.wait(timeout=max(1,p['deadline_epoch']-time.time()))
    assert trc==src==0,(trc,src,name)
    value=json.loads(raw);assert value['status']=='UNCOMPRESSED_STREAM_EXTRACTED'
    result=dict(name=name,status='ACTUAL_STREAM_EXIT0',transmitted_bytes=sent,wall_sec=time.time()-start,remote=value)
    write(job/(name+'-RESULT.json'),result);print(json.dumps(result,ensure_ascii=False),flush=True);return result
   except BaseException as exc:write(job/(name+'-FAILED.json'),dict(error=repr(exc),transmitted_bytes=sent,wall_sec=time.time()-start));abort.set();raise
   finally:
    for c in (tar,ssh):
     if c is not None and c.poll() is None:
      os.killpg(c.pid,signal.SIGTERM)
      try:c.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    write(job/(name+'-CLEANUP.json'),dict(children=[dict(pid=c.pid,exit=c.poll()) for c in (tar,ssh) if c is not None],source_deleted=False))
 try:
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(one,s) for s in p['streams']]
   for future in as_completed(futures):results.append(future.result())
  write(job/'STREAM-RESULT.json',dict(status='ALL_DECLARED_STREAMS_EXTRACTED_NO_MODEL_EXECUTION',results=results,source_deleted=False));return 0
 except BaseException as exc:
  abort.set();write(job/'STREAM-FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed=results));raise
 finally:signal.alarm(0);write(job/'STREAM-CLEANUP.json',dict(children=[dict(pid=c.pid,exit=c.poll()) for c in live],no_GPU=True))

if __name__=='__main__':raise SystemExit(main())

"""New-host direct static-asset transfer; password supplied on stdin, never stored."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,json,os
from pathlib import Path
import shlex,signal,socket,subprocess,sys,threading,time,traceback

OLD=r'''
from pathlib import Path
import json,os,signal,subprocess,sys,time
r=json.loads(sys.argv[1]);app=Path('/root/autodl-tmp/wam-tf-pf2-20261017');meta=app/'投机采样/results/runtime/p0-baai-static-export-20261017-r1';meta.mkdir(exist_ok=True)
os.sched_setaffinity(0,{r['CPU']});end=r['deadline_epoch']
def wr(name,value):
 with (meta/(r['name']+'-'+name+'.json')).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def pin(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rpartition(') ')[2].split();return dict(pid=pid,birth_ticks=int(v[19]),pgid=int(v[2]),uid=p.stat().st_uid)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original migration absolute deadline')));signal.alarm(max(1,int(end-time.time())))
wr('STARTED',dict(pin=pin(os.getpid()),deadline_epoch=end,CPU=r['CPU'],GPU=0))
args=['/bin/tar','--format=posix','--exclude=__pycache__','--exclude=*.pyc','--exclude=.git','-I','gzip -1',
 '--transform=s,^wam-tf-pf2-20261017/,WAM-egale/,','--transform=s,^wam-suite-assets-20261017-v1/,assets/community/,','-cf','-','-C','/root/autodl-tmp']
if r['name']=='envs':args+=['wam-tf-pf2-20261017/envs']
else:
 args+=['wam-tf-pf2-20261017/'+s for s in ['models','third_party','userlib','投机采样/experiments','投机采样/scripts','投机采样/configs','投机采样/wam-p0-worktree']]
 args+=['-C','/root','wam-suite-assets-20261017-v1']
child=None
try:
 child=subprocess.Popen(args,stdout=sys.stdout.buffer,stderr=sys.stderr.buffer,start_new_session=True);cp=pin(child.pid);wr('OWNED-TAR',cp)
 rc=child.wait(timeout=max(1,end-time.time()-20));assert rc==0,('actual static tar failed',rc)
 wr('RESULT',dict(status='STATIC_SOURCE_TAR_EXIT0',exit=rc,source_deleted=False,no_SHA=True))
except BaseException as e:wr('FAILED',dict(error=repr(e),source_deleted=False));raise
finally:
 signal.alarm(0)
 if child is not None and child.poll() is None:
  try:
   q=pin(child.pid)
   if all(q[k]==cp[k] for k in ('birth_ticks','uid','pgid')):os.killpg(cp['pgid'],signal.SIGTERM)
   child.wait(timeout=10)
  except (ProcessLookupError,subprocess.TimeoutExpired):pass
 wr('CLEANUP',dict(tar_exit=child.poll() if child else None,source_deleted=False))
'''

def write(path,value):
 tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)

def main():
 a=argparse.ArgumentParser();a.add_argument('--registration',required=True,type=Path);x=a.parse_args();r=json.loads(x.registration.read_text())
 root=(Path.home()/'cxz').resolve();assert Path.cwd().resolve()==root
 job=root/'jobs/baai-direct-migration-20261017-r1';job.mkdir(mode=0o700);assert time.time()<r['deadline_epoch']
 os.sched_setaffinity(0,{40,41,42,43});os.environ['CUDA_VISIBLE_DEVICES']='';secret=sys.stdin.buffer.readline(512).rstrip(b'\n');assert secret
 write(job/'REGISTRATION.json',r);write(job/'STARTED.json',dict(pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),at=time.time(),GPU=0,CPU=[40,41,42,43]))
 sockpath=job/'askpass.sock';server=socket.socket(socket.AF_UNIX);server.bind(str(sockpath));os.chmod(sockpath,0o600);server.listen(4);server.settimeout(.5)
 helper=job/'askpass.py';helper.write_text('#!/usr/bin/python3\nimport os,socket,sys\ns=socket.socket(socket.AF_UNIX);s.connect(os.environ["WAM_MIGRATION_AUTH_SOCKET"]);sys.stdout.buffer.write(s.recv(512));s.close()\n');helper.chmod(0o700)
 stop=threading.Event();abort=threading.Event();requests=[];children=[];lock=threading.Lock()
 def authentication():
  while not stop.is_set():
   try:c,_=server.accept()
   except socket.timeout:continue
   try:
    requests.append(time.time());assert len(requests)<=2;c.sendall(secret+b'\n')
   finally:c.close()
 auth=threading.Thread(target=authentication,daemon=True);auth.start()
 def phase(name,cpu):
  start=time.time();dest=root/'direct-r1';dest.mkdir(exist_ok=True)
  sshcmd=['ssh','-T','-x','-a','-o','PubkeyAuthentication=no','-o','PreferredAuthentications=password','-o','NumberOfPasswordPrompts=1','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(root/'jobs/AUTODL-HOST-KEYS'),'-p','40416','root@region-9.autodl.pro',shlex.join(['/root/miniconda3/bin/python','-B','-c',OLD,json.dumps(dict(name=name,CPU=cpu,deadline_epoch=r['deadline_epoch']))])]
  env=dict(os.environ,SSH_ASKPASS=str(helper),SSH_ASKPASS_REQUIRE='force',DISPLAY='migration-internal',WAM_MIGRATION_AUTH_SOCKET=str(sockpath),TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1')
  ssh=None;tar=None;total=0;last=0
  with (job/(name+'-ssh.stderr')).open('xb') as sl,(job/(name+'-tar.stderr')).open('xb') as tl:
   try:
    ssh=subprocess.Popen(sshcmd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=sl,env=env,start_new_session=True)
    tar=subprocess.Popen(['/bin/tar','-xzpf','-','-C',str(dest),'--no-same-owner'],stdin=subprocess.PIPE,stderr=tl,env=env,start_new_session=True)
    with lock:
     children.extend([ssh,tar]);write(job/(name+'-OWNED.json'),dict(processes=[dict(pid=p.pid,birth_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rpartition(') ')[2].split()[19]),pgid=os.getpgid(p.pid),uid=os.getuid()) for p in (ssh,tar)]))
    while True:
     assert time.time()<r['deadline_epoch'] and not abort.is_set(),'bounded transfer abort/deadline'
     data=ssh.stdout.read(2**20)
     if not data:break
     tar.stdin.write(data);total+=len(data)
     if time.time()-last>=20:
      v=dict(stage=name,status='ACTUAL_STATIC_COPY',compressed_bytes=total,wall_sec=time.time()-start);write(job/(name+'-STATUS.json'),v);print(json.dumps(v),flush=True);last=time.time()
    tar.stdin.close();trc=tar.wait(timeout=max(1,r['deadline_epoch']-time.time()));src=ssh.wait(timeout=30);assert trc==src==0,(name,trc,src)
    v=dict(status='STATIC_SOURCE_AND_EXTRACTION_EXIT0',stage=name,compressed_bytes=total,wall_sec=time.time()-start,source_deleted=False,no_SHA=True);write(job/(name+'-RESULT.json'),v);print(json.dumps(v),flush=True);return v
   except BaseException as e:
    abort.set();write(job/(name+'-FAILED.json'),dict(error=repr(e),compressed_bytes=total,source_deleted=False));raise
   finally:
    for p in (ssh,tar):
     if p is not None and p.poll() is None:
      os.killpg(p.pid,signal.SIGTERM)
      try:p.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    write(job/(name+'-CLEANUP.json'),dict(children=[dict(pid=p.pid,exit=p.poll()) for p in (ssh,tar) if p],source_deleted=False))
 results=[]
 try:
  with ThreadPoolExecutor(max_workers=2) as pool:
   for f in as_completed([pool.submit(phase,'envs',18),pool.submit(phase,'assets',19)]):results.append(f.result())
  write(job/'RESULT.json',dict(status='DIRECT_STATIC_COPY_COMPLETE_NO_MODEL_EXECUTION',phases=results,source_deleted=False));return 0
 except BaseException as e:write(job/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc(),source_deleted=False));raise
 finally:
  stop.set();auth.join(timeout=2);server.close();sockpath.unlink(missing_ok=True);secret=None
  write(job/'CLEANUP.json',dict(authentication_socket_removed=not sockpath.exists(),secret_not_written_to_file=True,children=[dict(pid=p.pid,exit=p.poll()) for p in children],source_deleted=False))

if __name__=='__main__':raise SystemExit(main())

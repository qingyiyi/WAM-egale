"""Preserve the newly closed full-shard namespace and its original receipts locally."""
import argparse,getpass,json,os
from pathlib import Path
import shlex,signal,tarfile,time

REMOTE=r'''
from pathlib import Path
import io,json,os,signal,stat,sys,tarfile,time
p=json.loads(REG);source=Path(p['source']);meta=Path(p['remote_metadata']);meta.mkdir();os.sched_setaffinity(0,{18})
def rd(f):return json.loads(f.read_text())
def wr(n,v):
 with (meta/n).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def active(pin):
 q=Path('/proc')/str(pin['pid'])
 if not q.exists():return False
 s=(q/'stat').read_text().rpartition(') ')[2].split();return int(s[19])==pin['birth_ticks'] and q.stat().st_uid==pin['uid'] and s[0] not in ('Z','X')
def identity(f):
 s=f.lstat();assert stat.S_ISREG(s.st_mode);return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original migration absolute deadline')));signal.alarm(max(1,int(p['deadline_epoch']-time.time())))
try:
 while True:
  assert time.time()<p['deadline_epoch']
  needed=['RESUME-RESULT.json','FINAL-OWNED-CLEANUP.json','GUARD-FINAL.json','LAUNCH-TERMINAL.json']
  if all((source/n).is_file() for n in needed):
   outcome=rd(source/'RESUME-RESULT.json');close=rd(source/'FINAL-OWNED-CLEANUP.json');guard=rd(source/'GUARD-FINAL.json');launch=rd(source/'LAUNCH-TERMINAL.json')
   assert close['all_owned_groups_empty'] and guard['status']=='CLEAN' and launch['status']=='SUPERVISED_CHILDREN_EXITED'
   if not active(rd(source/'SUPERVISOR-PROCESS.json')):break
  print(json.dumps(dict(kind='WAIT_REAL_COMPLETE_SHARD_AND_SUPERVISOR_EXIT',at=time.time())),flush=True);time.sleep(10)
 entries=[];total=0
 for f in sorted(source.rglob('*')):
  if not f.is_file():continue
  assert not f.is_symlink();v=identity(f);assert v['size']<=64*2**20;total+=v['size'];assert total<=p['source_cap_bytes'];entries.append(dict(name=source.name+'/'+str(f.relative_to(source)),source=str(f),identity=v))
 manifest=dict(schema='closed-resume-original-manifest/v1',entries=entries,count=len(entries),source_bytes=total,outcome=outcome,no_SHA=True,source_deleted=False)
 raw=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).encode();assert len(raw)<=16*2**20
 print(json.dumps(dict(kind='BEGIN_FROZEN_TAR',files=len(entries),bytes=total)),flush=True)
 with tarfile.open(mode='w|',fileobj=sys.stdout.buffer,format=tarfile.PAX_FORMAT) as tar:
  m=tarfile.TarInfo('SOURCE-MANIFEST.json');m.size=len(raw);tar.addfile(m,io.BytesIO(raw))
  for e in entries:
   f=Path(e['source']);assert identity(f)==e['identity'];m=tarfile.TarInfo(e['name']);m.size=e['identity']['size'];m.mtime=e['identity']['mtime_ns']/1e9
   with f.open('rb') as stream:tar.addfile(m,stream)
   assert identity(f)==e['identity']
 for e in entries:assert identity(Path(e['source']))==e['identity']
 wr('RESULT.json',dict(status='FROZEN_SOURCE_STREAM_COMPLETE',files=len(entries),bytes=total,source_deleted=False,no_SHA=True))
except BaseException as e:wr('FAILED.json',dict(error=repr(e),source_deleted=False));raise
'''

def write(f,v):
 with f.open('x') as out:json.dump(v,out,ensure_ascii=False,indent=2);out.write('\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--registration',type=Path,required=True);x=a.parse_args();p=json.loads(x.registration.read_text());job=x.job.resolve();assert not job.exists();job.mkdir()
 write(job/'REGISTRATION.json',p);write(job/'STARTED.json',dict(pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),at=time.time(),GPU=0))
 import paramiko
 c=paramiko.SSHClient();c.load_host_keys(p['known_hosts']);c.set_missing_host_key_policy(paramiko.RejectPolicy());ch=None;secret=getpass.getpass('AutoDL password for final raw backup (RAM only): ')
 try:
  try:c.connect('region-9.autodl.pro',port=40416,username='root',password=secret,look_for_keys=False,allow_agent=False,timeout=30,auth_timeout=30)
  finally:secret=None
  c.get_transport().set_keepalive(10);ch=c.get_transport().open_session(window_size=8*2**20,max_packet_size=256*2**10);ch.settimeout(60)
  ch.exec_command(shlex.join(['/root/miniconda3/bin/python','-B','-c','REG='+repr(json.dumps(p))+'\n'+REMOTE]));stream=ch.makefile('rb')
  while True:
   event=json.loads(stream.readline(65536));print(json.dumps(event),flush=True)
   if event['kind']=='BEGIN_FROZEN_TAR':break
  pending=job/'new-shards-originals.tar.pending';total=0;last=0
  with pending.open('xb') as out:
   while True:
    data=stream.read(2**20)
    if not data:break
    total+=len(data);assert total<=p['archive_cap_bytes'] and time.time()<p['deadline_epoch'];out.write(data)
    if time.time()-last>20:print(json.dumps(dict(status='COPYING_FINAL_RAW',bytes=total)),flush=True);last=time.time()
   out.flush();os.fsync(out.fileno())
  error=ch.makefile_stderr('rb').read(16384).decode();assert ch.recv_exit_status()==0,error
  with tarfile.open(pending,'r:') as tar:
   members=tar.getmembers();manifest=json.loads(tar.extractfile(members[0]).read());expected={e['name']:e for e in manifest['entries']};seen=set();body=0
   for m in members[1:]:assert m.isfile() and m.name in expected and m.name not in seen and m.size==expected[m.name]['identity']['size'];seen.add(m.name);body+=m.size
  assert len(seen)==event['files'] and body==event['bytes']==manifest['source_bytes']
  with c.open_sftp() as s:
   with s.open(p['remote_metadata']+'/RESULT.json') as f:receipt=json.loads(f.read())
  assert receipt['status']=='FROZEN_SOURCE_STREAM_COMPLETE' and receipt['bytes']==body
  archive=job/'new-shards-originals.tar';pending.rename(archive)
  write(job/'SOURCE-MANIFEST.json',manifest);write(job/'RESULT.json',dict(status='COMPLETE_NEW_SHARDS_BACKUP_AND_ENTRY_AUDIT',archive=str(archive),files=len(seen),source_bytes=body,tar_bytes=total,outcome=manifest['outcome'],source_deleted=False,no_SHA=True,remote_receipt=receipt));print(json.dumps(dict(status='COMPLETE_NEW_SHARDS_BACKUP_AND_ENTRY_AUDIT',files=len(seen),bytes=body,outcome=manifest['outcome']['status'])),flush=True)
 except BaseException as e:write(job/'FAILED.json',dict(error=repr(e),source_deleted=False));raise
 finally:
  if ch:ch.close()
  c.close();write(job/'CLEANUP.json',dict(owned_SSH_closed=True,GPU_created=0,source_deleted=False))
if __name__=='__main__':raise SystemExit(main())

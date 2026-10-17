"""Copy the two already frozen evidence archives, without digests or model tests."""
from pathlib import Path
import argparse,json,os,shlex,subprocess,time

ROOT=Path(__file__).resolve().parents[3]
HOST='flagfft-dev.liulibo.tech-platform_research.dx-calc1.ws@ssh.platform-multi.baai.ac.cn'
RUNTIME=ROOT/'投机采样/results/runtime'

REMOTE=r'''
import json,os,pathlib,sys,tarfile,time
os.sched_setaffinity(0,{42})
base=(pathlib.Path.home()/'cxz').resolve(strict=True)
kind=sys.argv[1]; expected=int(sys.argv[2]); assert kind in ('previous','new-shards')
out=base/'jobs/baai-evidence-migration-20261018-r1';out.mkdir(parents=True,exist_ok=True)
def write(name,v):
 p=out/(kind+'-'+name+'.json')
 with p.open('x') as f:json.dump(v,f,indent=2)
archive=base/'backups'/(kind+'-originals.tar');archive.parent.mkdir(exist_ok=True)
start=time.time();size=0
write('STARTED',dict(pid=os.getpid(),started_epoch=start,expected_bytes=expected,GPU=0,no_SHA=True))
try:
 with archive.open('xb') as f:
  while True:
   b=sys.stdin.buffer.read(2**20)
   if not b:break
   size+=len(b);assert size<=expected;f.write(b)
 assert size==expected
 count=0;payload=0;stage=base/'evidence-stage'
 with tarfile.open(archive,'r:') as tar:
  for m in tar:
   part=pathlib.PurePosixPath(m.name)
   assert not part.is_absolute() and '..' not in part.parts
   if m.isdir():continue
   assert m.isfile(), 'unsupported evidence archive member type'
   if kind=='previous':
    dest=stage/part if part.parts[0]=='runtime' else base/'migration-originals'/part
   else:
    dest=(base/'migration-originals/new-shards-source-manifest.json') if m.name=='SOURCE-MANIFEST.json' else stage/'runtime'/part
   assert dest.resolve().is_relative_to(base)
   dest.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(m) as source:
    if dest.exists():
     assert dest.is_file() and dest.stat().st_size==m.size
     with dest.open('rb') as old:
      while True:
       block=source.read(2**20)
       if not block:break
       assert old.read(len(block))==block, 'existing archive member differs'
    else:
     with dest.open('xb') as target:
      copied=0
      while True:
       block=source.read(2**20)
       if not block:break
       copied+=len(block);target.write(block)
      assert copied==m.size
     dest.chmod(m.mode & 0o777)
   count+=1;payload+=m.size
 write('RESULT',dict(status='COPIED_ARCHIVE_AND_ALL_REGULAR_MEMBERS',archive=str(archive),archive_bytes=size,files=count,payload_bytes=payload,wall_sec=time.time()-start,source_deleted=False,no_SHA=True))
 print(json.dumps(dict(stage=kind,status='COMPLETE_ARCHIVE_AND_ENTRY_EXTRACTION',files=count,payload_bytes=payload,wall_sec=time.time()-start)),flush=True)
except BaseException as e:
 write('FAILED',dict(status='PARTIAL_NO_SOURCE_DELETION',error=repr(e),received_bytes=size,wall_sec=time.time()-start));raise
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args();job=args.job.resolve(strict=True)
 registration=json.loads((job/'REGISTRATION.json').read_text());deadline=registration['deadline_epoch']
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 archives=[('previous',RUNTIME/'p0-autodl-migration-backup-local-cpu1-v2-20261017-r2/all-runtime-originals.tar'),('new-shards',RUNTIME/'p0-baai-new-shards-backup-local-cpu1-v1-20261017-r1/new-shards-originals.tar')]
 ssh=['ssh','-T','-x','-a','-i','/rjs/cxz/.ssh/id_ed25519','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(RUNTIME/'p0-baai-server-inventory-readonly-local-cpu1-v1-20261017-r1/HOST-KEYS'),'-p','2222',HOST]
 receipts=[]
 try:
  for kind,archive in archives:
   assert time.time()+300<deadline
   size=archive.stat().st_size
   command='cd "$HOME/cxz" && PYTHONDONTWRITEBYTECODE=1 python3 -B -c '+shlex.quote(REMOTE)+' '+shlex.quote(kind)+' '+str(size)
   with archive.open('rb') as source,(job/(kind+'-ssh.log')).open('xb') as log:
    child=subprocess.Popen(ssh+[command],stdin=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    stat=Path('/proc')/str(child.pid)/'stat';birth=int(stat.read_text().rsplit(')',1)[1].split()[19])
    (job/(kind+'-OWNED.json')).write_text(json.dumps(dict(pid=child.pid,birth_ticks=birth,pgid=child.pid,uid=os.getuid(),archive=str(archive),archive_bytes=size)))
    try:rc=child.wait(timeout=max(1,deadline-time.time()-60))
    except subprocess.TimeoutExpired:
     current=int(stat.read_text().rsplit(')',1)[1].split()[19]) if stat.exists() else None
     if current==birth:os.killpg(child.pid,15)
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:
      if stat.exists() and int(stat.read_text().rsplit(')',1)[1].split()[19])==birth:os.killpg(child.pid,9)
      child.wait(timeout=15)
     raise
   receipts.append(dict(kind=kind,returncode=rc,source_deleted=False))
   print(json.dumps(receipts[-1]),flush=True);assert rc==0
  (job/'RESULT.json').write_text(json.dumps(dict(status='BOTH_FROZEN_EVIDENCE_ARCHIVES_COPIED',receipts=receipts,no_SHA=True),indent=2))
 finally:
  (job/'CLEANUP.json').write_text(json.dumps(dict(status='LOCAL_COPY_DRIVER_EXITING',receipts=receipts,source_deleted=False,no_background_children=True),indent=2))

if __name__=='__main__':main()

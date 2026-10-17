"""Stream every stopped runtime original to local storage; no remote temporary tar or deletion."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import getpass
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import shutil
import signal
import tarfile
import time

REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import io,json,os,signal,stat,sys,tarfile,time
P=json.loads(REGISTRATION);app=Path('/root/autodl-tmp/wam-tf-pf2-20261017')
stage=Path(P['remote_stage']);meta=Path(P['remote_metadata']);meta.mkdir()
os.sched_setaffinity(0,{16});os.environ['CUDA_VISIBLE_DEVICES']=''
def rd(p):
 p=Path(p);assert p.stat().st_size<=8*2**20
 return json.loads(p.read_text())
def wr(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def proc(pid):
 try:
  p=Path('/proc')/str(pid);v=(p/'stat').read_text().rpartition(') ')[2].split()
  return dict(pid=pid,birth_ticks=int(v[19]),pgid=int(v[2]),uid=p.stat().st_uid,state=v[0])
 except OSError:return None
def alive(pin):
 v=proc(pin['pid'])
 return bool(v and v['birth_ticks']==pin['birth_ticks'] and v['state'] not in ('Z','X'))
def guard():assert time.time()<P['deadline_epoch'],'migration archive lifecycle cap'
def identity(path):
 v=path.lstat();assert stat.S_ISREG(v.st_mode),'regular source files only'
 return dict(size=v.st_size,mtime_ns=v.st_mtime_ns,inode=v.st_ino,device=v.st_dev,mode=v.st_mode&0o777)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('archive absolute deadline')))
signal.alarm(max(1,int(P['deadline_epoch']-time.time())))
wr(meta/'STARTED.json',dict(pin=proc(os.getpid()),CPU=[16],GPU=0,source_commit=P['source_code_commit'],deadline_epoch=P['deadline_epoch']))
try:
 while True:
  guard()
  if (stage/'RESULT.json').is_file():
   result=rd(stage/'RESULT.json')
   assert result['status']=='PAUSED_AFTER_COMPLETE_ADMITTED_CASES_AND_REAL_CLEANUP',result['status']
   pins=[rd(stage/'LAUNCHED.json')['pin'],rd(stage/'OWNED-pause.json'),rd(stage/'OWNED-rescue.json')]
   if not any(alive(v) for v in pins):break
  print(json.dumps(dict(kind='WAIT_STAGE_COMPLETE_AND_ALL_CONTROLLERS_EXIT',at=time.time())),flush=True);time.sleep(15)
 # No new producer may start while a frozen archive is built.
 for name in ('p0-tf-suite-goal-remote-chain-v2-20261017-r1','p0-tf-suite-spatial-remote-chain-v2-20261017-r1'):
  j=app/'投机采样/results/runtime'/name
  assert rd(j/'CAMPAIGN-RESULT.json')['status']=='PAUSED_PARTIAL'
  assert not alive(rd(j/'CAMPAIGN-STARTED.json'))
 rotation=app/'投机采样/results/runtime/p0-tf-suite-cache-rotation-cpu1-v2-20261017-r1'
 assert rd(rotation/'FINAL-OWNED-CLEANUP.json')['all_groups_empty'] is True
 assert not rd(rotation/'STOP-REPORT.json')['deleted_cache_files']
 assert not alive(rd(rotation/'STARTED.json'))
 roots={'runtime':app/'投机采样/results/runtime','configs':app/'投机采样/configs'}
 entries=[];links=[];total=0
 def add(path,name):
  global total
  guard();v=path.lstat()
  if stat.S_ISLNK(v.st_mode):links.append(dict(name=name,source=str(path),target=os.readlink(path)));return
  if not stat.S_ISREG(v.st_mode):return
  ident=identity(path);assert ident['size']<=P['per_file_cap_bytes'],'source exceeds registered per-file cap'
  total+=ident['size'];assert total<=P['source_cap_bytes']
  entries.append(dict(name=name,source=str(path),identity=ident));assert len(entries)<=P['entry_cap']
 for alias,root in roots.items():
  if not root.exists():continue
  for parent,dirs,files in os.walk(root,followlinks=False):
   for d in dirs:
    path=Path(parent)/d
    if path.is_symlink():links.append(dict(name=alias+'/'+str(path.relative_to(root)),source=str(path),target=os.readlink(path)))
   for f in sorted(files):
    path=Path(parent)/f;add(path,alias+'/'+str(path.relative_to(root)))
 # Community target metadata is small; weights are reproducible and excluded.
 asset=Path('/root/wam-suite-assets-20261017-v1')
 if asset.exists():
  for parent,dirs,files in os.walk(asset,followlinks=False):
   for f in files:
    path=Path(parent)/f
    if path.suffix in ('.json','.md','.txt') and not path.is_symlink():add(path,'suite_metadata/'+str(path.relative_to(asset)))
 plan=rd(Path(P['remote_Long'])/'plan.json');selected=Path(plan['selected']['path'])
 assert selected.is_file() and str(selected) in {e['source'] for e in entries},'selected Long TF checkpoint must be archived'
 manifest=dict(schema='stopped-runtime-migration-manifest/v1',entries=entries,symlink_references_only=links,
  count=len(entries),payload_bytes=total,selected_Long_checkpoint=str(selected),stage_result=result,
  source_commit=P['source_code_commit'],no_SHA=True,remote_weights_not_copied=True,source_deletion=False)
 raw=(json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(raw)<=P['manifest_cap_bytes']
 wr(meta/'MANIFEST-SUMMARY.json',dict(count=len(entries),payload_bytes=total,manifest_bytes=len(raw),selected_Long_checkpoint=str(selected)))
 print(json.dumps(dict(kind='BEGIN_TAR_STREAM',count=len(entries),payload_bytes=total,manifest_bytes=len(raw))),flush=True)
 started=time.monotonic()
 with tarfile.open(mode='w|',fileobj=sys.stdout.buffer,format=tarfile.PAX_FORMAT) as tar:
  info=tarfile.TarInfo('SOURCE-MANIFEST.json');info.size=len(raw);info.mode=0o600;tar.addfile(info,io.BytesIO(raw))
  for entry in entries:
   guard();assert time.monotonic()-started<P['copy_wall_cap_sec']
   path=Path(entry['source']);assert identity(path)==entry['identity'],'source changed before stream'
   info=tarfile.TarInfo(entry['name']);info.size=entry['identity']['size'];info.mode=0o600
   info.mtime=entry['identity']['mtime_ns']/1e9
   with path.open('rb') as f:tar.addfile(info,f)
   assert identity(path)==entry['identity'],'source changed during stream'
 for entry in entries:guard();assert identity(Path(entry['source']))==entry['identity'],'source changed after stream'
 wr(meta/'RESULT.json',dict(status='COMPLETE_UNCHANGED_SOURCE_STREAM',files=len(entries),bytes=total,
  selected_Long_checkpoint=str(selected),wall_copy_sec=time.monotonic()-started,no_SHA=True,source_deleted=False))
except BaseException as e:
 wr(meta/'FAILED.json',dict(status='FAILED_OR_PARTIAL_NO_RETRY',error=repr(e),source_deleted=False));raise
'''

def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--registration',type=Path,required=True);x=a.parse_args()
    p=json.loads(x.registration.read_text());job=x.job.resolve();assert not job.exists()
    assert len(os.sched_getaffinity(0))==1 and os.getenv('CUDA_VISIBLE_DEVICES')==''
    assert p['lifecycle_wall_cap_sec']==9000 and p['source_cap_bytes']==4*2**30 and p['no_source_deletion'] is True
    job.mkdir(parents=True);payload=job/'payload';payload.mkdir();started=time.time()
    p['deadline_epoch']=min(started+9000,p['original_master_work_stop_epoch'])
    write(job/'REGISTRATION.json',p);write(job/'STARTED.json',dict(pid=os.getpid(),uid=os.getuid(),
        birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),CPU=1,GPU=0,at=datetime.now(timezone.utc).isoformat(),deadline=p['deadline_epoch']))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('local migration archive lifecycle')))
    signal.alarm(max(1,int(p['deadline_epoch']-time.time())))
    import paramiko
    c=None;channel=None;copied=0;files=0;manifest=None;last_notice=0
    try:
        c=paramiko.SSHClient();c.load_host_keys(p['known_hosts']);c.set_missing_host_key_policy(paramiko.RejectPolicy())
        secret=getpass.getpass('AutoDL SSH password (RAM only): ')
        try:c.connect('region-9.autodl.pro',port=40416,username='root',password=secret,look_for_keys=False,allow_agent=False,timeout=30,auth_timeout=30)
        finally:secret=None
        t=c.get_transport();t.set_keepalive(10);channel=t.open_session(window_size=128*2**20,max_packet_size=2**20);channel.settimeout(60)
        code='REGISTRATION='+repr(json.dumps(p,ensure_ascii=False))+'\n'+REMOTE
        channel.exec_command(shlex.join(['/root/miniconda3/bin/python','-B','-c',code]))
        stream=channel.makefile('rb')
        while True:
            line=stream.readline(65536);assert line,'remote ended before archive gate'
            event=json.loads(line);print(json.dumps(event,ensure_ascii=False),flush=True)
            if event['kind']=='BEGIN_TAR_STREAM':break
        assert event['payload_bytes']<=p['source_cap_bytes'] and event['count']<=p['entry_cap']
        copy_start=time.time();seen=set()
        with tarfile.open(fileobj=stream,mode='r|') as tar:
            first=next(iter(tar));assert first.name=='SOURCE-MANIFEST.json' and first.isfile() and first.size<=p['manifest_cap_bytes']
            raw=tar.extractfile(first).read();manifest=json.loads(raw);write(job/'SOURCE-MANIFEST.json',manifest)
            expected={e['name']:e for e in manifest['entries']};assert len(expected)==event['count']
            assert manifest['payload_bytes']==event['payload_bytes']
            for member in tar:
                if member is first:continue
                name=PurePosixPath(member.name)
                assert member.isfile() and not name.is_absolute() and '..' not in name.parts and '.' not in name.parts
                assert member.name in expected and member.name not in seen
                entry=expected[member.name];assert member.size==entry['identity']['size']<=p['per_file_cap_bytes']
                assert time.time()-copy_start<p['copy_wall_cap_sec'] and time.time()<p['deadline_epoch']
                assert shutil.disk_usage(job).free>=p['local_free_floor_bytes']+member.size
                dest=payload.joinpath(*name.parts);dest.parent.mkdir(parents=True,exist_ok=True);n=0
                source=tar.extractfile(member)
                with dest.open('xb') as f:
                    while True:
                        chunk=source.read(2**20)
                        if not chunk:break
                        f.write(chunk);n+=len(chunk)
                    f.flush();os.fsync(f.fileno())
                assert n==member.size;os.utime(dest,ns=(entry['identity']['mtime_ns'],entry['identity']['mtime_ns']))
                copied+=n;files+=1;seen.add(member.name)
                if time.time()-last_notice>=30:
                    notice=dict(status='COPYING_STOPPED_ALL_RUNTIME_ORIGINALS',files=files,bytes=copied,total_files=len(expected),total_bytes=manifest['payload_bytes'])
                    temp=job/'status.pending.json';temp.write_text(json.dumps(notice)+'\n');os.replace(temp,job/'status.json');print(json.dumps(notice),flush=True);last_notice=time.time()
        # Drain remaining tar block padding so the remote stdout cannot block its terminal receipt.
        remainder=stream.read(1024*1024);assert len(remainder)<=1024*1024
        error=channel.makefile_stderr('rb').read(16384).decode();assert channel.recv_exit_status()==0,error
        assert seen==set(expected) and copied==manifest['payload_bytes']
        with c.open_sftp() as s:
            with s.open(p['remote_metadata']+'/RESULT.json') as f:receipt=json.loads(f.read())
        assert receipt['status']=='COMPLETE_UNCHANGED_SOURCE_STREAM' and receipt['files']==files and receipt['bytes']==copied
        write(job/'REMOTE-STREAM-RESULT.json',receipt)
        result=dict(status='COMPLETE_LOCAL_ORIGINAL_RUNTIME_BACKUP',files=files,bytes=copied,wall_sec=time.time()-started,
            payload=str(payload),stage_result=manifest['stage_result'],selected_Long_checkpoint=manifest['selected_Long_checkpoint'],
            no_SHA=True,verification='entry count, exact byte counts and stable original source stat identities; no independent digest',
            source_deleted=False,remote_weights_not_copied=True,permissions=p['permissions'])
        write(job/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return 0
    except BaseException as exc:
        write(job/'FAILED.json',dict(status='PARTIAL_OR_FAILED_BACKUP_NO_RETRY',error=repr(exc),files=files,bytes=copied,source_deleted=False));raise
    finally:
        signal.alarm(0)
        if channel is not None:channel.close()
        if c is not None:c.close()
        write(job/'CLEANUP.json',dict(owned_SSH_closed=True,GPU_processes_created=0,source_deleted=False))

if __name__=='__main__':raise SystemExit(main())

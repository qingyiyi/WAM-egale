"""Large-read archive transfer, followed by local header/size audit; original lifetime retained."""
from __future__ import annotations
import argparse
import getpass
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import shutil
import signal
import sys
import tarfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tf_fullsuite_pause_v1.archive import REMOTE

def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--registration',type=Path,required=True);x=a.parse_args()
    p=json.loads(x.registration.read_text());job=x.job.resolve();assert not job.exists()
    old=Path(p['previous_local_job']);oldreg=json.loads((old/'REGISTRATION.json').read_text())
    assert (old/'FAILED.json').is_file() and json.loads((old/'CLEANUP.json').read_text())['owned_SSH_closed'] is True
    assert p['deadline_epoch']==oldreg['deadline_epoch'] and len(os.sched_getaffinity(0))==1 and os.getenv('CUDA_VISIBLE_DEVICES')==''
    retained=sum(f.stat().st_size for f in old.rglob('*') if f.is_file())
    cap=p['storage_cap_bytes']-retained-64*2**20;assert cap>2*2**30
    job.mkdir();write(job/'REGISTRATION.json',p);write(job/'STARTED.json',dict(pid=os.getpid(),uid=os.getuid(),
        birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),
        start_epoch=time.time(),original_deadline_epoch=p['deadline_epoch'],previous_attempt_retained_bytes=retained,CPU=1,GPU=0))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original archive absolute deadline')))
    signal.alarm(max(1,int(p['deadline_epoch']-time.time())))
    import paramiko
    c=None;ch=None;total=0;last=0;started=time.time()
    try:
        c=paramiko.SSHClient();c.load_host_keys(p['known_hosts']);c.set_missing_host_key_policy(paramiko.RejectPolicy())
        secret=getpass.getpass('AutoDL SSH password (RAM only): ')
        try:c.connect('region-9.autodl.pro',port=40416,username='root',password=secret,look_for_keys=False,allow_agent=False,timeout=30,auth_timeout=30)
        finally:secret=None
        # Close only an exact, still-live previous CPU stream producer, if SSH close did not already do so.
        control="""from pathlib import Path
import json,os,signal,time
j=Path('/root/wam-migration-archive-control-20261017-v1');s=json.loads((j/'STARTED.json').read_text());pin=s['pin'];p=Path('/proc')/str(pin['pid']);sent=False
if p.exists():
 v=(p/'stat').read_text().rpartition(') ')[2].split()
 if int(v[19])==pin['birth_ticks'] and p.stat().st_uid==pin['uid']==0 and int(v[2])==pin['pgid']:
  os.kill(pin['pid'],signal.SIGTERM);sent=True
  end=time.time()+10
  while p.exists() and time.time()<end:time.sleep(.1)
out=dict(previous_pin=pin,SIGTERM_sent=sent,observed_process_present=p.exists(),metadata={n:json.loads((j/n).read_text()) for n in ('RESULT.json','FAILED.json') if (j/n).is_file()})
print(json.dumps(out))
"""
        _,out,err=c.exec_command(shlex.join(['/root/miniconda3/bin/python','-B','-c',control]),timeout=30)
        raw=out.read(65536);error=err.read(8192).decode();assert out.channel.recv_exit_status()==0,error
        write(job/'PREVIOUS-REMOTE-STREAM-OBSERVATION.json',json.loads(raw))
        t=c.get_transport();t.set_keepalive(10);ch=t.open_session(window_size=8*2**20,max_packet_size=256*2**10);ch.settimeout(60)
        code='REGISTRATION='+repr(json.dumps(p,ensure_ascii=False))+'\n'+REMOTE
        ch.exec_command(shlex.join(['/root/miniconda3/bin/python','-B','-c',code]));stream=ch.makefile('rb')
        while True:
            event=json.loads(stream.readline(65536));print(json.dumps(event),flush=True)
            if event['kind']=='BEGIN_TAR_STREAM':break
        assert event['payload_bytes']<=p['source_cap_bytes'] and event['count']<=p['entry_cap']
        pending=job/'all-runtime-originals.tar.pending'
        with pending.open('xb') as f:
            while True:
                chunk=stream.read(2**20)
                if not chunk:break
                total+=len(chunk);assert total<=cap and time.time()<p['deadline_epoch']
                assert shutil.disk_usage(job).free>=p['local_free_floor_bytes']+len(chunk)
                f.write(chunk)
                if time.time()-last>=20:
                    v=dict(status='COPYING_SINGLE_TAR_STREAM',tar_bytes=total,source_payload_bytes=event['payload_bytes'],wall_sec=time.time()-started)
                    temp=job/'status.pending.json';temp.write_text(json.dumps(v)+'\n');os.replace(temp,job/'status.json');print(json.dumps(v),flush=True);last=time.time()
            f.flush();os.fsync(f.fileno())
        error=ch.makefile_stderr('rb').read(16384).decode();assert ch.recv_exit_status()==0,error
        with c.open_sftp() as s:
            with s.open(p['remote_metadata']+'/RESULT.json') as f:receipt=json.loads(f.read())
        assert receipt['status']=='COMPLETE_UNCHANGED_SOURCE_STREAM'
        with tarfile.open(pending,'r:') as tar:
            first=tar.next();assert first.name=='SOURCE-MANIFEST.json' and first.size<=p['manifest_cap_bytes']
            manifest=json.loads(tar.extractfile(first).read());expected={e['name']:e for e in manifest['entries']}
            seen=set();body_bytes=0
            for member in tar:
                if member is first:continue
                name=PurePosixPath(member.name)
                assert member.isfile() and not name.is_absolute() and '..' not in name.parts
                assert member.name in expected and member.name not in seen
                assert member.size==expected[member.name]['identity']['size']<=p['per_file_cap_bytes']
                seen.add(member.name);body_bytes+=member.size
        assert seen==set(expected) and len(seen)==receipt['files']==event['count']
        assert body_bytes==receipt['bytes']==manifest['payload_bytes']==event['payload_bytes']
        archive=job/'all-runtime-originals.tar';pending.rename(archive)
        fd=os.open(job,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        write(job/'SOURCE-MANIFEST.json',manifest);write(job/'REMOTE-STREAM-RESULT.json',receipt)
        result=dict(status='COMPLETE_LOCAL_TAR_BACKUP_AND_ENTRY_AUDIT',archive=str(archive),files=len(seen),source_bytes=body_bytes,tar_bytes=total,
            selected_Long_checkpoint=manifest['selected_Long_checkpoint'],stage_result=manifest['stage_result'],
            wall_sec=time.time()-started,previous_attempt_retained_bytes=retained,original_deadline_epoch=p['deadline_epoch'],
            verification='complete tar, all headers/names/entry byte counts and stable source stat; no independent digest',
            no_SHA=True,source_deleted=False,permissions=p['permissions'])
        write(job/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return 0
    except BaseException as exc:
        write(job/'FAILED.json',dict(status='PARTIAL_OR_FAILED_NO_RETRY',error=repr(exc),tar_bytes=total,source_deleted=False));raise
    finally:
        signal.alarm(0)
        if ch is not None:ch.close()
        if c is not None:c.close()
        write(job/'CLEANUP.json',dict(owned_SSH_closed=True,GPU_processes_created=0,source_deleted=False))

if __name__=='__main__':raise SystemExit(main())

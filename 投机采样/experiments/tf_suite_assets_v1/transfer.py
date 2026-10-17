"""Bounded authenticated transfer of already produced assets, without digests."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import shlex
import signal
import time
import paramiko

ROOT=Path(__file__).resolve().parents[3]

def write(path,data):
    temp=path.with_name(path.name+'.pending')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--source-manifest',type=Path,required=True)
    ap.add_argument('--asset-root',type=Path,required=True);ap.add_argument('--source-commit',required=True)
    ap.add_argument('--wall-cap',type=int,default=3600)
    a=ap.parse_args();j=a.job.resolve()
    if j.exists() or len(os.sched_getaffinity(0))!=1 or not 300<=a.wall_cap<=3600:raise ValueError('fresh CPU1 transfer job')
    j.mkdir(parents=True);start=time.time();source=json.loads(a.source_manifest.read_text())
    deadline=start+a.wall_cap
    write(j/'REGISTRATION.json',dict(goal='stream complete pinned assets via large SSH window',source_commit=a.source_commit,
        source_manifest=str(a.source_manifest.resolve()),GPU=0,CPU=1,origin_epoch=start,deadline_epoch=deadline,
        storage_cap_bytes=21*2**30,credential_storage=False,no_hash=True,
        cleanup='close owned SSH channels and join before deadline; no simulator or GPU process',
        pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19])))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded transfer lifecycle')))
    signal.alarm(a.wall_cap)
    c=paramiko.SSHClient();c.load_host_keys(str(ROOT/'投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy());completed=[]
    try:
        secret=getpass.getpass('AutoDL SSH password: ')
        c.connect('region-9.autodl.pro',port=40416,username='root',password=secret,timeout=30,auth_timeout=30,
            look_for_keys=False,allow_agent=False)
        secret=None;c.get_transport().set_keepalive(10)
        c.get_transport().default_window_size=128*2**20
        for entry in source['assets']:
            s=entry['suite'];dest=a.asset_root.resolve()/s/'publisher/transformer/diffusion_pytorch_model.safetensors'
            if dest.exists():raise FileExistsError('final asset already exists')
            part=dest.with_name(dest.name+'.ssh-stream-'+j.name);dest.parent.mkdir(parents=True,exist_ok=True)
            if part.exists():raise FileExistsError('no automatic transfer retry')
            code="import os,pathlib,sys; p=pathlib.Path(sys.argv[1]); st=p.stat(); assert (st.st_size,st.st_mtime_ns)==(int(sys.argv[2]),int(sys.argv[3])); f=p.open('rb');\nwhile True:\n b=f.read(1048576)\n if not b:break\n sys.stdout.buffer.write(b)\nsys.stdout.buffer.flush(); assert (p.stat().st_size,p.stat().st_mtime_ns)==(st.st_size,st.st_mtime_ns)"
            argv=['taskset','-c','16','/root/miniconda3/bin/python','-B','-c',code,entry['path'],str(entry['size']),str(entry['mtime'])]
            ch=c.get_transport().open_session(window_size=128*2**20,max_packet_size=2**20);ch.settimeout(30);ch.exec_command(shlex.join(argv))
            n=0;last=0
            try:
                with part.open('xb') as out:
                    while True:
                        if time.time()>=deadline:raise TimeoutError('asset transfer deadline')
                        block=ch.recv(2**20)
                        if not block:break
                        n+=len(block)
                        if n>entry['size']:raise ValueError('source stream exceeded registered bytes')
                        out.write(block)
                        if time.time()-last>=5:
                            write(j/'status.json',dict(status='TRANSFERRING_REAL_PINNED_ASSET',suite=s,bytes=n,
                                expected_bytes=entry['size'],completed_suites=completed,wall_sec=time.time()-start,
                                MiB_sec=n/max(.001,time.time()-start)/2**20));last=time.time()
                    out.flush();os.fsync(out.fileno())
                err=ch.recv_stderr(8192).decode(errors='replace');rc=ch.recv_exit_status()
                if rc!=0 or n!=entry['size']:raise RuntimeError('incomplete authenticated source stream: '+str(rc)+' '+err[:500])
                os.link(part,dest);part.unlink();completed.append(s)
                write(j/(s+'-IMPORTED.json'),dict(status='COMPLETE_SIZE_AND_SOURCE_STAT_NO_DIGEST',suite=s,
                    destination=str(dest),bytes=n,source=entry['path'],source_mtime_ns=entry['mtime'],
                    cryptographic_identity='UNKNOWN_NOT_COMPUTED',GPU=0))
            finally:ch.close()
        write(j/'RESULT.json',dict(status='COMPLETE_PINNED_ASSET_TRANSFER',completed_suites=completed,
            wall_sec=time.time()-start,CPU=1,GPU=0,credential_storage=False,no_digest=True))
        return 0
    except BaseException as exc:
        write(j/'FAILED.json',dict(status='PARTIAL_ASSET_TRANSFER',error=type(exc).__name__,completed_suites=completed,
            wall_sec=time.time()-start,retained_partial_files=True));raise
    finally:
        c.close();signal.alarm(0)
        write(j/'CLEANUP.json',dict(owned_channels_closed=True,remote_GPU_processes_created=0,local_process_exits_next=True))

if __name__=='__main__':raise SystemExit(main())

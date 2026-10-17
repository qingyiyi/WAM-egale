"""Import a completed CPU bank and bind only its host path, then retire zero-work remote preparation."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import getpass
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]

def read(p):
    p=Path(p);assert p.stat().st_size<=2**20
    return json.loads(p.read_text())

def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def bind(job,registration):
    import torch
    p=read(registration);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    raw=job/'REMOTE-bank.pt';assert raw.stat().st_size<=64*2**20
    data=torch.load(raw,map_location='cpu',weights_only=True)
    ready=read(p['asset_ready']);prompts=read(p['prompts'])['object']['public_task_prompts']
    assert (data['schema'],data['suite'],data['target_checkpoint_id'])==(
        'libero-suite-original-T5-bank/v1','libero_object','Wjjjh/lingbot-va-libero-object')
    assert data['checkpoint_path']==p['remote_model_path'] and ready['publisher_revision']==p['publisher_revision']
    assert data['sequence_length']==512 and data['dtype']=='bfloat16'
    assert data['output_shape']==[1,512,4096] and data['torch_version']==str(torch.__version__)
    assert data['public_task_prompts']==prompts and set(data['embeddings'])==set(prompts.values())|{''}
    for x in data['embeddings'].values():
        assert x.device.type=='cpu' and x.dtype==torch.bfloat16 and tuple(x.shape)==(1,512,4096)
        assert bool(torch.isfinite(x).all())
    target=Path(ready['model_path']).resolve(strict=True)
    assert (target/'transformer/diffusion_pytorch_model.safetensors').stat().st_size==10177831668
    data['checkpoint_path']=str(target)  # metadata only; every embedding is preserved
    out=job/'object/bank.pt';out.parent.mkdir()
    with out.open('xb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
    write(job/'BANK-BINDING.json',dict(status='COMPLETE_HOST_PATH_BINDING_OF_ORIGINAL_CPU_BANK',
        source=str(raw),bank=str(out),old_checkpoint_path=p['remote_model_path'],new_checkpoint_path=str(target),
        embedding_values_changed=False,T5_calls=0,target_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
        token_length=512,dtype='bfloat16',source_identity='UNKNOWN_NOT_HASHED',
        deployment_equivalence='PENDING_INLINE_ACTUAL_COMPLETE_EPISODE'))

REMOTE_RETIRE=r'''
import pathlib,json,os,signal,time,sys
j=pathlib.Path('/root/wam-suite-research-20261017-v3/object-chain-r1')
p=json.loads((j/'CAMPAIGN-STARTED.json').read_text());s=json.loads((j/'status.json').read_text())
assert s['status']=='WAITING_REAL_ASSETS_BANK_OR_PRIOR_CLEANUP' and s['active_steps']==s['completed_steps']==[]
assert not (j/'stages').exists() and p['uid']==0 and p['source_code_commit']=='8eaf1306216d5344b2d171e048b1ee98ad321454'
proc=pathlib.Path('/proc')/str(p['pid']);st=proc.joinpath('stat').read_text().rpartition(') ')[2].split()
assert int(st[19])==p['birth_ticks'] and proc.stat().st_uid==0 and int(st[2])==p['pgid']==p['pid']
out=j/'LOCAL-IDLE-GPU-SUPERSEDED.json';assert not out.exists()
out.write_text(json.dumps(dict(reason='user authorizes idle local GPU; Object unchanged suite now assigned locally before first episode',
    new_local_job=sys.argv[1],GPU_models_started=0,episodes_started=0,at_epoch=time.time(),pin=p),ensure_ascii=False,indent=2)+'\n')
os.kill(p['pid'],signal.SIGTERM);end=time.monotonic()+30
while time.monotonic()<end:
 f=proc/'stat';v=f.read_text().rpartition(') ')[2].split() if f.exists() else None
 if not v or int(v[19])!=p['birth_ticks'] or v[0] in ('Z','X'):break
 time.sleep(.2)
else:raise RuntimeError('owned preparation controller did not exit')
assert (j/'CAMPAIGN-RESULT.json').is_file()
print(json.dumps(dict(status='REMOTE_ZERO_EPISODE_PREPARATION_RETIRED',pin=p,
    terminal=json.loads((j/'CAMPAIGN-RESULT.json').read_text())['status'],new_local_job=sys.argv[1]),ensure_ascii=False))
'''

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--job',type=Path,required=True)
    a.add_argument('--registration',type=Path,required=True);a.add_argument('--bind-only',action='store_true')
    x=a.parse_args();job=x.job.resolve();p=read(x.registration)
    if x.bind_only:bind(job,x.registration);return 0
    assert not job.exists() and len(os.sched_getaffinity(0))==1 and os.getenv('CUDA_VISIBLE_DEVICES')==''
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert p['wall_cap_sec']==900 and p['storage_cap_bytes']==128*2**20
    assert Path(p['local_campaign_job']).is_dir() and p['permissions']==dict(
        engineering_quality_screen=False,independent_test_complete=False,noninferiority_established=False,formal_performance_permission=False)
    job.mkdir(parents=True);start=time.time();write(job/'REGISTRATION.json',p)
    write(job/'STARTED.json',dict(pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),
        CPU=1,GPU=0,at=datetime.now(timezone.utc).isoformat(),deadline_epoch=start+900))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bank import900s cap')))
    signal.alarm(900);client=None;child=None
    try:
        import paramiko
        client=paramiko.SSHClient();client.load_host_keys(str(ROOT/'投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'))
        client.set_missing_host_key_policy(paramiko.RejectPolicy());secret=getpass.getpass('AutoDL SSH password (RAM only): ')
        try:client.connect('region-9.autodl.pro',port=40416,username='root',password=secret,look_for_keys=False,allow_agent=False,timeout=30,auth_timeout=30)
        finally:secret=None
        transport=client.get_transport();transport.set_keepalive(10)
        with paramiko.SFTPClient.from_transport(transport,window_size=128*2**20,max_packet_size=2**20) as s:
            s.get_channel().settimeout(60)
            receipt_path=p['remote_bank_parent']+'/RESULT.json'
            assert s.stat(receipt_path).st_size<=2**20
            with s.open(receipt_path) as stream:receipt=json.loads(stream.read().decode())
            assert receipt['status']=='COMPLETE_SELECTED_CPU_BANKS' and receipt['suites']==['object']
            write(job/'REMOTE-PRODUCER-RESULT.json',receipt)
            path=p['remote_bank_parent']+'/object/bank.pt';before=s.stat(path)
            assert before.st_size<=60*2**20  # raw and rebound copies plus receipts stay within128MiB
            temp=job/'REMOTE-bank.pt.pending'
            with temp.open('xb') as f:s.getfo(path,f);f.flush();os.fsync(f.fileno())
            after=s.stat(path);assert (before.st_size,before.st_mtime)==(after.st_size,after.st_mtime) and temp.stat().st_size==before.st_size
            temp.rename(job/'REMOTE-bank.pt');write(job/'TRANSFER.json',dict(source=path,bytes=before.st_size,mtime=before.st_mtime,no_SHA=True))
        with (job/'binding.log').open('xb') as log:
            child=subprocess.Popen([p['va_python'],'-B',str(Path(__file__).resolve()),'--job',str(job),'--registration',str(x.registration.resolve()),'--bind-only'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
            write(job/'OWNED-BINDING-CHILD.json',dict(pid=child.pid,uid=os.getuid(),
                birth_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().rpartition(') ')[2].split()[19]),
                owner_job=str(job),purpose='CPU-only metadata binding, no new encoding'))
            assert child.wait(timeout=max(1,900-(time.time()-start)-30))==0,'bank binding failed; preserve source and log'
        assert read(job/'BANK-BINDING.json')['status']=='COMPLETE_HOST_PATH_BINDING_OF_ORIGINAL_CPU_BANK'
        _,out,err=client.exec_command(shlex.join(['/root/miniconda3/bin/python','-B','-c',REMOTE_RETIRE,p['local_campaign_job']]),timeout=60)
        raw=out.read().decode();error=err.read().decode();assert out.channel.recv_exit_status()==0,error[:2000]
        write(job/'REMOTE-PREPARATION-RETIRED.json',json.loads(raw))
        write(job/'RESULT.json',dict(status='COMPLETE_LOCAL_BANK_AND_EXCLUSIVE_SUITE_ASSIGNMENT',
            bank=str(job/'object/bank.pt'),GPU=0,wall_sec=time.time()-start,source_commit=p['source_code_commit'],
            model_started=False,complete_episodes=0,permissions=p['permissions']))
        return 0
    except BaseException as e:
        write(job/'FAILED.json',dict(status='FAILED_OR_PARTIAL_PREPARATION_NO_RETRY',error=repr(e),wall_sec=time.time()-start,
            bank_partial_retained=True,permissions=p['permissions']));raise
    finally:
        signal.alarm(0)
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        if client is not None:client.close()
        write(job/'CLEANUP.json',dict(owned_SSH_closed=True,owned_binding_child_returncode=None if child is None else child.poll(),GPU_processes_created=0))

if __name__=='__main__':raise SystemExit(main())

"""Bounded asset offload and metadata supervision for the registered campaigns.

The one password is supplied at a terminal and retained only in process RAM.
No model tests, file digests, new machines, or model-selection actions exist.
"""
from __future__ import annotations
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from datetime import datetime
import paramiko

ROOT=Path(__file__).resolve().parents[3]
APP='/root/autodl-tmp/wam-tf-pf2-20261017'
RUN=APP+'/投机采样/results/runtime/'

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)

OFFLOAD_AND_STAGE=r'''
from pathlib import Path
import json,os,time,subprocess
r=Path('/root/autodl-tmp/wam-tf-pf2-20261017');cache=Path('/root/wam-suite-assets-20261017-v1')
job=r/'投机采样/results/runtime/p0-tf-suite-spatial-assets-remote-cpu1-v1-20261017-r1'
launch=job.with_name(job.name+'-launch')
if launch.exists():print('ALREADY_REGISTERED');raise SystemExit(0)
weight=cache/'object/publisher/transformer/diffusion_pytorch_model.safetensors';st=weight.stat()
assert st.st_size==10177831668 and st.st_mtime_ns==EXPECTED_MTIME_NS and st.st_uid==os.getuid()
assert weight.resolve(strict=True)==weight
producer=r/'投机采样/results/runtime/p0-tf-suite-assets-remote-cpu1-v1-20261017-r2'
assert json.loads((producer/'RESULT.json').read_text())['status']=='COMPLETE_REGISTERED_SUITE_ASSETS'
pin=json.loads((producer/'REGISTRATION.json').read_text())
try:current=int(Path('/proc/'+str(pin['pid'])+'/stat').read_text().rpartition(') ')[2].split()[19])
except FileNotFoundError:current=None
assert current!=pin['birth_ticks'],'original asset producer remains live'
refs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  for fd in (p/'fd').iterdir():
   try:
    if str(weight) == os.readlink(fd):refs.append(str(fd))
   except (OSError,FileNotFoundError):pass
  if str(weight) in (p/'maps').read_text():refs.append(str(p/'maps'))
 except (FileNotFoundError,ProcessLookupError):continue
assert not refs,('owned offloaded asset still open',refs)
launch.mkdir()
receipt=dict(status='OFFLOADED_OWNED_OBJECT_CACHE_AFTER_COMPLETE_LOCAL_IMPORT',path=str(weight),size=st.st_size,
 mtime_ns=st.st_mtime_ns,at=time.time(),input_source='complete authenticated local import receipt',no_live_open_or_mapped_reference_observed=True,
 unique_weight_preserved_on_local=True,hash_executed=False)
(launch/'OBJECT-CACHE-OFFLOAD-PLAN.json').write_text(json.dumps(receipt,indent=2))
weight.unlink()
receipt.update(deleted_exact_owned_cache=True,reclaimed_bytes=st.st_size)
(launch/'OBJECT-CACHE-OFFLOAD-RESULT.json').write_text(json.dumps(receipt,indent=2))
argv=['timeout','--signal=TERM','--kill-after=30s','7230s','taskset','-c','16','/root/miniconda3/bin/python','-B',
 str(r/'投机采样/experiments/tf_suite_assets_v1/stage.py'),'--job',str(job),'--asset-root','/root/wam-spatial-assets-20261017-v1',
 '--shared-components',str(r/'models/lingbot-va-posttrain-libero-long-modelscope-20260922'),
 '--source-commit','f69629341bd26198ceecc2905e745656120b2fc3','--wall-cap','7200','--suites','spatial','--transport-host','hf-mirror.com']
with (launch/'producer.log').open('xb') as out:p=subprocess.Popen(argv,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),cwd=r,start_new_session=True)
pin=dict(pid=p.pid,birth_ticks=int(Path('/proc/'+str(p.pid)+'/stat').read_text().rpartition(') ')[2].split()[19]),pgid=os.getpgid(p.pid),argv=argv,at=time.time())
(launch/'STARTED.json').write_text(json.dumps(pin,indent=2));print(json.dumps(pin))
'''

PUBLISH_SPATIAL=r'''
from pathlib import Path
import json
src=Path('/root/wam-spatial-assets-20261017-v1/spatial');dst=Path('/root/wam-suite-assets-20261017-v1/spatial')
if (dst/'ASSET-READY.json').exists():print('SPATIAL_ALREADY_READY');raise SystemExit(0)
if not (src/'ASSET-READY.json').exists():print('SPATIAL_WAITING_DOWNLOAD');raise SystemExit(0)
d=json.loads((src/'ASSET-READY.json').read_text());assert (src/'publisher/transformer/diffusion_pytorch_model.safetensors').stat().st_size==10177831668
for p in ('vae','text_encoder','tokenizer'):assert (src/'runtime'/p).resolve(strict=True)==(dst/'runtime'/p).resolve(strict=True)
(dst/'runtime/transformer').symlink_to(src/'runtime/transformer',target_is_directory=True)
d.update(model_path=str(dst/'runtime'),component_scaffold=str(dst/'COMPONENT-SCAFFOLD.json'))
with (dst/'ASSET-READY.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print('SPATIAL_READY_CANONICAL_RUNTIME_PUBLISHED')
'''

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--master',type=Path,required=True)
    a.add_argument('--source-commit',required=True);x=a.parse_args();j=x.job.resolve();j.mkdir(parents=True,exist_ok=False)
    master=json.loads((x.master/'REGISTRATION.json').read_text());deadline=datetime.fromisoformat(master['absolute_work_stop']).timestamp()
    save(j/'REGISTRATION.json',dict(goal='owned temporary cache offload, Spatial asset production, metadata supervision',
        source_commit=x.source_commit,deadline_epoch=deadline,master=str(x.master.resolve()),CPU=1,GPU=0,
        pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),
        credential_storage=False,source_weights_SHA=False,new_GPU_purchase=False,metadata_storage_cap_bytes=256*2**20,
        cleanup='close task SSH; asset producers separately timeout-bounded; all campaign guards remain independent'))
    secret=getpass.getpass('AutoDL SSH password: ');c=None;offloaded=False;last_mirror=0;last_error=None;missing=0
    def connect():
        z=paramiko.SSHClient();z.load_host_keys(str(ROOT/'投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'))
        z.set_missing_host_key_policy(paramiko.RejectPolicy());z.connect('region-9.autodl.pro',port=40416,username='root',
            password=secret,look_for_keys=False,allow_agent=False,timeout=30,auth_timeout=30)
        z.get_transport().set_keepalive(10);return z
    def remote(code):
        _,o,e=c.exec_command(shlex.join(['/root/miniconda3/bin/python','-B','-c',code]),timeout=45)
        text=o.read().decode(errors='replace');err=e.read().decode(errors='replace');rc=o.channel.recv_exit_status()
        if rc:raise RuntimeError('remote owned resource action failed: '+err[-2500:])
        return text
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('owned bridge stopped')))
    try:
        while time.time()<deadline:
            try:
                if c is None or not c.get_transport().is_active():
                    if c is not None:c.close()
                    c=connect()
                imported=ROOT/'投机采样/results/runtime/p0-tf-suite-assets-import-cpu1-v1-20261017-r3/object-IMPORTED.json'
                local=ROOT/'models/community-libero-20261017-v1/object/publisher/transformer/diffusion_pytorch_model.safetensors'
                if not offloaded and imported.is_file() and local.is_file() and local.stat().st_size==10177831668:
                    manifest=ROOT/'投机采样/results/runtime/p0-tf-suite-assets-import-cpu1-v1-20261017-r1/SOURCE.json'
                    entry=next(a for a in json.loads(manifest.read_text())['assets'] if a['suite']=='object')
                    code=OFFLOAD_AND_STAGE.replace('EXPECTED_MTIME_NS',str(entry['mtime']))
                    result=remote(code);save(j/'OFFLOAD-AND-SPATIAL-STAGE.json',dict(at=time.time(),result=result))
                    offloaded=True
                if offloaded:remote(PUBLISH_SPATIAL)
                if time.time()-last_mirror>=120:
                    manifests=['p0-tf-fullsuite-long-v2-20261017-r2','p0-tf-suite-banks-remote-cpu4-v2-20261017-r2',
                        'p0-tf-suite-goal-chain-v1-20261017-r1','p0-tf-suite-spatial-chain-v1-20261017-r1',
                        'p0-tf-suite-spatial-assets-remote-cpu1-v1-20261017-r1']
                    with c.open_sftp() as s:
                        for n in manifests:
                            for name in ('status.json','RESULT.json','FAILED.json','CAMPAIGN-RESULT.json','GUARD-FINAL.json','SUPERVISOR-RESULT.json'):
                                r=RUN+n+'/'+name
                                try:st=s.stat(r)
                                except OSError:continue
                                if st.st_size>8*2**20:raise RuntimeError('metadata mirror file budget')
                                dest=j/'remote'/n/name;dest.parent.mkdir(parents=True,exist_ok=True)
                                temp=dest.with_name(dest.name+'.transfer');s.get(r,str(temp));os.replace(temp,dest)
                    last_mirror=time.time()
                save(j/'status.json',dict(status='SUPERVISING_REGISTERED_RESOURCES',at=time.time(),owned_Object_cache_offloaded=offloaded,
                    last_remote_metadata_mirror=last_mirror,credential_storage=False,GPU_processes_directly_started=0,last_error=last_error))
                missing=0
                states=[]
                for s in ('goal','spatial'):
                    p=j/'remote'/('p0-tf-suite-'+s+'-chain-v1-20261017-r1')/'CAMPAIGN-RESULT.json'
                    states.append(p.is_file())
                local_terminal=ROOT/'投机采样/results/runtime/p0-tf-suite-object-chain-v1-20261017-r1/CAMPAIGN-RESULT.json'
                if all(states) and local_terminal.is_file():break
            except (OSError,paramiko.SSHException,RuntimeError) as e:
                last_error=repr(e);missing+=1;save(j/'OBSERVED-ERROR.json',dict(at=time.time(),error=last_error,consecutive=missing))
                if c is not None:c.close();c=None
                if missing>=3:raise
            time.sleep(20)
        save(j/'RESULT.json',dict(status='RESOURCE_SUPERVISION_ENDED',owned_Object_cache_offloaded=offloaded,
            deadline_reached=time.time()>=deadline,finished_epoch=time.time(),instance_shutdown='NOT_REQUESTED_NOT_EXECUTED'))
    except BaseException as e:
        save(j/'FAILED.json',dict(status='RESOURCE_BRIDGE_PARTIAL',error=repr(e),owned_Object_cache_offloaded=offloaded,
            autonomous_campaign_guards_unchanged=True,at=time.time()));raise
    finally:
        secret=None
        if c is not None:c.close()
        save(j/'CLEANUP.json',dict(task_SSH_closed=True,owned_GPU_processes=0,instance_shutdown='NOT_REQUESTED_NOT_EXECUTED'))

if __name__=='__main__':main()

"""Stage the bounded resume only after real local backup completion; password stays in RAM."""
from __future__ import annotations
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import time

REMOTE=r'''
from pathlib import Path
import json,os,shutil,subprocess,time
R=json.loads(REGISTRATION);app=Path(R['remote_app']);job=Path(R['remote_job']);stage=Path(R['remote_stage'])
def rd(f):return json.loads(Path(f).read_text())
def wr(f,v):
 with Path(f).open('x') as out:json.dump(v,out,ensure_ascii=False,indent=2);out.write('\n')
assert not job.exists(),'one fresh job; no automatic retry'
coverage=rd(stage/'MIGRATION-COVERAGE.json');old=Path(R['prior_job']);p=rd(old/'plan.json')
assert rd(old/'GPU-OWNED-CLEANUP.json')['all_owned_groups_empty']
assert rd(old/'GUARD-FINAL.json')['status']=='CLEAN'
assert not coverage['partial_cases_no_automatic_retry'] and len(coverage['complete_cases'])==189
assert time.time()<R['preparation_absolute_deadline_epoch']
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
assert not compute.stdout.strip(),'no change to another GPU consumer'
assert len(gpu.stdout.strip().splitlines())==2 and shutil.disk_usage(app).free>=5*2**30
complete=coverage['complete_cases'];immediate=sorted(v['case_id'] for v in complete if v['source_job']==str(old))
assert len(immediate)==166 and len(complete)-len(immediate)==23
closed=[rd(old/f'server-worker{i}'/'LIVE-COUNTER.json') for i in range(4)]
assert all(v['active_target_forwards']==0 for v in closed)
debit=p['prior_target_forward_debit']+sum(v['target_forward_attempts'] for v in closed)
p.update(job=str(job),source_code_commit=R['source_code_commit'],prior_job=str(old),prior_completed_case_ids=immediate,
 blocked_prior_partial_case_ids=[],prior_target_forward_debit=debit)
p['worker_forward_caps']={str(i):sum(s['case_id'] not in immediate and s['execution_worker_index']==i for s in p['cases'])*12000+250 for i in range(4)}
assert sum(p['worker_forward_caps'].values())+debit<=p['job_forward_attempt_cap']
ids=[v['case_id'] for v in coverage['unstarted_cases']]
r=dict(schema='tf-long-multigeneration-shard-resume/v1',source_code_commit=R['source_code_commit'],
 frozen_mathematical_source='d5448bbc5b02465dcc8d7c6df5bb4497d40cf3d9',stage_job=str(stage),
 prior_complete_source_cases=complete,unstarted_case_ids=ids,shards=[ids[i:i+10] for i in range(0,len(ids),10)],
 original_authorization_origin=p['authorization_origin'],original_work_stop=p['absolute_work_stop'],original_cleanup_deadline=p['absolute_cleanup_deadline'],
 prior_target_forward_debit=debit,series_evidence_roots=sorted({v['source_job'] for v in complete})+[str(job)],
 permissions=p['frozen_permissions'],no_new_training=True,no_old_episode_rerun=True,no_smoke_tests_or_SHA=True,
 expected_complete_episode_cap=933,CPU_server_client=16,CPU_control=1,physical_GPUs=2,logical_model_workers=4,
 per_episode_cap_sec=1200,per_episode_target_forward_cap=4000,series_target_forward_cap=p['job_forward_attempt_cap'],
 series_storage_cap_bytes=p['new_evidence_cap_bytes'],cleanup='original independent birth-pinned guard, all four model groups, own supervisor; no instance shutdown',
 stop='STOP_AFTER_SHARD waits complete 10-case/30-episode barrier; hard safety/resource limits retain PARTIAL/no retry',
 evidence_path=str(job),input_manifest='prior stage source map, raw complete567, selected TF_ON, same bank/assets, explicit unstarted311',
 terminal='complete registered shards or explicit partial; no certificate/formal timing')
job.mkdir();wr(job/'plan.json',p);wr(job/'RESUME-REGISTRATION.json',r);wr(job/'BASELINE-BACKUP-RECEIPT.json',BACKUP)
wr(job/'RESOURCE-ADMISSION.json',dict(GPU=gpu.stdout,compute=compute.stdout,disk_free_bytes=shutil.disk_usage(app).free,observed_epoch=time.time(),no_model_forward=True))
wr(job/'DEPLOYMENT-COMPLETE.json',dict(source_code_commit=R['source_code_commit'],mathematical_source='d5448bbc5b02465dcc8d7c6df5bb4497d40cf3d9',
 raw_source_bytes_verified=R['verified_unchanged_original_sources'],standalone_tests=False,SHA=False,source_assets_reused=True,permissions=p['frozen_permissions']))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
with (job/'supervisor.log').open('xb') as log:
 child=subprocess.Popen([p['va_python'],'-B',str(app/'投机采样/experiments/tf_fullsuite_wave_resume_v1/launch.py'),'--job',str(job)],
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=env)
f=Path('/proc')/str(child.pid);v=(f/'stat').read_text().rpartition(') ')[2].split();pin=dict(pid=child.pid,birth_ticks=int(v[19]),pgid=int(v[2]),uid=f.stat().st_uid)
wr(job/'SUPERVISOR-PROCESS.json',pin)
print(json.dumps(dict(status='NEW_SHARD_SUPERVISOR_STARTED_NOT_MODEL_READY',remote_job=str(job),pin=pin,prior_attempt_debit=debit,
 original_authorization_origin=p['authorization_origin'],original_work_stop=p['absolute_work_stop'],original_cleanup_deadline=p['absolute_cleanup_deadline'],
 pending_cases=len(ids),first_shard=ids[:10],worker_forward_caps=p['worker_forward_caps'],GPU=gpu.stdout,permissions=p['frozen_permissions'])))
'''


def main():
    a=argparse.ArgumentParser();a.add_argument('--registration',required=True,type=Path);a.add_argument('--job',required=True,type=Path);x=a.parse_args()
    r=json.loads(x.registration.read_text());job=x.job.resolve();assert not job.exists();assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==1
    backup_dir=Path(r['local_backup_job']);backup=json.loads((backup_dir/'RESULT.json').read_text())
    assert backup['status']=='COMPLETE_LOCAL_TAR_BACKUP_AND_ENTRY_AUDIT' and json.loads((backup_dir/'CLEANUP.json').read_text())['owned_SSH_closed']
    assert Path(backup['archive']).is_file() and Path(backup['archive']).stat().st_size==backup['tar_bytes']
    job.mkdir();(job/'REGISTRATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    import paramiko
    c=paramiko.SSHClient();c.load_host_keys(r['known_hosts']);c.set_missing_host_key_policy(paramiko.RejectPolicy())
    secret=getpass.getpass('AutoDL SSH password (RAM only): ')
    try:
        try:c.connect('region-9.autodl.pro',port=40416,username='root',password=secret,look_for_keys=False,allow_agent=False,timeout=30,auth_timeout=30)
        finally:secret=None
        root=Path(__file__).resolve().parents[3];verified=[]
        with c.open_sftp() as s:
            for rel in r['verified_unchanged_original_sources']:
                raw=(root/rel).read_bytes()
                with s.open(str(Path(r['remote_app'])/rel),'rb') as f:assert f.read()==raw,'frozen original source differs: '+rel
                verified.append(rel)
            remote=Path(r['remote_app'])/'投机采样/experiments/tf_fullsuite_wave_resume_v1'
            try:s.mkdir(str(remote))
            except OSError:raise RuntimeError('fresh wrapper namespace already exists; no overwrite')
            for name in ('runner.py','launch.py'):
                with s.open(str(remote/name),'wb') as f:f.write((Path(__file__).parent/name).read_bytes())
        code='REGISTRATION='+repr(json.dumps(r,ensure_ascii=False))+'\nBACKUP='+repr(backup)+'\n'+REMOTE
        _,out,err=c.exec_command(shlex.join(['/root/miniconda3/bin/python','-B','-c',code]),timeout=90)
        raw=out.read(65536);error=err.read(16384).decode();status=out.channel.recv_exit_status()
        assert status==0,error
        result=json.loads(raw);(job/'DEPLOY-RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        (job/'FAILED.json').write_text(json.dumps(dict(error=repr(exc),no_retry=True))+'\n');raise
    finally:
        c.close();(job/'CLEANUP.json').write_text(json.dumps(dict(owned_SSH_closed=True,no_GPU_processes_local=True))+'\n')

if __name__=='__main__':raise SystemExit(main())

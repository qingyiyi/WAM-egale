"""Register the finite three-suite schedule and immutable hardware assignments."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
REMOTE=Path('/root/autodl-tmp/wam-tf-pf2-20261017')
FALSE=dict(engineering_quality_screen=False,independent_test_complete=False,
    noninferiority_established=False,formal_performance_permission=False)
REVISIONS=dict(object='489806feb4074eee71f97180948e8b85e33ebeda',goal='42455fd87bc5b9c4613cc13b03be2f2d80605efe',
    spatial='933ab666f8cd21ff71d8765dea9fdd110512034f')

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--source-commit',required=True)
    x=a.parse_args();job=x.job.resolve();job.mkdir(parents=True,exist_ok=False)
    origin=datetime.now(timezone(timedelta(hours=8)));hard=origin+timedelta(hours=96);work=hard-timedelta(minutes=15)
    common=dict(schema='tf-suite-campaign/v1',source_code_commit=x.source_commit,
        authorization_origin=origin.isoformat(),absolute_work_stop=work.isoformat(),absolute_cleanup_deadline=hard.isoformat(),
        permissions=FALSE,episode_count_cap=1560,target_forward_attempt_cap=6241000,research_storage_cap_bytes=5*2**30,
        new_model_free_MiB=22000,training_fixed=True,no_smoke=True,no_independent_tests=True,no_SHA=True,
        no_additional_GPU_purchase=True,independent_TEST=False,formal_performance=False)
    entries=[]
    for s in ('object','goal','spatial'):
        remote=s!='object';root=REMOTE if remote else ROOT
        suitejob=root/'投机采样/results/runtime'/('p0-tf-suite-'+s+'-chain-v1-20261017-r1')
        bankjob='p0-tf-suite-banks-remote-cpu4-v2-20261017-r2' if remote else 'p0-tf-suite-banks-cpu7-v1-20261017-r1'
        if s=='object':profiles=[dict(server=[8,9],client=[10,11],control=[7]),dict(server=[12,13],client=[14,15],control=[7])]
        elif s=='goal':profiles=[dict(server=[0,1],client=[2,3],control=[16]),dict(server=[4,5],client=[6,7],control=[16])]
        else:profiles=[dict(server=[8,9],client=[10,11],control=[16]),dict(server=[12,13],client=[14,15],control=[16])]
        envs=root/'envs' if remote else Path('/rjs/cxz/miniconda3/envs')
        p={**common, 'job':str(suitejob),'root_path':str(root),'suite':s,'publisher_revision':REVISIONS[s],
            'gpu_index':1 if s=='spatial' else 0,'physical_GPU_hardware':'AutoDL A100-PCIe40G' if remote else 'local A100-SXM4-40G',
            'CPU_profiles':profiles,'training_CPU':[18,19] if s=='spatial' else [16,17] if remote else [2,3],
            'base_port':19320+10*('object','goal','spatial').index(s),'base_master_port':29820+10*('object','goal','spatial').index(s),
            'va_python':str(envs/'va/bin/python'),'client_python':str(envs/'libero-va/bin/python'),
            'LIBERO_CONFIG_PATH':str(root/'投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r1/libero-config'),
            'target_source_reference':str(root/'投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r1/target-model-source-reference.bytes'),
            'asset_ready':str(root/'models/community-libero-20261017-v1'/s/'ASSET-READY.json'),
            'bank':str(root/'投机采样/results/runtime'/bankjob/s/'bank.pt'),
            'prompts':str(root/'投机采样/results/runtime/p0-tf-suite-prompts-cpu1-v1-20261017-r1/PROMPTS.json'),
            'wait_for_closed_jobs':[str(root/'投机采样/results/runtime/p0-tf-fullsuite-long-v2-20261017-r2')] if remote else []}
        local_plan=job/(s+'-registration.json');save(local_plan,p)
        if not remote:save(suitejob/'registration.json',p)
        entries.append(dict(suite=s,host='AutoDL' if remote else 'local',job=str(suitejob),registration=str(local_plan),
            state='REGISTERED_NOT_YET_STARTED',all_500_case_arms_same_physical_GPU=True))
    save(job/'REGISTRATION.json',dict(goal='three independently trained same-recipe TF64 drafts and full T/C/P coverage',
        source_code_commit=x.source_commit,authorization_origin=origin.isoformat(),absolute_work_stop=work.isoformat(),
        absolute_cleanup_deadline=hard.isoformat(),total_new_complete_episode_cap=4680,total_target_forward_cap=18723000,
        research_storage_cap_bytes=15*2**30,asset_storage_cap_bytes=40*2**30,hardware_assignment=entries,
        full_coverage_is_descriptive_not_independent_TEST=True,old_Long_job_unchanged=True,permissions=FALSE))
    print(job)

if __name__=='__main__':main()

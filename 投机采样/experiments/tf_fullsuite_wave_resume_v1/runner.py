"""Resume only unstarted Long cases in complete 30-episode shards; frozen math reused."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from tf_fullsuite_eval_v2.protocol import load,read,write,require,PERMISSIONS,episode_dir
from tf_fullsuite_eval_v2.runner import Runner,process


def validate(job,p):
    r=read(job/'RESUME-REGISTRATION.json')
    require(r['schema']=='tf-long-multigeneration-shard-resume/v1','independent resume schema required')
    require(r['source_code_commit']==p['source_code_commit'] and r['frozen_mathematical_source']=='d5448bbc5b02465dcc8d7c6df5bb4497d40cf3d9','source identity differs')
    stage=Path(r['stage_job']);coverage=read(stage/'MIGRATION-COVERAGE.json')
    require(read(stage/'RESULT.json')['status']=='PAUSED_AFTER_COMPLETE_ADMITTED_CASES_AND_REAL_CLEANUP','prior stage not drained')
    require(not coverage['partial_cases_no_automatic_retry'],'no automatic retry of partial cases')
    complete=coverage['complete_cases'];pending=coverage['unstarted_cases']
    require(len(complete)==189 and len(pending)==311 and len({v['case_id'] for v in complete+pending})==500,'frozen stage coverage differs')
    require(r['prior_complete_source_cases']==complete and r['unstarted_case_ids']==[v['case_id'] for v in pending],'resume manifest differs from actual stage')
    require(all(v['source_job']!=str(job) for v in complete),'old receipts cannot be assigned to fresh source')
    require(r['shards']==[r['unstarted_case_ids'][i:i+10] for i in range(0,len(pending),10)],'fixed complete 30 episode shards required')
    require(r['permissions']==PERMISSIONS and r['original_authorization_origin']==p['authorization_origin']
        and r['original_work_stop']==p['absolute_work_stop'] and r['original_cleanup_deadline']==p['absolute_cleanup_deadline'],'budget or permission reset')
    backup=read(job/'BASELINE-BACKUP-RECEIPT.json')
    require(backup['status']=='COMPLETE_LOCAL_TAR_BACKUP_AND_ENTRY_AUDIT' and backup['source_deleted'] is False
        and backup['stage_result']['complete_episodes']==567,'old stage must be secured before more GPU work')
    old=read(Path(p['prior_job'])/'plan.json')
    roots=sorted({v['source_job'] for v in complete})
    require(r['series_evidence_roots']==roots+[str(job)],'all generation storage accounting required')
    exact_debit=old['prior_target_forward_debit']+sum(read(Path(p['prior_job'])/f'server-worker{i}'/'LIVE-COUNTER.json')['target_forward_attempts'] for i in range(4))
    require(p['prior_target_forward_debit']==r['prior_target_forward_debit']==exact_debit,'all prior attempts must be debited exactly')
    require(not (job/'episodes').exists() and not (job/'RUN-STARTED.json').exists(),'one fresh lifecycle, no retry')
    require(r['no_new_training'] and r['no_old_episode_rerun'] and r['no_smoke_tests_or_SHA'],'scope differs')
    return r


class ShardRunner(Runner):
    def __init__(self,job,plan,registration):
        super().__init__(job,plan);self.r=registration;self.shard_results=[];self.series_check=0

    def guard(self,stage):
        super().guard(stage)
        if time.monotonic()-self.series_check>=10:
            self.series_check=time.monotonic();total=0
            for folder in self.r['series_evidence_roots']:
                for f in Path(folder).rglob('*'):
                    if f.is_file():
                        try:total+=f.stat().st_size
                        except FileNotFoundError:continue
            require(total<=self.p['new_evidence_cap_bytes'],'all generation cumulative evidence cap')
            attempts=self.p['prior_target_forward_debit']
            for i in range(4):
                f=self.job/f'server-worker{i}'/'LIVE-COUNTER.json'
                if f.is_file():attempts+=read(f)['target_forward_attempts']
            require(attempts<=self.p['job_forward_attempt_cap'],'all generation attempt cap')

    def waves(self):
        for number,ids in enumerate(self.r['shards']):
            self.guard('full_30_episode_shard_admission')
            if (self.job/'STOP_AFTER_SHARD').exists():
                self.event('USER_STAGE_PAUSE_AT_COMPLETE_SHARD',next_shard=number);break
            counts=[sum(self.p['cases'][cid]['execution_worker_index']==i for cid in ids) for i in range(4)]
            reserve=max(counts)*3*(1200+120)
            if time.time()+reserve>self.work or time.time()+reserve+900>self.hard:
                self.event('SHARD_NOT_STARTED_ORIGINAL_RESERVE_UNAVAILABLE',shard=number,case_ids=ids);break
            started=time.time();self.atomic_new(self.job/f'SHARD-{number:03d}-STARTED.json',dict(case_ids=ids,at=started))
            def worker(index):
                for cid in ids:
                    if self.p['cases'][cid]['execution_worker_index']!=index:continue
                    if self.hard_event.is_set():return
                    result=self.run_case(cid)
                    if result['status']!='COMPLETE_RECORDED_CASE':return
            with ThreadPoolExecutor(max_workers=4,thread_name_prefix='complete-shard') as pool:
                futures=[pool.submit(worker,index) for index in range(4)]
                while not all(f.done() for f in futures):
                    if self.hard_event.is_set():self.cleanup(gpu_only=True)
                    time.sleep(.2)
                for future in futures:future.result()
            completed=[cid for cid in ids if self.results.get(cid,{}).get('status')=='COMPLETE_RECORDED_CASE']
            receipt=dict(status='COMPLETE_FULL_CASE_SHARD' if len(completed)==len(ids) else 'PARTIAL_NO_RETRY',
                shard=number,case_ids=ids,complete_case_ids=completed,complete_episodes=3*len(completed),
                full_30_episode_shard=len(ids)==10,wall_sec=time.time()-started,permissions=PERMISSIONS)
            self.atomic_new(self.job/f'SHARD-{number:03d}-RESULT.json',receipt);self.shard_results.append(receipt)
            if self.hard_event.is_set() or len(completed)!=len(ids):break

    def run_report(self):
        # Do not run a two-generation report which would lose the oldest 23 cases.
        rows=[]
        sources={v['case_id']:v['source_job'] for v in self.r['prior_complete_source_cases']}
        for cid,result in self.results.items():
            if result['status']=='COMPLETE_RECORDED_CASE':sources[cid]=str(self.job)
        for cid,source in sorted(sources.items()):
            for arm in ('T','C','P'):
                f=episode_dir(source,cid,arm)/'EPISODE-TERMINAL.json';t=read(f)
                require(t['complete_episode'] is True and t['quality_terminal'] is True,'raw complete coverage required')
                rows.append(dict(case_id=cid,arm=arm,source=str(f),success=t['success'],env_steps=t['env_steps'],failure800_penalty=t['failure800_penalty']))
        self.report_result=dict(status='MULTIGENERATION_RAW_COVERAGE_RECORDED_NO_QUALITY_CERTIFICATE',returncode=0,
            complete_cases=len(sources),complete_episodes=len(rows),episodes=rows,shards=self.shard_results,
            four_frozen_permissions=PERMISSIONS,formal_performance=False)
        write(self.job/'REPORT-TERMINAL.json',self.report_result)


def main():
    a=argparse.ArgumentParser();a.add_argument('--job',required=True,type=Path);x=a.parse_args();job=x.job.resolve(strict=True)
    try:
        p=load(job);r=validate(job,p);runner=ShardRunner(job,p,r);native_rc=runner.execute()
        c=read(job/'FINAL-OWNED-CLEANUP.json');prior={v['case_id']:v['source_job'] for v in r['prior_complete_source_cases']}
        for cid,result in runner.results.items():
            if result['status']=='COMPLETE_RECORDED_CASE':prior[cid]=str(job)
        partial=[cid for cid,v in runner.results.items() if v['status']!='COMPLETE_RECORDED_CASE']
        result=dict(status='COMPLETE_ALL_500_RAW_CASES_NO_QUALITY_CERTIFICATE' if len(prior)==500 and not partial and c['all_owned_groups_empty'] else
            'PAUSED_AFTER_COMPLETE_SHARDS' if not partial and not runner.hard_event.is_set() and c['all_owned_groups_empty'] else 'PARTIAL_OR_FAILED_NO_RETRY',
            native_supervisor_status=read(job/'SUPERVISOR-RESULT.json')['status'],native_returncode=native_rc,
            complete_cases=len(prior),complete_episodes=3*len(prior),new_complete_cases=len(prior)-189,
            partial_case_ids_no_retry=partial,complete_source_cases=[dict(case_id=cid,source_job=source) for cid,source in sorted(prior.items())],
            unstarted_case_ids=[cid for cid in range(500) if cid not in prior and cid not in partial],
            all_owned_groups_empty=c['all_owned_groups_empty'],backup_of_new_data='PENDING_SEPARATE_INCREMENTAL_ARCHIVE',
            original_deadline_unchanged=True,permissions=PERMISSIONS,performance_claim=False)
        write(job/'RESUME-RESULT.json',result)
        return 0 if not partial and not runner.hard_event.is_set() and c['all_owned_groups_empty'] else 2
    except BaseException as exc:
        f=job/'RESUME-ENTRY-FAILED.json'
        if not f.exists():write(f,dict(error=repr(exc),traceback=traceback.format_exc(),permissions=PERMISSIONS))
        raise

if __name__=='__main__':raise SystemExit(main())

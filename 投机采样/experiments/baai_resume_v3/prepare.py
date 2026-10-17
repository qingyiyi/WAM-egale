"""Carry original clocks, debit and invalid attempts into a distinct host job."""
from pathlib import Path
import argparse,copy,json,os,time


def main():
    a=argparse.ArgumentParser();a.add_argument('--source-commit',required=True);x=a.parse_args();base=(Path.home()/'cxz').resolve(strict=True);os.sched_setaffinity(0,{43})
    assert Path.cwd().resolve()==base and time.time()<1792272260.1204956
    app=base/'WAM-egale';runtime=app/'投机采样/results/runtime';old=runtime/'p0-tf-long-baai-resume-v2-20261018-r2'
    def read(p):return json.loads(Path(p).read_text())
    def write(p,d):
        with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)
    closed=read(old/'SUPERVISOR-RESULT.json');assert closed['complete_episodes_new']==0 and closed['all_owned_groups_empty'] and read(old/'GUARD-FINAL.json')['status']=='CLEAN'
    partial=[]
    for p in (old/'episodes').glob('case-*'):
        if any((p/a).exists() for a in ['T','C','P']):partial.append(int(p.name[5:]))
    assert sorted(partial)==[220,225,226,227]
    # Keep these original infra-invalid attempts blocked; never silently retry.
    p=copy.deepcopy(read(old/'plan.json'));job=runtime/'p0-tf-long-baai-resume-v3-20261018-r3';job.mkdir(exist_ok=False)
    roots=p['historical_source_roots']+[dict(logical_source_job=str(old),physical_source_job=str(old))]
    debit=sum(read(Path(r['physical_source_job'])/f'server-worker{i}/LIVE-COUNTER.json')['target_forward_attempts'] for r in roots for i in range(read(Path(r['physical_source_job'])/'plan.json')['max_GPU_workers']))
    blocked=sorted(set(partial)|set(p['blocked_prior_partial_case_ids']));pending=[v['case_id'] for v in p['cases'] if v['case_id'] not in set(p['prior_completed_case_ids'])|set(blocked)]
    queues={i:[cid for cid in pending if p['cases'][cid]['execution_worker_index']==i] for i in range(4)};shards=[]
    while any(queues.values()):
        s=[]
        for j in range(10):
            for k in range(4):
                i=(2*len(shards)+j+k)%4
                if queues[i]:s.append(queues[i].pop(0));break
        shards.append(sorted(s))
    p.update(job=str(job),source_code_commit=x.source_commit,code_namespace=str(app/'投机采样/experiments/baai_resume_v3'),historical_source_roots=roots,prior_target_forward_debit=debit,
        blocked_prior_partial_case_ids=blocked,unstarted_case_ids=pending,migration_shards=shards,worker_forward_caps={str(i):sum(p['cases'][cid]['execution_worker_index']==i for cid in pending)*12000+250 for i in range(4)},
        evidence_path=str(job),previous_host_attempt_status='PARTIAL_CLIENT_IMPORT_BEFORE_ANY_ENV_ACTION_NO_RETRY',client_library_path_fix='Python3.10 CPU client and Python3.12 CUDA server use their own libtorch_python; numerical source unchanged')
    assert sum(p['worker_forward_caps'].values())+debit<=p['job_forward_attempt_cap']
    assert (base/'deps/glvnd-ubuntu24-1.7.0/usr/lib/x86_64-linux-gnu/libGL.so.1').is_file()
    write(job/'plan.json',p);write(job/'SOURCE-FREEZE.json',dict(source_code_commit=x.source_commit,base='844738b2cc0590cf383d7fa0a29fc6d56bf32105',change='interpreter-specific runtime library paths only',old_failed_job=str(old),original_receipts_rewritten=False,no_SHA=True))
    write(job/'DEPLOYMENT-COMPLETE.json',dict(status='EXISTING_COMPLETE_ASSETS_WITH_DISTINCT_PATH_CONTROLLER',source_code_commit=x.source_commit,no_model_tests=True,no_SHA=True,permissions=p['permissions'],original_inline_math_and_RNG_checks_unchanged=True))
    print(json.dumps(dict(job=str(job),pending_cases=len(pending),blocked_no_retry_case_ids=blocked,prior_actual_attempts=debit,original_work_stop=p['absolute_work_stop'],original_cleanup=p['absolute_cleanup_deadline'])))

if __name__=='__main__':main()

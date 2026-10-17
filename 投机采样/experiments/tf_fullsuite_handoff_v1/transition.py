"""Bounded registered transition to four independent real-episode workers.

This is orchestration, not a benchmark or model test. All completed evidence
is read from its original producer namespace; no episode is retried.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

OWN=Path(__file__).resolve().parent
EVAL=OWN.parent/'tf_fullsuite_eval_v2'


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def proc(pid):
    try:
        s=Path('/proc')/str(pid)
        a=(s/'stat').read_text().rpartition(') ')[2].split()
        return dict(pid=pid,birth_ticks=int(a[19]),pgid=int(a[2]),state=a[0],uid=s.stat().st_uid)
    except (OSError,ValueError,IndexError):
        return None


def matches(pin):
    x=proc(pin['pid'])
    return x is not None and x['birth_ticks']==pin['birth_ticks'] and x['pgid']==pin['pgid']==pin['pid'] and x['uid']==pin.get('uid',os.getuid())


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--job',required=True,type=Path)
    a=ap.parse_args();job=a.job.resolve(strict=True)
    template=read(job/'plan-template.json');prior=Path(template['prior_job']).resolve(strict=True)
    p0=read(prior/'plan.json');prior_pin=read(prior/'LAUNCH-RESULT.json')['runner']
    started=time.time();deadline=min(started+7200,datetime.fromisoformat(p0['absolute_work_stop']).timestamp()-900)
    write(job/'TRANSITION-STARTED.json',dict(pid=os.getpid(),birth_ticks=proc(os.getpid())['birth_ticks'],
        started_unix=started,deadline_unix=deadline,source_code_commit=template['source_code_commit'],control_source_commit=template['handoff_source_commit'],
        kind='owned complete-case orchestration only',new_model_calls_before_old_cleanup=0))
    result=None
    try:
        require(not (job/'plan.json').exists(),'never replace an activated plan')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with (job/'handoff.log').open('xb') as stream:
            child=subprocess.Popen([template['va_python'],'-B',str(OWN/'complete_cases.py'),
                '--prior-job',str(prior),'--control-dir',str(job/'handoff-control')],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                env=env,close_fds=True,start_new_session=True)
        pin=dict(**proc(child.pid),owner_job=str(job),role='bounded_handoff_control')
        write(job/f'OWNED-{child.pid}.json',pin)
        try:
            rc=child.wait(timeout=max(1,min(5500,deadline-time.time()-600)))
        except subprocess.TimeoutExpired:
            if matches(pin):os.killpg(pin['pgid'],signal.SIGTERM)
            try:rc=child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if matches(pin):os.killpg(pin['pgid'],signal.SIGKILL)
                rc=child.wait(timeout=10)
        write(job/'CONTROL-OWNED-CLEANUP.json',dict(pin=pin,reaped=True,returncode=rc,
            observed_process=proc(pin['pid']),source='actual wait return; no fabricated exit0'))
        if rc!=0:
            result=dict(status='HANDOFF_NOT_COMPLETED_PRIOR_STATUS_REQUIRES_EVIDENCE',control_returncode=rc,
                raw_control_result=read(job/'handoff-control/RESULT.json') if (job/'handoff-control/RESULT.json').exists() else None,
                no_new_GPU_worker=True,no_episode_retry=True)
            return 2
        # The prior original supervisor writes its own unmodified PARTIAL,
        # report and cleanup. That status is not relabelled COMPLETE here.
        while time.time()<deadline:
            c=prior/'GPU-OWNED-CLEANUP.json';g=prior/'GUARD-FINAL.json';s=prior/'SUPERVISOR-RESULT.json'
            if c.exists() and g.exists() and s.exists():
                clean=read(c)
                if clean.get('all_owned_work_exited') is True and clean.get('all_owned_groups_empty') is True and read(g).get('status')=='CLEAN':
                    break
            time.sleep(.3)
        else:raise TimeoutError('original supervisor cleanup did not close inside transition cap')
        complete,blocked,seen=[],[],[]
        for row in template['cases']:
            i=row['case_id'];d=prior/'episodes'/('case-%04d'%i)
            if not d.exists():continue
            if not any((d/a).exists() for a in ('T','C','P')):continue
            seen.append(i);ok=True
            for arm in ('T','C','P'):
                e=d/arm
                names=('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json')
                if not all((e/n).exists() for n in names):ok=False;continue
                t,f=read(e/names[0]),read(e/names[-1])
                ok=ok and t.get('complete_episode') is True and f.get('full_episode_terminal') is True and f.get('async_drained') is True
            (complete if ok else blocked).append(i)
        require(not blocked,'controlled full-case handoff left partial cases; do not auto-retry')
        debit=0
        for i in range(p0['max_GPU_workers']):
            live=read(prior/('server-worker%d'%i)/'LIVE-COUNTER.json')
            require(live['active_target_forwards']==0,'old target work remains active')
            debit+=live['target_forward_attempts']
        p=dict(template)
        p.update(prior_completed_case_ids=complete,blocked_prior_partial_case_ids=blocked,
            prior_target_forward_debit=debit,status='REGISTERED_COMPLETE_CASE_REUSE_NO_RETRIES',
            prior_original_supervisor_status=read(prior/'SUPERVISOR-RESULT.json')['status'],
            activated_at=datetime.now(timezone.utc).isoformat())
        pending=[r for r in p['cases'] if r['case_id'] not in set(complete)|set(blocked)]
        p['worker_forward_caps']={str(i):sum(r['execution_worker_index']==i for r in pending)*12000+250 for i in range(4)}
        require(sum(p['worker_forward_caps'].values())+debit<=p['job_forward_attempt_cap'],'fixed series forward cap')
        extra=Path(p['prior_attempted_transition_job'])
        outside=sum(f.stat().st_size for f in extra.rglob('*') if f.is_file())
        # The frozen evaluator scans prior+new. A named sparse accounting
        # reserve makes it also count this closed external preparation job.
        with (job/'EXTERNAL-PREPARATION-BUDGET.reserve').open('xb') as f:f.truncate(outside)
        write(job/'EXTERNAL-PREPARATION-BUDGET.json',dict(external_root=str(extra),
            external_logical_bytes=outside,reserve_is_accounting_only_not_tensor_data=True,
            actual_disk_allocation_must_be_reported_separately=True))
        p['external_preparation_evidence_bytes']=outside
        write(job/'plan.json',p)
        write(job/'PRIOR-READONLY-REUSE.json',dict(status='FULL_RAW_THREE_ARM_CASES_ONLY',
            prior_job=str(prior),completed_case_ids=complete,blocked_partial_case_ids=blocked,
            old_supervisor_status=p['prior_original_supervisor_status'],prior_debit=debit,
            prior_files_changed=False,four_permissions=p['frozen_permissions']))
        r=subprocess.run([p['va_python'],'-B',str(EVAL/'launch.py'),'--job',str(job)],
            stdin=subprocess.DEVNULL,capture_output=True,text=True,env=env,timeout=30)
        result=dict(status='FOUR_WORKER_RUNNER_STARTED' if r.returncode==0 else 'FOUR_WORKER_LAUNCH_FAILED',
            returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
            prior_complete_cases=len(complete),remaining_cases=len(pending),prior_forward_debit=debit,
            new_worker_forward_caps=p['worker_forward_caps'],no_episode_retry=True,
            original_absolute_deadline=p['absolute_cleanup_deadline'])
        return r.returncode
    except BaseException as exc:
        result=dict(status='TRANSITION_FAILED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        # An unexpected control termination must not strand the original CPU
        # runner paused. Check its exact previously registered lifetime first.
        observed=proc(prior_pin['pid'])
        if (job/'handoff-control/RUNNER-PAUSED-ADMITTED-CASES.json').exists() and observed is not None and observed['state'] in ('T','t') and matches(prior_pin):
            os.kill(prior_pin['pid'],signal.SIGCONT)
        orphan_cleanup=[]
        for f in (job/'handoff-control').glob('OWNED-MANUAL-*.json'):
            item=read(f)
            if item.get('controller_owner_job')!=str(job) or item.get('owner_job')!=str(prior):continue
            if matches(item) and proc(item['pid'])['state'] not in ('Z','X'):
                os.killpg(item['pgid'],signal.SIGTERM);stop=time.time()+10
                while matches(item) and proc(item['pid'])['state'] not in ('Z','X') and time.time()<stop:time.sleep(.1)
                if matches(item) and proc(item['pid'])['state'] not in ('Z','X'):
                    os.killpg(item['pgid'],signal.SIGKILL)
            orphan_cleanup.append(dict(pid=item['pid'],birth_ticks=item['birth_ticks'],observed=proc(item['pid']),
                only_birth_pinned_manual_control_groups=True))
        write(job/'MANUAL-CHILD-POSTCONTROL-OBSERVATION.json',dict(records=orphan_cleanup,
            no_original_server_or_foreign_process_signal=True))
        if result is not None:
            result.update(wall_sec=time.time()-started,finished_at=datetime.now(timezone.utc).isoformat())
            write(job/'TRANSITION-RESULT.json',result)


if __name__=='__main__':
    raise SystemExit(main())

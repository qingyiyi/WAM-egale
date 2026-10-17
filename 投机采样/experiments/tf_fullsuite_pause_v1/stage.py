"""Bounded stage-pause supervision; no new case, model, test or mathematical path."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OWN=Path(__file__).resolve().parent

def read(path):
    path=Path(path);assert path.stat().st_size<=8*2**20
    return json.loads(path.read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def proc(pid):
    try:
        p=Path('/proc')/str(pid);v=(p/'stat').read_text().rpartition(') ')[2].split()
        return dict(pid=pid,birth_ticks=int(v[19]),pgid=int(v[2]),uid=p.stat().st_uid,state=v[0])
    except (OSError,ValueError,IndexError):return None

def matches(pin):
    v=proc(pin['pid'])
    return bool(v and all(v[k]==pin[k] for k in ('birth_ticks','pgid','uid')) and v['pgid']==v['pid'])

def living(pin):return matches(pin) and proc(pin['pid'])['state'] not in ('Z','X')

def rescue(job):
    reg=read(job/'REGISTRATION.json');old=Path(reg['prior_job']);pin=read(job/'OWNED-pause.json')
    control=job/'control';limit=time.time()+30
    while not (control/'STARTED.json').is_file():
        assert living(pin) and time.time()<limit,'pause did not establish its identity'
        time.sleep(.1)
    oldpin=read(control/'STARTED.json')['prior_runner']
    write(control/'RESCUE-GUARD-STARTED.json',dict(**proc(os.getpid()),deadline_epoch=reg['control_deadline_epoch']))
    signals=[];reason='NORMAL_CONTROLLER_FINALLY'
    while True:
        if (control/'CONTROL-FINALLY.json').is_file() and not (matches(oldpin) and proc(oldpin['pid'])['state'] in ('T','t')):
            break
        if not living(pin) or time.time()>=reg['control_deadline_epoch']:
            reason='CONTROLLER_EXIT_OR_CONTROL_DEADLINE_RESCUE'
            if living(pin):
                os.killpg(pin['pgid'],signal.SIGTERM);signals.append(dict(pid=pin['pid'],signal='SIGTERM'))
                until=time.time()+15
                while living(pin) and time.time()<until:time.sleep(.2)
                if living(pin):os.killpg(pin['pgid'],signal.SIGKILL);signals.append(dict(pid=pin['pid'],signal='SIGKILL'))
            # This is an actual infrastructure failure, never a natural task failure.
            if not (old/'STOP_REQUESTED').exists():
                (old/'STOP_REQUESTED').write_text('migration pause control failure; retain PARTIAL, no retry\n')
            for f in control.glob('OWNED-MANUAL-*.json'):
                row=read(f)
                if row.get('controller_owner_job')!=str(job) or row.get('owner_job')!=str(old):continue
                if living(row):
                    os.killpg(row['pgid'],signal.SIGTERM);signals.append(dict(pid=row['pid'],signal='SIGTERM'))
            if matches(oldpin) and proc(oldpin['pid'])['state'] in ('T','t'):
                os.kill(oldpin['pid'],signal.SIGCONT);signals.append(dict(pid=oldpin['pid'],signal='SIGCONT'))
            break
        time.sleep(.5)
    write(job/'RESCUE-GUARD-FINAL.json',dict(reason=reason,signals=signals,
        prior_runner=proc(oldpin['pid']),foreign_processes_signalled=False))
    return 0

def supervise(job):
    reg=read(job/'REGISTRATION.json');old=Path(reg['prior_job'])
    assert os.sched_getaffinity(0)=={16} and os.getenv('CUDA_VISIBLE_DEVICES')==''
    assert reg['control_wall_cap_sec']==6000 and reg['no_new_cases'] and reg['no_new_model_worker']
    assert reg['source_code_commit'] not in ('','PENDING_NOT_EXECUTABLE')
    assert time.time()<reg['control_deadline_epoch']
    write(job/'STAGE-STARTED.json',dict(**proc(os.getpid()),at=datetime.now(timezone.utc).isoformat(),
        source_code_commit=reg['source_code_commit'],permissions=reg['permissions']))
    children=[]
    def spawn(role,argv):
        with (job/(role+'.log')).open('xb') as log:
            child=subprocess.Popen(list(map(str,argv)),stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
                start_new_session=True,close_fds=True)
        pin=proc(child.pid);assert pin and pin['pgid']==child.pid
        write(job/('OWNED-'+role+'.json'),dict(pin,owner_job=str(job),role=role));children.append((child,pin))
        return child,pin
    pause,pp=spawn('pause',[sys.executable,'-B',OWN/'pause.py','--prior-job',old,'--control-dir',job/'control'])
    guard,gp=spawn('rescue',[sys.executable,'-B',OWN/'stage.py','--job',job,'--rescue'])
    result=None
    try:
        pause.wait(timeout=max(1,reg['control_deadline_epoch']-time.time()-300))
        assert pause.returncode==0,'pause control did not complete; retain its FAILED'
        until=reg['control_deadline_epoch']
        while time.time()<until:
            names=('SUPERVISOR-RESULT.json','ALL-OWNED-CLEANUP.json','GUARD-FINAL.json')
            if all((old/n).is_file() for n in names):
                close=read(old/'ALL-OWNED-CLEANUP.json');g=read(old/'GUARD-FINAL.json')
                if close.get('all_owned_groups_empty') is True and g.get('status')=='CLEAN':
                    pins=[read(f) for f in old.glob('OWNED-*.json')]
                    extra=read(old/'GUARD-PROCESS.json');extra.setdefault('uid',os.getuid());pins.append(extra)
                    sys.path.insert(0,str(OWN.parent))
                    from tf_fullsuite_eval_v2.runner import forensic_groups
                    proof=forensic_groups(pins)
                    if proof['scan_complete'] and not any(x['members'] for x in proof['groups']):
                        write(job/'ORIGINAL-REAL-OWNED-CLEANUP.json',proof);break
            time.sleep(1)
        else:raise TimeoutError('original partial supervisor/guard cleanup did not close')
        guard.wait(timeout=max(1,min(30,reg['control_deadline_epoch']-time.time())))
        assert guard.returncode==0
        p=read(old/'plan.json');prior=Path(p['prior_job']);prior_ids=set(p['prior_completed_case_ids'])
        complete=[];unstarted=[];blocked=[]
        for cid in range(500):
            source=prior if cid in prior_ids else old;folder=source/'episodes'/('case-%04d'%cid)
            attempted=any((folder/a).exists() for a in ('T','C','P'));valid=True
            for arm in ('T','C','P'):
                e=folder/arm;names=('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json')
                if not all((e/n).is_file() for n in names):valid=False;continue
                t,c,a,f=[read(e/n) for n in names]
                valid=valid and t.get('complete_episode') is True and t.get('quality_terminal') is True
                valid=valid and all(v.get('case_id')==cid and v.get('arm')==arm for v in (t,c,a,f))
                valid=valid and f.get('full_episode_terminal') is True and f.get('async_drained') is True
            entry=dict(case_id=cid,task=cid%10,init=cid//10,source_job=str(source))
            if valid:complete.append(entry)
            elif attempted:blocked.append(entry)
            else:unstarted.append(entry)
        write(job/'MIGRATION-COVERAGE.json',dict(complete_cases=complete,unstarted_cases=unstarted,
            partial_cases_no_automatic_retry=blocked,full_case_resume_only=True,old_supervisor_status=read(old/'SUPERVISOR-RESULT.json')['status'],
            independent_TEST=False,formal_performance=False,permissions=reg['permissions']))
        result=dict(status='PAUSED_AFTER_COMPLETE_ADMITTED_CASES_AND_REAL_CLEANUP',
            complete_cases=len(complete),complete_episodes=3*len(complete),unstarted_cases=len(unstarted),partial_cases=len(blocked),
            episode_truncation=False,original_supervisor_status=read(old/'SUPERVISOR-RESULT.json')['status'],
            evidence_backup='PENDING_NOT_YET_SAFE_TO_DESTROY_INSTANCE',instance_shutdown='NOT_REQUESTED_NOT_EXECUTED',permissions=reg['permissions'])
        return 0
    except BaseException as exc:
        result=dict(status='FAILED_OR_PARTIAL_STAGE_CONTROL_NO_RETRY',error=repr(exc),
            evidence_backup='UNKNOWN_NOT_SAFE_TO_DESTROY_INSTANCE',permissions=reg['permissions']);raise
    finally:
        if result is not None:write(job/'RESULT.json',result)
        write(job/'PARENT-CHILD-OBSERVATION.json',dict(children=[dict(pin=pin,returncode=c.poll(),observed=proc(pin['pid'])) for c,pin in children],
            note='rescue independently retains its registered deadline if control remains live'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--rescue',action='store_true');a=p.parse_args()
    return rescue(a.job.resolve(strict=True)) if a.rescue else supervise(a.job.resolve(strict=True))

if __name__=='__main__':raise SystemExit(main())

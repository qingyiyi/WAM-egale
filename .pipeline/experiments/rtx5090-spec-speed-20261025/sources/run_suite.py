"""One actual full-episode timing panel; no shortened episodes or hidden retries."""
from pathlib import Path
import argparse,json,os,subprocess,time,traceback
from owned import identity,launch,stop
P=Path('/root/autodl-tmp/cxz/spec-speed-20261025')
parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--attempt',default='01');parser.add_argument('--server-attempt',required=True);parser.add_argument('--start-index',type=int,default=0);a=parser.parse_args()
R=P/'results'/a.suite;J=R/('panel-'+a.attempt);J.mkdir(parents=True,exist_ok=False)
registration=json.loads((P/'REGISTRATION.json').read_text())
from datetime import datetime
deadline=datetime.fromisoformat(registration['absolute_work_stop']).timestamp()
state=dict(status='WAITING_SERVER',start=time.time(),suite=a.suite,attempt=a.attempt,completed=[],attempts=[],deadline=deadline,preparation_only_repairs='No completed or executed partial episode automatically retried')
active=None;server=json.loads((R/'ACTIVE-SERVER.json').read_text())
server_status=R/('server-'+a.server_attempt)/'STATUS.json'
def save():
    path=J/'STATUS.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
def read(path):
    return json.loads(path.read_text()) if path.exists() else {}
def gpu_processes():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True,capture_output=True,timeout=15)
    return dict(returncode=r.returncode,output=r.stdout.strip(),error=r.stderr.strip())
try:
    save();ready_limit=min(time.time()+900,deadline)
    while read(server_status).get('status') not in ['READY','SERVING']:
        if read(server_status).get('status')=='FAILED':raise RuntimeError('Server preparation failed: '+str(read(server_status).get('error')))
        if time.time()>ready_limit:raise TimeoutError('Server setup deadline')
        time.sleep(2)
    cases=(json.loads((P/'manifests/robotwin-selected49.json').read_text()) if a.suite=='robotwin' else list(range(10)))
    sequence=[]
    for i,case in enumerate(cases):
        for arm in ['TCP','CPT','PTC'][i%3]:sequence.append((i,case,arm))
    cpus=sorted(os.sched_getaffinity(0));client_cores=cpus[2:4]
    for index,(i,case,arm) in enumerate(sequence):
        if index<a.start_index:continue
        cap=1800 if a.suite=='robotwin' else 1200
        if time.time()+cap+180>deadline:state['status']='PARTIAL_NO_WHOLE_EPISODE_BUDGET';break
        if os.statvfs(P).f_bavail*os.statvfs(P).f_frsize<5*2**30:raise RuntimeError('5GiB data disk floor')
        state.update(status='RUNNING_COMPLETE_EPISODE',active_index=index,active_case=case,active_arm=arm);save()
        episode=J/'episodes'/f'{i:02d}-{arm}'
        if a.suite=='robotwin':
            c=dict(case,arm=arm,episode_id=case['task']+'-valid049-'+arm);path=J/f'case-{index:02d}.json';path.write_text(json.dumps(c))
            invocation=f'{P}/client310/bin/python3.10 -u {P}/client_robotwin.py {path} {episode}'
        else:invocation=f'{P}/client310/bin/python3.10 -u {P}/client_libero.py {a.suite} {case} {arm} {episode}'
        command=['bash','-c',f'source {P}/runtime_env.sh; exec taskset -c {",".join(map(str,client_cores))} {invocation}']
        active,row=launch(f'{a.suite}-panel{a.attempt}-{index:02d}',command)
        before=read(server_status);record=dict(index=index,case=case,arm=arm,output=str(episode),start=time.time(),process=row,teacher_before=before.get('teacher_attempted',0),gpu_before=gpu_processes());state['attempts'].append(record);save()
        while active.poll() is None:
            current=read(server_status)
            if time.time()-record['start']>cap or time.time()>deadline:
                record['stop_reason']='REGISTERED_TIMEOUT';record['cleanup']=stop(row);active.wait(timeout=20);break
            if current.get('teacher_attempted',0)-record['teacher_before']>(8000 if a.suite=='robotwin' else 4000):
                record['stop_reason']='FORWARD_CAP';record['cleanup']=stop(row);active.wait(timeout=20);break
            time.sleep(2)
        record.update(returncode=active.returncode,end=time.time(),teacher_after=read(server_status).get('teacher_attempted',0),gpu_after=gpu_processes());active=None
        if a.suite=='robotwin':
            terminal=read(episode/'STATUS.json');complete=terminal.get('status') in ['COMPLETE_SUCCESS','COMPLETE_ALGORITHMIC_FAILURE']
        else:
            f=episode/'episodes.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else [];terminal=rows[-1] if rows else {};complete=len(rows)==1 and terminal.get('status')=='completed'
        record.update(terminal=terminal,complete=complete,lifecycle_seconds=record['end']-record['start']);save()
        if not complete or record['returncode']!=0:
            state.update(status='PARTIAL_OR_INVALID',reason='Preserved full attempt; no automatic episode retry');break
        state['completed'].append(index);save()
    else:state['status']='COMPLETE_PREDEFINED_PANEL'
except BaseException as e:
    state.update(status='FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:
    if active is not None and active.poll() is None:state['client_cleanup']=stop(state['attempts'][-1]['process'])
    # A setup-only client failure can be repaired without reloading this ready model.
    last=state['attempts'][-1] if state['attempts'] else None
    zero_teacher=last is not None and last.get('teacher_after',0)==last.get('teacher_before',0)
    if state['status']=='PARTIAL_OR_INVALID' and zero_teacher and time.time()+300<deadline:
        state['ready_server_retained_for_zero_teacher_setup_repair']=True
    else:state['server_cleanup']=stop(server)
    state.update(end=time.time(),lifecycle_seconds=time.time()-state['start'],gpu_final=gpu_processes());save()

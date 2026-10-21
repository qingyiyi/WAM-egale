"""Bounded Target48 -> CPU audit -> fixed GPU PARENT2000/CAL. No robot auto-retry."""
from pathlib import Path
import os,subprocess,json,time,datetime,signal,traceback,fcntl
B=Path(__file__).resolve().parent;R=B.parent;PY='/root/cxz/envs/va/bin/python'
GPU='GPU-4acf84ff-31ce-dead-c2a4-2a122b123b8c'
os.sched_setaffinity(0,{18,19})
reg=json.loads((B/'PLAN.json').read_text());deadline=datetime.datetime.fromisoformat(reg['deadline']).timestamp()
state=dict(status='PREPARING_CHILD_CONT',start=time.time(),jobs=[],GPU_lifecycle_seconds=0.,
           downstream_robot_started=False,permissions={k:False for k in ['engineering_quality_screen','independent_test_complete','noninferiority_established','formal_performance_permission']})
def save():
 p=B/'STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
def ident(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
def alive(x):
 try:return ident(x['pid'])==x
 except (FileNotFoundError,ProcessLookupError):return False
save();lease=None
try:
 assert json.loads((R/'parent-audit-complete18-20261021/RESULT.json').read_text())['status']=='PASS_COMPLETE_FIXED18_REACHED_SUPERVISION'
 state['status']='RUNNING_FIXED_CHILD_CONT_INTERFACE';save()
 for name,cap,gpu in [('CHILD2500',300,True),('CONT3000',300,True)]:
    assert time.time()+cap+60<deadline
    lease=(R/'GPU1.lease').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=json.loads((R/'GPU-LEDGER.json').read_text());assert ledger['consumed_seconds']+cap+60<=ledger['cap_seconds']
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(x.startswith(GPU) for x in apps.splitlines())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'),CUDA_CACHE_PATH=str(B/'cache/cuda'))
    for p in ['tmp','cache/cuda']:(B/p).mkdir(parents=True,exist_ok=True)
    with (B/f'{name}.log').open('x') as f:p=subprocess.Popen([PY,str(R/'tf-migration-20261021/cal_fixed_branch.py'),str(B/(name+'-PLAN.json')),str(B/name)],cwd=B,env=env,stdout=f,stderr=f,start_new_session=True)
    row=dict(name=name,owned=ident(p.pid),start=time.time(),gpu=gpu);state['jobs'].append(row);save()
    try:rc=p.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        row['timeout']=True
        if alive(row['owned']):os.killpg(row['owned']['pgid'],signal.SIGTERM)
        try:rc=p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if alive(row['owned']):os.killpg(row['owned']['pgid'],signal.SIGKILL)
            rc=p.wait(timeout=15)
    row.update(returncode=rc,end=time.time(),lifecycle_seconds=time.time()-row['start'],root_reaped=True,identity_alive=alive(row['owned']));save()
    if gpu:
        state['GPU_lifecycle_seconds']+=row['lifecycle_seconds']
        with (R/'GPU-LEDGER.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads((R/'GPU-LEDGER.json').read_text())
            ledger['consumed_seconds']+=row['lifecycle_seconds'];ledger['jobs'].append(dict(namespace=str(B),**row,GPU_lifecycle_seconds=row['lifecycle_seconds']))
            temp=R/'GPU-LEDGER.json.tmp';temp.write_text(json.dumps(ledger,indent=2));temp.replace(R/'GPU-LEDGER.json')
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close();lease=None
    if rc or row.get('timeout'):raise RuntimeError(name+' failed; no automatic retry')
 state['status']='COMPLETE_FIXED_CHILD_CONT_INTERFACE_ONLY'
except BaseException as e:state.update(status='FAILED_OR_BLOCKED_DOWNSTREAM',error=repr(e),traceback=traceback.format_exc())
finally:
 if lease:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
 state['end']=time.time();state['wall_seconds']=time.time()-state['start'];save()

"""New, explicitly authorized four-episode diagnosis; no retention override."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import subprocess

OWN=Path(__file__).resolve().parent
EXP=OWN.parent
ROOT=EXP.parent.parent
RUNTIME=ROOT/'投机采样/results/runtime'
WORKTREE=ROOT/'投机采样/wam-p0-worktree'
BASE=RUNTIME/'p0-draft-freeze-quality-v1-local-20261016-r1'
SNAPSHOT=ROOT/'review_exports/WAM-egale'
COMMIT='b0dfac3a02f65de33a1217246b83003615166640'
ARCH={'TF_CONT':'transformer64'}
ARMS=('T','TF_CONT')
PERMISSIONS=dict(engineering_quality_screen=False,independent_test_complete=False,
    noninferiority_established=False,formal_performance_permission=False)

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value,*,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        temp=path.with_name(path.name+'.pending');temp.write_text(raw);temp.replace(path)
    else:
        with path.open('x') as stream:stream.write(raw)

def cases():
    return [dict(arm=a,task=4,init=26,group='known_risk_diagnostic',rep=r,text_bank=True,
        key=f'{a}-task4-init26-repeat{r}') for a in ARMS for r in range(2)]

def validate(p,job,*,verify=True):
    if (p.get('schema')!='task4-direction-diagnostic/v1' or p.get('job')!=str(job)
            or p.get('commit')!=COMMIT or p.get('cases')!=cases() or p.get('arms')!=list(ARMS)
            or p.get('horizon')!=800 or p.get('seed')!=20260922
            or p.get('mode')!='verifier_k2_b2_bonus' or p.get('selector')!='prefix_then_cost'
            or p.get('threshold')!=0.007839680526450805 or p.get('max_run_wall_sec')!=2400
            or p.get('cleanup_reserve_sec')!=300 or p.get('output_budget_GiB')!=1
            or p.get('permissions')!=PERMISSIONS or p.get('workers')!=1
            or p.get('performance_claim') is not False or p.get('absolute_no_AutoDL') is not True):
        raise ValueError('new diagnostic contract differs')
    if verify:
        if any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('source changed after registration')
        if sha(job/'PREREGISTRATION.json')!=p['preregistration_sha256']:raise ValueError('registration changed')
        if sha(p['selected']['TF_CONT']['checkpoint'])!=p['selected']['TF_CONT']['checkpoint_sha256']:
            raise ValueError('fixed TF fixture checkpoint changed')
        for path,stat in p['target_weight_stats'].items():
            s=Path(path).stat()
            if s.st_size!=stat['bytes'] or s.st_mtime_ns!=stat['mtime_ns']:raise ValueError('target stat changed')
        for key in ('bank','real_CAL_data'):
            path=Path(p[key] if key=='bank' else p[key]['path'])
            stat=p['bank_stat'] if key=='bank' else p['real_CAL_data']
            s=path.stat()
            if s.st_size!=stat['bytes'] or s.st_mtime_ns!=stat['mtime_ns']:raise ValueError(key+' stat changed')
    return p

def load(job,*,verify=True):
    job=Path(job).resolve(strict=True);return validate(read(job/'plan.json'),job,verify=verify)

def prepare(job):
    job=Path(job).resolve()
    if not job.name.startswith('p0-task4-direction-diagnostic-v1-') or (job/'plan.json').exists():
        raise ValueError('fresh independent diagnostic namespace required')
    head=subprocess.check_output(['git','-C',str(SNAPSHOT),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:raise ValueError('pinned checkout differs')
    p=read(BASE/'plan.json')
    inventory=p['source_sha256']
    pinned=0
    for path,value in inventory.items():
        path=Path(path)
        if sha(path)!=value:raise ValueError('old frozen inventory changed '+str(path))
        counterpart=SNAPSHOT/path.relative_to(ROOT)
        if path.suffix=='.py':
            if not counterpart.is_file() or path.read_bytes()!=counterpart.read_bytes():
                raise ValueError('Python differs from fixed public commit '+str(path))
            pinned+=1
    all_sources={**inventory,**{str(f):sha(f) for f in sorted(OWN.glob('*.py'))}}
    # Isolated diagnosis has a new allowance; closed old jobs/retention are read-only.
    now=datetime.now(timezone.utc)
    hard=now+timedelta(seconds=2400)
    prereg=dict(goal='Locate teacher/draft/path/scheduler/decode/submission/environment divergence on known task4 risk',
        commit=COMMIT,created_at=now.isoformat(),cases=cases(),full_episodes=4,pairs=2,
        scientific_role='known, previously unblinded diagnostic DEV; NOT independent TEST',
        arm_order=list(ARMS),within_arm_order=[0,1],paired_by_repeat=True,
        fixed_TF_checkpoint=p['selected']['TF_CONT'],all_old_retention_unchanged=True,
        permissions=PERMISSIONS,threshold=p['threshold'],production_API_unchanged=True,
        observation_only_generated_AST_reverse_exact=True,extra_target_forwards_for_logging=0,
        real_CAL_adapter_check_rows=20237,required_log_scope='every complete policy chunk and all action scheduler updates',
        bounded_tensor_scope='compact action states/directions plus existing episode artifacts; no per-query 30-layer KV copies',
        GPU='local NVIDIA A100-SXM4-40GB',GPU_workers=1,shared_quality_GPU_allowed=True,
        CPU_worker_affinity=[12,13,14,15],CPU_server_threads=2,CPU_client_threads=2,guard_CPUs=[0,1],
        max_run_wall_sec=2400,cleanup_reserve_sec=300,storage_cap_bytes=2**30,
        hard_absolute_deadline=hard.isoformat(),task_work_deadline=(hard-timedelta(seconds=300)).isoformat(),
        cleanup='pinned own roots and descendants only; independent attached watchdog; never signal other users',
        timeout='PARTIAL/FAILED with all completed and interrupted episodes retained; no auto retry',
        training_trigger='specific reachable recursive/target mismatch evidence with a falsifiable control; local error alone insufficient',
        evidence_path=str(job),terminal_status='REGISTERED_NOT_STARTED',
        original_frozen_sources=len(inventory),pinned_Python_sources=pinned)
    write(job/'PREREGISTRATION.json',prereg)
    p.update(schema='task4-direction-diagnostic/v1',profile='task4_diagnostic',job=str(job),commit=COMMIT,
        cases=cases(),arms=list(ARMS),planned_full_episodes=4,site='local',
        global_started=now.isoformat(),absolute_work_stop=prereg['task_work_deadline'],
        absolute_cleanup_deadline=prereg['hard_absolute_deadline'],max_run_wall_sec=2400,
        cleanup_reserve_sec=300,output_budget_GiB=1,workers=1,absolute_no_AutoDL=True,
        job_target_forward_cap=16000,per_episode_target_forward_cap=4000,per_episode_wall_sec=900,
        port=19568,master_port=29568,source_sha256=all_sources,permissions=PERMISSIONS,
        preregistration=str(job/'PREREGISTRATION.json'),preregistration_sha256=sha(job/'PREREGISTRATION.json'))
    validate(p,job);write(job/'plan.json',p)
    return p

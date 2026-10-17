"""Explicit completion of the same two diagnostic pairs, cumulative budget."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from task4_direction_diagnostic_v1.protocol import ROOT,RUNTIME,WORKTREE,COMMIT,PERMISSIONS,ARCH,sha,read,write

OWN=Path(__file__).resolve().parent
EXP=OWN.parent
FIRST=RUNTIME/'p0-task4-direction-diagnostic-v1-20261016-r1'
ARMS=('TF_CONT',)

def cases():return [x for x in read(FIRST/'PREREGISTRATION.json')['cases'] if x['arm']=='TF_CONT']

def validate(p,job,*,verify=True):
    if (p.get('schema')!='task4-direction-diagnostic-completion/v1' or p.get('job')!=str(job)
            or p.get('commit')!=COMMIT or p.get('cases')!=cases() or p.get('arms')!=['TF_CONT']
            or p.get('max_run_wall_sec')!=3600 or p.get('cumulative_diagnostic_cap_sec')!=3600
            or p.get('horizon')!=800 or p.get('seed')!=20260922
            or p.get('threshold')!=0.007839680526450805 or p.get('permissions')!=PERMISSIONS
            or p.get('mode')!='verifier_k2_b2_bonus' or p.get('selector')!='prefix_then_cost'
            or p.get('performance_claim') is not False):raise ValueError('completion contract differs')
    if verify:
        if any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('frozen completion source changed')
        for k,v in p['prior_evidence_sha256'].items():
            if sha(k)!=v:raise ValueError('prior actual Attempt evidence changed')
        if sha(job/'PREREGISTRATION.json')!=p['preregistration_sha256']:raise ValueError('completion registration changed')
        for k,st in p['target_weight_stats'].items():
            s=Path(k).stat()
            if (s.st_size,s.st_mtime_ns)!=(st['bytes'],st['mtime_ns']):raise ValueError('target weight stat changed')
        if sha(p['selected']['TF_CONT']['checkpoint'])!=p['selected']['TF_CONT']['checkpoint_sha256']:
            raise ValueError('fixed TF changed')
    return p

def load(job,*,verify=True):
    job=Path(job).resolve();return validate(read(job/'plan.json'),job,verify=verify)

def prepare(job):
    p=read(FIRST/'plan.json');job=Path(job).resolve()
    if job.exists():raise FileExistsError(job)
    failed=read(FIRST/'FAILED.json');clock=read(FIRST/'CLOCK-RESULT.json')
    if (not clock['all_observed_owned_live_processes_exited']
            or not read(FIRST/'FINAL-OWNED-CLEANUP.json')['all_roots_reaped']
            or read(FIRST/'status.json')['completed']!=[x['key'] for x in p['cases'] if x['arm']=='T']):
        raise ValueError('two full Targets and closed original FAILED required')
    start=datetime.fromisoformat(p['global_started']);hard=start+timedelta(seconds=3600)
    if datetime.now(timezone.utc)>=hard-timedelta(seconds=300):raise TimeoutError('same experiment cumulative budget expired')
    evidence=[FIRST/'FAILED.json',FIRST/'FINAL-OWNED-CLEANUP.json',FIRST/'CLOCK-RESULT.json',FIRST/'CLOCK-GUARD.json',
        FIRST/'plan.json',FIRST/'PREREGISTRATION.json',FIRST/'T-READY.json',
        FIRST/'arms/TF_CONT/FULL-REAL-CAL-PARITY.json']
    for c in p['cases']:
        if c['arm']=='T':
            folder=FIRST/'episodes'/c['key']
            evidence.extend(folder/x for x in ('FINISH_READY.json','CLIENT-CLEANUP.json','COUNTER.json','client/episodes.jsonl','client/repro_metadata.json'))
    sources={**p['source_sha256'],**{str(f):sha(f) for f in OWN.glob('*.py')}}
    prereg=dict(goal='Complete ONLY the two originally planned TF full diagnostics after observation binding failure',
        commit=COMMIT,created_at=datetime.now(timezone.utc).isoformat(),cases=cases(),
        fixed_full_Target_reference_job=str(FIRST),independent_TEST=False,original_FAILED_preserved=True,
        quality_permissions=PERMISSIONS,existing_full_GPU_CAL_reused=True,new_CAL_rows=0,
        correction='inspect.unwrap(original).__globals__ instead of decorator wrapper globals',
        old_attempt_charge_sec=clock['elapsed_quality_process_seconds'],cumulative_cap_sec=3600,
        budget_origin='earliest actual first Attempt birth AND conservative first registration absolute clock; no reset',
        absolute_cleanup_deadline=hard.isoformat(),absolute_work_stop=(hard-timedelta(seconds=300)).isoformat(),
        GPU='local A100-SXM4-40GB',CPU_worker_affinity=[12,13,14,15],server_threads=2,client_threads=2,
        single_worker=True,storage_cap_bytes=2**30,cleanup_reserve_sec=300,
        cleanup='independent cumulative watchdog; pinned own processes only',
        timeout='original and completion FAILED/PARTIAL retained; no further retry',
        terminal_status='REGISTERED_NOT_STARTED',evidence_path=str(job),
        explanation='20–40 minutes was an estimate; explicit revised finite allowance completes same requested pairs, no new research arm or GPU rental')
    job.mkdir();write(job/'PREREGISTRATION.json',prereg)
    p.update(schema='task4-direction-diagnostic-completion/v1',job=str(job),arms=['TF_CONT'],cases=cases(),
        planned_full_episodes=2,max_run_wall_sec=3600,cumulative_diagnostic_cap_sec=3600,
        absolute_cleanup_deadline=prereg['absolute_cleanup_deadline'],absolute_work_stop=prereg['absolute_work_stop'],
        source_sha256=sources,prior_evidence_sha256={str(f):sha(f) for f in evidence},
        preregistration=str(job/'PREREGISTRATION.json'),preregistration_sha256=sha(job/'PREREGISTRATION.json'),
        prior_job=str(FIRST),reference_target_READY=str(FIRST/'T-READY.json'),
        existing_CAL_certificate=str(FIRST/'arms/TF_CONT/FULL-REAL-CAL-PARITY.json'),
        existing_CAL_sha256=sha(FIRST/'arms/TF_CONT/FULL-REAL-CAL-PARITY.json'),
        target_reference_cases=[c for c in read(FIRST/'plan.json')['cases'] if c['arm']=='T'])
    validate(p,job);write(job/'plan.json',p);return p

"""Bounded full-episode current-TF data collection and later fixed DEV."""
from datetime import datetime,timedelta,timezone
import subprocess
from pathlib import Path
from tf_effect_closure_v1.protocol import ROOT,EXP,RT,BASE,COMMIT,PERMISSIONS,read,write,sha,require,environment
OWN=Path(__file__).resolve().parent
FIRST=RT/'p0-tf-effect-collector-qualification-v1-20261016-r1'
QUAL=RT/'p0-tf-effect-collector-qualification-v1-20261016-r3'
RISK=((1,29),(3,26),(4,26),(5,28),(6,27),(8,25),(0,28),(6,25))

def cases(collect):
    def row(arm,t,i,role):return dict(arm=arm,task=t,init=i,role=role,rep=0,key=f'{arm}-{role}-t{t}-i{i}-r0')
    if collect:return [row('TF_CONT',t,i,'FIT' if i<32 else 'CAL') for i in (30,31,32) for t in range(10)]
    rows=[row(a,t,i,'fresh_DEV') for a in ('T','TF_CONT','TF_ON','MLP_ON') for i in (36,37) for t in range(10)]
    rows += [row(a,t,i,'risk_DEV') for a in ('T','TF_ON') for t,i in RISK]
    require(len(rows)==96,'finite DEV size differs');return rows

def inventory(q):
    files=list(OWN.glob('*.py'))+list((EXP/'tf_effect_collector_v1').glob('*.py'))+list((EXP/'tf_effect_train_v1').glob('*.py'))
    files += [EXP/'tf_effect_closure_v1/protocol.py',EXP/'tf_effect_closure_guard_v2/clock.py']
    return {**q['source_sha256'],**{str(f):sha(f) for f in files}}

def prepare(job,*,collect,audit=None,training=None,trigger=None):
    job=Path(job).resolve();require(job.parent==RT and not job.exists(),'fresh isolated job required')
    require(subprocess.check_output(['git','-C',str(ROOT/'review_exports/WAM-egale'),'rev-parse','HEAD'],text=True).strip()==COMMIT,'fixed public base differs')
    q=read(BASE/'plan.json')
    for f,h in q['source_sha256'].items():require(sha(f)==h,'old frozen source changed '+f)
    require((OWN/'server_entry.py').exists() and (OWN/'client_entry.py').exists(),'reviewed full campaign source required')
    first=read(FIRST/'PREREGISTRATION.json');global_hard=datetime.fromisoformat(first['created_at'])+timedelta(hours=12)
    now=datetime.now(timezone.utc);cap=14400 if collect else 18000;hard=min(now+timedelta(seconds=cap),global_hard)
    require((hard-now).total_seconds()>900,'global12h envelope exhausted')
    evidence=[QUAL/'WORKER-RESULT.json',QUAL/'SUPERVISOR-RESULT.json',QUAL/'CLOCK-RESULT.json']
    require(read(evidence[0])['status']=='PASS_FULL_RECORDED_EPISODE_COLLECTOR_EQUIVALENCE_BASELINE_REUSED'
        and read(evidence[0])['reset_and_final_full_valid_KV_RNG_exact'] is True
        and read(evidence[0])['chunk_RNG_exact'] is True,'full ON recorded episode versus original OFF not qualified')
    require(read(evidence[1])['status']=='COMPLETE' and read(evidence[2])['all_observed_owned_live_processes_exited'],'qualification not cleanly complete')
    require(audit is not None,'full qualification CPU audit required');audit=Path(audit).resolve(strict=True)
    require(read(audit)['status']=='PASS_FULL_50_CHUNK_QUALIFICATION_AUDIT_BASELINE_REUSED','collector numeric audit not passed');evidence.append(audit)
    identity=RT/'p0-tf-effect-split-identity-all38-cpu-v1-20261016-r1/RESULT.json'
    require(read(identity)['all_new_initial_states_exact_unique_within_task'] is True,'exact initial-state split overlap')
    master=RT/'p0-tf-effect-closure-v1-20261016-r1/PREREGISTRATION.json'
    require(read(master)['collection_cases']==cases(True) and read(master)['DEV_cases_frozen_before_new_data']==cases(False),'prospectively fixed split/cases changed')
    evidence += [identity,master]
    selected={'TF_CONT':q['selected']['TF_CONT']};arms=['TF_CONT'];gpu_cal_jobs={}
    if not collect:
        require(training is not None,'complete new training required')
        training=Path(training).resolve(strict=True);f=read(training/'FINAL-RESULT.json')
        require(f.get('status')=='PASS_FIXED_FINAL_TF_EFFECT_TRAINING','complete finite training required')
        selection=read(training/'selected.json')
        for arm in ('TF_ON','MLP_ON'):
            cell=selection['conditions'][arm]
            selected[arm]={**cell,'condition':arm,'training_dir':str(training)}
            gpu_cal_jobs[arm]=str(job/'cal'/arm)
        arms=['T','TF_CONT','TF_ON','MLP_ON'];evidence += [training/'FINAL-RESULT.json',training/'selected.json']
    p=dict(schema='tf-effect-full-campaign/v1',job=str(job),commit=COMMIT,kind='collection' if collect else 'DEV',
        collect=collect,arms=arms,cases=cases(collect),reference_plan=q,selected=selected,gpu_cal_jobs=gpu_cal_jobs,
        goal='one current-TF complete FIT20/CAL10 data opportunity' if collect else 'fixed TF-cont/TF-on paired complete DEV and known-risk regression',
        collector_qualification_evidence=[str(f) for f in evidence[:4]],
        created_at=now.isoformat(),source_sha256=inventory(q),evidence_sha256={str(f):sha(f) for f in evidence},
        permissions=PERMISSIONS,performance_claim=False,horizon=800,seed=20260922,
        mode=q['mode'],selector=q['selector'],threshold=q['threshold'],bank=q['bank'],
        max_run_wall_sec=cap,cleanup_reserve_sec=300,absolute_cleanup_deadline=hard.isoformat(),
        absolute_work_stop=(hard-timedelta(seconds=300)).isoformat(),global_cleanup_deadline=global_hard.isoformat(),
        global_budget_origin=str(FIRST/'PREREGISTRATION.json'),global_budget_no_reset=True,
        storage_cap_bytes=(2 if collect else 3)*2**30,collected_teacher_cap_bytes=512*2**20,
        disk_floor_bytes=5*2**30,initial_free_GiB=22,runtime_free_GiB=.5,
        job_target_forward_cap=(120000 if collect else 280000),per_episode_target_forward_cap=4000,
        per_episode_wall_sec=1200,model_ready_wall_sec=900,planned_full_episodes=len(cases(collect)),
        port=19668,master_port=29668,workers=1,shared_GPU_allowed=True,GPU='local A100-SXM4-40GB',
        CPU_affinity=[12,13,14,15],server_affinity=[12,13],client_affinity=[14,15],server_threads=2,client_threads=2,
        EGL=True,single_worker_reason='two target workers need at least44GiB beyond available34GiB; one persistent model and one EGL client',
        row_selection='root_and_actually_reached_parent',FIT_inits=[30,31],CAL_inits=[32],
        DEV_inits=[36,37],known_risk_pairs=[list(v) for v in RISK],TEST_inits_unread=list(range(40,50)),
        formal_performance_permission=False,no_120=True,no_AutoDL=True,no_architecture_sweep=True,
        no_threshold_change=True,no_policy_or_engine_change=True,strict_paired_initial_state_and_RNG=True,
        decision_rule=dict(TF_success_at_least_target=True,TF_success_at_least_matched_MLP=True,
            both_success_step_ratio_P50_max=1.10,P90_max=1.20,max_ratio=1.30,no_new_risk_failure=True,
            quality_first=True,no_CAL_checkpoint_selection=True),
        cleanup='independent actual-birth watchdog plus strictly pinned owned roots only',
        timeout='full completed episodes retained; interrupted episode infrastructure PARTIAL, not algorithm failure; no auto retry',
        terminal_status='REGISTERED_NOT_STARTED',evidence_path=str(job))
    if trigger is not None:p['training_trigger_certificate']=str(Path(trigger).resolve(strict=True))
    job.mkdir();write(job/'PREREGISTRATION.json',p);write(job/'plan.json',{**p,'preregistration_sha256':sha(job/'PREREGISTRATION.json')})
    return p

def load(job,verify=True):
    job=Path(job).resolve(strict=True);p=read(job/'plan.json')
    require(p['schema']=='tf-effect-full-campaign/v1' and p['job']==str(job) and p['commit']==COMMIT,'campaign identity differs')
    require(p['permissions']==PERMISSIONS and p['performance_claim'] is False and p['cases']==cases(p['collect']),'frozen cases/permissions differ')
    require(sha(job/'PREREGISTRATION.json')==p['preregistration_sha256'],'registration changed')
    if verify:
        for f,h in p['source_sha256'].items():require(sha(f)==h,'frozen source changed '+f)
        for f,h in p['evidence_sha256'].items():require(sha(f)==h,'qualification evidence changed '+f)
        for f,st in p['reference_plan']['target_weight_stats'].items():
            s=Path(f).stat();require((s.st_size,s.st_mtime_ns)==(st['bytes'],st['mtime_ns']),'target changed')
        for arm,row in p['selected'].items():require(sha(row['checkpoint'])==row['checkpoint_sha256'],'candidate changed '+arm)
    return p

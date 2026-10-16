"""Prospective today50 DEV protocol. Original usability contracts stay frozen."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = EXP.parent.parent
RUNTIME = ROOT/'投机采样/results/runtime'
WORKTREE = ROOT/'投机采样/wam-p0-worktree'
ARMS = ('T', 'TF_CONT', 'MLP_CONT')
ARCH = {'TF_CONT': 'transformer64', 'MLP_CONT': 'mlp96'}
ROOT_JOB = RUNTIME/'p0-draft-freeze-v1-20261016-r1'
BASE_JOB = RUNTIME/'p0-tf-usable-quality-v1-local-i33-20261016-r4'
TF_TRAIN = RUNTIME/'p0-tf-usable-train-v1-20261016-r1'
MLP_TRAIN = RUNTIME/'p0-tf-usable-mlp-continue-v1-20261016-r1'
GLOBAL_STARTED = '2026-10-16T12:20:29+08:00'
WORK_STOP = '2026-10-16T18:15:29+08:00'
FINAL_CLEANUP = '2026-10-16T18:20:29+08:00'
THRESHOLD = 0.007839680526450805
RISK = ((1,29),(3,26),(4,26),(5,28),(6,27),(8,25),(0,28),(6,25))

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value, *, replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        tmp=path.with_name(path.name+'.pending');tmp.write_text(raw);tmp.replace(path)
    else:
        with path.open('x') as out:out.write(raw)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cases():
    def row(arm,task,init,group,rep=0):
        return dict(arm=arm,task=task,init=init,group=group,rep=rep,text_bank=True,
                    key=f'{arm}-{group}-t{task}-i{init}-r{rep}')
    rows=[row('T',task,33,'repeatability',rep) for rep in range(2) for task in (2,8)]
    for arm in ARMS:
        rows += [row(arm,task,35,'fresh_DEV') for task in range(10)]
        if arm != 'MLP_CONT':rows += [row(arm,task,init,'risk_DEV') for task,init in RISK]
    assert len(rows)==50 and len({r['key'] for r in rows})==50
    return rows

def source_files():
    # Bind the actual unchanged inference tree plus isolated adapters/launcher.
    baseline=read(BASE_JOB/'plan.json')['source_sha256']
    paths=set(map(Path,baseline))
    paths.update(OWN.glob('*.py'))
    paths.update((EXP/'tf_usable_mlp_continue_v1').glob('*.py'))
    paths.update((EXP/'tf_usable_train_v1').glob('*.py'))
    paths.update((EXP/'draft_freeze_clock_v1').glob('*.py'))
    for train in (TF_TRAIN,MLP_TRAIN):
        paths.update(map(Path,read(train/'PLAN.json')['source_sha256']))
    return sorted(paths)

def training_gate(selected, *, deep=False):
    import torch
    from tf_usable_train_v1.adapter import load_selected as tf_load
    from tf_usable_mlp_continue_v1.adapter import load_selected as mlp_load
    for arm,folder,architecture,expected,loader in (
        ('TF_CONT',TF_TRAIN,'transformer64','PASS_PRIMARY_TF_CONT_500_COMPLETE',tf_load),
        ('MLP_CONT',MLP_TRAIN,'mlp96','PASS_MATCHED_MLP_CONT_500_COMPLETE',mlp_load)):
        f=read(folder/'FINAL-RESULT.json')
        if (f['status']!=expected or f['complete_runs']!=1 or f['total_updates']!=3000
                or not f['full_CAL_new_adapter_bitwise_exact'] or not f['normalization_exact_parent']
                or (folder/'FAILED.json').exists() or (folder/'STOP_REQUESTED').exists()):
            raise ValueError('fixed last500 training incomplete: '+arm)
        cell=folder/f'{arm}-seed20260923'/'checkpoint.pt'
        item=selected[arm]
        if Path(item['checkpoint'])!=cell or sha(cell)!=item['checkpoint_sha256']:
            raise ValueError('fixed trained checkpoint identity differs: '+arm)
        if deep:
            model=loader(folder,architecture,device='cpu',expected_channels=range(7))
            if model.checkpoint_sha256!=item['checkpoint_sha256']:
                raise ValueError('strict candidate loader differs: '+arm)
            del model
    if torch.cuda.is_initialized():raise ValueError('CPU preparation must not initialize CUDA')

def selected():
    rows={}
    for arm,folder in (('TF_CONT',TF_TRAIN),('MLP_CONT',MLP_TRAIN)):
        checkpoint=folder/f'{arm}-seed20260923'/'checkpoint.pt'
        record=read(checkpoint.parent/'RESULT.json')
        rows[arm]=dict(checkpoint=str(checkpoint),checkpoint_sha256=record['checkpoint_sha256'],
                       architecture=ARCH[arm],bytes=checkpoint.stat().st_size)
    return rows

def validate(p,job, *, verify=True):
    if (p.get('schema')!='draft-freeze-complete-dev/v1' or p.get('profile')!='today50'
            or p.get('job')!=str(job) or p.get('site')!='local'
            or p.get('cases')!=cases() or p.get('arms')!=list(ARMS)
            or p.get('horizon')!=800 or p.get('seed')!=20260922
            or p.get('mode')!='verifier_k2_b2_bonus' or p.get('selector')!='prefix_then_cost'
            or p.get('threshold')!=THRESHOLD or p.get('observation_protocol')!='unchanged_legacy_cached_observations'
            or p.get('absolute_work_stop')!=WORK_STOP or p.get('absolute_cleanup_deadline')!=FINAL_CLEANUP
            or p.get('global_started')!=GLOBAL_STARTED or p.get('max_run_wall_sec')!=14400
            or p.get('workers')!=1 or p.get('performance_claim') is not False
            or p.get('output_budget_GiB')!=3 or p.get('cleanup_reserve_sec')!=300):
        raise ValueError('registered today50 protocol differs')
    registration=read(ROOT_JOB/'PREREGISTRATION.json')
    for arm,folder in (('TF_CONT',TF_TRAIN),('MLP_CONT',MLP_TRAIN)):
        value=p['selected'][arm]
        expected=registration['fixed_TF_checkpoint_sha256' if arm=='TF_CONT' else 'fixed_MLP_checkpoint_sha256']
        if (value['architecture']!=ARCH[arm] or value['checkpoint_sha256']!=expected
                or Path(value['checkpoint'])!=folder/f'{arm}-seed20260923'/'checkpoint.pt'):
            raise ValueError('candidate differs from prospectively fixed checkpoint: '+arm)
    if verify:
        base=read(BASE_JOB/'plan.json')
        if any(sha(k)!=v for k,v in base['source_sha256'].items()):
            raise ValueError('morning original inference source changed')
        inventory={str(x):sha(x) for x in source_files()}
        if inventory!=p['source_sha256']:
            raise ValueError('isolated/current source inventory changed')
        if sha(ROOT_JOB/'PREREGISTRATION.json')!=p['preregistration_sha256']:
            raise ValueError('prospective selection/experiment manifest changed')
        for path,stat in p['target_weight_stats'].items():
            s=Path(path).stat()
            if s.st_size!=stat['bytes'] or s.st_mtime_ns!=stat['mtime_ns']:
                raise ValueError('original target weightstat changed')
        for arm,item in p['selected'].items():
            if sha(item['checkpoint'])!=item['checkpoint_sha256']:
                raise ValueError('draft bytes changed: '+arm)
        for key in ('bank','real_CAL_data'):
            path=Path(p[key] if key=='bank' else p[key]['path'])
            stat=p['bank_stat'] if key=='bank' else p['real_CAL_data']
            s=path.stat()
            if s.st_size!=stat['bytes'] or s.st_mtime_ns!=stat['mtime_ns']:
                raise ValueError('bank/CAL stat changed')
    return p

def prepare(job):
    job=Path(job).resolve()
    if job.exists() or not job.name.startswith('p0-draft-freeze-quality-v1-local-'):
        raise ValueError('fresh isolated local job required')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(WORK_STOP):
        raise TimeoutError('today work deadline passed')
    p=read(BASE_JOB/'plan.json')
    choices=selected();training_gate(choices,deep=True)
    p.update(schema='draft-freeze-complete-dev/v1',profile='today50',job=str(job),site='local',
        cases=cases(),arms=list(ARMS),selected=choices,init=None,
        training_dirs=dict(TF_CONT=str(TF_TRAIN),MLP_CONT=str(MLP_TRAIN)),
        global_started=GLOBAL_STARTED,absolute_work_stop=WORK_STOP,absolute_cleanup_deadline=FINAL_CLEANUP,
        max_run_wall_sec=14400,combined_two_site_GPU_seconds_cap=14400,
        job_target_forward_cap=100000,per_episode_target_forward_cap=4000,
        port=19557,master_port=29557,OMP_NUM_THREADS=2,client_threads=2,
        observation_protocol='unchanged_legacy_cached_observations',
        init_role='fresh35DEV; known-risk25to29DEV; repeat33DEV; no TEST40to49',
        preregistration=str(ROOT_JOB/'PREREGISTRATION.json'),
        preregistration_sha256=sha(ROOT_JOB/'PREREGISTRATION.json'),
        selection=read(ROOT_JOB/'PREREGISTRATION.json')['decision_rule'],
        source_sha256={str(x):sha(x) for x in source_files()},
        formal_performance_permission=False,independent_test_complete=False,
        original_morning_FAILED_preserved=True,no_old_T_reuse=True,
        planned_full_episodes=50,absolute_no_AutoDL=True)
    validate(p,job)
    job.mkdir(parents=True)
    write(job/'plan.json',p)
    return p

def load(job, *, verify=True):
    job=Path(job).resolve(strict=True)
    return validate(read(job/'plan.json'),job,verify=verify)

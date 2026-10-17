"""Finite TF effect closure; immutable old evidence and opt-in new namespaces."""
from datetime import datetime, timedelta, timezone
import hashlib, json, subprocess
from pathlib import Path

OWN=Path(__file__).resolve().parent; EXP=OWN.parent; ROOT=EXP.parent.parent
RT=ROOT/'投机采样/results/runtime'; WORKTREE=ROOT/'投机采样/wam-p0-worktree'
BASE=RT/'p0-task4-direction-diagnostic-completion-v1-20261016-r2'
SOURCE_EPISODE=BASE/'episodes/TF_CONT-task4-init26-repeat0'
COMMIT='b0dfac3a02f65de33a1217246b83003615166640'
PERMISSIONS=dict(engineering_quality_screen=False,independent_test_complete=False,
    noninferiority_established=False,formal_performance_permission=False)

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,s):
    if not x:raise ValueError(s)
def write(p,x,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        q=p.with_name(p.name+'.tmp');q.write_text(raw);q.replace(p)
    else:
        with p.open('x') as f:f.write(raw)

def prepare_qualification(job):
    job=Path(job).resolve();require(job.parent==RT and not job.exists(),'new isolated runtime required')
    require(subprocess.check_output(['git','-C',str(ROOT/'review_exports/WAM-egale'),'rev-parse','HEAD'],text=True).strip()==COMMIT,'fixed public base differs')
    q=read(BASE/'plan.json')
    for f,v in q['source_sha256'].items():require(sha(f)==v,'old frozen source changed: '+f)
    paths=list(OWN.glob('*.py'))+list((EXP/'tf_effect_collector_v1').glob('*.py'))
    require((EXP/'tf_effect_collector_v1/collector.py').exists(),'reviewed collector required')
    sources={**q['source_sha256'],**{str(f):sha(f) for f in paths}}
    refs=[SOURCE_EPISODE/'COUNTER.json',SOURCE_EPISODE/'client/COMMAND-MAPPING.json',
          SOURCE_EPISODE/'client/initial_observations.npz',BASE/'arms/TF_CONT/TARGET-INITIAL-RNG.pt',
          RT/'p0-tf-supervision-alignment-cpu-v1-20261016-r1/RESULT.json']
    refs+=sorted((SOURCE_EPISODE/'direction_trace').glob('chunk-*.pt'))
    refs+=list((SOURCE_EPISODE/'server_output').rglob('*.pt'))
    require(len(list((SOURCE_EPISODE/'direction_trace').glob('chunk-*.pt')))==50,'full original TF episode required')
    now=datetime.now(timezone.utc);cap=2700;reserve=180
    p=dict(schema='tf-effect-closure-job/v1',job=str(job),kind='qualification',commit=COMMIT,
        goal='Qualify observer without changing O3: replay all50 policy and cache requests from one complete recorded TF episode OFF/ON',
        created_at=now.isoformat(),reference_plan=q,source_episode=str(SOURCE_EPISODE),
        source_sha256=sources,evidence_stat={str(f):dict(bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in refs},
        max_run_wall_sec=cap,cleanup_reserve_sec=reserve,
        absolute_cleanup_deadline=(now+timedelta(seconds=cap)).isoformat(),
        absolute_work_stop=(now+timedelta(seconds=cap-reserve)).isoformat(),
        target_forward_cap=6000,storage_cap_bytes=2**30,disk_floor_bytes=5*2**30,
        GPU='local A100-SXM4-40GB',shared_GPU_allowed=True,workers=1,CPU_affinity=[12,13],CPU_threads=2,
        initial_free_GiB=22,master_port=29667,port=19667,permissions=PERMISSIONS,
        real_robot_episode_count=0,recorded_full_episode_replay_passes=2,policy_chunks_per_pass=50,
        gates=['all recorded actions/video/physical commands/paths/counts reproduced before observer claim',
               'OFF/ON actions, all valid30-layer KV, RNG, routes and target counts exact per request',
               'O3 events enabled; trace/state_teacher remain None; zero extra teacher forwards',
               'all teacher rows match actual saved original verifier tensors and score arithmetic'],
        timeout='PARTIAL/FAILED retained; no auto retry, no reset of this cap',
        no_smoke=True,no_training=True,no_120=True,no_TEST=True,no_AutoDL=True,
        cleanup='independent CPU clock and exact owned PID identity/ancestry; never foreign processes',
        evidence_path=str(job),terminal_status='REGISTERED_NOT_STARTED',performance_claim=False)
    job.mkdir();write(job/'PREREGISTRATION.json',p)
    write(job/'plan.json',{**p,'preregistration_sha256':sha(job/'PREREGISTRATION.json')})
    return p

def load(job,verify=True):
    job=Path(job).resolve(strict=True);p=read(job/'plan.json')
    require(p['schema']=='tf-effect-closure-job/v1' and p['job']==str(job) and p['commit']==COMMIT,'job identity differs')
    require(p['permissions']==PERMISSIONS and p['performance_claim'] is False,'old permissions unchanged')
    require(sha(job/'PREREGISTRATION.json')==p['preregistration_sha256'],'registration changed')
    if verify:
        for f,h in p['source_sha256'].items():require(sha(f)==h,'frozen source changed '+f)
        for f,st in p.get('evidence_stat',{}).items():
            s=Path(f).stat();require((s.st_size,s.st_mtime_ns)==(st['bytes'],st['mtime_ns']),'input changed '+f)
        for f,st in p['reference_plan']['target_weight_stats'].items():
            s=Path(f).stat();require((s.st_size,s.st_mtime_ns)==(st['bytes'],st['mtime_ns']),'target weight changed')
    return p

def environment(p):
    import os
    q=p['reference_plan']
    env={k:v for k,v in os.environ.items() if not k.startswith(('WAM_P0_','P0_','LINGBOT_','PREFIX_ACCEPT_','RESET_TEXT_BANK_','FORMAL_SPEED_','TARGET_TRACE_','DIRECTION_DIAG_','TF_EFFECT_'))}
    env.update(CUDA_VISIBLE_DEVICES='0',LINGBOT_SEED=str(q['seed']),PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
        LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',LINGBOT_ATTN_MODE='torch',LINGBOT_NSYS_PROFILE='off',
        LINGBOT_VA_LIBERO_SUITE='libero_10',LINGBOT_VA_LIBERO_CHECKPOINT=q['target_checkpoint_path'],
        LINGBOT_VA_LIBERO_CHECKPOINT_ID=q['target_checkpoint_id'],LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=q['registry'],
        WAM_VAE_DEVICE='cuda',WAM_P0_PHASE_PROFILE='0',WAM_P0_ACTION_DRAFT_MODE='off',WAM_P0_ACTION_DRAFT_SELECTOR='global',
        WAM_P0_ACTION_DRAFT_THRESHOLD=str(q['threshold']),WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=q['selected']['TF_CONT']['checkpoint'],
        RESET_TEXT_BANK_ENABLED='1',PREFIX_ACCEPT_ENABLED='1',PREFIX_ACCEPT_GPU_APPROVED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(p['master_port']),TF_EFFECT_JOB=p['job'])
    return env

"""One-shot execution contract; no torch/CUDA/network imports."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import os
import time

BASE = Path(__file__).resolve().parent.parent/'single_worker_chc_gpu_v4'
import sys
if str(BASE) not in sys.path:sys.path.append(str(BASE))

from contract import ROOT, WORKTREE, ROWS, digest, job_path

OWN = Path(__file__).resolve().parent
PREFIX = 'p0-shared-prefix-v3-'
REVISION = 'zero-copy-cfg2-shared-prefix-gpu-v3'
ENABLED = 'P0_SHARED_PREFIX_V3_ENABLED'
APPROVED = 'P0_SHARED_PREFIX_V3_GPU_APPROVED'
FLAG = ENABLED
COUNTER_JOB = 'P0_SHARED_PREFIX_V3_COUNTER_JOB'
STAGES = {'kernel':0,'capture':52,'resource_probe':1,'equivalence_HC':29,'equivalence_S':29,
          'warmup_R':29,'warmup_S':29,
          **{f'OFF_{b}_{a}':29 for b in range(6) for a in ('R','S')},
          **{f'invalidation_{i:02d}':5 for i in range(1,13)},
          **{f'isolation_{a}':2 for a in ('R','HC','S')},'finish':2}
assert sum(STAGES.values()) == 585


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write(path, value, *, exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x') as f:f.write(body)
    else:
        tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
        with tmp.open('x') as f:f.write(body)
        os.replace(tmp,path)


def validate(p):
    exact={'gpu_run_approved':True,'training':False,'RL':False,'joint_training':False,
           'GPU_model_worker_cap':1,'GPU_attempt_cap':1,'hard_target_forward_cap':715,
           'GPU_hard_limit_minutes':36,'new_full_episode_cap':0,'EGL_initialization_cap':0,
           'complete_policy_chunk_cap':0,'executed_environment_step_cap':0,
           'batch':14,'layers':30,'bonus':False,'selector':'global','cache_mode':'original',
           'CPU_preflight_required':True,'no_retries':True,'no_resident_peer':True,
           'execution_revision':REVISION,'resource_wait_explicitly_authorized':True,
           'resource_poll_interval_sec':180,'cleanup_reserve_seconds':15,
           'counter_persistence_enabled':True,'GPU_compute_mode_changed':False}
    for k,v in exact.items():
        if type(p.get(k)) is not type(v) or p[k]!=v:raise ValueError('scope mismatch: '+k)
    if p['stage_caps']!=STAGES or p['rows']!=list(ROWS):raise ValueError('stage/batch contract')
    job=Path(p['job']).resolve()
    if job.parent!=ROOT/'投机采样/results/runtime' or not job.name.startswith(PREFIX):raise ValueError('new private job required')
    if p['worktree']!=str(WORKTREE) or len(p['ports'])!=1 or not 1024<p['ports'][0]<65535:raise ValueError('single worker path/port')
    if p['main_orders']!=[['R','S'],['S','R']]*3:raise ValueError('six predeclared R/S crossover orders')
    actual_paths={str(f) for f in OWN.iterdir() if f.is_file() and f.suffix in ('.py','.mjs','.md')}
    if set(p['implementation_sha256'])!=actual_paths:raise ValueError('new revision implementation identity')
    if any(k in p for k in ('CPU_test_fault','CPU_exit_after_one_command','CPU_only')):
        raise ValueError('CPU fixture fields forbidden in GPU plan')
    if p.get('aggregate_prior_target_intents')!=183 or p.get('aggregate_prior_GPU_seconds')!=507.2913444358856:raise ValueError('prior consumption must be preserved')
    if p['aggregate_prior_target_intents']+p['hard_target_forward_cap']>900 or p['aggregate_prior_GPU_seconds']+2160>2700:raise ValueError('aggregate budget')
    if type(p.get('diagnostic_nonbitwise_approved')) is not bool or p.get('direct_attention_micro_call_cap')!=256:raise ValueError('numerical diagnostic/micro scope')
    if p['diagnostic_nonbitwise_approved'] and not p.get('diagnostic_authorization_text'):raise ValueError('explicit diagnostic authorization required')
    return p


def enabled_plan(job, *, cpu_gate=False):
    if os.environ.get(ENABLED)!='1' or os.environ.get(APPROVED)!='1':raise ValueError('new revision default-disabled')
    job=Path(job).resolve();p=validate(json.loads(job_path(job,'plan.json').read_text()))
    if str(job)!=p['job']:raise ValueError('wrong job identity')
    if cpu_gate:
        g=json.loads(job_path(job,'preflight/CPU-GATE.json').read_text())
        if g['status']!='PASS' or g['plan_sha256']!=digest(job/'plan.json'):raise ValueError('fresh CPU gate missing/changed')
        if g['implementation_sha256']!=p['implementation_sha256']:raise ValueError('CPU implementation generation')
    freeze(p)
    return p


def read_cpu_plan(job):
    cpu_enabled();job=Path(job).resolve()
    if job.parent==ROOT/'投机采样/results/runtime' and job.name.startswith(PREFIX):
        return enabled_plan(job)
    # Actual termination fixtures are CPU-only children of this new plan.
    parent=job.parent.parent
    if job.parent.name!='workers' or not job.name.startswith('case-'):
        raise ValueError('new private CPU fixture required')
    root=enabled_plan(parent);p=json.loads((job/'cpu-plan.json').read_text())
    allowed={'job','CPU_only','GPU_run_approved','gpu_run_approved','parent_plan_sha256',
             'CPU_test_fault','CPU_exit_after_one_command','CPU_fixture_dtype'}
    if (p.get('job')!=str(job) or p.get('CPU_only') is not True
        or p.get('GPU_run_approved') is not False or p.get('gpu_run_approved') is not False
        or p.get('parent_plan_sha256')!=digest(parent/'plan.json')
        or {k:v for k,v in p.items() if k not in allowed}!={k:v for k,v in root.items() if k not in allowed}):
        raise ValueError('CPU fixture parent plan/scope mismatch')
    freeze(p);return p


def cpu_enabled():
    if os.environ.get(ENABLED)!='1' or os.environ.get(APPROVED)!='1':
        raise ValueError('new revision default-disabled CPU counter')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU visibility required')


def counter_enabled():
    if os.environ.get('CUDA_VISIBLE_DEVICES')=='':return cpu_enabled()
    context=os.environ.get(COUNTER_JOB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or not context:
        raise ValueError('CPU visibility or admitted GPU counter context required')
    # No CUDA import: fresh CPU gate, one-shot claim and PID ownership required.
    return gpu_entry_gate(context)


def counter_identity(job, *, CPU=False):
    job=Path(job).resolve()
    plan_file=job/('cpu-plan.json' if CPU and job.parent.name=='workers' else 'plan.json')
    return dict(job=str(job),plan_sha256=digest(plan_file),
                kind='CPU-only-shared-prefix' if CPU else 'GPU-supervised-shared-prefix-v1')


def gpu_entry_gate(job, *, claim=False):
    """Pre-Torch one-shot entry. Old approval, stale gates and replay fail closed."""
    job=Path(job).resolve();p=enabled_plan(job,cpu_gate=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('single GPU visibility required')
    if any((job/name).exists() for name in ['FAILED.json','FINAL_RESULT.json','STOP_REQUESTED']):
        raise ValueError('terminal/stopped job cannot be reused')
    a=json.loads((job/'GPU-ATTEMPT.json').read_text())
    if (a.get('job')!=str(job) or a.get('execution_revision')!=REVISION or a.get('attempt')!=1
            or abs(a['monotonic_deadline']-a['monotonic_start']-2160)>1e-6):
        raise ValueError('fresh supervised GPU attempt required')
    if time.monotonic()>=a['monotonic_deadline']-15:raise TimeoutError('GPU entry cleanup-reserved deadline')
    marker=job/'worker/GPU-ENTRY.json'
    if claim:
        write(marker,dict(job=str(job),execution_revision=REVISION,pid=os.getpid(),timestamp=stamp(),
              CPU_gate_plan_sha256=digest(job/'plan.json'),one_shot=True),exclusive=True)
    else:
        e=json.loads(marker.read_text())
        if e.get('pid')!=os.getpid() or e.get('job')!=str(job) or e.get('execution_revision')!=REVISION:
            raise ValueError('GPU entry ownership marker required')
    return p


def freeze(p):
    for path,h in {**p['frozen_sha256'],**p['implementation_sha256']}.items():
        if digest(path)!=h:raise ValueError('frozen source changed: '+path)
    if (ROOT/'.pipeline/cycle.yaml').exists():raise ValueError('protected cycle absence changed')
    if digest(p['frozen_A_off'])!=p['frozen_A_off_sha256']:raise ValueError('A_off changed')
    s=p['snapshot_reference'];st=Path(s['path']).stat()
    if (st.st_size,st.st_mtime_ns)!=(s['bytes'],s['mtime_ns']):raise ValueError('fixture changed')
    for w in p['target_weight_inventory']:
        st=Path(w['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(w['bytes'],w['mtime_ns']):raise ValueError('target stat changed')


class Ledger:
    """Bound each stage before attempted public target math, no refunds/retries."""
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.deadline=None;self.reserved=0;self.attempted=0
        self.seen=set();self.stage=None;self.stage_attempted=0;self.completed=0
    def begin(self,key):
        if key not in STAGES or key in self.seen:raise ValueError('unknown/retried stage')
        self.seen.add(key);self.reserved+=STAGES[key];self.stage=key;self.stage_attempted=0
        if self.reserved>585:raise RuntimeError('allocated budget exhausted')
    def before_forward(self):
        if self.deadline is not None and self.clock()>=self.deadline:raise TimeoutError('GPU deadline before forward')
        if self.stage not in STAGES or self.stage_attempted>=STAGES[self.stage] or self.attempted>=715:raise RuntimeError('target forward cap')
        self.stage_attempted+=1;self.attempted+=1
    def after_forward(self):self.completed+=1
    def summary(self):
        return dict(target_reserved=self.reserved,target_attempted=self.attempted,
                    target_completed=self.completed,stage=self.stage,stage_attempted=self.stage_attempted,
                    allocated_cap=585,target_hard_cap=715,extra_micro_calls=0,
                    EGL=0,complete_episodes=0,complete_policy_chunks=0,environment_steps=0)

"""Default-off Long task2–9 TRAIN collection. No inherited GPU authority."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo
from paths import ROOT, OWN, S0, WORKTREE

RESEARCH=ROOT/'投机采样'
VA='/rjs/cxz/miniconda3/envs/va/bin/python'
CLIENT='/rjs/cxz/miniconda3/envs/libero-va/bin/python'
ENABLED='MORNING_ONPOLICY_LONG8_V4_ENABLED'
APPROVED='MORNING_ONPOLICY_LONG8_V4_GPU_APPROVED'
PREFIX='p0-morning-onpolicy-long8-v4-'
MODES={'COLLECT':'verifier_k2_b2_bonus','OLD':'verifier_k2_b2_bonus'}
SELECTORS={a:'prefix_then_cost' for a in MODES}
JOB_PARENT=Path('/var/tmp/cxz-wam-quality-20261014')

def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def metadata(path):
    s=Path(path).stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def write(path,value,*,exclusive=False,durable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path if exclusive else path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush()
        if durable:os.fsync(f.fileno())
    if not exclusive:os.replace(tmp,path)

def assignments(stage='collection'):
    if stage!='collection':raise ValueError('TRAIN collection only')
    return [[dict(id=f'collect-t{t}-i17',arm='COLLECT',task=t,init=17,split='train',cap=4000)
             for t in pair] for pair in ((2,3),(4,5),(6,7),(8,9))]

def fixed_slots(stage='collection'):
    if stage!='collection':raise ValueError('TRAIN collection only')
    return [dict(slot=i,arm=a,checkpoint='NEW',teacher=bool(i),optimized=True)
            for i,a in enumerate(('TEACHER_OFF','TEACHER_ON'))]

def implementation():return {str(p):digest(p) for p in OWN.glob('*.py')}

def freeze(p,full=False):
    for key in ('implementation_sha256','critical_source_sha256','frozen_sha256'):
        for path,h in p[key].items():
            if digest(path)!=h:raise ValueError('frozen input changed: '+path)
    for v in p['target_weight_inventory']:
        s=Path(v['path']).stat()
        if (s.st_size,s.st_mtime_ns)!=(v['bytes'],v['mtime_ns']):raise ValueError('target inventory changed')
    manifest=Path(p['target_weight_content_manifest'])
    if digest(manifest)!=p['target_weight_content_manifest_sha256']:
        raise ValueError('current target content-hash manifest changed')
    rows=json.loads(manifest.read_text())['weight_files']
    if (len(rows)!=len(p['target_weight_inventory'])
        or [(x['path'],x['bytes'],x['mtime_ns']) for x in rows]
           !=[(x['path'],x['bytes'],x['mtime_ns']) for x in p['target_weight_inventory']]
        or any(len(x['sha256'])!=64 for x in rows)):
        raise ValueError('target content manifest/inventory mismatch')

def plan(job,gpu=False,full=False):
    job=Path(job).resolve()
    if job.parent!=JOB_PARENT or not job.name.startswith(PREFIX):raise ValueError('isolated onpolicy job required')
    p=json.loads((job/'plan.json').read_text())
    exact=dict(stage='collection',full_episode_cap=2,EGL_initialization_cap=2,model_worker_cap=3,
        fixed_model_workers=1,fixed_target_cap=320,fixed_full_chunk_cap=4,target_cap=8320,parallel_EGL_workers=2,
        aggregate_GPU_seconds_cap=46311.19392736256,aggregate_target_cap=226051,
        aggregate_model_launches_cap=50,
        GPU_initial_free_bytes=33*2**30,GPU_runtime_free_floor_bytes=2*2**30,
        resource_samples=3,initial_resource_wait_seconds=30,no_retries=True,
        shared_quality_authorized=True,exclusive_required=False,speed_comparison_deferred=True,profile_OFF=True,
        full_trace_OFF=True,training=False,chunk_events_ON=True,teacher_OFF=False,
        disk_initial_min_bytes=15*2**30,disk_runtime_floor_bytes=12*2**30,new_output_bytes_cap=2*2**30,
        no_deployment=True,collect_all_completed_episodes_regardless_of_success=True,
        new_numeric_OFF_ON_required=True,previous_native_oracles_current_proof=False,
        cold_original_ctor_install_required=True)
    if any(type(p.get(k)) is not type(v) or p[k]!=v for k,v in exact.items()):raise ValueError('finite collection scope changed')
    if type(p.get('wave_index')) is not int or not 0<=p['wave_index']<4:raise ValueError('one of four wave indices required')
    if p['assignments']!=[assignments()[p['wave_index']]] or p['fixed_slots']!=fixed_slots() or len(p['snapshots'])!=2:raise ValueError('frozen split/oracle mismatch')
    if set(p['implementation_sha256'])!=set(implementation()):raise ValueError('source set changed')
    if (type(p.get('GPU_hard_limit_seconds')) is not int or not 0<p['GPU_hard_limit_seconds']<=1500
        or p['aggregate_prior_GPU_seconds']+p['GPU_hard_limit_seconds']>p['aggregate_GPU_seconds_cap']
        or p['aggregate_prior_target_intents']+8320>p['aggregate_target_cap']
        or p['aggregate_prior_model_launches']+3>p['aggregate_model_launches_cap']):
        raise ValueError('aggregate cap exhausted')
    from oracle_reuse import validate_carry,validate_oracles,authority
    authority(p);validate_carry(p);validate_oracles(p)
    if not p['candidate_research_only_not_quality_admitted']:raise ValueError('candidate not deployment eligible')
    freeze(p,full)
    if gpu:
        if os.getenv(ENABLED)!='1' or os.getenv(APPROVED)!='1' or os.getenv('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('default-off GPU boundary')
        a=json.loads((job/'AUTHORIZATION.json').read_text());g=json.loads((job/'preflight/CPU-GATE.json').read_text())
        if (a.get('authorized') is not True or a.get('plan_sha256')!=digest(job/'plan.json') or
                g.get('status')!='PASS' or g.get('plan_sha256')!=a['plan_sha256'] or
                a.get('CPU_gate_sha256')!=digest(job/'preflight/CPU-GATE.json') or g.get('CUDA_initialized')
                or a.get('source_carry_sha256')!=p['source_carry_sha256']
                or a.get('source_carry_audit_sha256')!=p['source_carry_audit_sha256']
                or a.get('GPU_UUID')!=p['GPU_UUID']):raise ValueError('root authorization/CPU gate mismatch')
    return p

def worker_entry(kind='episode'):
    job=Path(os.environ['P0_SPEC_E2E_JOB']).resolve();p=plan(job,gpu=True)
    out=Path(os.environ['P0_SPEC_E2E_OUT']).resolve()
    if kind=='fixed':
        if out!=job/'fixed/worker0':raise ValueError('single new numeric worker path')
        cap=320
    else:
        item=next((v for wave in p['assignments'] for v in wave if v['id']==os.environ['DRAFT_TRAIN_ID']),None)
        if item is None or out!=job/'campaigns'/item['arm']/'workers'/item['id']:raise ValueError('fixed assignment required')
        cap=item['cap']
    if any((job/f).exists() for f in ('FAILED.json','GPU-COLLECTION.json','STOP_REQUESTED')):raise ValueError('terminal collection')
    a=json.loads((job/'GPU-ATTEMPT.json').read_text())
    if time.monotonic()>=a['monotonic_deadline']-25:raise TimeoutError('collection deadline')
    r=json.loads((out/'FORWARD-RESERVATION.json').read_text())
    if r['cap']!=cap or r['plan_sha256']!=digest(job/'plan.json'):raise ValueError('durable reservation required')
    import ownership
    end=time.monotonic()+3
    while True:
        pins={x['pid']:ownership.Process(**{**x,'argv':tuple(x['argv'])}) for x in json.loads((job/'OWNED-PROCESSES.json').read_text())}
        if ownership.is_owned(ownership.read_process(os.getpid()),pins):break
        if time.monotonic()>end:raise ValueError('owned worker required')
        time.sleep(.01)
    for k,v in (('MUJOCO_GL','egl'),('PYOPENGL_PLATFORM','egl'),('MUJOCO_EGL_DEVICE_ID','0')):
        if os.getenv(k)!=v:raise ValueError('EGL provenance required')
    write(out/'GPU-ENTRY.json',dict(pid=os.getpid(),kind=kind,cap=cap,at=stamp()),exclusive=True)
    return job,out,p

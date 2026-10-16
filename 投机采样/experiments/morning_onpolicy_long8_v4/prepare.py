"""Prepare one paired Long wave; never launch GPU or inherit authorization."""
import argparse,json,os,subprocess
from datetime import datetime
from pathlib import Path
from experiment import ROOT,OWN,RESEARCH,VA,JOB_PARENT,assignments,fixed_slots,implementation,digest,write,stamp
from oracle_reuse import AUDIT,AUDIT_SHA,CARRY,CARRY_SHA,validate_carry,validate_oracles,authority
SOURCE_PLAN=Path('/var/tmp/cxz-wam-quality-20261014/p0-overnight-onpolicy-coldfix-v1-20261015-prepared/plan.json')
SOURCE_PLAN_SHA='b96066bfee72aa988cb57814fb3089f73e52c178bd112367df2aeae2679dea20'
TARGET_HASH_MANIFEST=ROOT/'.pipeline/research/speculative-sampling/evidence/fusion-withdraw-draft-onpolicy-20261014/MORNING-CURRENT-TEACHER-WEIGHT-SHA256.json'
TARGET_HASH_MANIFEST_SHA='adb14003a14bcbd4a0507ed2608e93fc13fa71fc20a7dc34e97786ca99446c76'
CANDIDATE=Path('/var/tmp/cxz-wam-quality-20261014/p0-draft-expanded-data-v1-20261014/B_MIXED_HISTORICAL_C-seed20260922/checkpoint.pt')
CANDIDATE_SHA='03e1354fc6bf4e9838607590466e3de70efc8b10a959f96665502bae68d3de49'
CURRENT=Path('/var/tmp/cxz-wam-quality-20261014/p0-draft-root-retention-v1-20261014/dataset.pt')
CURRENT_SHA='51c2947fa9af27d3b65beafd03a4c6a4d542907b44819f82d50b3a95d24ee044'
def fresh_sources():
    dirs=('single_worker_chc_gpu_v4','speculative_e2e_v3','fixed_load_cost','prompt_kv_gpu_v4',
          'bonus_acceptance_v2','bonus_e2e_v1','overnight_onpolicy_v1',
          'draft_teacher_train_v1','draft_architecture_v1','draft_root_retention_v1','trace_preserving_train_v1')
    files=[]
    for d in dirs:files.extend((OWN.parent/d).glob('*.py'))
    files.extend((RESEARCH/'wam-p0-worktree/wan_va').rglob('*.py'))
    files.extend((RESEARCH/'scripts').glob('p0_*.py'))
    files.extend((RESEARCH/'configs').glob('libero_checkpoint_registry.json'))
    files.extend([RESEARCH/'configs/current_speculative_candidate.json',ROOT/'.pipeline/rules/structured/project/data-collection-egl-parallel.yaml'])
    native=OWN.parent/'shared_prefix_torch_v2'
    binaries=list(native.glob('_shared_torch_v1*.so'))
    if len(binaries)!=1:raise ValueError('one original shared-KV native binary required')
    files.extend(binaries+[native/'SOURCE-PROVENANCE.json',native/'FROZEN-INPUTS.json'])
    # Fresh scope, never inherit unrelated operator source paths from old plans.
    return {str(p.resolve()):digest(p) for p in sorted(set(files))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--wave',type=int,required=True,choices=range(4))
    ap.add_argument('--prior-terminal',type=Path)
    ap.add_argument('--retry-of',type=Path,help='unsupported in v4; failed jobs need a separately reviewed recovery protocol')
    ap.add_argument('--limit-seconds',type=int,default=1500)
    ap.add_argument('--stop-at',required=True,help='timezone-aware absolute GPU stop ISO timestamp')
    a=ap.parse_args()
    if a.retry_of is not None:raise ValueError('v4 forbids automatic whole-wave retry; preserve FAILED job and seek new reviewed recovery plan')
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or os.sched_getaffinity(0)!={8}:raise ValueError('CPU8 only, CUDA hidden')
    if datetime.fromisoformat(a.stop_at).utcoffset() is None:raise ValueError('timezone-aware stop required')
    if digest(CARRY)!=CARRY_SHA or digest(AUDIT)!=AUDIT_SHA:raise ValueError('exact final closed carry required')
    if digest(SOURCE_PLAN)!=SOURCE_PLAN_SHA:raise ValueError('old source plan changed')
    previous_pass=CARRY if a.wave==0 else a.prior_terminal
    if previous_pass is None:raise ValueError('previous closed PASS wave terminal required')
    previous_pass=previous_pass.resolve()
    if a.wave==0 and a.prior_terminal is not None:raise ValueError('wave0 starts from final overnight ledger')
    if a.wave>0 and previous_pass.name!='GPU-COLLECTION.json':raise ValueError('prior complete wave required')
    carry_path=previous_pass
    if a.job.parent.resolve()!=JOB_PARENT or not a.job.name.startswith(f'p0-morning-onpolicy-long8-v4-w{a.wave}-'):
        raise ValueError('isolated wave-specific job name required')
    for prior_job in a.job.parent.glob(f'p0-morning-onpolicy-long8-v4-w{a.wave}-*'):
        if (prior_job/'GPU-COLLECTION.json').exists():raise ValueError('PASS wave already exists; never resample')
        failed=prior_job/'FAILED.json'
        if failed.exists():
            raise ValueError('FAILED wave exists; v4 forbids automatic resampling and preserves its evidence')
        if (prior_job/'plan.json').exists() and not failed.exists():
            raise ValueError('already prepared or running wave; no duplicate job')
    job=a.job.resolve();job.mkdir(exist_ok=False);(job/'preflight').mkdir()
    carry=json.loads(carry_path.read_text());old=json.loads(SOURCE_PLAN.read_text())
    for path,h in ((AUDIT,AUDIT_SHA),(CARRY,CARRY_SHA),(SOURCE_PLAN,SOURCE_PLAN_SHA),
                   (TARGET_HASH_MANIFEST,TARGET_HASH_MANIFEST_SHA),(CANDIDATE,CANDIDATE_SHA),(CURRENT,CURRENT_SHA)):
        if digest(path)!=h:raise ValueError('frozen source differs')
    p={k:v for k,v in old.items() if k not in ('frozen_sha256','critical_source_sha256','implementation_sha256')}
    for key in ('oracle_source_plan_sha256','oracle_artifact_sha256','oracle_compute_copies','reused_oracle_new_GPU_chunks','oracle_reuse_only_exact_frozen_generator',
                'night_authority','night_authority_sha256','night_addendum','night_addendum_sha256','old_original_implementation_sha256',
                'historical_four_GPU_oracles','source_carry_plan_sha256','held_out_inits','future_only_inits'):
        p.pop(key,None)
    p.update(job=str(job),created_at=stamp(),revision='morning-long8-identity-gated-v4',phase='long8_same_query_train_init17',
        GPU_absolute_stop_at=a.stop_at,GPU_UUID='GPU-754f96ec-4567-7af2-7048-7e8957a6eb36',
        wave_index=a.wave,assignments=[assignments()[a.wave]],tasks=[x['task'] for x in assignments()[a.wave]],inits=[17],
        full_episode_cap=2,EGL_initialization_cap=2,
        model_worker_cap=3,fixed_model_workers=1,fixed_target_cap=320,fixed_full_chunk_cap=4,target_cap=8320,
        GPU_hard_limit_seconds=a.limit_seconds,
        aggregate_prior_GPU_seconds=carry['aggregate_GPU_seconds'],aggregate_prior_target_intents=carry['aggregate_target_intents'],
        aggregate_prior_model_launches=38 if a.wave==0 else carry['aggregate_model_launches'],
        aggregate_GPU_seconds_cap=40311.19392736256+6000,
        aggregate_target_cap=192771+33280,aggregate_model_launches_cap=50,
        resource_samples=3,exclusive_required=False,shared_quality_authorized=True,ports=[19991,19992,29891,29892],
        source_carry=str(carry_path),source_carry_sha256=digest(carry_path),
        previous_pass_terminal=str(previous_pass),previous_pass_terminal_sha256=digest(previous_pass),
        retry_of=None,retry_of_sha256=None,
        source_carry_audit=str(AUDIT),source_carry_audit_sha256=AUDIT_SHA,
        target_weight_content_manifest=str(TARGET_HASH_MANIFEST),
        target_weight_content_manifest_sha256=TARGET_HASH_MANIFEST_SHA,
        source_carry_usage='closed independently audited overnight ledger; no inherited GPU authorization',
        new_numeric_OFF_ON_required=True,previous_native_oracles_current_proof=False,cold_original_ctor_install_required=True,
        attempts_max=1,seed=old['frozen_reference']['seed'],
        cold_scope='each new wave source requires cold ctor/reset and four native OFF/ON complete chunk proof before its paired full episodes',
        reservation_failure_policy='normal successful phase charges valid closed counters; any phase failure charges every launched reservation full cap while preserving observed valid counts; no refunds',
        critical_source_sha256=fresh_sources(),implementation_sha256=implementation())
    p['frozen_reference']['task_index']=None
    p['frozen_sha256']={str(x):digest(x) for x in (AUDIT,carry_path,carry_path.parent/'plan.json',previous_pass,previous_pass.parent/'plan.json',
                                                   TARGET_HASH_MANIFEST,CANDIDATE,CURRENT,SOURCE_PLAN)}
    authority(p);validate_carry(p);validate_oracles(p)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    with (job/'preflight/CPU-COLD.log').open('xb') as f:
        subprocess.run([VA,'-B',str(OWN/'cold_cpu_check.py'),'--out',str(job/'preflight/CPU-COLD.json')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    with (job/'preflight/CPU-COLLECTOR.log').open('xb') as f:
        subprocess.run([VA,'-B',str(OWN/'cpu_collector_check.py'),'--out',str(job/'preflight/CPU-COLLECTOR.json')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    p['frozen_sha256'][str(job/'preflight/CPU-COLLECTOR.json')]=digest(job/'preflight/CPU-COLLECTOR.json')
    p['frozen_sha256'][str(job/'preflight/CPU-COLD.json')]=digest(job/'preflight/CPU-COLD.json')
    write(job/'plan.json',p,exclusive=True,durable=True)
    print(json.dumps(dict(status='PREPARED_NO_AUTH_NO_GPU',job=str(job),plan_sha256=digest(job/'plan.json'))))

if __name__=='__main__':main()

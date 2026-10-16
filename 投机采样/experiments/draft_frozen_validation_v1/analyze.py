"""Real complete-shard and all-four-shard DEV analysis; never launches work."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

OWN=Path(__file__).resolve().parent
EXP=OWN.parent
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from draft_frozen_validation_v1.protocol import load,read,write
from tf_usable_quality_v1.analyze import episode
from tf_usable_quality_v1.capture_client import CAMERAS
from tf_usable_aggregate_v1.aggregate import summarize,paired

MANIFEST_SHA='b0ce5e0fae5fed7efea51ce4592717602c0cf259aba6c3466c16e3c40d66e612'
SHARD_STATUS='COMPLETE30_FROZEN_VALIDATION_SHARD_ANALYSIS'
COHORT_STATUS='COMPLETE120_EXPANDED_DEV_DESCRIPTIVE_COHORT'
PARTIAL_STATUS='PARTIAL_FROZEN_VALIDATION_COHORT'
CAL_STATUS='PASS_FULL_REAL_CAL_SAME_DEVICE_ADAPTER_PARITY'
COLLECTION_STATUS='PASS_COMPLETE_FROZEN_VALIDATION_SHARD_COLLECTION'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def permissions():
    return dict(engineering_quality_screen=False,independent_test_complete=False,
                noninferiority_established=False,formal_performance_permission=False)

def no_failures(job):
    # Old failures are referenced by path/hash; no old failure files are copied
    # into a fresh shard. Any failure/STOP inside THIS shard remains fatal.
    markers=sorted(str(p.relative_to(job)) for p in job.rglob('*FAILED*.json'))
    markers+=sorted(str(p.relative_to(job)) for p in job.rglob('STOP_REQUESTED'))
    require(not markers,'current shard failure/STOP markers: '+repr(markers))

def manifest_doc(path):
    path=Path(path).resolve(strict=True)
    require(path==OWN/'DEV-MANIFEST.json' and sha(path)==MANIFEST_SHA,'frozen DEV manifest/path differs; TEST is forbidden')
    m=read(path)
    require(m.get('schema')=='draft-frozen-validation-dev-manifest/v1'
        and m['full_episodes']==120 and m['task_init_groups']==40
        and m['benchmark']=='libero_10' and m['horizon']==800
        and m['draft_seed']==20260923 and m['draft_total_updates']==3000,
        'fixed expanded DEV contract differs')
    require(len(m['shards'])==4 and len({s['id'] for s in m['shards']})==4,'four distinct whole shards required')
    seen=set()
    for shard in m['shards']:
        rows=shard['cases'];groups={(r['task'],r['init']) for r in rows}
        require(len(rows)==30 and len(groups)==10 and len({r['key'] for r in rows})==30
            and all(r['shard']==shard['id'] and r['group']=='expanded_DEV' and r['rep']==0
                    and r['init']==shard['init'] for r in rows),'manifest full30 case binding differs')
        require(not seen&groups,'shards overlap actual task/init identities');seen|=groups
        require(Counter(r['arm'] for r in rows)==Counter({a:10 for a in m['arms']})
            and all({r['arm'] for r in rows if (r['task'],r['init'])==g}==set(m['arms']) for g in groups),
            'three real arms per task/init required')
    require(len(seen)==40,'forty groups, not120 independent groups')
    return m

def plan_and_cases(job):
    p=load(job)
    m=manifest_doc(p['manifest_path'])
    require(p['manifest_sha256']==MANIFEST_SHA,'plan manifest SHA differs')
    shards=[s for s in m['shards'] if s['id']==p['shard']]
    require(len(shards)==1 and p['cases']==shards[0]['cases'],'plan must use exact manifest case sequence')
    require(p['horizon']==800 and p['seed']==m['seed'] and p['performance_claim'] is False,'original full-task semantics differ')
    for arm,digest in m['fixed_checkpoint_sha256'].items():
        require(p['selected'][arm]['checkpoint_sha256']==digest,'fixed same-seed3000 checkpoint differs: '+arm)
    registration=Path(p['registration']).resolve(strict=True)
    require(sha(registration)==p['registration_sha256'],'prospective cohort registration differs')
    source=Path(episode.__code__.co_filename).resolve()
    coverage=Path(episode.__globals__['coverage'].__code__.co_filename).resolve()
    require(source==EXP/'tf_usable_quality_v1/analyze.py'
        and coverage==EXP/'tf_usable_quality_v1/saved_outputs.py'
        and p['source_sha256'].get(str(source))==sha(source)
        and p['source_sha256'].get(str(coverage))==sha(coverage),
        'actual original episode/full-saved helper source differs')
    return p,m

def arrays(path):
    with np.load(path,allow_pickle=False) as f:
        result={k:np.array(f[k],copy=True) for k in f.files}
    require(all(v.dtype.kind in 'buifc' and bool(np.isfinite(v).all()) for v in result.values()),'nonfinite saved state/observation')
    return result

def same_arrays(a,b):
    return set(a)==set(b) and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype
        and np.ascontiguousarray(a[k]).tobytes()==np.ascontiguousarray(b[k]).tobytes() for k in a)

def tensor_fingerprint(x):
    require(isinstance(x,torch.Tensor) and x.device.type=='cpu' and x.dtype==torch.uint8
        and x.ndim==1 and x.numel()>0,'actual CPU/CUDA RNG tensor must be saved CPU uint8')
    return dict(shape=list(x.shape),dtype=str(x.dtype),bytes_sha256=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())

def pair_initial(job,reference,candidate,rng_equal):
    a=job/'episodes'/reference['key'];b=job/'episodes'/candidate['key']
    x,y=dict(reference['initial_pairing']),dict(candidate['initial_pairing'])
    x.pop('initial_obs_hashes');y.pop('initial_obs_hashes')
    physical=same_arrays(arrays(a/'client/initial_state.npz'),arrays(b/'client/initial_state.npz'))
    reset=read(a/'RESET.json')['state']==read(b/'RESET.json')['state']
    obs_a=arrays(a/'client/initial_observations.npz');obs_b=arrays(b/'client/initial_observations.npz')
    require(set(obs_a)==set(obs_b)==set(CAMERAS),'both actual initial cameras required')
    cameras={}
    for name in CAMERAS:
        aa,bb=obs_a[name],obs_b[name]
        require(aa.shape==bb.shape==(128,128,3) and aa.dtype==bb.dtype==np.uint8,'camera shape/dtype differs')
        d=np.abs(aa.astype(np.int16)-bb.astype(np.int16))
        cameras[name]=dict(exact=bool(np.array_equal(aa,bb)),changed_channels=int(np.count_nonzero(d)),
            changed_pixels=int(np.count_nonzero(np.any(d!=0,axis=2))),max_abs=int(d.max()),
            reference_sha256=hashlib.sha256(aa.tobytes()).hexdigest(),candidate_sha256=hashlib.sha256(bb.tobytes()).hexdigest())
    physical_RNG=x==y and physical and reset and rng_equal
    return dict(shard=candidate['shard'],group=candidate['group'],task=candidate['task'],init=candidate['init'],
        reference=reference['arm'],candidate=candidate['arm'],reference_key=reference['key'],candidate_key=candidate['key'],
        metadata_initial_sim_controller_python_numpy_rng_equal=x==y,actual_saved_sim_controller_arrays_bitwise_equal=physical,
        full_target_reset_KV_VAE_CPU_CUDA_RNG_state_equal=reset,actual_target_initial_RNG_equal=rng_equal,
        physical_RNG_equal=physical_RNG,camera_pixels=cameras,all_initial_camera_pixels_exact=all(v['exact'] for v in cameras.values()),
        all_exact=physical_RNG and all(v['exact'] for v in cameras.values()),
        limits='Saved physical/controller arrays and actual arm-level initial Torch RNG; raw Python/NumPy RNG and unsaved simulator internals UNKNOWN. Initial pixels do not certify subsequent history.')

def numeric_checks(summary,comparisons):
    ratio=comparisons['TF_CONT_vs_T']['both_success_steps_ratio']
    return dict(TF_success_at_least_T=summary['TF_CONT']['successes']>=summary['T']['successes'],
        TF_success_at_least_equal_update_MLP=summary['TF_CONT']['successes']>=summary['MLP_CONT']['successes'],
        common_success_nonempty=ratio['n']>0,length_P50=ratio['p50'] is not None and ratio['p50']<=1.10,
        length_P90=ratio['p90'] is not None and ratio['p90']<=1.20,
        length_max=ratio['max'] is not None and ratio['max']<=1.30)

def summaries(rows,arms):
    summary={a:summarize([r for r in rows if r['arm']==a]) for a in arms}
    comparisons={f'{a}_vs_{b}':paired(rows,a,b,False) for a,b in
        (('TF_CONT','T'),('MLP_CONT','T'),('TF_CONT','MLP_CONT'))}
    return summary,comparisons

def compute_shard(job,p,m):
    """Read every real saved value; no writes or calls to old analyze()."""
    no_failures(job)
    collection=read(job/'COLLECTION.json')
    require(collection['status']==COLLECTION_STATUS and collection['full_episodes']==30
        and collection['completed']==[r['key'] for r in p['cases']]
        and collection['all_own_processes_cleaned'] is True,'complete30 original-horizon collection required')
    ready={a:read(job/f'{a}-READY.json') for a in m['arms']}
    rng={a:torch.load(job/'arms'/a/'TARGET-INITIAL-RNG.pt',map_location='cpu',weights_only=True) for a in m['arms']}
    for a in m['arms']:
        require(set(rng[a])=={'CPU','CUDA'} and read(job/'arms'/a/'OWNED-CLEANUP.json')['all_roots_reaped'] is True,
            'actual target RNG or arm cleanup missing: '+a)
        require(ready[a]['target_initial_rng']==ready['T']['target_initial_rng']
            and ready[a]['ready_rng']==ready[a]['target_initial_rng']
            and ready[a]['full_real_CAL_rows']==(0 if a=='T' else 20237),'READY initial RNG/fullCAL identity')
        for key in ('CPU','CUDA'):
            require(tensor_fingerprint(rng[a][key])==ready[a]['target_initial_rng'][key.lower()]
                and torch.equal(rng[a][key],rng['T'][key]),'actual RNG bytes and READY fingerprint differ')
        if a!='T':
            v=read(job/'arms'/a/'FULL-REAL-CAL-PARITY.json')
            require(v['status']==CAL_STATUS and v['rows']==20237 and v['CAL_episodes']==20
                and v['complete_CAL'] is True and v['same_device_bitwise_exact'] is True
                and v['finite'] is True and v['CPU_rng_unchanged'] is True and v['CUDA_rng_unchanged'] is True
                and v['synthetic_queries']==0 and v['target_forwards']==0
                and v['checkpoint_sha256']==m['fixed_checkpoint_sha256'][a]
                and v['architecture']==p['selected'][a]['architecture'],'full actual same-GPU CAL interface differs')
    rows=[]
    for case in p['cases']:
        r=episode(job,case,ready[case['arm']])
        r.update(site=p['site'],shard=p['shard'],group=case['group'],rep=case['rep'])
        require(type(r['env_steps']) is int and 0<r['env_steps']<=800 and 0<r['target_forwards']<=p['per_episode_target_forward_cap'],
            'actual full task bounds differ')
        rows.append(r)
    require(len(rows)==30 and sum(r['target_forwards'] for r in rows)==collection['total_target_forwards']<=p['job_target_forward_cap'],
        'complete shard actual target accounting differs')
    groups={(r['task'],r['init']) for r in rows};pairs=[]
    for task,init in sorted(groups):
        cells={r['arm']:r for r in rows if (r['task'],r['init'])==(task,init)}
        require(set(cells)==set(m['arms']),'all three arms required per actual group')
        for a in m['arms']:
            if a!='T':pairs.append(pair_initial(job,cells['T'],cells[a],True))
    require(len(groups)==10 and len(pairs)==20,'ten groups/twenty initial pairings required')
    physical=all(r['physical_RNG_equal'] for r in pairs);pixel=all(r['all_initial_camera_pixels_exact'] for r in pairs)
    certificate=dict(status='COMPLETE20_INITIAL_PAIRING_AUDIT',pairs=20,complete_episodes=30,task_init_groups=10,
        physical_RNG_pairing_passed=physical,all_initial_camera_pixels_exact=pixel,comparisons=pairs,
        no_image_injection=True,old_FAILED_and_permissions_unchanged=True)
    summary,comparisons=summaries(rows,m['arms']);checks=numeric_checks(summary,comparisons)
    no_failures(job)
    return dict(schema='draft-frozen-validation-shard-analysis/v1',status=SHARD_STATUS,full_episodes=30,
        task_init_groups=10,shard=p['shard'],manifest_path=p['manifest_path'],manifest_sha256=MANIFEST_SHA,
        plan_sha256=sha(job/'plan.json'),registration_sha256=p['registration_sha256'],analysis_source_sha256=sha(Path(__file__)),
        fixed_checkpoint_sha256=m['fixed_checkpoint_sha256'],arms=summary,comparisons=comparisons,episode_rows=rows,
        initial_pairing=certificate,physical_RNG_pairing_passed=physical,strict_initial_camera_pixels_passed=pixel,
        numeric_checks_descriptive_only=checks,numeric_screen_observed=all(checks.values()),
        strict_numeric_and_initial_pair_screen=all(checks.values()) and physical and pixel,
        original_saved_output_values_actually_audited=True,total_target_forwards=collection['total_target_forwards'],
        scope='One expanded_DEV shard, not independent TEST; final supervisor/clock cleanup is required separately.',
        permissions=permissions(),overall_quality_passed=False,global_default_changed=False,performance_claim=False,
        original_today50_and_morning_FAILED_unchanged=True,
        limitations=['Same-seed23 equal3000 updates is a controlled comparison, not TF superiority or noninferiority.',
            'Shared-GPU wall is diagnostic, never formal speed.','Do not pool today35, old33/34, risk25to29, or old13 into this cohort.',
            'Complete subsequent history/full simulator determinism is not established by initial pixel/state equality.'])

def shard_report(result):
    lines=['# 扩大DEV单分片完整30局','',f"分片：{result['shard']}；10个task/init组、三臂各10完整局。四权限false。",'',
        '| 方案 | 成功/10 | 全部真实步数P50/P90 | 失败800均值 | target/chunk |',
        '|---|---:|---:|---:|---:|']
    for arm,s in result['arms'].items():
        d=s['actual_env_steps'];lines.append(f"| {arm} | {s['successes']}/10 | {d['p50']:.1f}/{d['p90']:.1f} | {s['failure800_penalty']['mean']:.2f} | {s['target_forwards_per_chunk']:.3f} |")
    cert=result['initial_pairing'];lines+=['',f"物理/RNG：{sum(r['physical_RNG_equal'] for r in cert['comparisons'])}/20；相机严格一致：{sum(r['all_initial_camera_pixels_exact'] for r in cert['comparisons'])}/20。",'',
        '主TF/T共同成功长度分布：'+json.dumps(result['comparisons']['TF_CONT_vs_T']['both_success_steps_ratio'],ensure_ascii=False),
        '失败真实步数与固定800罚分分别记录；所有pair和四格完整保留。',
        '这是DEV描述，非TEST/非劣/正式性能许可；最终仍须监督器完整清理及失败优先终态。']
    return '\n'.join(lines)+'\n'

def analyze(job):
    job=Path(job).resolve(strict=True);affinity=os.sched_getaffinity(0)
    os.sched_setaffinity(0,sorted(affinity)[:2]);torch.set_num_threads(2)
    try:
        require(not torch.cuda.is_initialized(),'CPU-only analysis must not initialize CUDA')
        p,m=plan_and_cases(job);result=compute_shard(job,p,m)
        write(job/'PAIRING-PIXELS.json',result['initial_pairing'])
        require(result['physical_RNG_pairing_passed'],'saved physical/reset/actual-RNG mismatch; preserve real pair differences')
        no_failures(job)
        with (job/'REPORT.md').open('x') as f:f.write(shard_report(result))
        no_failures(job)
        return result
    finally:os.sched_setaffinity(0,affinity)

def terminal_shard(job):
    job=Path(job).resolve(strict=True);no_failures(job);p,m=plan_and_cases(job)
    result=read(job/'RESULT.json');collection=read(job/'COLLECTION.json')
    require(result['status']==SHARD_STATUS and result['full_episodes']==30 and result['task_init_groups']==10
        and result['manifest_sha256']==MANIFEST_SHA and result['plan_sha256']==sha(job/'plan.json')
        and result['original_saved_output_values_actually_audited'] is True
        and result['permissions']==permissions() and result['overall_quality_passed'] is False,
        'complete source-bound shard RESULT required')
    require(collection['completed']==[r['key'] for r in p['cases']] and collection['full_episodes']==30
        and read(job/'FINAL-OWNED-CLEANUP.json')['all_roots_reaped'] is True,'full shard final cleanup required')
    for a in m['arms']:require(read(job/'arms'/a/'OWNED-CLEANUP.json')['all_roots_reaped'] is True,'arm terminal cleanup')
    clock=read(job/'CLOCK-RESULT.json');guard=read(job/'CLOCK-GUARD.json')
    require(clock['status']=='CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS'
        and clock['job']==p['job'] and guard['job']==p['job']
        and clock['all_observed_owned_live_processes_exited'] is True
        and clock['supervisor_exit_observation']['live'] is False
        and clock['original_quality_RESULT_present'] is True and clock['original_quality_FAILED_present'] is False
        and clock['external_stop_reason'] is None and clock['error'] is None and not clock['signals']
        and clock['no_foreign_signals'] is True and clock['clock']==guard['clock'],'actual clean independent clock terminal required')
    charged=clock['elapsed_quality_process_seconds']
    require(isinstance(charged,(int,float)) and math.isfinite(charged) and charged>=0,'finite nonnegative actual charged lifetime required')
    require(charged<=clock['clock']['hard_stop_monotonic']-clock['clock']['process_birth_monotonic'],
        'actual GPU-job charged lifetime exceeded independent hard clock')
    # Re-read real saved outputs and raw pairing, without rewriting either
    # earlier RESULT or PAIRING/REPORT. No placeholder result can satisfy this.
    again=compute_shard(job,p,m)
    require(all(result.get(k)==v for k,v in again.items()),'actual raw shard recomputation differs from original RESULT')
    require(read(job/'PAIRING-PIXELS.json')==again['initial_pairing']
        and (job/'REPORT.md').read_text()==shard_report(again),'actual pairing/report differs from complete raw recomputation')
    require(again['physical_RNG_pairing_passed'],'actual physical/RNG mismatch')
    no_failures(job)
    names=('plan.json','RESULT.json','COLLECTION.json','PAIRING-PIXELS.json','REPORT.md',
        'FINAL-OWNED-CLEANUP.json','CLOCK-GUARD.json','CLOCK-RESULT.json')
    return dict(job=str(job),plan=p,result=again,clock=clock,charged_seconds=charged,
        evidence_sha256={name:sha(job/name) for name in names})

def merge(shard_jobs,manifest):
    """Return PARTIAL on any missing/invalid shard; never pool passing subsets."""
    affinity=os.sched_getaffinity(0);os.sched_setaffinity(0,sorted(affinity)[:2]);torch.set_num_threads(2)
    base=dict(schema='draft-frozen-validation-cohort-analysis/v1',manifest=str(Path(manifest)),
        expected_full_episodes=120,expected_task_init_groups=40,permissions=permissions(),
        overall_quality_passed=False,global_default_changed=False,performance_claim=False,
        old_FAILED_unchanged=True,TEST_outcomes_read=False,no_GPU_or_SSH_or_new_episode=True)
    receipts=[];issues=[]
    try:
        require(not torch.cuda.is_initialized(),'merge is CPU-only')
        m=manifest_doc(manifest);ids=[s['id'] for s in m['shards']]
        require(isinstance(shard_jobs,dict) and set(shard_jobs)==set(ids),'explicit all-four shard mapping required')
        for sid in ids:
            try:
                receipt=terminal_shard(shard_jobs[sid]);require(receipt['plan']['shard']==sid,'shard mapping identity differs')
                receipts.append(receipt)
            except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                issues.append(dict(shard=sid,error=repr(exc)))
        require(not issues,'one or more complete shard terminals rejected')
        first=receipts[0]['plan'];mapping={sid:str(Path(shard_jobs[sid]).resolve()) for sid in ids};cumulative=0.
        for receipt in receipts:
            p=receipt['plan']
            require(all(p[k]==first[k] for k in ('registration','registration_sha256','selected','source_sha256',
                'cohort_started_at','cohort_cleanup_deadline','cohort_gpu_job_cap_seconds','cohort_output_budget_GiB','cohort_metadata_root','shard_jobs')),
                'shards must bind one cohort/registration/frozen weight and engine')
            require({k:str(Path(v).resolve()) for k,v in p['shard_jobs'].items()}==mapping,'cohort complete shard map differs')
            require(p['cohort_gpu_job_cap_seconds']==28800 and p['cohort_output_budget_GiB']==5
                and p['output_budget_GiB']==3,'registered cohort/shard resource caps differ')
            require(float(p['charged_gpu_job_seconds_before_shard'])==cumulative,'actual prior charged ledger mismatch')
            cumulative+=receipt['charged_seconds']
        require(cumulative<=28800,'all launched shard GPU-job lifetime charges exceed8h')
        start=datetime.fromisoformat(first['cohort_started_at']);end=datetime.fromisoformat(first['cohort_cleanup_deadline'])
        require(start.tzinfo is not None and end.tzinfo is not None and 0<(end-start).total_seconds()<=43200,'registered cohort12h wall cap differs')
        require(all(datetime.fromisoformat(x['clock']['clock']['absolute_cleanup_deadline'])<=end for x in receipts),
            'a shard independent hard clock exceeds cohort wall deadline')
        metadata_root=Path(first['cohort_metadata_root']).resolve(strict=True)
        require(metadata_root==Path(first['registration']).resolve().parent
            and metadata_root.parent==EXP.parent/'results/runtime'
            and metadata_root.name.startswith('p0-draft-frozen-validation-v1-cohort-'),
            'explicit isolated cohort metadata output root differs')
        roots=[metadata_root]+[Path(x['job']).resolve() for x in receipts]
        require(all(not a.is_relative_to(b) and not b.is_relative_to(a)
                    for i,a in enumerate(roots) for b in roots[i+1:]),'cohort output roots overlap or nest')
        file_bytes=[]
        for receipt in receipts:
            size=sum(p.stat().st_size for p in Path(receipt['job']).rglob('*') if p.is_file())
            require(size<=3*1024**3,'actual shard output cap exceeded');file_bytes.append(size)
        metadata_bytes=sum(p.stat().st_size for p in metadata_root.rglob('*') if p.is_file())
        require(sum(file_bytes)+metadata_bytes<=5*1024**3,'actual cohort output including metadata cap exceeded')
        rows=[r for receipt in receipts for r in receipt['result']['episode_rows']]
        require(len(rows)==120 and len({(r['task'],r['init']) for r in rows})==40
            and len({r['key'] for r in rows})==120,'all120 actual rows/40 groups required')
        pairs=[r for receipt in receipts for r in receipt['result']['initial_pairing']['comparisons']]
        require(len(pairs)==80,'all80 initial candidate/reference pairs required')
        summary,comparisons=summaries(rows,m['arms']);checks=numeric_checks(summary,comparisons)
        physical=all(r['physical_RNG_equal'] for r in pairs);pixel=all(r['all_initial_camera_pixels_exact'] for r in pairs)
        for receipt in receipts:no_failures(Path(receipt['job']))
        return dict(base,status=COHORT_STATUS,complete=True,full_episodes=120,task_init_groups=40,
            relative_to_T_initial_pairs=80,per_candidate_pairs=40,arms=summary,comparisons=comparisons,
            episode_rows=rows,initial_pairing_comparisons=pairs,physical_RNG_pairing_passed=physical,
            strict_initial_camera_pixels_passed=pixel,numeric_checks_descriptive_only=checks,
            numeric_screen_observed=all(checks.values()),strict_numeric_and_initial_pair_screen=all(checks.values()) and physical and pixel,
            actual_charged_GPU_job_seconds=cumulative,registered_cohort_wall_seconds=(end-start).total_seconds(),
            actual_shard_output_bytes=file_bytes,actual_cohort_metadata_output_bytes=metadata_bytes,
            actual_cohort_output_bytes=sum(file_bytes)+metadata_bytes,
            all120_saved_values_and_pairings_recomputed_on_CPU=True,
            shard_receipts=[dict(shard=x['plan']['shard'],job=x['job'],charged_seconds=x['charged_seconds'],evidence_sha256=x['evidence_sha256']) for x in receipts],
            per_shard={x['plan']['shard']:dict(arms=x['result']['arms'],comparisons=x['result']['comparisons']) for x in receipts},
            recommendation='EXPANDED_DEV_DESCRIPTIVE_ONLY_NO_AUTOMATIC_TEST_OR_SPEED_PERMISSION',
            limitations=['Forty fixed task/init groups, not120 independent groups; no3ppNI/TF superiority claim.',
                'No old13/old50/today35/old33to34 pooling.','Initial equality cannot guarantee equal subsequent environment history.'])
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        if not issues:issues=[dict(error=repr(exc))]
        return dict(base,status=PARTIAL_STATUS,complete=False,issues=issues,
            validated_terminal_shards=[x['plan']['shard'] for x in receipts],
            no_pooled_quality_or_partial_zero_filling=True,selection='NONE_PENDING_ALL_FOUR_REAL_COMPLETE_TERMINALS')
    finally:os.sched_setaffinity(0,affinity)

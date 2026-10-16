"""Complete50 real-data analysis; strata and original permissions stay separate."""
from __future__ import annotations
from collections import Counter
import os
from pathlib import Path
import numpy as np
import torch
from draft_freeze_quality_v1.protocol import ARMS, ARCH, load, read, write
from tf_usable_quality_v1.analyze import episode
from tf_usable_quality_v1.capture_client import CAMERAS
from tf_usable_aggregate_v1.aggregate import summarize, paired

def require(ok,msg):
    if not ok:raise ValueError(msg)

def same_arrays(a,b):
    return (set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape
            and np.ascontiguousarray(a[k]).tobytes()==np.ascontiguousarray(b[k]).tobytes() for k in a))

def arrays(path):
    with np.load(path,allow_pickle=False) as v:return {k:np.array(v[k],copy=True) for k in v.files}

def compare_initial(job,base,new,rng_equal):
    t=job/'episodes'/base['key']/'client';c=job/'episodes'/new['key']/'client'
    x,y=dict(base['initial_pairing']),dict(new['initial_pairing'])
    x.pop('initial_obs_hashes');y.pop('initial_obs_hashes')
    saved= same_arrays(arrays(t/'initial_state.npz'),arrays(c/'initial_state.npz'))
    a,b=arrays(t/'initial_observations.npz'),arrays(c/'initial_observations.npz')
    require(set(a)==set(b)==set(CAMERAS),'actual two camera inventory required')
    cameras={}
    for name in CAMERAS:
        diff=np.abs(a[name].astype(np.int16)-b[name].astype(np.int16))
        cameras[name]=dict(exact=bool(np.array_equal(a[name],b[name])),
            changed_pixels=int(np.count_nonzero(np.any(diff,axis=2))),
            changed_channels=int(np.count_nonzero(diff)),max_abs=int(diff.max()))
    physical=x==y and saved and rng_equal
    return dict(group=new['group'],task=new['task'],init=new['init'],reference=base['arm'],
        candidate=new['arm'],reference_key=base['key'],candidate_key=new['key'],
        physical_RNG_equal=physical,actual_saved_sim_controller_arrays_exact=saved,
        actual_target_initial_RNG_equal=rng_equal,camera_pixels=cameras,
        all_exact=physical and all(v['exact'] for v in cameras.values()),
        scope='Saved initial state/controller and actual initial Torch RNG; unsaved simulator internals UNKNOWN')

def analyze(job):
    job=Path(job).resolve();p=load(job)
    # Controller only: server workers have already exited. Bound real CPU audit.
    affinity=os.sched_getaffinity(0);os.sched_setaffinity(0,sorted(affinity)[:2])
    torch.set_num_threads(2)
    try:
        require(not torch.cuda.is_initialized(),'CPU analysis must not allocate CUDA')
        collection=read(job/'COLLECTION.json')
        require(collection['status']=='PASS_COMPLETE_DRAFT_FREEZE_COLLECTION'
            and collection['full_episodes']==50
            and collection['completed']==[r['key'] for r in p['cases']]
            and collection['all_own_processes_cleaned'] is True,'complete50 collection required')
        ready={a:read(job/f'{a}-READY.json') for a in ARMS}
        rng={a:torch.load(job/'arms'/a/'TARGET-INITIAL-RNG.pt',map_location='cpu',weights_only=True) for a in ARMS}
        equal={a:set(rng[a])==set(rng['T'])=={'CPU','CUDA'}
            and all(torch.equal(rng[a][k],rng['T'][k]) for k in ('CPU','CUDA')) for a in ARMS}
        for a in ARMS:
            require(equal[a] and ready[a]['target_initial_rng']==ready['T']['target_initial_rng']
                and ready[a]['ready_rng']==ready[a]['target_initial_rng']
                and read(job/'arms'/a/'OWNED-CLEANUP.json')['all_roots_reaped'] is True,
                'actual initial target RNG/arm cleanup differs: '+a)
            if a!='T':
                cal=read(job/'arms'/a/'FULL-REAL-CAL-PARITY.json')
                require(cal['status']=='PASS_FULL_REAL_CAL_SAME_DEVICE_ADAPTER_PARITY'
                    and cal['architecture']==ARCH[a] and cal['CAL_episodes']==20
                    and cal['rows']==20237 and cal['complete_CAL'] is True
                    and cal['same_device_bitwise_exact'] is True and cal['synthetic_queries']==0
                    and cal['finite'] is True and cal['CPU_rng_unchanged'] is True
                    and cal['CUDA_rng_unchanged'] is True and cal['target_forwards']==0
                    and cal['checkpoint_sha256']==p['selected'][a]['checkpoint_sha256'],
                    'full real BF16 GPU CAL interface not exact: '+a)
        rows=[]
        for case in p['cases']:
            r=episode(job,case,ready[case['arm']])
            r.update(site='local',group=case['group'],rep=case['rep']);rows.append(r)
        require(sum(r['target_forwards'] for r in rows)==collection['total_target_forwards'],
                'all50 target count differs')
        fresh=[r for r in rows if r['group']=='fresh_DEV'];risk=[r for r in rows if r['group']=='risk_DEV']
        repeats=[r for r in rows if r['group']=='repeatability']
        require(len(fresh)==30 and len(risk)==16 and len(repeats)==4,'registered strata counts differ')
        comparisons={k:paired(fresh,a,b,False) for k,a,b in (
            ('TF_CONT_vs_T','TF_CONT','T'),('MLP_CONT_vs_T','MLP_CONT','T'),
            ('TF_CONT_vs_MLP_CONT','TF_CONT','MLP_CONT'))}
        risk_comparison=paired(risk,'TF_CONT','T',False)
        fresh_summary={a:summarize([r for r in fresh if r['arm']==a]) for a in ARMS}
        risk_summary={a:summarize([r for r in risk if r['arm']==a]) for a in ('T','TF_CONT')}
        pairs=[]
        for group,group_rows in (('fresh_DEV',fresh),('risk_DEV',risk)):
            for task,init in sorted({(r['task'],r['init']) for r in group_rows}):
                cells={r['arm']:r for r in group_rows if (r['task'],r['init'])==(task,init)}
                for a in (ARMS[1:] if group=='fresh_DEV' else ('TF_CONT',)):
                    pairs.append(compare_initial(job,cells['T'],cells[a],equal[a]))
        repeat_pairs=[]
        for task in (2,8):
            a,b=sorted([r for r in repeats if r['task']==task],key=lambda r:r['rep'])
            cmp=compare_initial(job,a,b,True)
            ca=np.load(job/'episodes'/a['key']/'client/commands.npy',allow_pickle=False)
            cb=np.load(job/'episodes'/b['key']/'client/commands.npy',allow_pickle=False)
            cmp.update(old_success=a['success'],new_success=b['success'],
                old_env_steps=a['env_steps'],new_env_steps=b['env_steps'],
                complete_commands_bitwise_equal=ca.dtype==cb.dtype and ca.shape==cb.shape
                    and ca.tobytes()==cb.tobytes())
            repeat_pairs.append(cmp)
        require(len(pairs)==28,'all28 current initial pairings required')
        physical=all(v['physical_RNG_equal'] for v in pairs)
        pixel=all(v['all_exact'] for v in pairs)
        pair_certificate=dict(status='COMPLETE28_INITIAL_PAIRING_AUDIT',pairs=28,complete_episodes=50,
            physical_RNG_pairing_passed=physical,all_initial_camera_pixels_exact=pixel,
            comparisons=pairs,repeatability_pairs=repeat_pairs,old_FAILED_unchanged=True)
        write(job/'PAIRING-PIXELS.json',pair_certificate)
        require(physical,'current saved physical/controller/targetRNG pairings differ')
        ratio=comparisons['TF_CONT_vs_T']['both_success_steps_ratio']
        checks=dict(fresh_TF_success_at_least_T=fresh_summary['TF_CONT']['successes']>=fresh_summary['T']['successes'],
            fresh_TF_success_at_least_equal_update_MLP=fresh_summary['TF_CONT']['successes']>=fresh_summary['MLP_CONT']['successes'],
            fresh_common_success_nonempty=ratio['n']>0,
            fresh_length_P50=ratio['p50'] is not None and ratio['p50']<=1.10,
            fresh_length_P90=ratio['p90'] is not None and ratio['p90']<=1.20,
            fresh_length_max=ratio['max'] is not None and ratio['max']<=1.30,
            risk_no_new_TF_failure=risk_comparison['four_cells']['reference_1_candidate_0']==0)
        numeric=all(checks.values())
        result=dict(schema='draft-freeze-complete50-analysis/v1',status='COMPLETE50_PROSPECTIVE_DEV_ANALYSIS',
            full_episodes=50,fresh_DEV=fresh_summary,risk_DEV=risk_summary,
            comparisons=comparisons,risk_TF_CONT_vs_T=risk_comparison,repeatability=repeat_pairs,
            episode_rows=rows,engineering_numeric_checks=checks,numeric_screen_observed=numeric,
            strict_current_initial_pixel_screen=pixel,physical_RNG_pairing_passed=physical,
            strict_current_numeric_and_initial_pair_screen=numeric and pixel,
            retained_primary_for_prospective_independent_TEST=numeric and physical,
            recommendation='RETAIN_FIXED_TF64_CONT_FOR_TEST' if numeric else 'UNMET_DEV_REQUIREMENTS_DO_NOT_EXPAND_AUTOMATICALLY',
            permissions=dict(engineering_quality_screen=False,independent_test_complete=False,
                noninferiority_established=False,formal_performance_permission=False),
            global_default_changed=False,original_morning_FAILED_and_PARTIAL_unchanged=True,
            total_target_forwards=sum(r['target_forwards'] for r in rows),performance_claim=False,
            warnings=['All50 include known-risk and repeated target: never report as50 independent quality groups.',
                'Numeric DEV retention is not the original strict certificate or3ppNI/architecture superiority.',
                'Initial camera equality does not prove equal subsequent sensor history or full simulator state.',
                'All saved output tensors actually checked by frozen original episode helper; no rawvelocity cost recompute.',
                'No image injection, fresh-observation forcing, PF/threshold/tree change or system optimization.'])
        lines=['# 今天draft结构/训练配方：完整50条DEV','','原FAILED/PARTIAL及四权限均保持；fresh、风险、重复分别统计。','',
            '| fresh init35方案 | 成功/10 | 失败800均值 | target/chunk |','|---|---:|---:|---:|']
        for a,s in fresh_summary.items():
            lines.append(f"|{a}|{s['successes']}/10|{s['failure800_penalty']['mean']:.2f}|{s['target_forwards_per_chunk']:.3f}|")
        lines+=['',f"TF/T共同成功步数比：{ratio}",f"风险8组配对：{risk_comparison['four_cells']}",
            f"28初始像素严格一致：{sum(v['all_exact'] for v in pairs)}/28；物理/RNG：{physical}",
            f"事前数值条件：{checks}",f"建议：{result['recommendation']}",
            '目标重复的完整commands/success/相机一致结果见RESULT；不能据有限重复认定根因或保证未来。']
        with (job/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
        return result
    finally:
        os.sched_setaffinity(0,affinity)

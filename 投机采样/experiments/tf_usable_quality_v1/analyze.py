"""Four-arm complete-task DEV quality, physical/RNG pairing, camera audit."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

EXP=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(EXP))
from tf_usable_quality_v1.protocol import ARMS,ARCH,load,read,write
from tf_usable_quality_v1.saved_outputs import coverage
from tf_usable_quality_v1.capture_client import CAMERAS


def require(condition,message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    if not values:
        return dict(n=0,mean=None,p50=None,p90=None,min=None,max=None)
    x=np.asarray(values,dtype=np.float64)
    require(bool(np.isfinite(x).all()),'nonfinite metric')
    return dict(n=len(values),mean=float(x.mean()),p50=float(np.quantile(x,.5)),
                p90=float(np.quantile(x,.9)),min=float(x.min()),max=float(x.max()))


def only_record(folder):
    manifest=folder/'episodes.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    require(len(rows)==1,'one actual complete episode required')
    return rows[0]


def hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def episode(job,case,ready):
    out=job/'episodes'/case['key']
    folder=out/'client'
    record=only_record(folder)
    require(record.get('status')=='completed' and record.get('benchmark')=='libero_10'
        and record.get('task_index')==case['task'] and record.get('episode_index')==case['init']
        and record.get('libero_horizon')==800 and record.get('max_policy_chunks') is None
        and record.get('truncated_after_max_policy_chunks') is False
        and (record.get('success') or record.get('horizon_reached')),
        'complete original-horizon DEV identity/result differs')
    require(record['env_steps']>0 and record['infer_chunks']>0,'empty task trajectory')
    require(len(record['action_shapes'])==record['infer_chunks']
            and all(shape==[7,4,4] for shape in record['action_shapes']),
            'complete policy action geometry differs')
    meta=read(folder/'repro_metadata.json')
    with np.load(folder/'initial_observations.npz',allow_pickle=False) as images:
        require(set(images.files)==set(CAMERAS)==set(meta['initial_obs_hashes']),
                'both initial camera pixel arrays must match metadata inventory')
        for name in CAMERAS:
            array=images[name]
            require(array.shape==(128,128,3) and array.dtype==np.uint8
                    and hash_array(array)==meta['initial_obs_hashes'][name],
                    'actual uint8 initial camera pixels/hash differ: '+name)
    commands=np.load(folder/'commands.npy',allow_pickle=False)
    require(commands.shape==(record['env_steps'],7) and commands.dtype==np.float32
            and bool(np.isfinite(commands).all()) and meta['command_count']==len(commands)
            and meta['no_policy_chunk_truncation'] is True,'actual commands do not cover full task')
    for which in ('initial','final'):
        with np.load(folder/f'{which}_state.npz',allow_pickle=False) as state:
            require(set(state.files)==set(meta[f'{which}_state_hashes']),'state artifact field inventory differs')
            for name in state.files:
                array=state[name]
                require(bool(np.isfinite(array).all()) and hash_array(array)==meta[f'{which}_state_hashes'][name],
                        'initial/final state artifact differs from recorded metadata')
    reset=read(out/'RESET.json')
    require(reset['status']=='PASS_FULL_RESET_BANK_AND_EMPTY_STATE' and reset['RNG_unchanged'] is True
        and reset['row']==case and reset['state']['frame_st_id']==0
        and len(reset['state']['KV'])==30 and reset['state']['init_latent'] is None
        and 'VAE' in reset['state'] and 'prompt' in reset['state']
        and reset['state']['cpu_rng']==ready['target_initial_rng']['cpu']
        and reset['state']['cuda_rng']==ready['target_initial_rng']['cuda'],
        'full bank reset/RNG proof differs')
    counter=read(out/'COUNTER.json')
    require(counter['arm']==case['arm'] and counter['key']==case['key']
        and counter['target_forwards']==counter['returned_forwards']
        and read(out/'FINISH_READY.json')['status']=='PASS'
        and read(out/'CLIENT-CLEANUP.json')['all_roots_reaped'] is True,
        'closed full-task accounting required')
    requests=counter['requests']
    counts=Counter(request['kind'] for request in requests)
    require(all(r['ok'] for r in requests) and counts['reset']==1
        and counts['policy']==record['infer_chunks']
        and counts['cache']==record['cache_infer_calls']
        and sum(r['target_forwards'] for r in requests)==counter['target_forwards'],
        'model requests/target forwards do not cover episode')
    for request in requests:
        context=request['trace_context']
        require(context['task_index']==case['task'] and context['episode_index']==case['init']
                and context['parallel_worker_id']==case['key']
                and context['episode_attempt_id']==record['episode_attempt_id'],
                'model request identity differs')
    accepted=Counter()
    for tree in counter['acceptance']:
        context=tree['context']
        require(context['task_index']==case['task'] and context['episode_index']==case['init']
            and context['parallel_worker_id']==case['key']
            and context['episode_attempt_id']==record['episode_attempt_id']
            and tree['actual_depth'] in (0,1,2),
            'acceptance tree identity differs')
        accepted[tree['actual_depth']]+=1
    trees=sum(accepted.values())
    if case['arm']=='T':
        require(trees==0,'target-only arm unexpectedly used speculation')
    else:
        require(trees>0 and trees==counter['phase_counters'].get('tree_count',0),
                'speculative tree count incomplete')
    final=read(out/'FINAL-SERVER-STATE.json')
    require(final['frame_st_id']>=0 and len(final['KV'])==30
            and all(set(layer)=={'mask','id','is_pred','k','v'} for layer in final['KV'])
            and {'VAE','init_latent','cpu_rng','cuda_rng','prompt','negative','action_mask',
                 'q01','q99','action_norm_method'}<=set(final),
            'complete final KV/VAE/model state snapshot required')
    saved=coverage(out/'server_output',record,counter)
    summary=dict(arm=case['arm'],task=case['task'],init=case['init'],key=case['key'],episode_attempt_id=record['episode_attempt_id'],
        success=bool(record['success']),horizon_reached=bool(record['horizon_reached']),
        env_steps=int(record['env_steps']),infer_chunks=int(record['infer_chunks']),
        target_forwards=int(counter['target_forwards']),target_forwards_per_chunk=counter['target_forwards']/record['infer_chunks'],
        trees=trees,acceptance_depth_counts={str(i):accepted[i] for i in range(3)},
        accepted_depth_sum=sum(k*v for k,v in accepted.items()),
        mean_accepted_depth=None if not trees else sum(k*v for k,v in accepted.items())/trees,
        episode_wall_diagnostic_sec=float(record['elapsed_sec']),reset_wall_diagnostic_sec=reset['reset_wall_sec'],
        horizon_penalty_score=int(record['env_steps']) if record['success'] else 800,
        score_semantics='actual policy env_steps for success; fixed 800 penalty for failure; not measured time',
        initial_pairing={k:meta[k] for k in ('seed','model_xml_sha256','init_state_sha256',
                                           'initial_state_hashes','initial_obs_hashes','initial_rng')},
        phase_counters=counter['phase_counters'],saved_output_audit=saved,performance_claim=False)
    return summary


def paired(candidate,reference,by_arm,init):
    rows=[]
    cells=Counter()
    ratios=[]
    for task in range(10):
        a,b=by_arm[candidate][task],by_arm[reference][task]
        cells[f"reference_{int(b['success'])}_candidate_{int(a['success'])}"]+=1
        ratio=a['env_steps']/b['env_steps'] if a['success'] and b['success'] else None
        if ratio is not None:
            ratios.append(ratio)
        rows.append(dict(task=task,init=init,candidate_success=a['success'],reference_success=b['success'],
                         candidate_steps=a['env_steps'],reference_steps=b['env_steps'],both_success_steps_ratio=ratio))
    return dict(candidate=candidate,reference=reference,pairs=10,
        success_difference_pp=10*(sum(r['candidate_success'] for r in rows)-sum(r['reference_success'] for r in rows)),
        four_cells={f'reference_{r}_candidate_{c}':cells[f'reference_{r}_candidate_{c}'] for r in (0,1) for c in (0,1)},
        both_success_step_ratio=distribution(ratios),rows=rows,
        limitation='Ten task/init pairs are engineering DEV; no 3pp noninferiority claim')


def initial_pairing(job,p,by_arm,ready):
    comparisons=[];passed=True
    baseline=ready['T']['target_initial_rng']
    rng_equal=all(ready[arm]['target_initial_rng']==baseline
                  and ready[arm]['ready_rng']==baseline for arm in ARMS)
    initial_rng_tensors={}
    for arm in ARMS:
        initial_rng_tensors[arm]=torch.load(job/'arms'/arm/'TARGET-INITIAL-RNG.pt',
                                            map_location='cpu',weights_only=True)
        require(set(initial_rng_tensors[arm])=={'CPU','CUDA'},'actual target initial RNG tensors absent')
    rng_tensor_equal=all(all(torch.equal(initial_rng_tensors[arm][key],initial_rng_tensors['T'][key])
                             for key in ('CPU','CUDA')) for arm in ARMS)
    passed &= rng_equal and rng_tensor_equal
    for task in range(10):
        t=by_arm['T'][task]
        base_folder=job/'episodes'/t['key']/'client'
        with np.load(base_folder/'initial_observations.npz',allow_pickle=False) as base:
            reference={name:np.array(base[name],copy=True) for name in CAMERAS}
        base_reset=read(job/'episodes'/t['key']/'RESET.json')['state']
        with np.load(base_folder/'initial_state.npz',allow_pickle=False) as states:
            reference_state={name:np.array(states[name],copy=True) for name in states.files}
        for arm in ARMS[1:]:
            new=by_arm[arm][task]
            t_pair=dict(t['initial_pairing']);new_pair=dict(new['initial_pairing'])
            t_pair.pop('initial_obs_hashes');new_pair.pop('initial_obs_hashes')
            initial_equal=new_pair==t_pair
            with np.load(job/'episodes'/new['key']/'client'/'initial_state.npz',allow_pickle=False) as states:
                physical_arrays_exact=(set(states.files)==set(reference_state)
                    and all(np.array_equal(states[name],reference_state[name])
                            and states[name].shape==reference_state[name].shape
                            and states[name].dtype==reference_state[name].dtype
                            for name in reference_state))
            reset_equal=read(job/'episodes'/new['key']/'RESET.json')['state']==base_reset
            with np.load(job/'episodes'/new['key']/'client'/'initial_observations.npz',allow_pickle=False) as images:
                pixel={}
                for name in CAMERAS:
                    difference=np.abs(images[name].astype(np.int16)-reference[name].astype(np.int16))
                    pixel[name]=dict(exact=bool(np.array_equal(images[name],reference[name])),
                        changed_channels=int(np.count_nonzero(difference)),
                        changed_pixels=int(np.count_nonzero(np.any(difference!=0,axis=2))),
                        max_abs=int(difference.max()),mean_abs=float(difference.mean()),
                        T_hash=hash_array(reference[name]),candidate_hash=hash_array(images[name]))
            row=dict(task=task,init=p['init'],reference='T',candidate=arm,
                initial_sim_controller_python_numpy_rng_equal=initial_equal,
                actual_initial_sim_controller_arrays_exact=physical_arrays_exact,
                full_target_reset_KV_VAE_CPU_CUDA_rng_equal=reset_equal,
                camera_pixels=pixel,physical_RNG_equal=initial_equal and physical_arrays_exact and reset_equal,
                all_exact=initial_equal and physical_arrays_exact and reset_equal
                    and all(v['exact'] for v in pixel.values()))
            comparisons.append(row);passed &= row['physical_RNG_equal']
    certificate=dict(status='PASS_PHYSICAL_RNG_PAIRING_CAMERA_REPORTED' if passed else 'FAIL_PHYSICAL_RNG_PAIRING',
        pairs=30,complete_episodes=40,all_target_initial_rng_equal=rng_equal,
        all_actual_target_CPU_CUDA_rng_tensors_equal=rng_tensor_equal,
        all_task_initial_pairing_equal=all(r['initial_sim_controller_python_numpy_rng_equal'] for r in comparisons),
        all_actual_initial_sim_controller_arrays_exact=all(r['actual_initial_sim_controller_arrays_exact'] for r in comparisons),
        all_reset_states_equal=all(r['full_target_reset_KV_VAE_CPU_CUDA_rng_equal'] for r in comparisons),
        all_initial_camera_pixels_exact=all(all(v['exact'] for v in r['camera_pixels'].values()) for r in comparisons),
        physical_RNG_pairing_passed=passed,
        comparisons=comparisons,no_reference_observation_injected=True,
        original_strict_pixel_qualification_requires_all_initial_camera_pixels_exact=True,
        old_T_not_reused=True,performance_claim=False)
    write(job/'PAIRING-PIXELS.json',certificate)
    if not passed:
        with (job/'PAIRING-FAIL-REPORT.md').open('x') as stream:
            stream.write('# 新40初始物理/RNG配对失败\n\n逐task像素差异与物理/RNG见 PAIRING-PIXELS.json；保留每局原始像素，不重试。\n')
        raise ValueError('fresh four-arm initial sim/controller/target-RNG/reset not paired')
    return certificate


def summarize(task_rows):
    rr=list(task_rows.values());trees=sum(r['trees'] for r in rr)
    return dict(episodes=10,successes=sum(r['success'] for r in rr),
        success_rate=sum(r['success'] for r in rr)/10,
        env_steps_all=distribution([r['env_steps'] for r in rr]),
        env_steps_success=distribution([r['env_steps'] for r in rr if r['success']]),
        horizon_penalty_score=distribution([r['horizon_penalty_score'] for r in rr]),
        target_forwards=sum(r['target_forwards'] for r in rr),policy_chunks=sum(r['infer_chunks'] for r in rr),
        target_forwards_per_chunk=sum(r['target_forwards'] for r in rr)/sum(r['infer_chunks'] for r in rr),
        trees=trees,acceptance_depth_counts={str(i):sum(r['acceptance_depth_counts'][str(i)] for r in rr) for i in range(3)},
        mean_accepted_depth=None if not trees else sum(r['accepted_depth_sum'] for r in rr)/trees,
        episode_wall_diagnostic=distribution([r['episode_wall_diagnostic_sec'] for r in rr]),
        performance_claim=False)


def analyze(job):
    torch.set_num_threads(2)
    job=Path(job).resolve();p=load(job)
    if (job/'STOP_REQUESTED').exists() or (job/'FAILED.json').exists():
        raise RuntimeError('failure/STOP requested before complete analysis')
    collection=read(job/'COLLECTION.json')
    require(collection.get('status')=='PASS_COMPLETE_40_TF_USABLE_DEV_TASK_COLLECTION'
        and collection.get('completed')==[r['key'] for r in p['cases']]
        and collection.get('full_episodes')==40 and collection.get('all_own_processes_cleaned') is True
        and isinstance(collection.get('total_target_forwards'),int)
        and collection['total_target_forwards']<=p['job_target_forward_cap'],
        'all forty full original tasks and complete target accounting required')
    ready={arm:read(job/f'{arm}-READY.json') for arm in ARMS}
    for arm in ARMS:
        require(ready[arm]['target_initial_rng']==ready['T']['target_initial_rng']
            and ready[arm]['ready_rng']==ready[arm]['target_initial_rng']
            and ready[arm]['full_real_CAL_rows']==(0 if arm=='T' else 20237)
            and read(job/'arms'/arm/'OWNED-CLEANUP.json')['all_roots_reaped'] is True,
            'fresh target-only initial RNG or arm cleanup/CAL count differs')
        if arm in ARCH:
            parity=read(job/'arms'/arm/'FULL-REAL-CAL-PARITY.json')
            require(parity.get('status')=='PASS_FULL_REAL_CAL_SAME_DEVICE_ADAPTER_PARITY'
                and parity.get('rows')==20237 and parity.get('complete_CAL') is True
                and parity.get('synthetic_queries')==0 and parity.get('same_device_bitwise_exact') is True
                and parity.get('checkpoint_sha256')==p['selected'][arm]['checkpoint_sha256'],
                'explicit draft lacks complete actual-CAL same-GPU API equivalence')
    rows=[episode(job,r,ready[r['arm']]) for r in p['cases']]
    by_arm={arm:{r['task']:r for r in rows if r['arm']==arm} for arm in ARMS}
    require(all(len(by_arm[a])==10 for a in ARMS),'missing complete fresh arm/task')
    require(sum(r['target_forwards'] for r in rows)==collection['total_target_forwards'],
            'complete RAM model accounting differs from job total')
    certificate=initial_pairing(job,p,by_arm,ready)
    summary={arm:summarize(by_arm[arm]) for arm in ARMS}
    comparisons={f'{candidate}_vs_{reference}':paired(candidate,reference,by_arm,p['init'])
                 for candidate,reference in (('TF_CHILD','T'),('TF_CONT','T'),
                     ('TF_CONT','TF_CHILD'),('TF_CONT','MLP_CHILD'),('MLP_CHILD','T'))}
    ratio=comparisons['TF_CONT_vs_T']['both_success_step_ratio']
    checks=dict(success_at_least_T=summary['TF_CONT']['successes']>=summary['T']['successes'],
        success_at_least_matched_MLP=summary['TF_CONT']['successes']>=summary['MLP_CHILD']['successes'],
        nonempty_T_both_success=ratio['n']>0,
        T_both_success_P50=ratio['p50'] is not None and ratio['p50']<=1.10,
        T_both_success_P90=ratio['p90'] is not None and ratio['p90']<=1.20,
        T_both_success_max=ratio['max'] is not None and ratio['max']<=1.30)
    numeric_screen=all(checks.values())
    strict_pixel_pairing_pass=certificate['all_initial_camera_pixels_exact']
    recommend=numeric_screen and certificate['physical_RNG_pairing_passed'] and strict_pixel_pairing_pass
    result=dict(status='COMPLETE_40_TF_USABLE_DEV_ANALYSIS',full_episodes=40,task_init_pairs=10,
        arms=summary,comparisons=comparisons,episode_rows=rows,pairing_certificate=certificate,
        engineering_selection=dict(checks=checks,numeric_screen_passed=numeric_screen,
            physical_RNG_pairing_passed=certificate['physical_RNG_pairing_passed'],
            strict_pixel_pairing_pass=strict_pixel_pairing_pass,
            strict_engineering_screen_passed=recommend,site_provisional_only=True,
            complete_two_site_20pair_joint_selection=False,formal_speed_admission=False,
            recommend_TF_for_expanded_quality=recommend,
            recommendation='TF_CONT_TO_EXPANDED_QUALITY' if recommend else 'NO_AUTOMATIC_DRAFT_SELECTION',
            frozen_rule=p['selection'],production_preset_changed=False),
        all_complete_horizon800=True,all_original_saved_outputs_audited=True,
        total_target_forwards=collection['total_target_forwards'],performance_claim=False,shared_GPU=True,
        limitations=[f"DEV init{p['init']} is development data; not heldout TEST40–49.",
            'Ten task pairs cannot prove success noninferiority within three percentage points.',
            'Shared-GPU wall times are diagnostic and never formal speed results.',
            'Architecture comparison includes matched seed23 2500-update MLP and TF; TF_CONT has 500 more updates.',
            'Failure horizon score is a fixed 800-step penalty, distinct from recorded environment steps.'])
    lines=['# 新40完整 DEV 结果','',
           '四臂同批各10完整局；逐task物理/控制器/target RNG严格配对，两个相机像素差异逐项披露。仅作工程筛选。','',
           '| 方案 | 成功/10 | 环境步数P50/P90 | target/chunk | 平均接受深度 |',
           '|---|---:|---:|---:|---:|']
    for arm,s in summary.items():
        d=s['env_steps_all'];depth='—' if s['mean_accepted_depth'] is None else f"{s['mean_accepted_depth']:.4f}"
        lines.append(f"| {arm} | {s['successes']}/10 | {d['p50']:.1f}/{d['p90']:.1f} | {s['target_forwards_per_chunk']:.3f} | {depth} |")
    lines+=['',f"TF/T双方成功步数比：{json.dumps(ratio,ensure_ascii=False)}。",'',
            '数值门：'+json.dumps(checks,ensure_ascii=False)+'。',
            f'严格相机像素配对：{strict_pixel_pairing_pass}；物理/target-RNG配对：{certificate["physical_RNG_pairing_passed"]}。',
            '单站10对仅临时工程筛选，必须两站合计20对后统一判断；本站不能独自授予正式计时许可。','',
            '建议：'+('TF_CONT进入扩大质量评测，当前部署preset不变。' if recommend else '未自动选定最终draft；不调阈值或事后更换seed。'),'',
            '逐task失胜、失败800步罚分、接受深度和target计数见RESULT.json；共享GPU墙钟仅诊断。']
    if (job/'STOP_REQUESTED').exists() or (job/'FAILED.json').exists():
        raise RuntimeError('failure/STOP requested before final report')
    with (job/'REPORT.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    analyze(parser.parse_args().job)

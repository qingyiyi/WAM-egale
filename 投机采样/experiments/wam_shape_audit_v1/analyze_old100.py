"""Descriptive CPU analysis; original FAILED remains authoritative."""
import json
import os
from pathlib import Path
import sys
import numpy as np

def read(p):return json.loads(p.read_text())
def main():
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    archive=Path(sys.argv[1]);root=archive/'job';rows=[];raw={}
    plan=read(root/'plan.json')
    for case in plan['assignments']:
        d=root/'episodes'/case['id'];q=read(d/'QUALIFIED.json');m=read(d/'client/repro_metadata.json')
        e=read(d/'client/episodes.jsonl');ready=read(d/'READY.json')
        assert e['status']=='completed' and not e['truncated_after_max_policy_chunks'] and e['libero_horizon']==800
        assert all(q[k]==case[k] for k in ('id','arm','task','init'))
        assert q['success']==e['success'] and q['env_steps']==e['env_steps']
        assert read(d/'CLEANUP.json')['all_roots_reaped'] is True
        with np.load(d/'client/initial_state.npz',allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
        raw[q['task'],q['init'],q['arm']]=(q,m,state)
        row={k:q[k] for k in ('id','arm','task','init','success','env_steps','target_intents','infer_chunks')}
        row.update(penalty800=q['env_steps'] if q['success'] else 800,selector=ready['selector'],mode=ready['mode'])
        for key in ('elapsed_sec','outer_episode_wall_sec','reset_infer_wall_sec','policy_infer_wall_sec','cache_infer_wall_sec'):row[key]=e.get(key)
        rows.append(row)
    pairs=[]
    for task in range(10):
        for init in range(25,30):
            a,am,aa=raw[task,init,'A_OFF'];t,tm,ta=raw[task,init,'T']
            physical_keys=('seed','model_xml_sha256','init_state_sha256','initial_state_hashes','initial_rng')
            equality={k:am.get(k)==tm.get(k) for k in physical_keys}
            arrays=set(aa)==set(ta) and all(aa[k].dtype==ta[k].dtype and aa[k].shape==ta[k].shape and aa[k].tobytes()==ta[k].tobytes() for k in aa)
            pairs.append(dict(task=task,init=init,T_success=t['success'],A_OFF_success=a['success'],T_steps=t['env_steps'],A_OFF_steps=a['env_steps'],physical_metadata_equal=equality,physical_arrays_exact=arrays,camera_hashes_equal=a['initial_obs_hashes']==t['initial_obs_hashes'],different_cameras=[k for k in t['initial_obs_hashes'] if a['initial_obs_hashes'].get(k)!=t['initial_obs_hashes'][k]],both_success_steps_ratio=a['env_steps']/t['env_steps'] if a['success'] and t['success'] else None))
    totals={}
    for arm in ('T','A_OFF'):
        selected=[r for r in rows if r['arm']==arm]
        totals[arm]=dict(episodes=len(selected),successes=sum(r['success'] for r in selected),env_steps=sum(r['env_steps'] for r in selected),penalty800_mean=float(np.mean([r['penalty800'] for r in selected])),failures=[dict(task=r['task'],init=r['init'],steps=r['env_steps']) for r in selected if not r['success']],selector=sorted({r['selector'] for r in selected}),mode=sorted({r['mode'] for r in selected}))
        for key in ('elapsed_sec','outer_episode_wall_sec','reset_infer_wall_sec','policy_infer_wall_sec','cache_infer_wall_sec'):totals[arm][key+'_mean']=float(np.mean([r[key] for r in selected]))
    ratios=[r['both_success_steps_ratio'] for r in pairs if r['both_success_steps_ratio'] is not None]
    summary=dict(status='DESCRIPTIVE_COMPLETE100_ORIGINAL_STRICT_FAILED',original_error=read(sorted(root.glob('RUN-*-RESULT.json'))[-1])['error'],independent_test_complete=False,noninferiority_established=False,formal_performance_permission=False,engineering_quality_screen='NOT_CERTIFIED',target_initial_CPU_CUDA_RNG_recorded=False,camera_arrays_saved=0,camera_pixel_difference_quantifiable=False,physical_arrays_equal_pairs=sum(p['physical_arrays_exact'] for p in pairs),physical_metadata_and_env_rng_equal_pairs=sum(all(p['physical_metadata_equal'].values()) for p in pairs),camera_hash_equal_pairs=sum(p['camera_hashes_equal'] for p in pairs),camera_mismatch_pairs=[{'task':p['task'],'init':p['init'],'cameras':p['different_cameras']} for p in pairs if not p['camera_hashes_equal']],totals=totals,paired_counts=dict(both_success=sum(p['T_success'] and p['A_OFF_success'] for p in pairs),target_win=sum(p['T_success'] and not p['A_OFF_success'] for p in pairs),spec_win=sum(p['A_OFF_success'] and not p['T_success'] for p in pairs),both_fail=sum(not p['T_success'] and not p['A_OFF_success'] for p in pairs)),SR_spec_minus_T=(totals['A_OFF']['successes']-totals['T']['successes'])/50,both_success_steps_ratio=dict(n=len(ratios),p50=float(np.quantile(ratios,.5)),p90=float(np.quantile(ratios,.9)),max=max(ratios)),per_task={str(t):{a:sum(r['success'] for r in rows if r['task']==t and r['arm']==a) for a in ('T','A_OFF')} for t in range(10)},pairs=pairs,episodes=rows,note='No exact observation/RNG pairing certificate; no formal hardware speed comparison; T selector field global; A_OFF uses prefix_then_cost + bonus. Missing camera arrays prevent pixel-magnitude diagnosis. Original FAILED retained.')
    dest=archive.parent/'OLD100-DESCRIPTIVE.json'
    with dest.open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pairs','episodes')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

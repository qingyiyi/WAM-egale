"""Read actual completed DEV evidence and source metadata, never TEST outcomes."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np

OWN=Path(__file__).resolve().parent
ROOT=OWN.parents[2]
RUNTIME=ROOT/'投机采样/results/runtime'
FRESH=RUNTIME/'p0-target-trace-draft-fresh-dev-v1-20261015-r2'
PARENT=RUNTIME/'p0-target-trace-parent-ablation-v1-20261015-r1'
TRAIN=RUNTIME/'p0-target-trace-draft-finetune-v1-20261015-r1'
TAU=.007839680526450805
SIGMA_EDGES=(0.,.01,.03,.1,.3,.6,1.000001)

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def stats(v):
    a=np.asarray(v,dtype=np.float64)
    return dict(n=len(v),mean=float(a.mean()) if len(v) else None,
        p50=float(np.quantile(a,.5)) if len(v) else None,
        p90=float(np.quantile(a,.9)) if len(v) else None,
        max=float(a.max()) if len(v) else None)
def stat(p):
    p=Path(p);s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sigma_bin(s):
    for lo,hi in zip(SIGMA_EDGES,SIGMA_EDGES[1:]):
        if lo<=s<hi:return f'[{lo},{hi})'
    raise ValueError('sigma outside actual schedule')
def tree_stats(trees):
    accepted=[];first=[];depths=Counter();bins=defaultdict(list)
    for t in trees:
        assert t['threshold']==TAU and t['effective_selector']=='prefix_then_cost'
        depth=int(t['actual_depth']);depths[depth]+=1
        f=float(t['first_costs'][t['actual_parent']]);first.append(f/TAU)
        chosen=[]
        if depth>=1:chosen.append(f)
        if depth>=2:chosen.append(float(t['second_costs'][t['actual_parent']][t['actual_child']]))
        assert all(0<=x<=TAU for x in chosen)
        accepted+=chosen;bins[sigma_bin(t['sigma'])].append((depth,f/TAU))
    return dict(trees=len(trees),depth_counts={str(k):depths[k] for k in range(3)},
        mean_depth=sum(k*v for k,v in depths.items())/len(trees) if trees else None,
        selected_first_cost_over_tau=stats(first),accepted_recorded_cost=stats(accepted),
        accepted_edges_within_1pct_tau=sum(x>=.99*TAU for x in accepted),
        accepted_edges_within_5pct_tau=sum(x>=.95*TAU for x in accepted),
        accepted_edges_within_10pct_tau=sum(x>=.9*TAU for x in accepted),
        by_sigma={b:dict(trees=len(v),mean_depth=sum(x[0] for x in v)/len(v),
            chosen_first_cost_over_tau=stats([x[1] for x in v])) for b,v in bins.items()},
        cost_construction_recomputed_from_target_tensor=False)

def episode(job,arm,task):
    folder=job/'episodes'/f'{arm}-t{task}-i24';client=folder/'client'
    rows=[json.loads(l) for l in (client/'episodes.jsonl').read_text().splitlines() if l.strip()]
    assert len(rows)==1
    r=rows[0];assert r['status']=='completed' and r['episode_index']==24 and r['task_index']==task
    counter=read(folder/'COUNTER.json');commands=np.load(client/'commands.npy',allow_pickle=False)
    assert commands.shape==(r['env_steps'],7) and np.isfinite(commands).all()
    grip=commands[:,6];sgn=np.sign(grip);runs=[];start=0
    for i in range(1,len(sgn)+1):
        if i==len(sgn) or sgn[i]!=sgn[start]:
            runs.append(dict(start=start,end_exclusive=i,sign=int(sgn[start])));start=i
    by=defaultdict(list)
    for t in counter['acceptance']:by[int(t['context']['rollout_chunk'])].append(t)
    chunk=[]
    for i in sorted(by):
        x=tree_stats(by[i]);x['rollout_chunk']=i
        # This is a sum of local recorded proxies, not the true trajectory error.
        x['sum_selected_local_proxy']=sum(t['first_costs'][t['actual_parent']]*(t['actual_depth']>=1)
            +(t['second_costs'][t['actual_parent']][t['actual_child']] if t['actual_depth']>=2 else 0.) for t in by[i])
        chunk.append(x)
    return dict(arm=arm,task=task,init=24,success=bool(r['success']),env_steps=r['env_steps'],chunks=r['infer_chunks'],
        target_forwards=counter['target_forwards'],acceptance=tree_stats(counter['acceptance']),chunk_rows=chunk,
        gripper_command_only=dict(positive=int((grip>0).sum()),negative=int((grip<0).sum()),zero=int((grip==0).sum()),
            sign_switches=max(0,len(runs)-1),runs=runs,actual_gripper_position_or_contact_not_recorded=True),
        motion_command_abs_mean=np.abs(commands[:,:6]).mean(0).tolist(),
        available_files=sorted(str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and 'server_output' not in p.parts),
        source=dict(counter=stat(folder/'COUNTER.json'),repro_metadata=stat(client/'repro_metadata.json'),commands=stat(client/'commands.npy')))

def source_exposure():
    """Only plan/source JSON; no episode, RESULT, or proposed TEST tensor reads."""
    bases=[RUNTIME]
    for p in Path('/var/tmp').iterdir():
        if p.is_dir() and p.name.startswith(('cxz-wam-quality','cxz-formal-e2e','cxz-libero-quality')):bases.append(p)
    plan_paths=[];errors=[]
    for base in bases:
        def error(e):errors.append(str(e))
        for directory,dirs,files in os.walk(base,onerror=error):
            dirs[:]=[d for d in dirs if d not in {'server_output','models','.git','__pycache__','traces','trace','observation','old100','archive','client'}]
            for name in ('plan.json','PLAN.json'):
                p=Path(directory)/name
                if name in files and p.stat().st_size<=16*1024**2:plan_paths.append(p)
    init_keys={'init','init_index','init_idx','episode_index','inits','init_indices','episode_indices'}
    def values(obj,path='$'):
        out=[]
        if isinstance(obj,dict):
            for k,v in obj.items():
                q=path+'.'+k
                if k in init_keys:
                    z=[v] if isinstance(v,int) else v if isinstance(v,list) else []
                    z=[x for x in z if isinstance(x,int) and not isinstance(x,bool)]
                    if z:out.append(dict(field=q,inits=sorted(set(z))))
                out+=values(v,q)
        elif isinstance(obj,list):
            for i,v in enumerate(obj):
                if isinstance(v,(dict,list)):out+=values(v,f'{path}[{i}]')
        return out
    scanned=[];high=[];claims=[]
    for p in sorted(set(plan_paths)):
        try:v=read(p)
        except (OSError,ValueError) as e:errors.append(f'{p}: {e}');continue
        fields=values(v);hi=[x for x in fields if any(30<=i<50 for i in x['inits'])]
        if hi:
            # Stat directory names only: a plan is not evidence of an executed policy.
            high.append(dict(plan=str(p),fields=hi,top_level_entries=sorted(x.name for x in p.parent.iterdir()),
                explicit_history_claim=v.get('historical_exposure'),results_opened=False))
        if v.get('historical_exposure'):claims.append(dict(plan=str(p),claim=v['historical_exposure']))
        scanned.append(str(p))
    proof=read(TRAIN/'DATA-SOURCE-PROOF.json')
    return dict(status='METADATA_ONLY_PROPOSED_TEST_BLINDNESS_UNKNOWN',plans_scanned=len(scanned),plans=scanned,
        init30_to49_plan_declarations=high,historical_claims=claims,errors=errors,
        current_training=dict(target_fit_max_init=max(x%100 for x in proof['target_fit_episode_ids']),
            same_query_max_init=max(x%100 for x in proof['same_query_episode_ids']),
            cal_inits=sorted({x%100 for x in proof['cal_episode_ids']}),provenance=proof['same_query_provenance']),
        historical_claim_origin='libero_scale_v1/prepare.py:64 hard-coded conservative label; not execution evidence',
        prepared_benchmark500_has_only_plan_and_preflight=sorted(x.name for x in Path('/var/tmp/cxz-libero-quality-local/benchmark500-job').iterdir())==['plan.json','preflight.json'],
        proposed_TEST_inits=list(range(40,50)),TEST_outcomes_or_tensors_read=False,
        TEST_status='UNKNOWN_NO_INDEPENDENT_BLIND_TEST_CERTIFICATE',
        target_pretraining_exposure_certified=False,remote_complete_exposure_audit=False)

def inputs20():
    folder=FRESH/'episodes/FINETUNE_TF64-t7-i24';counter=read(folder/'COUNTER.json')
    policies=[x for x in counter['requests'] if x['kind']=='policy'];caches=[x for x in counter['requests'] if x['kind']=='cache']
    files={p.name:p for p in (folder/'server_output').rglob('*') if p.is_file()}
    assert len(files)==150 and len(policies)==50
    rows=[]
    for index in range(20):
        policy=policies[index];cache=next(x for x in caches if x['trace_context']['rollout_chunk']==index)
        frame=policy['server_frame_st_id_before'];assert frame==4*index
        names=(f'actions_{frame}.pt',f'latents_{frame}.pt',f'obs_data_{frame}.pt')
        assert set(names[:2])==set(policy['expected_saved_outputs']) and cache['expected_saved_outputs']==[names[2]]
        rows.append(dict(chunk=index,frame_st_id=frame,policy_context=policy['trace_context'],cache_context=cache['trace_context'],
            policy=policy,cache=cache,files={n:stat(files[n]) for n in names}))
    return dict(status='ACTUAL_CURRENT_TF_INPUT_CHAIN_AVAILABLE_REPLAY_NOT_YET_QUALIFIED',source_job=str(FRESH),
        arm='FINETUNE_TF64',task=7,init=24,chunks=rows,full_chunk_count=20,
        initial_observations=stat(folder/'client/initial_observations.npz'),
        actual_initial_target_rng=stat(FRESH/'arms/FINETUNE_TF64/TARGET-INITIAL-RNG.pt'),
        source_READY=stat(FRESH/'FINETUNE_TF64-READY.json'),source_RESET=stat(folder/'RESET.json'),
        source_commands=stat(folder/'client/commands.npy'),source_final_state=stat(folder/'FINAL-SERVER-STATE.json'),
        current_checkpoint=read(TRAIN/'selected.json')['architectures']['transformer64'],
        inference_query_tensor_saved=False,verify_teacher_saved=False,
        query_tensor_reconstruction_requires_actual_GPU_replay=True,
        bytes_for_20_chunk_artifacts=sum(x['bytes'] for r in rows for x in r['files'].values()),
        proposed_query_budget=dict(max_full_packed_calls=32,max_real_node_queries=224,max_counterfactual_target_calls=256,
            original_chunk_policy_calls_counted_separately=True),
        original_policy_cache_entry='VA_Server.infer(request) with original saved observation sequence and current checkpoint',
        comparison_entry='VA_Server._target_action_candidate_directions(same_states, same_timesteps, frame_st_id), B14 versus 7 sequential B2 calls',
        stop_if_original_actions_videos_or_acceptance_path_not_bitwise_reproduced=True,
        no_new_environment_or_robot_success_claim=True,synthetic_data=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise ValueError('explicit CPU-only environment required')
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    own=[]
    for job,arm in ((FRESH,'T'),(PARENT,'PARENT_TF64_SEED23'),(FRESH,'FINETUNE_TF64')):own.append(episode(job,arm,7))
    alltf=[episode(FRESH,'FINETUNE_TF64',t) for t in range(10)]
    import torch
    torch.set_num_threads(2)
    # Real completed DEV initial tensors only; no proposed TEST file is opened.
    images={};rng={}
    for job,arm in ((FRESH,'T'),(PARENT,'PARENT_TF64_SEED23'),(FRESH,'FINETUNE_TF64')):
        with np.load(job/'episodes'/f'{arm}-t7-i24/client/initial_observations.npz',allow_pickle=False) as z:images[arm]={k:z[k] for k in z.files}
        rng[arm]=torch.load(job/'arms'/arm/'TARGET-INITIAL-RNG.pt',map_location='cpu',weights_only=True)
    assert all(all(np.array_equal(images[x][k],images['T'][k]) for k in images['T']) for x in images)
    assert all(torch.equal(rng[x][k],rng['T'][k]) for x in rng for k in ('CPU','CUDA'))
    parity=read(FRESH/'arms/FINETUNE_TF64/FULL-REAL-CAL-PARITY.json')
    episodes=dict(status='COMPLETE_TASK7_RECORD_ONLY_DIAGNOSTIC',task7=own,current_TF_all_tasks=alltf,
        task7_initial_pixels_and_actual_target_rng_exact=True,full_per_step_physical_states_or_contact_available=False,
        TF_real_CAL_parity=parity,within_chunk_error_accumulation_observed=False)
    write(a.out/'EPISODE-AUDIT.json',episodes)
    exposure=source_exposure();write(a.out/'SOURCE-EXPOSURE.json',exposure)
    inputs=inputs20();write(a.out/'Q02-INPUT-MANIFEST.json',inputs)
    risk=[
      dict(id=1,status='STRUCTURAL_DISTRIBUTION_GAP_SUPPORTED_CURRENT_QUERY_SHIFT_UNMEASURED',
        support='FIT target-only + Aoff3906 + historicalTF11091; none generated by current childTF23.',
        counterevidence='Historical target/aux cover all10 tasks and real sigma schedules; not a missing-direction dataset.',
        missing=['currentTF state/previous/target distributions','same-query currentTF residual metrics'],
        stop='Do not trigger D01 solely from the word on-policy; first qualify Q02 query reconstruction and numerical controls.'),
      dict(id=2,status='REAL_SHAPE_NUMERICS_SUPPORTED_CURRENT_QUERY_ACCEPTANCE_FLIPS_UNKNOWN',
        support='Root real long2160 attention/projection replay shows B2/B6/B14 differences; not full-target causal proof.',
        counterevidence='CurrentTF CAL20237 CPU/GPU acceptance flips both directions=0 despite non-bitwise outputs; same-GPU adapter exact.',
        missing=['same-current-query B14 versus B2 CFG directions','route flips and projected successor drift'],
        stop='Any original replay mismatch, KV/RNG mutation or incomplete query identity stops before same-query claims.'),
      dict(id=3,status='SIGMA_AND_LOCAL_COST_OBSERVED_GLOBAL_ACCUMULATED_ERROR_UNKNOWN',
        support='COUNTER has sigma,next_sigma,context,depth and local accepted costs for every recorded tree.',
        counterevidence='All recorded accepted edges respect tau; task7 parent also makes many accepts yet succeeds.',
        missing=['all committed per-step x_t/direction/scheduler states','same-state target reference trajectory'],
        stop='Do not call sums of local proxy costs actual accumulated trajectory error or infer causation from sigma bins.'),
      dict(id=4,status='COMMAND_SEQUENCES_AVAILABLE_CONTACT_HYPOTHESIS_UNRESOLVED',
        support='Commands and raw observed camera windows exist; initial/final simulator/controller states exist.',
        counterevidence='No actual intermediate gripper qpos/contacts/object pose/native success predicates were recorded.',
        missing=['per-step contact and gripper position','object/eef relative pose','native task predicates'],
        stop='No contact-stage or grasp-failure cause from gripper command sign or image guess; need actual telemetry if later collected.'),
      dict(id=5,status='LOCAL_PROXY_NOT_A_TASK_SUCCESS_GUARANTEE_OBSERVED_CONDITION_CAUSE_UNKNOWN',
        support='Task7 child fails796 while T333/parent490 succeed under exact initial pixels and target RNG; rules remain valid.',
        counterevidence='Parent fails different task1; fixed TF architecture alone does not explain failure. Current draft has no prompt/cross-attn by design.',
        missing=['same-query conditional ablation','task-risk labels or certified sensitivity metric'],
        stop='No cross-attn/training/threshold search from this single failure; require new-FIT improvement over same-update old-data continuation.')]
    write(a.out/'RISK-ASSESSMENT.json',dict(risks=risk,D01_decision='HOLD_UNTIL_Q02_NUMERICAL_CONTROLS_AND_CURRENT_QUERY_EVIDENCE',
        D01_preparation_only=True,D01_not_started=True,D02_rule=read(OWN/'PLAN.json')['D02_if_later_admitted']))
    result=dict(status='COMPLETE_CPU_EVIDENCE_Q02_PREPARATION',source_plan=read(OWN/'PLAN.json'),
        artifacts=['EPISODE-AUDIT.json','SOURCE-EXPOSURE.json','Q02-INPUT-MANIFEST.json','RISK-ASSESSMENT.json'],
        CPU_threads=2,CPU_affinity=sorted(os.sched_getaffinity(0)),CUDA_initialized=torch.cuda.is_initialized(),GPU_operations=0,SSH_operations=0,
        proposed_TEST_outcomes_read=False,new_sampling=False,new_training=False,frozen_source_modified=False,
        seconds=time.monotonic()-start,source_sha256=sha(Path(__file__)))
    assert result['CUDA_initialized'] is False
    write(a.out/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_plan'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

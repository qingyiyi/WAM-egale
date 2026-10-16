"""Actual qualified Q02 task7 queries versus exact-chain historical CAL; CPU only."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

OWN=Path(__file__).resolve().parent;EXP=OWN.parent;ROOT=OWN.parents[2]
RT=ROOT/'投机采样/results/runtime'
Q02=RT/'p0-wam-quality-q02-replay-v1-20261016-r2'
TAU=.007839680526450805

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def require(v,s):
    if not v:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    require(os.getenv('CUDA_VISIBLE_DEVICES')=='','CPU-only explicit CUDA mask required')
    os.sched_setaffinity(0,{12,13})
    def stop(*_):raise TimeoutError('finite840s CPU diagnostic')
    signal.signal(signal.SIGALRM,stop);signal.alarm(840)
    import torch
    import numpy as np
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(EXP))
    from target_trace_draft_finetune_v1.adapter import load_checkpoint
    out=args.out.resolve();require(out.is_relative_to(RT) and not out.exists(),'fresh isolated diagnostic output required')
    out.mkdir();started=time.monotonic()
    try:
        result=read(Q02/'RESULT.json');w=read(Q02/'WORKER-RESULT.json');p=read(Q02/'plan.json')
        require(result['status']=='COMPLETE_Q02_REAL_INPUT_NUMERIC_DIAGNOSTIC'
            and result['all_owned_processes_reaped'] and result['final_NVML_own_GPU_processes_empty']
            and read(Q02/'FINAL-OWNED-CLEANUP.json')['all_roots_reaped']
            and result['analysis']==w and w['same_query_qualified']
            and (w['complete_policy_cache_chunks'],w['unique_original_chunks'],w['packed_tree_queries'],w['three_predictor_query_rows'])==(20,10,32,96)
            and w['all_per_query_KV_and_RNG_unchanged']
            and not any((Q02/n).exists() for n in ('FAILED.json','STOP_REQUESTED')),
            'complete qualified actual Q02 and owned cleanup required')
        require(w['route_or_depth_flips']==w['depth_flips']==0,'numeric flips: stop training-trigger analysis')
        cp=p['reference_plan']['selected']['transformer64']
        require(cp['seed']==20260923 and cp['checkpoint_sha256']=='9cb92a4862579edc9cfb99210b0535e620daba56dd5abcc094a7d735dbadab07','fixed currentTF23 required')
        rng=torch.get_rng_state().clone()
        predictor=load_checkpoint(cp['checkpoint'],architecture='transformer64',device='cpu',expected_channels=tuple(range(7)),expected_sha256=cp['checkpoint_sha256'])
        for source in (EXP/'target_trace_draft_finetune_v1/adapter.py',EXP/'target_trace_draft_train_v1/adapter.py',EXP/'draft_architecture_v1/models.py'):
            require(sha(source)==p['source_sha256'][str(source)],'frozen actual predictor source changed')
        ds=p['reference_plan']['real_CAL_data'];path=Path(ds['path']);st=path.stat()
        require((st.st_size,st.st_mtime_ns)==(ds['bytes'],ds['mtime_ns']),'actual CAL column pack changed')
        pack=torch.load(path,map_location='cpu',weights_only=False,mmap=True);c=pack['columns'];cal=pack['views']['cal20']
        require(len(cal)==20237,'all20 CAL view required')
        ix=cal[c['task'][cal]==7]
        require(set(c['init'][ix].tolist())=={18,19},'only historical CAL task7 init18/19')
        def key(values):return tuple(torch.as_tensor(values,dtype=torch.float32).tolist())
        historical_keys=[key(v) for v in c['scalars'][ix]]
        numeric={r['id']:r for r in w['numeric']}
        current=[];files=sorted((Q02/'queries').glob('*.pt'))
        require(len(files)==32 and {f.stem for f in files}==set(numeric),'32 actual query files must match numeric receipt')
        for f in files:
            q=torch.load(f,map_location='cpu',weights_only=True)
            require(q['schema']=='actual-current-TF-DEV-query-numeric-replay/v1'
                and q['checkpoint_sha256']==cp['checkpoint_sha256'] and len(q['teacher_rows'])==3
                and q['data_role']=='DEV_DIAGNOSTIC_NOT_TRAINING' and not q['KV_saved'],'actual teacher source differs')
            for t in q['teacher_rows']:
                require(t['node'] in (0,1,2) and t['data_role']=='DEV_DIAGNOSTIC_NOT_FIT','three real predictor-node rows required')
                scalars=torch.tensor([t['sigma'],t['next_sigma'],t['timestep']/1000.],dtype=torch.float32)
                nr=numeric[f.stem]
                costs=nr['packed_first_costs'] if t['node']==0 else nr['packed_second_costs'][t['node']-1]
                current.append(dict(id=f.stem+'-node'+str(t['node']),chunk=nr['chunk'],tree_index=nr['tree_index'],
                    node=t['node'],role='root' if t['node']==0 else 'counterfactual_first',key=key(scalars),
                    scalars=scalars,state=t['state'].reshape(112),previous=t['previous'].reshape(112),
                    target_packed=t['target_packed'].reshape(112),target_sequential=t['target_sequential'].reshape(112),
                    selected_branch=t['selected_branch'],draft_edge_committed=t['draft_edge_committed'],
                    GPU_actual_packed_costs=costs,source=str(f)))
        require(len(current)==96 and Counter(x['role'] for x in current)=={'root':32,'counterfactual_first':64},'fixed96 rows/roles')
        wanted={x['key'] for x in current}
        hist=[]
        for global_i,k in zip(ix.tolist(),historical_keys):
            if k not in wanted:continue
            hist.append(dict(id='CAL-row'+str(global_i),key=k,scalars=c['scalars'][global_i].clone(),
                state=c['state'][global_i].clone(),previous=c['previous'][global_i].clone(),
                target=c['target'][global_i].clone(),delta=float(c['delta'][global_i]),
                init=int(c['init'][global_i]),chunk=int(c['rollout_chunk'][global_i])))
        pools={k:[i for i,x in enumerate(hist) if x['key']==k] for k in wanted}
        require(all(pools[x['key']] for x in current),'unmatched current query chain: no bin-only fallback')

        @torch.no_grad()
        def evaluate(row,labels):
            state=row['state'].bfloat16();previous=row['previous'].bfloat16()
            require(torch.equal(state.float(),row['state'].float())
                and torch.equal(previous.float(),row['previous'].float()),'actual inputs must retain original BF16 values')
            sigma,next_sigma,normalized_time=row['scalars'].tolist();delta=abs(sigma-next_sigma)
            if 'delta' in row:require(float(torch.tensor(delta,dtype=torch.float32))==row['delta'],'exact chain scheduler delta differs')
            prediction=predictor.predict(state,previous,sigma,next_sigma,normalized_time*1000.).reshape(2,112).float()
            require(bool(torch.isfinite(prediction).all()),'finite actual predictions required')
            errors={name:torch.linalg.vector_norm(prediction-row[name].float()[None],dim=1).double().mul(delta).numpy() for name in labels}
            features=torch.cat((state.float(),previous.float(),row['scalars']))
            z=((features-predictor.feature_mean)/predictor.feature_std).double().numpy()
            return errors,z
        he=[];hz=[]
        for row in hist:
            errors,z=evaluate(row,['target']);he.append(errors['target']);hz.append(z)
        ce={name:[] for name in ('target_packed','target_sequential')};cz=[]
        for row in current:
            errors,z=evaluate(row,list(ce));cz.append(z)
            for name in ce:ce[name].append(errors[name])
        he=np.asarray(he);hz=np.asarray(hz);cz=np.asarray(cz);ce={k:np.asarray(v) for k,v in ce.items()}
        def weighted_stats(v,weights):
            v=np.asarray(v,dtype=float);weights=np.asarray(weights,dtype=float);weights=weights/weights.sum()
            order=np.argsort(v);cum=np.cumsum(weights[order])
            def q(t):return float(v[order[min(np.searchsorted(cum,t,side='left'),len(v)-1)]])
            return dict(mean=float(np.dot(v,weights)),p50=q(.5),p90=q(.9),max=float(v.max()),
                fixed_tau_pass=float(np.dot(v<=TAU,weights)))
        def group(indices):
            counts=Counter(current[i]['key'] for i in indices)
            hs=sorted({j for k in counts for j in pools[k]})
            hw=np.asarray([counts[hist[j]['key']]/(len(indices)*len(pools[hist[j]['key']])) for j in hs])
            baseline=weighted_stats(he[hs].min(1),hw)
            hzmean=np.sum(hz[hs]*hw[:,None],axis=0);cmean=cz[indices].mean(0)
            outside=[]
            for i in indices:
                z=hz[pools[current[i]['key']]]
                outside.append(((cz[i]<z.min(0))|(cz[i]>z.max(0))).mean())
            metrics={name:weighted_stats(v[indices].min(1),np.ones(len(indices))) for name,v in ce.items()}
            return dict(current_rows=len(indices),unique_historical_rows=len(hs),chains=len(counts),
                current_chunk_count=len({current[i]['chunk'] for i in indices}),historical_weighting='each exact chain weighted by current group chain frequency; no duplicated-row independent sample claim',
                current_CPU_BF16_draft_errors=metrics,historical_CPU_BF16_draft_error=baseline,
                tau_pass_difference_pp={k:100*(v['fixed_tau_pass']-baseline['fixed_tau_pass']) for k,v in metrics.items()},
                feature_description=dict(normalizer='unchanged parent FIT174 normalizer',
                    RMS_z_mean_difference=float(np.sqrt(np.mean((cmean-hzmean)**2))),
                    mean_fraction_coordinates_outside_exact_chain_historical_range=float(np.mean(outside)),
                    current_mean_abs_z=float(np.abs(cz[indices]).mean()),historical_weighted_mean_abs_z=float(np.sum(np.abs(hz[hs]).mean(1)*hw))),
                inference='purposive correlated task7 DEV queries; no population confidence interval, OOD certificate or causal policy-drift proof')
        groups={'all96':group(list(range(96))),'root32':group([i for i,x in enumerate(current) if x['node']==0]),
                'counterfactual_first64':group([i for i,x in enumerate(current) if x['node']>0])}
        chain_rows=[]
        for k in sorted(wanted,reverse=True):
            index=[i for i,x in enumerate(current) if x['key']==k]
            chain_rows.append(dict(sigma=k[0],next_sigma=k[1],timestep=k[2]*1000.,current_nodes=len(index),
                historical_CAL_rows=len(pools[k]),root_nodes=sum(current[i]['node']==0 for i in index),comparison=group(index)))
        by_chunk=[dict(chunk=k,comparison=group([i for i,x in enumerate(current) if x['chunk']==k])) for k in range(10)]
        rows=[]
        for i,row in enumerate(current):
            costs=np.asarray(row['GPU_actual_packed_costs'],dtype=float);cpu=ce['target_packed'][i]
            rows.append({k:v for k,v in row.items() if k not in ('state','previous','scalars','target_packed','target_sequential','source')})
            rows[-1].update(CPU_BF16_packed_teacher_errors=cpu.tolist(),CPU_BF16_sequential_teacher_errors=ce['target_sequential'][i].tolist(),
                CPU_best_pass=bool(cpu.min()<=TAU),GPU_recorded_best_pass=bool(costs.min()<=TAU),
                historical_exact_chain_rows=len(pools[row['key']]),source=row['source'])
        source_files=[Path(__file__),Q02/'RESULT.json',Q02/'WORKER-RESULT.json',Q02/'FINAL-OWNED-CLEANUP.json',Q02/'plan.json',
            EXP/'target_trace_draft_finetune_v1/adapter.py',EXP/'target_trace_draft_train_v1/adapter.py',EXP/'draft_architecture_v1/models.py']
        decision=dict(D01='HOLD_FOR_ROOT_REVIEW_OF_FINITE_TASK7_EVIDENCE',
            matched_query_gap_demonstrated=any(groups[k]['tau_pass_difference_pp']['target_packed']<0 for k in ('root32','counterfactual_first64')),
            explanation='Observed matched residual/feature differences justify at most evaluating one bounded balanced data-correction experiment; no automatic start or claim of global on-policy shift.',
            constraints='Selection is task7 first10chunks, not failure-time coverage; historical init18/19 versus current init24 and target-only versus counterfactual state construction remain confounded.')
        receipt=dict(status='COMPLETE_ACTUAL_Q02_MATCHED_TASK7_CPU_DIAGNOSTIC',Q02=str(Q02),checkpoint=cp,
            matching=dict(task=7,current_init=24,historical_CAL_init=[18,19],historical_CAL_task7_rows=len(ix),
                current_rows=96,root_rows=32,counterfactual_first_rows=64,current_rows_exactly_matched=96,
                unique_historical_matched_rows=len(hist),exact_chains=len(wanted),
                key='exact FP32 sigma,next_sigma,timestep/1000 used by original adapter; no tolerance/bin fallback'),
            groups=groups,exact_chain_comparisons=chain_rows,per_chunk=by_chunk,
            recorded_GPU_separate_diagnostic=dict(current_only=True,historical_per_row_GPU_predictions_available=False,
                current_CPU_vs_saved_GPU_best_pass_flips=sum(x['CPU_best_pass']!=x['GPU_recorded_best_pass'] for x in rows),
                source='actual packed_first_costs/packed_second_costs, not a new GPU evaluation',
                limitation='No per-row historical GPU model predictions are reconstructed; current packed and B2 sequential teacher labels remain separately reported.'),
            decision=decision,scope='fixed recorded-observation task7 risk-selected DEV diagnostic, not fullLong10/onpolicy population/closed-loopSR',
            no_new_training=True,no_new_GPU=True,no_AUTODL=True,no_TEST40_49_outcomes_read=True,
            historical_teacher_source=pack.get('target_source'),current_labels=['stored packed B14 CFG target','stored sequential B2 CFG counterfactual target'],
            resources=dict(elapsed_sec=time.monotonic()-started,CPU_threads=2,CPU_affinity=sorted(os.sched_getaffinity(0)),
                CUDA_initialized=torch.cuda.is_initialized(),CPU_rng_unchanged=torch.equal(rng,torch.get_rng_state()),
                dataset_load='mmap CPU, selected CAL task7 rows only; no FIT feature rows used',storage_cap_bytes=128*2**20),
            source_sha256={str(x):sha(x) for x in source_files})
        require(not receipt['resources']['CUDA_initialized'] and receipt['resources']['CPU_rng_unchanged'],'CPU RNG/CUDA boundary changed')
        write(out/'ROWS.json',rows);write(out/'COMPARISON.json',receipt)
        require(sum(x.stat().st_size for x in out.iterdir())<=128*2**20,'diagnostic output cap')
        print(json.dumps(dict(status=receipt['status'],matching=receipt['matching'],groups=groups,D01=decision),ensure_ascii=False))
    except BaseException as exc:
        write(out/'FAILED.json',dict(status='STOP_ACTUAL_Q02_MATCHED_DIAGNOSTIC',error=repr(exc),D01_admitted=False));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()

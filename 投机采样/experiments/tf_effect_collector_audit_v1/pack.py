"""Finite CPU audit/pack of the one registered current-TF FIT20/CAL10 pool.

Only collection=True cases are admitted. No DEV/TEST episode, model forward,
teacher, optimizer, simulator, or CUDA call is executed. Source shards and
counterfactual rows remain in their original archive and are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from tf_effect_collector_audit_v1 import audit as support

ROOT, RT, COMMIT, PERMISSIONS = support.ROOT, support.RT, support.COMMIT, support.PERMISSIONS
read, write, sha, stat, require = support.read, support.write, support.sha, support.stat, support.require
SCHEMA = 'tf-effect-full-collection-pack-job/v1'
GENERATOR_SHA = '3657388fdce0763093d1b704c88a03b6836954160d43dddb9f75337a054d64ac'
CHILD2500_SHA = '9cb92a4862579edc9cfb99210b0535e620daba56dd5abcc094a7d735dbadab07'
TAU = 0.007839680526450805
FIT_IDS = [t*100+i for t in range(10) for i in (30,31)]
CAL_IDS = [t*100+32 for t in range(10)]
KEYS = ('state','previous','target','scalars','delta','task','episode','node')
TOP_RECEIPTS = ('plan.json','PREREGISTRATION.json','COLLECTION.json','SUPERVISOR-RESULT.json',
    'CLOCK-RESULT.json','CLOCK-GUARD.json','FINAL-OWNED-CLEANUP.json','RUN-STARTED.json','TF_CONT-READY.json')
EPISODE_RECEIPTS = ('COUNTER.json','TEACHER-INDEX.json','RESET.json','FINISH_SERVER.json',
    'FINISH_READY.json','FINAL-SERVER-STATE.json','client/episodes.jsonl','client/COMMAND-MAPPING.json',
    'client/repro_metadata.json')


def digest_json(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def cases(plan):
    require(plan['schema']=='tf-effect-full-campaign/v1' and plan['collect'] is True
        and plan['commit']==COMMIT and plan['permissions']==PERMISSIONS
        and plan['performance_claim'] is False and plan['arms']==['TF_CONT'],
        'only the fixed collect=true TF_CONT campaign is admitted')
    rows=plan['cases']
    expected={(t,i,'FIT' if i in (30,31) else 'CAL') for t in range(10) for i in (30,31,32)}
    require(len(rows)==30 and len({r['key'] for r in rows})==30
        and {(r['task'],r['init'],r['role']) for r in rows}==expected
        and all(r['arm']=='TF_CONT' for r in rows), 'exact FIT20/CAL10 cases required; no DEV/TEST or replacement')
    require(plan['mode']=='verifier_k2_b2_bonus' and plan['selector']=='prefix_then_cost'
        and plan['threshold']==TAU and plan['horizon']==800,
        'unchanged production mode/selector/threshold/horizon required')
    generator=plan['selected']['TF_CONT']
    require(generator['checkpoint_sha256']==GENERATOR_SHA
        and generator['architecture']=='transformer64', 'fixed TF_CONT3000 generator identity differs')
    return rows


def prepare(args):
    support.runtime()
    job=args.job.resolve(); source=args.collection.resolve(strict=True)
    require(job.parent==RT and not job.exists(), 'new isolated CPU pack namespace required')
    require(300<=args.wall_cap_seconds<=1800 and 512<=args.storage_cap_mib<=1024,
        'CPU pack cap must be 300..1800 seconds and 512..1024 MiB')
    job.mkdir(); start=time.time()
    reg=dict(schema=SCHEMA,source_commit=COMMIT,collection=str(source),goal='audit complete30 and pack reached FIT20/CAL10 rows',
        birth_epoch=start,deadline_epoch=start+args.wall_cap_seconds,wall_cap_seconds=args.wall_cap_seconds,
        storage_cap_bytes=args.storage_cap_mib*2**20,CPU_threads=2,GPU=False,norm_atol=1e-6,
        cpu_affinity=sorted(os.sched_getaffinity(0)),permissions=PERMISSIONS,
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(support.__file__).resolve()):sha(support.__file__)},
        cleanup_plan='single CPU process, no child/model/CUDA; alarm plus caller external timeout',
        evidence_path=str(job),terminal_status='REGISTERED_NOT_PREPARED',
        no_training=True,no_new_teacher=True,no_DEV_TEST_episode_reads=True)
    write(job/'PREREGISTRATION.json',reg)
    try:
        support.alarm(reg)
        require(all((source/n).is_file() for n in TOP_RECEIPTS), 'collection is not fully terminal; no PARTIAL pack admission')
        plan=read(source/'plan.json'); rows=cases(plan)
        evidence=[source/n for n in TOP_RECEIPTS]
        artifacts=[source/'arms/TF_CONT/TARGET-INITIAL-RNG.pt']; shard_paths=[]
        for row in rows:
            out=source/'episodes'/row['key']
            require(all((out/n).is_file() for n in EPISODE_RECEIPTS), 'missing complete episode receipt: '+row['key'])
            evidence += [out/n for n in EPISODE_RECEIPTS]
            for name in ('initial_observations.npz','initial_state.npz','final_state.npz','commands.npy'):
                require((out/'client'/name).is_file(), 'actual initial/final physical/camera/command capture missing')
                artifacts.append(out/'client'/name)
            index=read(out/'TEACHER-INDEX.json')
            for item in index['shards']:
                shard=Path(item['path']).resolve(strict=True)
                proof=Path(item['counter_proof']).resolve(strict=True)
                require(shard.is_relative_to(source/'teacher') and proof==shard.with_suffix('.COUNTER.json'),
                    'teacher/proof escaped registered source collection')
                shard_paths.append(shard); artifacts.append(shard); evidence.append(proof)
            counter=read(out/'COUNTER.json')
            for item in counter['save_async']['kept_artifacts']:
                path=Path(item['path']).resolve(strict=True)
                require(path.is_relative_to(out/'server_output'), 'saved action/latent escaped source case')
                artifacts.append(path)
        require(len(shard_paths)==len(set(shard_paths)), 'teacher shard referenced by multiple episodes')
        actual=set((source/'teacher').rglob('chunk-*.pt'))
        require(actual==set(shard_paths), 'unreferenced, missing, or interrupted teacher shard prevents complete30 admission')
        require(not list((source/'teacher').rglob('*FAILED*')) and not list((source/'teacher').rglob('*.tmp')),
            'failed/incomplete teacher archive cannot be admitted')
        evidence += [Path(path).resolve(strict=True) for path in plan['evidence_sha256']]
        for path,expected in plan['evidence_sha256'].items():
            require(sha(path)==expected, 'registered qualification evidence changed')
        require(sha(plan['selected']['TF_CONT']['checkpoint'])==GENERATOR_SHA, 'small generator checkpoint bytes differ')
        evidence=list(dict.fromkeys(evidence)); artifacts=list(dict.fromkeys(artifacts))
        prepared=dict(**reg,case_manifest=rows,evidence_stat={str(p):stat(p) for p in evidence+artifacts},
            receipt_sha256={str(p):sha(p) for p in evidence},teacher_shards=[str(p) for p in shard_paths],
            preregistration_sha256=sha(job/'PREREGISTRATION.json'))
        write(job/'PLAN.json',prepared)
        write(job/'PREPARED.json',dict(status='PREPARED_FULL_30_TERMINAL_COLLECTION_CPU_ONLY',
            episodes=30,teacher_shards=len(shard_paths),model_calls=0,elapsed_sec=time.time()-start))
        support.storage(job,reg['storage_cap_bytes'])
        return prepared
    except BaseException as exc:
        write(job/'FAILED.json',dict(status='FAILED_CPU_COLLECTION_PACK_PREPARE',error=repr(exc),
            traceback=traceback.format_exc(),elapsed_sec=time.time()-start,permissions=PERMISSIONS))
        raise
    finally:
        signal.alarm(0)


def qualifications(plan):
    certificates=[]
    for path,expected in plan['evidence_sha256'].items():
        require(sha(path)==expected,'qualification evidence changed')
        certificates.append((Path(path),read(path)))
    gpu_statuses={'PASS_FULL_RECORDED_EPISODE_COLLECTOR_EQUIVALENCE',
        'PASS_FULL_RECORDED_EPISODE_COLLECTOR_EQUIVALENCE_BASELINE_REUSED'}
    cpu_statuses={'PASS_FULL_50_CHUNK_QUALIFICATION_AUDIT',
        'PASS_FULL_50_CHUNK_QUALIFICATION_AUDIT_BASELINE_REUSED'}
    gpu=[(p,x) for p,x in certificates if x.get('status') in gpu_statuses]
    cpu=[(p,x) for p,x in certificates if x.get('status') in cpu_statuses]
    require(len(gpu)==len(cpu)==1,'one actual GPU equivalence and one full50 CPU numeric qualification required')
    gp,g=gpu[0]; cp,c=cpu[0]
    require(c['collector_equivalence_passed'] is True and c['source_commit']==COMMIT
        and c['permissions']==PERMISSIONS and c['complete_chunks']==50
        and c['actual_teacher_trees']==904 and c['actual_teacher_rows']==2712
        and c['all_state_previous_target_proposal_bytes_exact'] is True
        and c['all_sigma_time_step_route_cost_flags_exact'] is True
        and c['counter_coverage_checked'] is True and c['new_teacher_forwards']==0
        and c['teacher_cpu_norm_max_abs']<=1e-6 and c['cpu_norm_uses_separate_FP32_operands_before_subtraction'] is True,
        'CPU complete numeric qualification gates failed')
    require(g['permissions']==PERMISSIONS and g['extra_teacher_forwards']==0
        and g['actual_o3_preserved'] is True and Path(c['qualification']).resolve()==gp.parent,
        'GPU/CPU qualifications do not bind the same real experiment')
    qp=read(gp.parent/'plan.json')
    if g['status']=='PASS_FULL_RECORDED_EPISODE_COLLECTOR_EQUIVALENCE_BASELINE_REUSED':
        require(c['status']=='PASS_FULL_50_CHUNK_QUALIFICATION_AUDIT_BASELINE_REUSED'
            and c['baseline_reused'] is True and c['new_ON_passes']==1 and c['new_OFF_passes']==0
            and c['all_intermediate_request_KV_equivalence_established'] is False
            and c['old_two_new_pass_allrequestKV_contract_passed'] is False
            and c['all_original_teacher_proposal_routes_exact'] is True
            and c['reset_and_final_full_valid_KV_RNG_exact'] is True and c['per_chunk_RNG_exact'] is True
            and g['baseline_reused'] is True and g['per_request_all_KV_comparison'] is False
            and g['final_all_valid_KV_RNG_exact'] is True and g['per_policy_original_RNG_routes_counts_exact'] is True
            and qp['OFF_baseline_reused'] is True and qp['per_request_KV_comparison'] is False
            and Path(qp['source_episode']).resolve()==RT/'p0-task4-direction-diagnostic-completion-v1-20261016-r2/episodes/TF_CONT-task4-init26-repeat0',
            'new baseline-reuse qualification scope/binding differs; old PARTIAL cannot become PASS')
        contract='baseline_reused_original_complete_M11_OFF_plus_new_full50_ON'
        all_intermediate_KV=False
    else:
        require(c['status']=='PASS_FULL_50_CHUNK_QUALIFICATION_AUDIT'
            and g['all_valid_KV_RNG_routes_counts_exact'] is True,
            'old two-new-pass contract requires its own actual CPU/GPU pass')
        contract='two_new_complete_OFF_ON_passes_with_each_request_valid_KV'
        all_intermediate_KV=True
    collector=str(EXP/'tf_effect_collector_v1/collector.py')
    require(qp['source_sha256'][collector]==plan['source_sha256'][collector]
        and sha(collector)==qp['source_sha256'][collector], 'production collector differs from qualified observer')
    require(any(Path(e['path']).resolve()==gp and e['sha256']==sha(gp) for e in c['evidence']),
        'CPU audit does not bind actual GPU worker certificate')
    return dict(GPU_qualification=str(gp),CPU_qualification=str(cp),same_observer_source=True,
        qualification_contract=contract,all_intermediate_request_KV_equivalence_established=all_intermediate_KV)


def terminal(source,plan,rows):
    collection=read(source/'COLLECTION.json'); supervisor=read(source/'SUPERVISOR-RESULT.json')
    cleanup=read(source/'FINAL-OWNED-CLEANUP.json'); clock=read(source/'CLOCK-RESULT.json')
    guard=read(source/'CLOCK-GUARD.json'); keys=[r['key'] for r in rows]
    require(collection['status']=='COMPLETE_REGISTERED_FULL_EPISODES' and collection['collect'] is True
        and collection['full_episodes']==30 and collection['completed']==keys
        and collection['permissions']==PERMISSIONS and collection['performance_claim'] is False,
        'actual complete30 COLLECTION terminal contract required')
    require(supervisor['status']=='COMPLETE' and supervisor['failure'] is None
        and supervisor['completed']==keys and supervisor['full_episodes']==30
        and supervisor['permissions']==PERMISSIONS and supervisor['cleanup']==cleanup
        and cleanup['all_roots_reaped'] is True, 'supervisor incomplete or owned cleanup missing')
    require(clock['status']=='CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS'
        and clock['all_observed_owned_live_processes_exited'] is True
        and clock['error'] is None and clock['external_stop_reason'] is None
        and clock['no_foreign_signals'] is True, 'independent collection clock not cleanly complete')
    require(guard['clock']['plan_sha256']==sha(source/'plan.json') and guard['no_budget_reset'] is True,
        'independent clock is not bound to frozen plan/budget')
    require(supervisor['elapsed_sec']<=plan['max_run_wall_sec']
        and clock['clock']['process_birth_monotonic']+clock['elapsed_quality_process_seconds']
            <=clock['clock']['hard_stop_monotonic']+1,'collection wall or absolute deadline exceeded')
    for name in ('FAILED.json','CLOCK-FAILED.json','CLOCK-STOP.json','TF_CONT-MONITOR-FAILED.json'):
        require(not (source/name).exists(),'retained failed/partial collection is not complete30: '+name)
    pins=list(supervisor['observed_pins'])+[x['pin'] for x in clock['process_observations']]
    pins += [guard['supervisor_pin'],guard['guard_pin']]
    unique={(p['pid'],p['start_ticks'],p['uid']):p for p in pins}
    observations=[support.live_pin(p) for p in unique.values()]
    require(observations and not any(x['live'] for x in observations),'a pinned collection process is still live')
    ready=read(source/'TF_CONT-READY.json')
    require(ready['status']=='READY' and ready['collect'] is True and ready['collector_installed'] is True
        and ready['setup_RNG_restored'] is True and ready['target_RNG_captured_before_draft_and_CAL'] is True
        and ready['predictor_checkpoint']==plan['selected']['TF_CONT']['checkpoint']
        and ready['O1_O2_O3'] is True and ready['selector']==plan['selector']
        and ready['synthetic_queries']==0, 'registered unchanged TF/O3 server not ready')
    require(collection['total_target_forwards']==supervisor['total_target_forwards'], 'top-level forward counts disagree')
    return collection,dict(status='PASS_COLLECTION_TERMINAL_AND_OWNED_CLEANUP',
        observed_owned_processes=observations,clock_is_not_quality_certificate=True)


def route(first,second,threshold):
    pairs=[]
    for p in range(2):
        for c in range(2):
            a,b=float(first[p]),float(second[p][c]);d=0 if a>threshold else 1 if b>threshold else 2
            pairs.append((p,c,d,a+b,a+b if d==2 else a if d==1 else 0.))
    if max(x[2] for x in pairs)==0:
        return min(pairs,key=lambda x:(-x[2],x[3],x[0],x[1]))[:3]
    return min(pairs,key=lambda x:(-x[2],x[4],x[0],x[1]))[:3]


def commands(torch,mapping,record,requests):
    require(mapping['schema']=='physical-action-to-environment-command/v1' and mapping['complete_episode'] is True
        and mapping['success']==record['success'] and mapping['env_steps']==record['env_steps']
        and mapping['termination_reason']==('success' if record['success'] else 'full_horizon'),
        'full physical command terminal mapping required')
    submitted,environment=mapping['submitted'],mapping['environment']
    require(len(submitted)==len(environment)==record['env_steps'],'all actually executed commands required')
    chunks=mapping['chunks'];n=record['infer_chunks']
    require(len(chunks)==n and [x['chunk'] for x in chunks]==list(range(n)),'physical chunks are not complete/in order')
    predictions=[torch.tensor(x['received_action'],dtype=torch.float32) for x in chunks]
    require(all(x.shape==(7,4,4) and bool(torch.isfinite(x).all()) for x in predictions),'invalid received physical actions')
    by_chunk=Counter();before=0
    for j,(a,b) in enumerate(zip(submitted,environment)):
        require(a['env_index']==b['env_index']==j and a['chunk']==b['chunk']
            and a['command']==b['command'],'submitted command differs at actual environment boundary')
        chunk=a['chunk'];require(0<=chunk<n,'command references unknown policy chunk')
        k=by_chunk[chunk];require(a['command_offset']==k,'per-chunk actual command offset differs')
        frame=(1 if chunk==0 else 0)+k//4; sub=k%4
        require(a['frame']==frame and a['subaction']==sub and frame<4,'physical action frame/subaction mismatch')
        pred=predictions[chunk]
        actual=torch.tensor(a['command'],dtype=torch.float32)
        support.same_bytes(torch,actual,pred[:,frame,sub],'decoded/submitted/environment command')
        require(bool(torch.isfinite(actual).all()),'nonfinite environment command')
        by_chunk[chunk]+=1
    policies=[r for r in requests if r['kind']=='policy']
    for chunk,(item,req) in enumerate(zip(chunks,policies)):
        require(item['context']==req['trace_context'] and item['env_steps_before']==before
            and item['context']['env_steps_before']==before,'policy command/request chronology mismatch')
        require(item['shape']==[7,4,4] and item['dtype']=='float32','physical production action contract changed')
        before+=by_chunk[chunk]
    require(before==record['env_steps'],'command length conservation failed')
    for req in requests:
        if req['kind']=='cache':
            k=req['trace_context']['rollout_chunk']
            require(req['trace_context']['env_steps_after']==sum(by_chunk[j] for j in range(k+1)),
                'cache does not bind actual executed prefix length')


def physical_artifacts(torch,out,record,mapping,ready,reset):
    import hashlib
    import numpy as np
    meta=read(out/'client/repro_metadata.json')
    require(meta['command_count']==record['env_steps'] and meta['no_policy_chunk_truncation'] is True
        and meta['actual_EGL_initialization_attempts']==1 and meta['initial_rng'] and meta['final_rng'],
        'complete physical/RNG reproduction metadata required')
    for filename,field in (('initial_state.npz','initial_state_hashes'),('final_state.npz','final_state_hashes'),
                           ('initial_observations.npz','initial_obs_hashes')):
        with np.load(out/'client'/filename,allow_pickle=False) as arrays:
            require(set(arrays.files)==set(meta[field]),'physical/camera capture field inventory differs')
            for key in arrays.files:
                x=np.ascontiguousarray(arrays[key])
                require(hashlib.sha256(x.tobytes()).hexdigest()==meta[field][key], 'actual physical/camera bytes differ from capture manifest')
                if filename=='initial_observations.npz':
                    require(x.dtype==np.uint8 and x.shape==(128,128,3),'actual initial camera geometry differs')
    actual=np.load(out/'client/commands.npy',allow_pickle=False)
    require(actual.dtype==np.float32 and actual.shape==(record['env_steps'],7),'actual command archive geometry differs')
    expected=torch.tensor([x['command'] for x in mapping['environment']],dtype=torch.float32)
    support.same_bytes(torch,torch.from_numpy(actual),expected,'lean/client/environment command archives')
    require(reset['state']['cpu_rng']==ready['target_initial_rng']['cpu']
        and reset['state']['cuda_rng']==ready['target_initial_rng']['cuda'], 'per-episode target RNG differs from initialization')
    state=read(out/'FINAL-SERVER-STATE.json')
    require(len(state['KV'])==30 and state['cpu_rng'] and state['cuda_rng'],'all30 valid KV/RNG state fingerprints required')
    for cache in state['KV']:
        require(cache['k']['shape']==cache['v']['shape'] and cache['k']['dtype']==cache['v']['dtype']=='torch.bfloat16',
            'final full valid KV fingerprint dtype/geometry differs')


def run(job):
    support.runtime();job=job.resolve(strict=True);p=read(job/'PLAN.json');reg=read(job/'PREREGISTRATION.json')
    require(p['schema']==SCHEMA and p['source_commit']==COMMIT and p['permissions']==PERMISSIONS,
        'CPU pack identity/permissions changed')
    require(not (job/'STARTED.json').exists() and not (job/'FAILED.json').exists(),'one CPU attempt; preserve failures')
    require(sha(job/'PREREGISTRATION.json')==p['preregistration_sha256'],'registration changed')
    for f,h in p['source_sha256'].items():require(sha(f)==h,'auditor source changed')
    start=time.time();write(job/'STARTED.json',dict(status='RUNNING_CPU_ONLY',pid=os.getpid(),permissions=PERMISSIONS))
    try:
        support.alarm(reg)
        for f,s in p['evidence_stat'].items():require(stat(f)==s,'source artifact changed: '+f)
        for f,h in p['receipt_sha256'].items():require(sha(f)==h,'source receipt changed: '+f)
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized(),'CPU pack acquired CUDA context')
        source=Path(p['collection']);plan=read(source/'plan.json');case_rows=cases(plan)
        qual=qualifications(plan);collection,cleanup=terminal(source,plan,case_rows)
        ready=read(source/'TF_CONT-READY.json')
        splits={name:{key:[] for key in (*KEYS,'state_reached')} for name in ('FIT','CAL')}
        episode_rows=[];raw_index=[];all_shards=set();max_error=0.;edge_flips=depth_flips=0;total_forwards=0
        for case in case_rows:
            key=case['key'];out=source/'episodes'/key;counter=read(out/'COUNTER.json')
            records=[json.loads(x) for x in (out/'client/episodes.jsonl').read_text().splitlines() if x.strip()]
            require(len(records)==1,'exactly one full terminal client record required')
            record=records[0];success=record['success'];env_steps=record['env_steps'];nchunks=record['infer_chunks']
            require(record['status']=='completed' and isinstance(success,bool) and isinstance(env_steps,int) and env_steps>0
                and isinstance(nchunks,int) and 1<=nchunks<=50
                and record['task_index']==case['task'] and record['episode_index']==case['init']
                and record['libero_horizon']==800 and record['max_policy_chunks'] is None
                and record['truncated_after_max_policy_chunks'] is False
                and (success or record['horizon_reached'] is True), 'partial/truncated/infrastructure interruption is not a complete episode')
            require(record['simulator_timestep_final']==record['simulator_timestep_start']+env_steps
                and (success or record['simulator_timestep_final']>=800), 'actual full horizon/step semantics differ')
            finish,finish_server=read(out/'FINISH_READY.json'),read(out/'FINISH_SERVER.json')
            require(finish==finish_server and finish['status']=='PASS' and finish['full_episode_terminal'] is True
                and finish['teacher_complete'] is True and finish['success']==success and finish['env_steps']==env_steps
                and finish['extra_teacher_forwards']==0 and finish['async_drained'] is True,
                'complete terminal server/teacher/async proof required')
            reset=read(out/'RESET.json')
            require(reset['status']=='PASS_FULL_RESET_BANK_AND_EMPTY_STATE' and reset['row']==case
                and reset['RNG_unchanged'] is True,'original reset/RNG proof required')
            require(counter['row']==case and counter['arm']=='TF_CONT' and counter['key']==key
                and counter['extra_teacher_forwards']==0
                and counter['target_forwards']==counter['returned_forwards'], 'episode target accounting/identity mismatch')
            reqs=counter['requests'];policies=[x for x in reqs if x['kind']=='policy'];caches=[x for x in reqs if x['kind']=='cache']
            require(reqs[0]['kind']=='reset' and sum(x['kind']=='reset' for x in reqs)==1
                and len(policies)==nchunks and len(caches)==nchunks-int(success)
                and len(reqs)==1+len(policies)+len(caches) and all(x['ok'] for x in reqs)
                and record['policy_infer_calls']==nchunks and record['cache_infer_calls']==len(caches)
                and record['reset_infer_calls']==1,'complete original reset/policy/cache request coverage required')
            require(sum(r['target_forwards'] for r in reqs)==counter['target_forwards'],'episode forward conservation failed')
            total_forwards+=counter['target_forwards']
            base=record['trace_context']
            require(base['campaign_id']==source.name and base['parallel_worker_id']==key
                and base['task_index']==case['task'] and base['episode_index']==case['init'], 'client actual identity mismatch')
            for req in reqs:
                require(all(req['trace_context'][k]==base[k] for k in base),'episode context changed between requests')
            require([r['trace_context']['rollout_chunk'] for r in policies]==list(range(nchunks)), 'policy chunk context is not unique/sequential')
            mapping=read(out/'client/COMMAND-MAPPING.json')
            commands(torch,mapping,record,reqs)
            physical_artifacts(torch,out,record,mapping,ready,reset)
            events=counter['O3_events']
            require(len(events)==nchunks and all(e['enabled'] and e['ok'] and e['closed'] for e in events),'O3 not preserved on every chunk')
            saved=counter['save_async'];kept=saved['kept_artifacts']
            names=[n for r in reqs for n in r['expected_saved_outputs']]
            require(saved['original_executor_drained'] is True and saved['obs_suppressed']==len(caches)
                and saved['kept']==len(kept)==2*nchunks
                and sorted(x['name'] for x in kept)==sorted(names),'original action/latent saved coverage incomplete')
            for artifact in kept:
                path=Path(artifact['path']);require(path.stat().st_size==artifact['bytes'] and artifact['suppressed'] is False,'saved artifact stat differs')
                value=torch.load(path,map_location='cpu',weights_only=True)
                require(isinstance(value,torch.Tensor) and bool(torch.isfinite(value).all()),'nonfinite/missing saved original action or video latent')
            require(not list((out/'server_output').rglob('obs_data_*.pt')),'unexpected copied full visual observations')
            index=read(out/'TEACHER-INDEX.json')
            require(index==counter['teacher'] and index['enabled'] is True and index['extra_teacher_forwards']==0
                and len(index['shards'])==nchunks and len(counter['chunks'])==nchunks,'complete teacher index/counter chunk coverage required')
            ep_trees=ep_rows=ep_reached=0;hist=Counter();phase_sum=Counter()
            for chunk,(ref,policy,chc) in enumerate(zip(index['shards'],policies,counter['chunks'])):
                path=Path(ref['path']).resolve();require(path not in all_shards,'duplicate global shard/context mapping')
                all_shards.add(path);pack=torch.load(path,map_location='cpu',weights_only=True)
                cols,tr=pack['columns'],pack['trees'];context=pack['context'];n=pack['tree_count'];nr=3*n
                require(pack['schema']=='tf-effect-o3-same-query-columns/v1'
                    and pack['status']=='COMPLETE_CHUNK_PENDING_EXTERNAL_COUNTER_CHECK'
                    and context==policy['trace_context']==chc['context'] and context['rollout_chunk']==chunk,
                    'teacher chunk is not its actual policy query/context')
                require(context['parallel_worker_id']==key and context['episode_attempt_id']==base['episode_attempt_id'], 'teacher/context episode mismatch')
                meta=pack['metadata']
                require(meta['source_commit']==COMMIT and meta['generator_checkpoint_sha256']==GENERATOR_SHA
                    and meta['generator_checkpoint']==plan['selected']['TF_CONT']['checkpoint']
                    and meta['generator_condition']=='TF_CONT3000' and meta['generator_total_updates']==3000
                    and meta['data_role_by_key'][key]==case['role'] and meta['target_checkpoint_id']==plan['reference_plan']['target_checkpoint_id'],
                    'teacher generator/role/target provenance differs')
                require(set(pack['source_dtypes'].values())=={'torch.bfloat16'},'qualified BF16 source direction/state dtype changed')
                require(pack['query_and_teacher_bytes_exact'] is True and pack['original_O3_active_at_every_capture'] is True
                    and pack['original_results_returned_unchanged'] is True and pack['extra_teacher_forwards']==0
                    and pack['row_count']==nr and pack['candidate_calls']==pack['execute_calls']==n
                    and ref['trees']==chc['tree_count']==n and ref['rows']==nr and chc['teacher_shard']=={k:ref[k] for k in chc['teacher_shard']},
                    'teacher actual-call/tree coverage mismatch')
                proof=read(ref['counter_proof'])
                require(proof['status']=='PASS_SHARD_EXTERNAL_TREE_COUNTER_COVERAGE' and proof['context']==context
                    and proof['actual_phase_tree_count']==proof['acceptance_tree_count']==n
                    and proof['source_trace_and_teacher_disabled'] is True and proof['extra_teacher_forwards']==0
                    and proof['key']==key and proof['arm']=='TF_CONT' and proof['role']==case['role']
                    and proof['shard']==chc['teacher_shard'] and proof['actual_O3_events']==chc['O3_events']
                    and len(chc['O3_events'])==1 and chc['O3_events'][0]==events[chunk],
                    'external actual counter/disabled-trace proof missing')
                require(chc['phase_counters']['tree_count']==n and chc['phase_counters']['committed_scheduler_update_count']==50
                    and chc['target_forwards']==policy['target_forwards'], 'full policy phase/target conservation failed')
                phase_sum.update(chc['phase_counters'])
                accepted=[r for r in counter['acceptance'] if r['context']==context]
                require(len(accepted)==n and all(len(v)==n for v in tr.values()),'per-tree acceptance geometry differs')
                for name,value in cols.items():
                    require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and len(value)==nr
                        and bool(torch.isfinite(value).all()),'invalid compact real-query column: '+name)
                require(cols['state'].shape==cols['previous'].shape==cols['target'].shape==(nr,112)
                    and cols['proposals'].shape==(nr,2,112) and cols['scalars'].shape==(nr,3),'teacher compact geometry differs')
                for name in ('state','previous','target','proposals','scalars','delta'):
                    require(cols[name].dtype==torch.float32,'teacher float column must remain CPU FP32')
                for name in ('tree_index','tree_step_index','step_index','node','task','init','episode','accepted_depth'):
                    require(cols[name].dtype==torch.int64,'teacher identity column must remain int64')
                for name in ('selected_branch','state_reached','draft_edge_committed'):
                    require(cols[name].dtype==torch.bool,'actual reachability/commit flags must remain bool')
                require(tr['sigma_path'].dtype==tr['target_timesteps'].dtype==tr['threshold'].dtype==torch.float64,
                    'original scalar triplets must retain exact float64 representation')
                next_tree_step=1
                for j,a in enumerate(accepted):
                    first,second=tr['first_costs'][j],tr['second_costs'][j]
                    support.same_bytes(torch,first,torch.tensor(a['first_costs'],dtype=torch.float32),'original six first scores')
                    support.same_bytes(torch,second,torch.tensor(a['second_costs'],dtype=torch.float32),'original six second scores')
                    require(float(tr['threshold'][j])==a['threshold']==TAU,'fixed threshold changed')
                    parent,child,depth=route(first.tolist(),second.tolist(),TAU)
                    require((int(tr['selected_parent'][j]),int(tr['selected_child'][j]),int(tr['accepted_depth'][j]))
                        ==(a['actual_parent'],a['actual_child'],a['actual_depth'])==(parent,child,depth), 'actual route is not the frozen legal prefix_then_cost route')
                    hist[depth]+=1;step=int(tr['start_step'][j]);sig=tr['sigma_path'][j].tolist();times=tr['target_timesteps'][j].tolist()
                    require(1<=step<=47 and step==next_tree_step and int(tr['O3_generation'][j])==j+1
                        and int(tr['frame_st_id'][j])==chc['frame_st_id']==a['frame_st_id']
                        and sig[0]==a['sigma'] and sig[1]==a['next_sigma'] and times[0]==a['timestep'],
                        'actual step/frame/sigma/time binding differs')
                    next_tree_step=step+depth+1
                    for node in range(3):
                        k=3*j+node;offset=0 if node==0 else 1;selected=node==0 or node-1==parent
                        reached=node==0 or (selected and depth>=1)
                        committed=(node==0 and depth>=1) or (node>0 and selected and depth==2)
                        expected=dict(tree_index=j,tree_step_index=step,step_index=step+offset,node=node,
                            task=case['task'],init=case['init'],episode=case['task']*100+case['init'],accepted_depth=depth,
                            selected_branch=selected,state_reached=reached,draft_edge_committed=committed)
                        for name,value in expected.items():require(cols[name][k].item()==value,'actual row identity/reached/commit flag differs: '+name)
                        support.same_bytes(torch,cols['scalars'][k],torch.tensor([sig[offset],sig[offset+1],times[offset]/1000.],dtype=torch.float32),'sigma/time input row')
                        support.same_bytes(torch,cols['delta'][k],torch.tensor(abs(sig[offset+1]-sig[offset]),dtype=torch.float32),'actual local delta')
                        require(0<=sig[offset+1]<sig[offset]<=1 and 0<=times[offset]<=1000,'invalid real scheduler input')
                        if node>0:
                            support.same_bytes(torch,cols['previous'][k],cols['proposals'][j*3,node-1],'recursive previous direction is not chosen first proposal')
                    prop=cols['proposals'][3*j:3*j+3].float();teacher=cols['target'][3*j:3*j+3].float()
                    costs=torch.linalg.vector_norm(prop-teacher[:,None,:],dim=2)*cols['delta'][3*j:3*j+3,None]
                    original=torch.cat((first[None],second),dim=0);error=float((costs-original).abs().max())
                    require(math.isfinite(error) and error<=p['norm_atol'],'CPU separately-FP32 six-norm error exceeds 1e-6')
                    max_error=max(max_error,error);edge_flips+=int(((costs<=TAU)!=(original<=TAU)).sum())
                    cpu_depth=0 if float(costs[0,parent])>TAU else 1 if float(costs[1+parent,child])>TAU else 2
                    depth_flips+=int(cpu_depth!=depth)
                reached=cols['state_reached'];require(reached.dtype==torch.bool and int(reached.sum())==pack['reached_row_count']==ref['reached_rows'], 'actual reached row coverage differs')
                for name in (*KEYS,'state_reached'):splits[case['role']][name].append(cols[name][reached].contiguous())
                ep_trees+=n;ep_rows+=nr;ep_reached+=int(reached.sum())
                raw_index.append(dict(path=str(path),stat=stat(path),context=context,counter_proof=ref['counter_proof'],
                    trees=n,all_rows=nr,reached_rows=int(reached.sum()),counterfactual_rows=nr-int(reached.sum())))
            ph=counter['phase_counters']
            require(ep_trees==index['trees']==ph['tree_count']==len(counter['acceptance'])
                and ep_rows==index['rows'] and ep_reached==index['reached_rows'],'episode row/tree/index conservation failed')
            for depth in (0,1,2):require(ph.get('accepted_depth_'+str(depth),0)==hist[depth],'episode accepted-depth histogram differs')
            require(ph['draft_forward_count']==3*ep_trees and ph['verifier_forward_count']==ep_trees
                and ph['logical_candidate_rows']==ph['verifier_target_row_count']==7*ep_trees
                and ph['verifier_cfg_rows']==14*ep_trees,'actual teacher/draft/CFG call conservation failed')
            draft=hist[1]+2*hist[2];bonus=hist[2];ordinary=50*nchunks-draft-bonus
            require(ph['committed_scheduler_update_count']==50*nchunks
                and ph.get('committed_draft_scheduler_update_count',0)==draft
                and ph.get('bonus_target_tree_count',0)==ph.get('bonus_target_scheduler_update_count',0)==bonus
                and ph['ordinary_target_scheduler_update_count']==ordinary
                and ph['tree_scheduler_transition_count']==6*ep_trees+ordinary+bonus
                and ph['target_forward_count']==ordinary-hist[0]-hist[1]
                and ph['terminal_target_forward_count']==nchunks,'all actual scheduler, bonus and fallback updates must conserve')
            for name,value in phase_sum.items():require(value<=ph.get(name,0),'policy phase sum exceeds whole episode phase: '+name)
            episode_rows.append(dict(key=key,arm='TF_CONT',task=case['task'],init=case['init'],role=case['role'],
                success=success,env_steps=env_steps,failure_penalty=env_steps if success else 800,
                termination_reason='success' if success else 'full_horizon',policy_chunks=nchunks,
                trees=ep_trees,all_teacher_rows=ep_rows,reached_rows=ep_reached,counterfactual_rows=ep_rows-ep_reached,
                accepted_depth={str(d):hist[d] for d in (0,1,2)},target_forwards=counter['target_forwards'],
                complete_terminal=True,saved_action_latent_count=len(kept),teacher_shards=nchunks,
                complete_command_mapping=True,evidence_path=str(out)))
            write(job/'status.json',dict(status='AUDITING_AND_PACKING_COMPLETE_CURRENT_TF_EPISODES_CPU_ONLY',
                complete_episodes=len(episode_rows),teacher_shards=len(all_shards)),replace=True)
            support.storage(job,p['storage_cap_bytes'])
        require(len(episode_rows)==30 and all_shards=={Path(x) for x in p['teacher_shards']}
            and total_forwards==collection['total_target_forwards'],'full registered collection/forward/shard conservation failed')
        final={name:{key:torch.cat(values,dim=0).contiguous() for key,values in columns.items()} for name,columns in splits.items()}
        for name,ids in (('FIT',FIT_IDS),('CAL',CAL_IDS)):
            require(set(final[name]['episode'].tolist())==set(ids) and bool(final[name]['state_reached'].all()),'reached-only exact FIT/CAL population mismatch')
            require(all(final[name][k].dtype==(torch.int64 if k in ('task','episode','node') else torch.float32) for k in KEYS),'training schema column dtype differs')
        total_rows=sum(len(x['state']) for x in final.values());require(0<total_rows<=225000,'complete reached-node row cap exceeded')
        raw_path=job/'RAW-SHARD-INDEX.json';write(raw_path,dict(schema='tf-effect-original-teacher-archive-index/v1',
            source_collection=str(source),shards=raw_index,counterfactual_archive_not_copied=True))
        manifest=dict(source_commit=COMMIT,generator_checkpoint=plan['selected']['TF_CONT']['checkpoint'],
            generator_checkpoint_sha256=GENERATOR_SHA,generator_condition='TF_CONT3000',generator_total_updates=3000,
            fit_episode_ids=FIT_IDS,cal_episode_ids=CAL_IDS,complete_episodes=30,all_episodes_terminal=True,
            extra_teacher_forwards=0,actual_o3_preserved=True,row_selection='root_and_actually_reached_parent',
            counterfactual_nodes_in_columns=False,production_mode='verifier_k2_b2_bonus',threshold=TAU,
            source_collection=str(source),row_order='registered case order, rollout_chunk, actual tree order, node0/1/2',
            training_initial_total_updates=2500,training_initial_checkpoint_sha256=CHILD2500_SHA,
            generator_and_training_initialization_are_distinct=True,qualifications=qual,
            FIT_rows=len(final['FIT']['state']),CAL_rows=len(final['CAL']['state']),
            episode_manifest=episode_rows,raw_shard_index=str(raw_path),raw_shard_index_sha256=sha(raw_path))
        write(job/'MANIFEST.json',manifest)
        output=job/'columns.pt';tmp=job/'columns.pt.tmp'
        require(not output.exists() and not tmp.exists(),'do not overwrite packed data')
        torch.save(dict(schema='tf-effect-onpolicy-columns/v1',FIT=final['FIT'],CAL=final['CAL'],manifest=manifest),tmp)
        require(tmp.stat().st_size<=512*2**20,'training columns exceed 512 MiB')
        os.replace(tmp,output)
        for f,s in p['evidence_stat'].items():require(stat(f)==s,'source artifact changed during CPU audit: '+f)
        evidence=[dict(path=f,sha256=h) for f,h in p['receipt_sha256'].items()]
        evidence += [dict(path=str(raw_path),sha256=sha(raw_path)),dict(path=str(job/'MANIFEST.json'),sha256=sha(job/'MANIFEST.json'))]
        certificate=dict(schema='tf-effect-data-audit/v1',status='PASS_FULL_30_EPISODE_REACHED_NODE_AUDIT',
            source_commit=COMMIT,input_pack=str(output),input_pack_stat=stat(output),manifest_sha256=digest_json(manifest),
            all_rows_reached=True,all_episodes_terminal=True,fit_cal_disjoint=True,no_development_or_test_rows=True,
            collector_equivalence_passed=True,extra_teacher_forwards=0,actual_o3_preserved=True,
            total_complete_episodes=30,FIT_episodes=20,CAL_episodes=10,FIT_rows=manifest['FIT_rows'],CAL_rows=manifest['CAL_rows'],
            success=sum(r['success'] for r in episode_rows),failure=sum(not r['success'] for r in episode_rows),
            env_steps=sum(r['env_steps'] for r in episode_rows),failure_penalty=sum(r['failure_penalty'] for r in episode_rows),
            total_teacher_trees=sum(r['trees'] for r in episode_rows),total_original_teacher_rows=sum(r['all_teacher_rows'] for r in episode_rows),
            total_reached_rows=total_rows,total_counterfactual_rows=sum(r['counterfactual_rows'] for r in episode_rows),
            total_target_forwards=total_forwards,teacher_cpu_norm_max_abs=max_error,cpu_norm_atol=p['norm_atol'],
            cpu_norm_bitwise_GPU_equivalence_claimed=False,cpu_norm_operands_separately_FP32=True,
            cpu_reduction_edge_gate_flips=edge_flips,cpu_reduction_depth_flips=depth_flips,
            production_acceptance_uses_original_GPU_scores=True,episode_results=episode_rows,
            qualifications=qual,cleanup=cleanup,evidence=evidence,permissions=PERMISSIONS,
            source_counterfactual_shards_retained=True,source_evidence_modified=False,
            training_steps=0,new_teacher_forwards=0,GPU=False,CUDA_initialized=False,
            elapsed_run_sec=time.time()-start,elapsed_registered_wall_sec=time.time()-p['birth_epoch'],
            performance_claim=False,robot_quality_certificate=False)
        require(not torch.cuda.is_initialized(),'CPU pack unexpectedly initialized CUDA')
        support.storage(job,p['storage_cap_bytes'])
        write(job/'DATA-AUDIT.json',certificate)
        write(job/'RESULT.json',dict(status=certificate['status'],input_pack=str(output),audit=str(job/'DATA-AUDIT.json'),
            FIT_rows=manifest['FIT_rows'],CAL_rows=manifest['CAL_rows'],episodes=30,permissions=PERMISSIONS))
        support.storage(job,p['storage_cap_bytes'])
        return certificate
    except BaseException as exc:
        write(job/'FAILED.json',dict(status='FAILED_CPU_FULL30_PACK_AUDIT',error=repr(exc),traceback=traceback.format_exc(),
            elapsed_run_sec=time.time()-start,elapsed_registered_wall_sec=time.time()-reg['birth_epoch'],permissions=PERMISSIONS))
        raise
    finally:
        signal.alarm(0)
        try:
            module=sys.modules.get('torch')
            write(job/'CPU-CLEANUP.json',dict(status='NO_CHILDREN_MODEL_OR_GPU_WORK_CREATED',pid=os.getpid(),
                CUDA_initialized=False if module is None else module.cuda.is_initialized(),no_foreign_signals=True))
        except Exception as exc:print('CPU_PACK_CLEANUP_MARKER_ERROR',repr(exc),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--collection',type=Path);parser.add_argument('--wall-cap-seconds',type=int,default=900)
    parser.add_argument('--storage-cap-mib',type=int,default=1024);args=parser.parse_args()
    if args.prepare:
        if args.collection is None:parser.error('--collection required for prepare')
        value=prepare(args);print(json.dumps(dict(status='PREPARED_CPU_ONLY',job=str(args.job.resolve()),deadline_epoch=value['deadline_epoch']),ensure_ascii=False))
    else:
        value=run(args.job);print(json.dumps(dict(status=value['status'],input_pack=value['input_pack'],FIT_rows=value['FIT_rows'],CAL_rows=value['CAL_rows']),ensure_ascii=False))


if __name__=='__main__':main()

"""Fixed per-suite TF64 training and referenced-shard data audits.

No torch, model, CUDA, collection or training operation runs at import.
Preparation seals a finite CPU job; execution has one fixed final checkpoint.
Legacy train entrypoints and hash/checkpoint loaders are never imported.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
if str(EXP) not in sys.path:
    sys.path.append(str(EXP))
from tf_suite_pipeline_v1.adapter import (SCHEMA, PLAN_SCHEMA, DATA_SCHEMA, INDEX_SCHEMA,
    SEED, TAU, ARCHITECTURE, PARAMETERS, CONDITIONS, UPDATES, PARENTS, PERMISSIONS,
    NORMALIZATION, MODELS, PRODUCTION, require, suite_short, model_source,
    load_checkpoint, validate_normalization, typed_equal)

TARGET_MATH = EXP/'target_trace_draft_train_v1/train.py'
MIXED_MATH = EXP/'target_trace_draft_finetune_v1/train.py'
REG_SCHEMA = 'tf-suite-stage-training-registration/v1'
STAGES = ('TARGET200', 'PARENT30', 'CURRENT30')
SOURCE_ENUM = dict(canonical_target_scheduler_CFG_after=0,
    O3_same_query_root_CFG_after=1, O3_same_query_parent_CFG_after=2)
ROLE_ENUM = dict(FIT=0, CAL=1)
DTYPE_ENUM = {'torch.bfloat16':0,'torch.float32':1}
FLOAT_COLUMNS = ('state', 'previous', 'target', 'scalars', 'delta', 'timestep')
INT_COLUMNS = ('task', 'init', 'episode', 'node', 'step_index', 'rollout_chunk',
    'tree_index', 'query', 'request_row_id', 'frame_st_id')
BOOL_COLUMNS = ('reached', 'state_reached', 'selected_branch', 'draft_edge_committed')
SMALL_COLUMNS = ('source', 'role', 'state_source_dtype', 'previous_source_dtype', 'target_source_dtype')
COLUMNS = FLOAT_COLUMNS + INT_COLUMNS + BOOL_COLUMNS + SMALL_COLUMNS
FINAL_STATUS = 'COMPLETE_FIXED_SUITE_TRAINING_STAGE_NO_ROBOT_CERTIFICATE'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    path = Path(path)
    require(path.stat().st_size <= 32*2**20, 'bounded JSON manifest required')
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value, *, current=False):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    require(len(raw) <= 32*2**20, 'bounded JSON output required')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.pending') if current else path
    with temp.open('wb' if current else 'xb') as stream:
        stream.write(raw)
    if current:
        temp.replace(path)


def stat(path):
    path = Path(path).resolve(strict=True)
    info = path.stat()
    require(path.is_file(), 'regular immutable input required')
    return dict(path=str(path), bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def unchanged(info):
    require(stat(info['path']) == info, 'registered input stat changed: '+info['path'])


def episode_ids(stage, split):
    require(stage in STAGES and split in ('FIT', 'CAL'), 'registered stage/split required')
    inits = ({'FIT': range(18), 'CAL': (18,19)} if stage == 'TARGET200' else
        {'FIT': (20,21), 'CAL': (22,)} if stage == 'PARENT30' else
        {'FIT': (23,24), 'CAL': (25,)})[split]
    return {task*100+init for task in range(10) for init in inits}


def pure_math():
    import torch
    namespace = dict(torch=torch, statistics=statistics, require=require,
        TAU=TAU, UPDATES=2000, BATCH=512)
    requests = ((TARGET_MATH, ('normalize','draw_indices','data_loss')),
        (MIXED_MATH, ('draw','decode','loss_value','predict_all','errors','metrics')))
    for path, names in requests:
        nodes = [n for n in ast.parse(path.read_bytes(), filename=str(path)).body
            if isinstance(n, ast.FunctionDef) and n.name in names]
        require({n.name for n in nodes} == set(names), 'original pure math inventory differs: '+str(path))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path)+':original-math', 'exec'), namespace)
    return {key:namespace[key] for _, names in requests for key in names}


def validate_columns(columns, *, suite, stage, split=None):
    """Validate only actual reached columns; never access counterfactual tensors."""
    import torch
    require(set(COLUMNS) <= set(columns), 'actual-column inventory incomplete')
    n = len(columns['state'])
    require(n > 0, 'empty actual-query shard/pool')
    for key in FLOAT_COLUMNS:
        value = columns[key]
        shape = (n,112) if key in ('state','previous','target') else (n,3) if key == 'scalars' else (n,)
        require(isinstance(value,torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == torch.float32 and tuple(value.shape) == shape
            and bool(torch.isfinite(value).all()), 'FP32 actual column differs: '+key)
    for key, dtype in [(k,torch.int64) for k in INT_COLUMNS] + [(k,torch.bool) for k in BOOL_COLUMNS] + [
            (k,torch.int8) for k in SMALL_COLUMNS]:
        value = columns[key]
        require(isinstance(value,torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == dtype and tuple(value.shape) == (n,), 'actual column type/shape differs: '+key)
    require(bool(columns['reached'].all() and columns['state_reached'].all()),
        'unreached counterfactual rows cannot enter any training/calibration view')
    for name in ('state','previous','target'):
        tag=columns[name+'_source_dtype']
        require(bool(((tag==0)|(tag==1)).all()), 'actual numerical dtype tag differs: '+name)
        bf=tag==0
        require(not bool(bf.any()) or torch.equal(columns[name][bf],columns[name][bf].bfloat16().float()),
            'claimed BF16 origin cannot represent stored values: '+name)
    task, init, episode = columns['task'], columns['init'], columns['episode']
    require(bool(((task>=0)&(task<10)&(init>=0)&(init<50)).all())
        and torch.equal(episode,task*100+init), 'actual episode/task/init encoding differs')
    allowed = episode_ids(stage,'FIT') | episode_ids(stage,'CAL') if split is None else episode_ids(stage,split)
    require(set(episode.tolist()) <= allowed, 'stage/split episode leakage')
    fit_ids = episode_ids(stage,'FIT')
    expected_role = torch.tensor([0 if int(i) in fit_ids else 1 for i in episode],dtype=torch.int8)
    require(torch.equal(columns['role'],expected_role), 'actual FIT/CAL role encoding differs')
    require(bool(((columns['step_index']>=1)&(columns['step_index']<=49)).all())
        and bool((columns['rollout_chunk']>=0).all()) and bool((columns['request_row_id']>=0).all()),
        'complete action scheduler/query index range differs')
    scalars = columns['scalars']
    require(bool(((scalars[:,:2]>=0)&(scalars[:,:2]<=1)).all())
        and bool((scalars[:,0] >= scalars[:,1]).all())
        and torch.equal(columns['delta'],(scalars[:,0]-scalars[:,1]).abs()),
        'authoritative sigma interval or delta differs')
    require(torch.equal(scalars[:,2],torch.tensor([float(t)/1000. for t in columns['timestep']],dtype=torch.float32)),
        'actual timestep and stored normalized time disagree')
    if stage == 'TARGET200':
        require(bool((columns['node']==-1).all() and (columns['tree_index']==-1).all()
            and (columns['source']==0).all() and columns['selected_branch'].all()
            and (~columns['draft_edge_committed']).all()), 'canonical target-only column provenance differs')
        local = columns['step_index']
    else:
        node = columns['node']
        require(bool(((node>=0)&(node<=2)).all() and (columns['tree_index']>=0).all()
            and columns['selected_branch'].all()), 'actual selected root/parent provenance differs')
        require('accepted_depth' in columns and columns['accepted_depth'].dtype == torch.int64
            and columns['accepted_depth'].shape == (n,), 'actual acceptance depth missing')
        depth=columns['accepted_depth']
        require(bool(((depth>=0)&(depth<=2)).all())
            and bool(((node==0)|(depth>=1)).all())
            and torch.equal(columns['source'],torch.where(node==0,1,2).to(torch.int8))
            and torch.equal(columns['draft_edge_committed'],((node==0)&(depth>=1))|((node>0)&(depth==2))),
            'reached/committed node differs from actual prefix depth')
        local = columns['tree_index']*3+node+1
    require(torch.equal(columns['query'],episode*1000000+columns['rollout_chunk']*10000+local)
        and len(torch.unique(columns['query'])) == n, 'actual query identity is duplicated or changed')
    return n


def load_dataset(path, *, suite, stage):
    import torch
    short = suite_short(suite)
    path = Path(path).resolve(strict=True)
    if path.suffix == '.json':
        index = read(path)
        require(index['schema'] == INDEX_SCHEMA and index['suite_short'] == short
            and index['suite'] == 'libero_'+short and index['source_stage'] == stage
            and index['status'] == 'PASS_COMPLETE_EPISODE_REACHED_ONLY_INDEX'
            and index['all_episodes_terminal'] is True and index['all_rows_reached'] is True
            and index['extra_teacher_forwards'] == 0 and index['permissions'] == PERMISSIONS,
            'complete actual-only referenced dataset index required')
        unchanged(index['collection_plan'])
        for episode in index['episodes']:
            for receipt in episode['raw_receipts'].values():
                unchanged(receipt)
        parts = dict(FIT=[],CAL=[])
        for entry in index['shards']:
            unchanged(entry['file'])
            payload = torch.load(entry['file']['path'],map_location='cpu',weights_only=True)
            require(payload['schema'] == 'tf-suite-draft-chunk-columns/v1'
                and payload['status'] == 'COMPLETE_POLICY_CHUNK_PENDING_EPISODE_AUDIT'
                and payload['suite_short'] == short and payload['suite'] == 'libero_'+short
                and payload['source_stage'] == stage and payload['extra_teacher_forwards'] == 0
                and payload['column_enums'] == dict(source=SOURCE_ENUM,role=ROLE_ENUM,source_dtype=DTYPE_ENUM),
                'registered shard identity differs')
            cols = payload['columns']
            count = validate_columns(cols,suite=short,stage=stage,split=entry['split'])
            require(count == entry['rows'] == payload['rows'] == payload['reached_rows'], 'shard row count differs')
            parts[entry['split']].append({k:cols[k] for k in COLUMNS})
        data = {split:{k:torch.cat([p[k] for p in items],0).contiguous() for k in COLUMNS}
            for split,items in parts.items()}
        manifest = index
    else:
        pack = torch.load(path,map_location='cpu',weights_only=True)
        require(pack['schema'] == DATA_SCHEMA and pack['source_stage'] == stage
            and suite_short(pack['suite']) == short, 'actual suite columns pack identity differs')
        data, manifest = {k:pack[k] for k in ('FIT','CAL')}, pack['manifest']
        require(manifest.get('all_episodes_terminal') is True and manifest.get('all_rows_reached') is True
            and manifest.get('extra_teacher_forwards') == 0, 'whole-episode source manifest missing')
        for split in data:
            validate_columns(data[split],suite=short,stage=stage,split=split)
    for split, cols in data.items():
        require(set(cols['episode'].tolist()) == episode_ids(stage,split),
            'complete stage episode coverage required; no historical missing-six exclusion')
        require(len(torch.unique(cols['query'])) == len(cols['query']), 'duplicate dataset queries')
        require(all(bool((cols['task']==task).any()) for task in range(10)), 'balanced task coverage required')
    require(not set(data['FIT']['episode'].tolist()) & set(data['CAL']['episode'].tolist()), 'FIT/CAL episode overlap')
    return data['FIT'],data['CAL'],manifest


def pack_collection(collection_job, job, registration):
    """Audit every full episode/shard and write a JSON reference index only.

    Original shards, failed episodes and counterfactual fields remain in their
    original files. No merged tensor pack or alternative split is written.
    """
    collection_job=Path(collection_job).resolve(strict=True)
    job=Path(job).resolve()
    require(not job.exists(), 'fresh immutable data-audit job required')
    r=read(registration)
    require(r['schema']=='tf-suite-data-pack-registration/v1' and r['cpu_threads'] in (1,2)
        and r['permissions']==PERMISSIONS and type(r['deadline_epoch']) in (int,float)
        and r['deadline_epoch']>time.time() and 2**20<=r['output_limit_bytes']<=64*2**20,
        'finite explicit CPU data-audit registration required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=r['cpu_threads'],
        'data audit must remain CPU-only within registered affinity')
    job.mkdir(parents=True,exist_ok=False)
    write(job/'PREREGISTRATION.json',r)
    write(job/'STARTED.json',dict(status='RUNNING_COMPLETE_EPISODE_REFERENCED_DATA_AUDIT',at=utc(),pid=os.getpid()))
    started=time.monotonic()
    deadline=min(r['deadline_epoch'],time.time()+min(r.get('maximum_wall_seconds',1800),3600))
    old_alarm=signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('registered suite data audit deadline')))
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    try:
        import torch
        require(not torch.cuda.is_initialized(), 'data audit cannot initialize CUDA')
        torch.set_num_threads(r['cpu_threads'])
        p=read(collection_job/'plan.json')
        phase=p['phase']
        require(phase in ('T_POOL','PARENT_DATA','CURRENT_DATA'), 'training data must originate from registered collection phase')
        stage=dict(T_POOL='TARGET200',PARENT_DATA='PARENT30',CURRENT_DATA='CURRENT30')[phase]
        short=suite_short(p['suite'])
        require(suite_short(r['suite'])==short and r['source_stage']==stage
            and p['frozen_permissions']==PERMISSIONS and p.get('no_SHA') is True,
            'registered source suite/phase/quality boundary differs')
        info=p['suite_info']
        require(info['suite']=='libero_'+short and isinstance(info['checkpoint_id'],str)
            and info['checkpoint_id'].startswith('Wjjjh/'), 'actual community suite target identity required')
        arm='T' if phase=='T_POOL' else 'C'
        expected=episode_ids(stage,'FIT')|episode_ids(stage,'CAL')
        rows=p['rows']
        require(len(rows)==len(expected) and {x['task']*100+x['init'] for x in rows}==expected,
            'all200/30 fixed collection episodes required; no missing-six exclusion')
        generator=None if phase=='T_POOL' else p['selected']
        if generator is not None:
            expected_condition='PARENT2000' if phase=='PARENT_DATA' else 'CONT3000'
            from tf_suite_pipeline_v1.adapter import validate_metadata
            validate_metadata(generator['metadata'],suite=short,condition=expected_condition)
            require(generator['condition']==expected_condition
                and generator['metadata']['target_checkpoint_id']==info['checkpoint_id'],
                'source collection generator condition/target identity differs')
        shards,episodes,seen_paths,seen_queries=[],[],set(),set()
        counts=dict(FIT=0,CAL=0)
        dtype_counts={name:{'torch.bfloat16':0,'torch.float32':0} for name in ('state','previous','target')}
        compact=Path(p['collector_output_dir']).resolve(strict=True)
        data_root=Path(p['suite_storage_root']).resolve(strict=True)
        require(compact.is_relative_to(data_root), 'original compact data must remain in shared suite budget root')
        for position,spec in enumerate(rows):
            require(time.time()<deadline,'registered data audit deadline')
            task,init=spec['task'],spec['init'];episode=task*100+init
            split='FIT' if episode in episode_ids(stage,'FIT') else 'CAL'
            out=collection_job/'episodes'/('case-%04d'%(init*10+task))/arm
            require(spec['row_id']==position and spec['case_id']==init*10+task and spec['arm']==arm
                and Path(spec['out']).resolve()==out.resolve(), 'actual registered episode routing differs')
            files={name:stat(out/name) for name in ('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json',
                'FINISH_READY.json','COLLECTION-AUDIT.json')}
            terminal,counter,audit,finish,collection=[read(files[k]['path']) for k in (
                'EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json','COLLECTION-AUDIT.json')]
            require(terminal['schema']=='tf-suite-full-episode-terminal/v1'
                and terminal['status']=='COMPLETE_RECORDED_FULL_EPISODE'
                and terminal['row_id']==position and terminal['case_id']==spec['case_id']
                and terminal['task']==task and terminal['init']==init and terminal['arm']==arm
                and terminal['complete_episode'] is True and terminal['quality_terminal'] is True
                and terminal['horizon']==800 and type(terminal['success']) is bool
                and 0<terminal['env_steps']<=800 and (terminal['success'] or terminal['horizon_reached'])
                and terminal['failure800_penalty']==(terminal['env_steps'] if terminal['success'] else 800)
                and terminal['permissions']==PERMISSIONS, 'full natural episode terminal missing or changed')
            require(finish['status'] in ('PASS','FAILED_COMPONENT_AUDIT')
                and finish['full_episode_terminal'] is True and finish['async_drained'] is True
                and finish['row_id']==position and finish['case_id']==spec['case_id'] and finish['arm']==arm
                and finish['permissions']==PERMISSIONS, 'full episode async-drain receipt missing')
            require(counter['schema']=='tf-suite-episode-counter/v1' and counter['row_id']==position
                and counter['suite']==p['suite'] and counter['phase']==phase
                and counter['policy_chunks']==terminal['policy_chunks']
                and counter['target_forward_attempts']==counter['target_forward_returned']
                and counter['target_forward_failed']==0 and counter['total_actual_g']==50*terminal['policy_chunks']
                and counter['collector_invalid'] is False and not counter['collector_errors']
                and counter['permissions']==PERMISSIONS, 'complete native counter/data qualification missing')
            inline=counter['adapter_inline_observation']
            if arm=='T':
                require(inline=='NOT_APPLICABLE_T_ONLY', 'target-only collection must not use a draft observer')
            else:
                require(isinstance(inline,dict) and inline.get('schema')=='tf-suite-inline-real-query-contract/v1'
                    and inline.get('observed_queries',0)>0
                    and len(inline.get('records',[]))==inline['observed_queries']
                    and inline.get('extra_teacher_forwards')==inline.get('extra_draft_forwards')==0
                    and all(record.get('status')=='PASS_OBSERVED_REAL_QUERY_CONTRACT'
                        for record in inline['records']),
                    'actual generator input/decode contract unresolved; preserve episode, reject training admission')
            require(audit['schema']=='tf-suite-episode-audit/v1' and audit['row_id']==position
                and audit['suite']==p['suite'] and audit['phase']==phase
                and audit['status']=='PASS_COMPLETE_EPISODE_COMPONENTS'
                and audit['collector_invalid'] is False
                and audit['collector_status']=='COMPLETE_POLICY_SHARDS_PENDING_DATASET_AUDIT'
                and audit['permissions']==PERMISSIONS, 'role/state/collector component qualification incomplete')
            require(collection['schema']=='tf-suite-whole-episode-collection/v1'
                and collection['status']=='PASS_COMPLETE_EPISODE_COLLECTION'
                and collection['suite']==p['suite'] and collection['phase']==phase
                and collection['row_id']==position and collection['case_id']==spec['case_id']
                and collection['task']==task and collection['init']==init and collection['arm']==arm
                and collection['role']==split and collection['full_episode_terminal'] is True
                and collection['training_admissible'] is True and not collection['collector_errors']
                and collection['success']==terminal['success'] and collection['env_steps']==terminal['env_steps']
                and collection['policy_chunks']==terminal['policy_chunks']
                and collection['no_extra_teacher_forwards'] is True and collection['permissions']==PERMISSIONS
                and collection['shards']==counter['collector_records'], 'whole episode collector qualification missing')
            packets=collection['shards']
            require(len(packets)==terminal['policy_chunks']>0, 'all complete policy chunks must have original shards')
            ep_rows=0
            for chunk,packet in enumerate(packets):
                path=Path(packet['path']).resolve(strict=True)
                require(path.is_relative_to(compact) and path not in seen_paths
                    and packet['status']=='COMPLETE_POLICY_CHUNK_PENDING_EPISODE_AUDIT'
                    and packet['extra_teacher_forwards']==0, 'duplicate/invalid shard or extra teacher call')
                file=stat(path)
                require(file['bytes']==packet['bytes'], 'original compact shard byte count changed')
                payload=torch.load(path,map_location='cpu',weights_only=True)
                require(payload['schema']=='tf-suite-draft-chunk-columns/v1'
                    and payload['status']==packet['status'] and payload['suite']==p['suite']
                    and payload['suite_short']==short and payload['source_stage']==stage
                    and payload['extra_teacher_forwards']==0 and payload['permissions']==PERMISSIONS
                    and payload['column_enums']==dict(source=SOURCE_ENUM,role=ROLE_ENUM,source_dtype=DTYPE_ENUM)
                    and payload['complete_episode_audited'] is False and payload['training_admissible'] is False,
                    'original shard pending-episode provenance/schema differs')
                ctx=payload['context']
                require(ctx==packet['context'] and ctx['row_id']==position and ctx['case_id']==spec['case_id']
                    and ctx['task']==task and ctx['init']==init and ctx['arm']==arm and ctx['out']==str(out)
                    and ctx['role']==split and ctx['rollout_chunk']==chunk, 'actual shard/context/ordinal differs')
                cols=payload['columns']
                n=validate_columns(cols,suite=short,stage=stage,split=split)
                require(n==packet['rows']==packet['reached_rows']==payload['rows']==payload['reached_rows']
                    and bool((cols['episode']==episode).all()) and bool((cols['rollout_chunk']==chunk).all())
                    and bool((cols['request_row_id']==position).all()), 'all actual shard rows must belong to this full episode/chunk')
                if stage=='TARGET200':
                    require(n==49 and cols['step_index'].tolist()==list(range(1,50)), 'all49 target query rows required')
                else:
                    details=payload['details'];tree_count=details['tree_count']
                    require(details['candidate_calls']==details['execute_calls']==tree_count
                        and details['raw_rows']==3*tree_count
                        and len(torch.unique(cols['tree_index']))==tree_count
                        and int((cols['node']==0).sum())==tree_count, 'all actual O3 tree/root coverage required')
                    observed=details['trees'];ti=cols['tree_index'];node=cols['node']
                    require(torch.equal(cols['accepted_depth'],observed['accepted_depth'][ti])
                        and torch.equal(cols['step_index'],observed['start_step'][ti]+(node>0).to(torch.int64))
                        and bool(((node==0)|(node-1==observed['selected_parent'][ti])).all()),
                        'actual selected path/depth/scheduler index differs from original tree audit')
                    for tree in range(tree_count):
                        mask=cols['tree_index']==tree;nodes=cols['node'][mask]
                        require(len(nodes)==1+int(observed['accepted_depth'][tree]>=1)
                            and int((nodes==0).sum())==1, 'all and only actual reached root/parent rows required')
                keys=set(cols['query'].tolist())
                require(not seen_queries.intersection(keys), 'actual query duplicated across shards')
                seen_queries.update(keys);seen_paths.add(path)
                for name in dtype_counts:
                    dtype_counts[name]['torch.bfloat16']+=int((cols[name+'_source_dtype']==0).sum())
                    dtype_counts[name]['torch.float32']+=int((cols[name+'_source_dtype']==1).sum())
                shards.append(dict(file=file,split=split,rows=n,episode=episode,rollout_chunk=chunk,
                    original_episode_collection_receipt=files['COLLECTION-AUDIT.json']))
                counts[split]+=n;ep_rows+=n
            episodes.append(dict(task=task,init=init,episode=episode,split=split,rows=ep_rows,
                success=terminal['success'],env_steps=terminal['env_steps'],failure800_penalty=terminal['failure800_penalty'],
                policy_chunks=terminal['policy_chunks'],raw_receipts=files))
            write(job/'status.json',dict(status='AUDITING_COMPLETE_EPISODES',complete_episodes=position+1,
                expected_episodes=len(rows),FIT_rows=counts['FIT'],CAL_rows=counts['CAL']),current=True)
        require(time.time()<deadline, 'registered data audit deadline')
        index=dict(schema=INDEX_SCHEMA,status='PASS_COMPLETE_EPISODE_REACHED_ONLY_INDEX',
            suite='libero_'+short,suite_short=short,source_stage=stage,source_phase=phase,
            collection_job=str(collection_job),collection_plan=stat(collection_job/'plan.json'),
            target_checkpoint_id=info['checkpoint_id'],target_checkpoint_path=info['model_path'],
            publisher_revision=info['publisher_revision'],generator_condition=None if generator is None else generator['condition'],
            generator_checkpoint=None if generator is None else generator['path'],
            generator_metadata=None if generator is None else generator['metadata'],
            FIT_episode_ids=sorted(episode_ids(stage,'FIT')),CAL_episode_ids=sorted(episode_ids(stage,'CAL')),
            episodes=episodes,shards=shards,FIT_rows=counts['FIT'],CAL_rows=counts['CAL'],
            actual_source_dtype_counts=dtype_counts,all_episodes_terminal=True,all_rows_reached=True,
            counterfactual_columns_used=False,extra_teacher_forwards=0,merged_tensor_pack_written=False,
            input_manifest_identity='raw receipt/row checks and stats; no cryptographic hash',
            permissions=PERMISSIONS,robot_quality_validated=False,no_SHA=True)
        write(job/'INDEX.json',index)
        final=dict(status='PASS_COMPLETE_REFERENCED_DATASET_AUDIT',index=str(job/'INDEX.json'),
            suite=index['suite'],source_stage=stage,complete_episodes=len(episodes),FIT_rows=counts['FIT'],CAL_rows=counts['CAL'],
            original_shard_bytes=sum(x['file']['bytes'] for x in shards),new_merged_data_bytes=0,
            all_failures_retained=True,all_rows_actual_reached=True,extra_teacher_forwards=0,
            elapsed_seconds=time.monotonic()-started,permissions=PERMISSIONS,robot_quality_validated=False,no_SHA=True)
        require(sum(x.stat().st_size for x in job.rglob('*') if x.is_file())<=r['output_limit_bytes'], 'pack metadata output cap exceeded')
        write(job/'RESULT.json',final)
        write(job/'FINAL-RESULT.json',final)
        write(job/'status.json',final,current=True)
        return final
    except BaseException as exc:
        failure=dict(status='FAILED_COMPLETE_SUITE_DATASET_AUDIT',error=repr(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started,partial_data_not_training_admissible=True,permissions=PERMISSIONS)
        write(job/'FAILED.json',failure)
        write(job/'status.json',failure,current=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_alarm)


def budget(job, plan):
    remaining = plan['deadline_epoch']-time.time()
    require(remaining>0, 'registered CPU job absolute deadline expired; no automatic continuation')
    size = sum(p.stat().st_size for p in Path(job).rglob('*') if p.is_file())
    require(size <= plan['output_limit_bytes'], 'registered CPU output storage cap exceeded')
    require(not (Path(job)/'STOP_REQUESTED').exists(), 'STOP_REQUESTED')
    return remaining


def seal_sources(job):
    paths = (OWN/'training.py',OWN/'adapter.py',MODELS,PRODUCTION,TARGET_MATH,MIXED_MATH)
    values = {str(p.resolve(strict=True)):p.read_text() for p in paths}
    write(Path(job)/'SOURCE-REFERENCES.json',dict(schema='tf-suite-raw-source-references/v1',
        source_texts=values,no_SHA=True))


def check_sources(job):
    sealed = read(Path(job)/'SOURCE-REFERENCES.json')
    require(sealed['schema'] == 'tf-suite-raw-source-references/v1' and sealed['no_SHA'] is True,
        'raw source reference schema differs')
    for path, original in sealed['source_texts'].items():
        require(Path(path).read_text() == original, 'registered raw source changed: '+path)


def prepare(job, registration):
    job, registration = Path(job).resolve(), Path(registration).resolve(strict=True)
    require(not job.exists(), 'fresh stage job required; previous failures/checkpoints are never overwritten')
    r = read(registration)
    require(r['schema'] == REG_SCHEMA and r['condition'] in CONDITIONS and r['seed'] == SEED
        and r['cpu_threads'] == 2 and r['permissions'] == PERMISSIONS,
        'fixed CPU2 primary-seed suite registration required')
    short, condition = suite_short(r['suite']),r['condition']
    now = time.time()
    cap = r.get('maximum_wall_seconds',3600 if condition=='PARENT2000' else 1800)
    require(type(cap) in (int,float) and 0<cap<=7200 and type(r['deadline_epoch']) in (int,float)
        and r['deadline_epoch']>now and type(r['output_limit_bytes']) is int
        and 2**20<=r['output_limit_bytes']<=512*2**20, 'finite registered CPU wall/storage cap required')
    require(isinstance(r.get('target_checkpoint_id'),str) and r['target_checkpoint_id']
        and isinstance(r.get('target_checkpoint_path'),str) and r['target_checkpoint_path'],
        'actual suite-specific target checkpoint identity required')
    target = stat(r['target_columns'])
    observer = None if condition=='PARENT2000' else stat(r['observer_columns'])
    parent = None if condition=='PARENT2000' else stat(r['parent_checkpoint'])
    parent_result = None if parent is None else stat(Path(parent['path']).parent/'RESULT.json')
    updates = 2000 if condition=='PARENT2000' else 500
    plan = dict(schema=PLAN_SCHEMA,registration=str(registration),registration_content=r,
        goal='fixed suite-specific TF64 final stage, no architecture/seed/acceptor search',
        suite='libero_'+short,suite_short=short,condition=condition,seed=SEED,
        architecture=ARCHITECTURE,parameter_count=PARAMETERS,target=target,observer=observer,
        parent=parent,parent_result=parent_result,parent_condition=PARENTS.get(condition),
        observer_source_stage=None if condition=='PARENT2000' else 'CURRENT30' if condition=='ON3000' else 'PARENT30',
        initial_total_updates=0 if condition=='PARENT2000' else UPDATES[PARENTS[condition]],
        updates=updates,total_updates=UPDATES[condition],batch=512,
        target_half=512 if condition=='PARENT2000' else 256,observer_half=0 if condition=='PARENT2000' else 256,
        learning_rate=1e-4 if condition=='PARENT2000' else 3e-5,weight_decay=.01,
        optimizer='AdamW',optimizer_state_reset=True,cpu_threads=2,inference_cpu_threads=1,
        normalization_source='TARGET200.FIT180',normalization_refit=False,
        normalization_fit_once=condition=='PARENT2000',selection='fixed_final_update_no_CAL_or_seed_selection',
        target_checkpoint_id=r['target_checkpoint_id'],target_checkpoint_path=r['target_checkpoint_path'],
        source_code_commit=r.get('source_code_commit','UNKNOWN_NO_HASH'),
        prepared_epoch=now,deadline_epoch=min(r['deadline_epoch'],now+cap),maximum_wall_seconds=cap,
        output_limit_bytes=r['output_limit_bytes'],permissions=PERMISSIONS,
        robot_quality_validated=False,deployed=False,no_GPU=True,no_new_teacher_or_environment=True)
    job.mkdir(parents=True,exist_ok=False)
    write(job/'PREREGISTRATION.json',r)
    write(job/'PLAN.json',plan)
    seal_sources(job)
    write(job/'PREPARED.json',dict(status='PREPARED_NO_MODEL_NO_TRAINING_NO_HASH',at=utc(),
        condition=condition,deadline_epoch=plan['deadline_epoch'],permissions=PERMISSIONS))
    return plan


def _save_checkpoint(path, payload):
    import torch
    path = Path(path)
    require(not path.exists(), 'immutable final checkpoint already exists')
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('xb') as stream:
        torch.save(payload,stream)
    pending.replace(path)


def predict_actual_rows(predictor, data):
    """Original production prediction on each CAL row's recorded actual dtype.

    The training draw, decoder and objective stay unchanged. This diagnostic
    avoids silently rounding a genuine FP32 production operand to BF16.
    """
    import torch
    values=torch.empty((len(data['state']),2,112),dtype=torch.float32)
    with torch.no_grad():
        for i in range(len(values)):
            state_dtype=torch.bfloat16 if int(data['state_source_dtype'][i])==0 else torch.float32
            previous_dtype=torch.bfloat16 if int(data['previous_source_dtype'][i])==0 else torch.float32
            sigma,next_sigma,_=data['scalars'][i].tolist()
            values[i]=predictor.predict(data['state'][i].reshape(7,4,4).to(state_dtype),
                data['previous'][i].reshape(7,4,4).to(previous_dtype),sigma,next_sigma,
                float(data['timestep'][i])).reshape(2,112).float()
    require(bool(torch.isfinite(values).all()), 'nonfinite fixed-final production CAL prediction')
    return values


def run(job):
    job = Path(job).resolve(strict=True)
    plan = read(job/'PLAN.json')
    require(plan['schema'] == PLAN_SCHEMA and plan['condition'] in CONDITIONS and plan['seed'] == SEED
        and plan['architecture'] == ARCHITECTURE and plan['permissions'] == PERMISSIONS,
        'fixed prepared suite stage required')
    condition=plan['condition'];is_parent=condition=='PARENT2000'
    require(plan['updates']==(2000 if is_parent else 500) and plan['total_updates']==UPDATES[condition]
        and plan['initial_total_updates']==(0 if is_parent else UPDATES[PARENTS[condition]])
        and plan['batch']==512 and plan['target_half']==(512 if is_parent else 256)
        and plan['observer_half']==(0 if is_parent else 256)
        and plan['learning_rate']==(1e-4 if is_parent else 3e-5) and plan['weight_decay']==.01
        and plan['cpu_threads']==2 and plan['inference_cpu_threads']==1
        and plan['normalization_refit'] is False and plan['normalization_fit_once'] is is_parent
        and plan['normalization_source']=='TARGET200.FIT180'
        and plan['selection']=='fixed_final_update_no_CAL_or_seed_selection'
        and plan['parent_condition']==PARENTS.get(condition)
        and plan['deadline_epoch']<=plan['prepared_epoch']+plan['maximum_wall_seconds']
        and plan['deadline_epoch']<=plan['registration_content']['deadline_epoch'],
        'fixed stage math, normalization, final selection or original wall budget differs')
    require(not any((job/n).exists() for n in ('STARTED.json','FAILED.json','FINAL-RESULT.json')),
        'one fresh execution per prepared stage; no automatic retry')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0))<=2,
        'CPU-only training requires hidden CUDA and at most two allowed cores')
    budget(job,plan)
    write(job/'STARTED.json',dict(status='RUNNING_FIXED_SUITE_CPU_STAGE',at=utc(),pid=os.getpid(),
        condition=plan['condition'],CPU_affinity=sorted(os.sched_getaffinity(0)),permissions=PERMISSIONS))
    started, completed_updates = time.monotonic(),0
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('fixed suite CPU deadline')))
    signal.setitimer(signal.ITIMER_REAL,budget(job,plan))
    try:
        import torch
        require(not torch.cuda.is_initialized(),'CPU stage cannot inherit a CUDA context')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        check_sources(job)
        for key in ('target','observer','parent','parent_result'):
            if plan[key] is not None:
                unchanged(plan[key])
        funcs = pure_math()
        fit,cal,target_manifest = load_dataset(plan['target']['path'],suite=plan['suite'],stage='TARGET200')
        require(target_manifest['target_checkpoint_id'] == plan['target_checkpoint_id']
            and target_manifest['target_checkpoint_path'] == plan['target_checkpoint_path'],
            'target data model identity differs')
        observed_fit = observed_cal = observer_manifest = None
        if plan['observer'] is not None:
            observed_fit,observed_cal,observer_manifest = load_dataset(plan['observer']['path'],
                suite=plan['suite'],stage=plan['observer_source_stage'])
            expected_generator='CONT3000' if plan['condition']=='ON3000' else 'PARENT2000'
            require(observer_manifest['target_checkpoint_id']==plan['target_checkpoint_id']
                and observer_manifest['generator_condition']==expected_generator,
                'actual observer target/generator provenance differs')
        source=model_source()
        if plan['condition']=='PARENT2000':
            norm=funcs['normalize'](fit)
            validate_normalization(norm)
            torch.manual_seed(SEED)
            predictor=source.Predictor(ARCHITECTURE,norm)
            draws=funcs['draw_indices'](fit,SEED)
            draw_pack=dict(target=draws)
        else:
            previous=read(plan['parent_result']['path'])
            require(previous['status']==FINAL_STATUS and previous['condition']==plan['parent_condition']
                and previous['updates_completed']==UPDATES[plan['parent_condition']]
                and previous['suite']==plan['suite'] and previous['permissions']==PERMISSIONS,
                'complete exact parent stage required')
            predictor=load_checkpoint(plan['parent']['path'],suite=plan['suite'],
                condition=plan['parent_condition'],device='cpu')
            require(predictor.metadata['target_checkpoint_id']==plan['target_checkpoint_id'], 'parent target model differs')
            generator=observer_manifest['generator_metadata']
            from tf_suite_pipeline_v1.adapter import validate_metadata
            validate_metadata(generator,suite=plan['suite'],condition=expected_generator)
            if condition=='CHILD2500':
                require(generator['train_job']==predictor.metadata['train_job'], 'parent data generator differs from this parent2000')
            elif condition=='CONT3000':
                require(str(Path(generator['train_job'])/'checkpoint.pt')==predictor.metadata['parent_checkpoint'],
                    'continuation observer pool must be this child common parent2000 data')
            else:
                require(generator['parent_checkpoint']==str(Path(predictor.metadata['train_job'])/'checkpoint.pt'),
                    'current CONT generator and ON must share the same CHILD2500 start')
            norm={k:getattr(predictor,k).detach().clone().cpu() for k in NORMALIZATION}
            fitted=funcs['normalize'](fit)
            require(all(typed_equal(norm[k],fitted[k]) for k in norm), 'normalizer must remain original TARGET FIT180 only')
            draw_pack=dict(target=funcs['draw'](fit,SEED+700000),
                observer=funcs['draw'](observed_fit,SEED+800000))
        require(sum(p.numel() for p in predictor.model.parameters())==PARAMETERS,'TF64 capacity differs')
        _save_checkpoint(job/'draws.pt',draw_pack)
        predictor.model.train()
        optimizer=torch.optim.AdamW(predictor.model.parameters(),lr=plan['learning_rate'],weight_decay=.01)
        logs=[]
        for i in range(plan['updates']):
            budget(job,plan)
            ids=draw_pack['target'][i]
            optimizer.zero_grad(set_to_none=True)
            if observed_fit is None:
                batch={k:fit[k][ids] for k in ('state','previous','target','scalars')}
                prediction=predictor.decode_batch(batch)
                loss=funcs['data_loss'](prediction,batch,predictor.residual_std)
            else:
                oi=draw_pack['observer'][i]
                batch={k:torch.cat((fit[k][ids],observed_fit[k][oi]),0)
                    for k in ('state','previous','target','scalars')}
                prediction=funcs['decode'](predictor,batch)
                loss=funcs['loss_value'](prediction,batch,predictor.residual_std)
            require(bool(torch.isfinite(loss)), 'nonfinite fixed-stage training loss')
            loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in predictor.model.parameters()),
                'nonfinite fixed-stage gradient')
            optimizer.step()
            completed_updates=i+1
            if completed_updates==1 or completed_updates% (250 if plan['updates']==2000 else 50)==0:
                logs.append(dict(update=completed_updates,loss=float(loss.detach()),seconds=time.monotonic()-started))
                write(job/'status.json',dict(status='RUNNING_FIXED_SUITE_CPU_STAGE',condition=plan['condition'],
                    updates_completed=completed_updates,elapsed_sec=time.monotonic()-started),current=True)
        predictor.model.eval()
        require(all(typed_equal(norm[k],getattr(predictor,k).detach().cpu()) for k in norm), 'frozen normalizer mutated')
        metadata={k:plan[k] for k in ('suite','suite_short','condition','seed','architecture','parameter_count',
            'initial_total_updates','updates','total_updates','batch','target_half','observer_half',
            'learning_rate','weight_decay','normalization_source','normalization_refit','normalization_fit_once',
            'selection','target_checkpoint_id','target_checkpoint_path','source_code_commit','parent_condition')}
        metadata.update(parent_checkpoint=None if plan['parent'] is None else plan['parent']['path'],
            target_columns=plan['target']['path'],observer_columns=None if plan['observer'] is None else plan['observer']['path'],
            observer_source_stage=plan['observer_source_stage'],optimizer_state_reset=True,
            train_job=str(job),training_updates_complete=True,permissions=PERMISSIONS,
            robot_quality_validated=False,deployed=False)
        checkpoint=job/'checkpoint.pt'
        _save_checkpoint(checkpoint,dict(schema=SCHEMA,architecture=ARCHITECTURE,specification=source.SPECS[ARCHITECTURE],
            model_state_dict=predictor.model.state_dict(),normalization=norm,metadata=metadata))
        loaded=load_checkpoint(checkpoint,suite=plan['suite'],condition=plan['condition'],device='cpu',expected_metadata=metadata)
        require(all(typed_equal(a,b) for a,b in zip(predictor.model.state_dict().values(),loaded.model.state_dict().values())),
            'saved/reloaded full model typed bytes differ')
        require(all(typed_equal(norm[k],getattr(loaded,k).cpu()) for k in norm), 'saved/reloaded normalizer differs')
        # One fixed-final full CAL pass, original production actual-dtype interface.
        # No CPU/GPU parity claim and no checkpoint/seed selection.
        cal_results={}
        rng=torch.get_rng_state().clone()
        training_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for name,view in (('TARGET_CAL20',cal),('OBSERVER_CAL10',observed_cal)):
                if view is None:
                    continue
                budget(job,plan)
                prediction=predict_actual_rows(loaded,view)
                _save_checkpoint(job/(name+'-predictions.pt'),prediction)
                cal_results[name]=dict(**funcs['metrics'](funcs['errors'](prediction,view),view),
                    meaning='fixed-final actual reached CAL direction proxy; not task success or trajectory bound',
                    input_dtype='per-row recorded original state/previous dtype, without forced BF16 rounding')
        finally:
            torch.set_num_threads(training_threads)
        require(torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized(), 'CPU CAL RNG/CUDA contract differs')
        check_sources(job)
        budget(job,plan)
        final=dict(status=FINAL_STATUS,condition=plan['condition'],suite=plan['suite'],suite_short=plan['suite_short'],
            checkpoint=str(checkpoint),metadata=metadata,updates_this_stage=completed_updates,
            updates_completed=plan['total_updates'],seed=SEED,parameters=PARAMETERS,CAL=cal_results,
            training_cpu_threads=2,cal_cpu_threads=1,training_threads_restored=True,
            one_final_checkpoint_only=True,CAL_used_for_selection=False,optimizer_state_reset=True,
            full_model_and_normalization_roundtrip_typed_bytes_exact=True,
            source_math_AST_reused=True,CAL_CPU_GPU_model_parity='UNKNOWN_NOT_TESTED',
            robot_quality_validated=False,permissions=PERMISSIONS,no_SHA=True,CUDA_initialized=False,
            extra_teacher_forwards=0,LIBERO_episodes=0,elapsed_seconds=time.monotonic()-started,loss_log=logs)
        write(job/'RESULT.json',final)
        write(job/'selected.json',dict(schema='tf-suite-fixed-final-selection/v1',checkpoint=str(checkpoint),
            suite=plan['suite'],condition=plan['condition'],metadata=metadata,
            selection='fixed_final_update_no_CAL_or_seed_selection',permissions=PERMISSIONS,
            robot_quality_validated=False))
        write(job/'FINAL-RESULT.json',final)
        write(job/'status.json',final,current=True)
        return final
    except BaseException as exc:
        failure=dict(status='FAILED_FIXED_SUITE_TRAINING_STAGE',condition=plan['condition'],suite=plan['suite'],
            updates_completed_this_stage=completed_updates,initial_total_updates=plan['initial_total_updates'],
            error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started,
            no_automatic_retry=True,permissions=PERMISSIONS)
        if not (job/'FAILED.json').exists():
            write(job/'FAILED.json',failure)
        write(job/'status.json',failure,current=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_alarm)


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--pack',action='store_true')
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--collection-job',type=Path)
    args=parser.parse_args()
    if args.prepare:
        require(args.registration is not None,'explicit registered CPU training input required')
        return prepare(args.job,args.registration)
    if args.pack:
        require(args.registration is not None and args.collection_job is not None,
            'registered complete collection and finite CPU audit budget required')
        return pack_collection(args.collection_job,args.job,args.registration)
    return run(args.job)


if __name__=='__main__':
    result=main()
    print(json.dumps({k:v for k,v in result.items() if k not in ('registration_content','loss_log')},ensure_ascii=False),flush=True)

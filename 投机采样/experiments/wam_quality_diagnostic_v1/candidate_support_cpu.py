"""Bounded audit of existing candidate FIT support; no new training/teacher/GPU."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import os
import signal
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RT = ROOT / '投机采样/results/runtime'
EXP = ROOT / '投机采样/experiments'
DATA = Path('/var/tmp/cxz-wam-quality-20261014/p0-morning-long10-train-six-v1-20261015-r1/dataset.pt')
Q02 = RT / 'p0-wam-quality-q02-replay-v1-20261016-r2'
TAU = .007839680526450805


def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    with Path(p).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def require(v, why):
    if not v: raise ValueError(why)
def fp32(x): return struct.unpack('f', struct.pack('f', float(x)))[0]
def chain(r): return (fp32(r['sigma']), fp32(r['next_sigma']), fp32(r['timestep'] / 1000.))
def tree_id(r):
    return (r['campaign_id'], r['parallel_worker_id'], r['episode_attempt_id'],
            r['task_index'], r['episode_index'], r['rollout_chunk'], r['tree_step_index'])
def reach(r):
    if r['node'] == 0: return 'actual_root_query'
    if not r['selected_branch']: return 'unselected_counterfactual_first'
    return 'selected_parent_prefix_committed' if r['accepted_depth'] >= 1 else 'selected_parent_edge_rejected'


def main(out):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only explicit mask required')
    os.sched_setaffinity(0, {12, 13})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('840s CPU audit cap')))
    signal.alarm(840)
    out = out.resolve()
    require(out.parent == RT and not out.exists(), 'fresh runtime output required')
    out.mkdir()
    started = time.monotonic()
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    rng = torch.get_rng_state().clone()
    terminal = dict(status='RUNNING')
    try:
        qr, qw, qp = read(Q02 / 'RESULT.json'), read(Q02 / 'WORKER-RESULT.json'), read(Q02 / 'plan.json')
        require(qr['status'] == 'COMPLETE_Q02_REAL_INPUT_NUMERIC_DIAGNOSTIC'
            and qr['all_owned_processes_reaped'] and qw['same_query_qualified']
            and qw['route_or_depth_flips'] == 0, 'qualified complete Q02 no-flip diagnostic required')
        pack = torch.load(DATA, map_location='cpu', weights_only=False)
        sources = {'current3906_Aoff': pack['current']['train_rows'],
                   'aux11091_historical_TF': pack['aux_rows']}
        require([len(v) for v in sources.values()] == [3906, 11091], 'fixed full FIT cardinality')
        allrows = sources['current3906_Aoff'] + sources['aux11091_historical_TF']
        require(all(r['split'] == 'train' and r['episode_index'] < 18 for r in allrows), 'FIT-only source')
        inventory = {}
        for name, rows in sources.items():
            trees = defaultdict(list)
            for row in rows: trees[tree_id(row)].append(row)
            for tree in trees.values():
                require(len(tree) == 3 and {r['node'] for r in tree} == {0, 1, 2}, 'complete three-node FIT trees required')
                require(len({r['accepted_depth'] for r in tree}) == 1, 'tree depth labels disagree')
                require(sum(r['selected_branch'] for r in tree if r['node']) == 1, 'one recorded selected parent required')
                for row in tree:
                    require(row['accepted_depth'] in (0, 1, 2), 'invalid recorded depth')
                    expected = row['accepted_depth'] >= 1 if row['node'] == 0 else row['selected_branch'] and row['accepted_depth'] == 2
                    require(bool(row['draft_edge_committed']) == bool(expected), 'recorded label semantic mismatch')
            inventory[name] = dict(rows=len(rows), trees=len(trees), all_three_nodes_per_tree=True,
                fields_present_all=sorted(set.intersection(*(set(r) for r in rows))),
                tasks=dict(Counter(r['task_index'] for r in rows)),
                episodes=sorted({(r['task_index'], r['episode_index']) for r in rows}),
                node_counts=dict(Counter(r['node'] for r in rows)),
                recorded_depth_trees=dict(Counter(v[0]['accepted_depth'] for v in trees.values())),
                recorded_reach_categories=dict(Counter(reach(r) for r in rows)),
                committed_nodes=sum(r['draft_edge_committed'] for r in rows),
                teacher_labels=dict(Counter(r['teacher_label'] for r in rows)))
        with (out / 'OLD-ROW-MANIFEST.csv').open('x') as f:
            writer = csv.writer(f)
            writer.writerow(['source','task','init','attempt','chunk','tree_step','node','sigma','next_sigma','timestep_normalized_FP32','selected','accepted_depth','draft_edge_committed','reach_category'])
            for name, rows in sources.items():
                for row in rows: writer.writerow([name,row['task_index'],row['episode_index'],row['episode_attempt_id'],row['rollout_chunk'],row['tree_step_index'],row['node'],*chain(row),int(row['selected_branch']),row['accepted_depth'],int(row['draft_edge_committed']),reach(row)])
        # One actual raw payload interface plus the existing full 3697-source audit;
        # this does not pretend to repeat the full raw numeric audit.
        raw_path = Path(pack['aux_sources'][0]['path'])
        raw = torch.load(raw_path, map_location='cpu', weights_only=False)
        raw_interface = dict(path=str(raw_path),top_fields=sorted(raw),
            row_fields=sorted(raw['rows'][0]), contract_fields=sorted(raw['contract']),
            source_audit=str(RT/'p0-data-numeric-audit-v1-20261015-r1/SAME-QUERY-RESULT.json'),
            all_raw_sources_reloaded_in_this_audit=False)
        current = []
        numeric = {r['id']: r for r in qw['numeric']}
        saved_metrics = {r['id']:r for r in read(RT/'p0-wam-quality-q02-matched-cpu-v1-20261016-r1/ROWS.json')}
        for p in sorted((Q02/'queries').glob('*.pt')):
            q = torch.load(p, map_location='cpu', weights_only=True)
            nr = numeric[p.stem]
            for t in q['teacher_rows']:
                require(t['data_role'] == 'DEV_DIAGNOSTIC_NOT_FIT', 'current96 are DEV only')
                r = {**t, 'id':p.stem+'-node'+str(t['node']), 'task_index':7,
                     'episode_index':24, 'accepted_depth':nr['packed_route'][2], 'source':str(p)}
                if 'accepted_depth' in t: require(t['accepted_depth']==r['accepted_depth'], 'actual depth binding')
                current.append(r)
        require(len(current) == 96, 'all actual96 current teacher rows')
        sys.path.insert(0, str(EXP))
        from target_trace_draft_finetune_v1.adapter import load_checkpoint
        cp = qp['reference_plan']['selected']['transformer64']
        predictor = load_checkpoint(cp['checkpoint'], architecture='transformer64', device='cpu',
            expected_channels=tuple(range(7)), expected_sha256=cp['checkpoint_sha256'])

        @torch.no_grad()
        def error(row, label='target'):
            s = row['state'].reshape(7,4,4).bfloat16()
            p = row['previous'].reshape(7,4,4).bfloat16()
            sigma, nxt, normalized = chain(row)
            prediction = predictor.predict(s,p,sigma,nxt,normalized*1000.).reshape(2,112).float()
            target = row[label].reshape(112).float()
            e = torch.linalg.vector_norm(prediction-target[None],dim=1).double()*abs(sigma-nxt)
            v = torch.linalg.vector_norm(p.float().reshape(112)-target).double()*abs(sigma-nxt)
            require(bool(torch.isfinite(e).all()) and bool(torch.isfinite(v)), 'finite actual row errors')
            return e.numpy(), float(v)
        def stats(values, weights=None):
            v=np.asarray(values,dtype=float)
            if not len(v): return dict(rows=0,mean=None,pass_rate=None)
            w=np.ones(len(v))/len(v) if weights is None else np.asarray(weights,dtype=float)
            w=w/w.sum()
            return dict(rows=len(v),mean=float(np.dot(w,v)),pass_rate=float(np.dot(w,v<=TAU)))
        current_errors, velocity = {}, {}
        for row in current:
            e,v=error(row,'target_packed');es,vs=error(row,'target_sequential')
            current_errors[row['id']]=float(e.min())
            require(np.allclose(e,np.asarray(saved_metrics[row['id']]['CPU_BF16_packed_teacher_errors']),rtol=0,atol=1e-12), 'actual current CPU path differs from saved96')
            velocity[row['id']]=dict(TF_packed=float(e.min()),vprev_packed=v,TF_sequential=float(es.min()),vprev_sequential=vs)
        old_error_cache={}
        comparisons=[]
        for name, rows in sources.items():
            for strict_reach in (False,True):
                for node_matching_mode in ("exact_node", "node_level"):
                    pool=defaultdict(list)
                    for i,row in enumerate(rows):
                        key=(row['task_index'],chain(row),row['node'] if node_matching_mode=='exact_node' else int(row['node']>0))+((reach(row),) if strict_reach else ())
                        pool[key].append(i)
                    for role in ('root','first'):
                        selected=[c for c in current if (c['node']==0)==(role=='root')]
                        wanted=Counter((c['task_index'],chain(c),c['node'] if node_matching_mode=='exact_node' else int(c['node']>0))+((reach(c),) if strict_reach else ()) for c in selected)
                        matched=[c for c in selected if pool[(c['task_index'],chain(c),c['node'] if node_matching_mode=='exact_node' else int(c['node']>0))+((reach(c),) if strict_reach else ())]]
                        pairs=[]
                        for key,count in wanted.items():
                            ids=pool.get(key,[])
                            if not ids: continue
                            for i in ids:
                                cachekey=(name,i)
                                if cachekey not in old_error_cache: old_error_cache[cachekey]=float(error(rows[i])[0].min())
                                pairs.append((old_error_cache[cachekey],count/len(ids)))
                        oldvalues,weights=zip(*pairs) if pairs else ([],[])
                        comparisons.append(dict(source=name,role=role,node_matching_mode=node_matching_mode,strict_recorded_reach_category=strict_reach,
                            current_rows=len(selected),matched_current_rows=len(matched),
                            unique_old_rows=len(pairs),exact_chains_matched=len({chain(c) for c in matched}),
                            current_matched_error=stats([current_errors[c['id']] for c in matched]),
                            old_exact_chain_reweighted_error=stats(oldvalues,weights),
                            unmatched_current_rows=len(selected)-len(matched),
                            historical_weighting='each exact task/FP32chain/node[/recorded-reach] pool weighted by current frequency',
                            caveat='old data is FIT seen by this model; current is purposive task7 DEV init24. Descriptive support only, not causal drift or independent generalization.'))
        vsummary={}
        for category,predicate in [('all96',lambda c:True),('root32',lambda c:c['node']==0),
            ('first64',lambda c:c['node']>0),('first_selected',lambda c:c['node']>0 and c['selected_branch']),
            ('first_committed',lambda c:c['node']>0 and c['draft_edge_committed']),
            ('first_not_committed',lambda c:c['node']>0 and not c['draft_edge_committed'])]:
            selected=[c for c in current if predicate(c)]
            vsummary[category]={k:stats([velocity[c['id']][k] for c in selected]) for k in ('TF_packed','vprev_packed','TF_sequential','vprev_sequential')}
        # Current full10 records have actual per-parent GPU costs. Old row payloads
        # do not; do not regenerate historical-generator scores from tensors.
        episode_counts=[]
        fresh=RT/'p0-target-trace-draft-fresh-dev-v1-20261015-r2'
        diagnostic=read(RT/'p0-fresh-dev-terminal-diagnostic-v1-20261015-r2/DIAGNOSTIC.json')
        require(diagnostic['status']=='COMPLETE_INDEPENDENT_TERMINAL30_DIAGNOSTIC','complete current DEV evidence')
        for task in range(10):
            p=fresh/f'episodes/FINETUNE_TF64-t{task}-i24/COUNTER.json';counter=read(p)
            cc=Counter()
            for tree in counter['acceptance']:
                require(tree['threshold']==TAU and tree['effective_selector']=='prefix_then_cost', 'fixed actual rule')
                d=tree['actual_depth'];cc['trees']+=1;cc['depth'+str(d)]+=1
                eligible=[float(x)<=TAU for x in tree['first_costs']]
                cc['first_candidate_queries']+=2
                cc['parent_edges_admissible']+=sum(eligible)
                cc['parent_edges_rejected']+=2-sum(eligible)
                cc['selected_parent_committed']+=int(d>=1)
                cc['selected_parent_rejected']+=int(d==0)
                cc['selected_second_edge_committed']+=int(d==2)
                for parent in range(2):
                    child_good=min(tree['second_costs'][parent])<=TAU
                    cc['admissible_parent_but_both_children_rejected']+=int(eligible[parent] and not child_good)
                cc['recorded_depth1_child_learning_opportunity_trees']+=int(d==1)
            episode_counts.append(dict(task=task,**dict(cc),source=str(p)))
        total=Counter()
        for row in episode_counts:
            for k,v in row.items():
                if isinstance(v,int) and k!='task':total[k]+=v
        require(total['trees']==3611, 'all current TF complete10 trees')
        current_keys=Counter(reach(c) for c in current)
        st=DATA.stat()
        write(out/'MANIFEST.json',dict(dataset=dict(path=str(DATA),bytes=st.st_size,mtime_ns=st.st_mtime_ns),
            raw_aux_original_inventory=pack['aux_sources'],checkpoint=cp,
            current96_sources=sorted({c['source'] for c in current}),
            reused_full_raw_audit=str(RT/'p0-data-numeric-audit-v1-20261015-r1/SAME-QUERY-RESULT.json'),
            audit_code=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            no_full_dataset_or_model_independent_hash=True))
        terminal=dict(status='COMPLETE_EXISTING_CANDIDATE_SUPPORT_CPU_AUDIT',old_inventory=inventory,
            raw_interface=raw_interface,current96_reach_categories=dict(current_keys),
            same_task_exact_FP32chain_node_comparisons=comparisons,
            current96_constant_velocity_comparison=vsummary,
            current10_episode_recorded_opportunities=episode_counts,current10_total=dict(total),
            current10_g_from_recorded_tree_depth=1+(total['depth1']+2*total['depth2'])/total['trees'],
            fixed_record_child_only_max_additional_g=total['depth1']/total['trees'],
            missing_old_fields=['per-parent first_costs/second_costs and threshold on each cached row',
                'historical-generator per-head proposals at every recursive node',
                'per-unselected-parent acceptance/local cost certificate',
                'complete observation/KV/conditioning tensors for arbitrary new same-query teacher re-evaluation',
                'contact/gripper-stage task-risk labels'],
            meanings=dict(teacher='Actual recorded same-query target direction for all3 predictor inputs; no relabeling as on-policy currentTF.',
                reach='Selected first node with recorded depth>=1 proves selected denoising parent prefix committed, not physical robot contact/success.',
                nonselected='Counterfactual queried states; local eligibility cannot be asserted from selected=false.',
                current_costs='Current10 counters contain actual saved GPU per-edge costs; depth uses each edge<=tau, accepted-edge sum only tie cost.'),
            D01='HOLD_NOT_AUTOMATICALLY_TRIGGERED',
            limitations=['Historical labels are already used in currentTF mixed FIT; lower old loss cannot establish covariate shift or new-data necessity.',
                'Current96 are task7 first10chunk purposive DEV; no population conclusion.',
                'Current10 counters supply opportunity denominators but no new teacher labels for missing candidate states.',
                'CPU vprev/TF errors are descriptive and do not replace GPU BF16 acceptance certification.',
                'Old DEV tensors share the loaded package but were never indexed/evaluated/exported; no future TEST results opened.'],
            elapsed_seconds=time.monotonic()-started,CPU_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=2,
            CPU_RNG_unchanged=torch.equal(rng,torch.get_rng_state()),CUDA_initialized=torch.cuda.is_initialized(),
            GPU_used=False,NVML_used=False,SSH_used=False,new_training=False,new_teacher=False,own_process_exits_after_report=True)
        require(terminal['CPU_RNG_unchanged'] and not terminal['CUDA_initialized'], 'CPU RNG or CUDA changed')
        write(out/'RESULT.json',terminal)
        size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        require(size<32*2**20,'32MiB evidence cap')
        print(json.dumps(dict(status=terminal['status'],out=str(out),seconds=terminal['elapsed_seconds'],bytes=size,
            comparisons=comparisons,totals=dict(total),velocity=vsummary),ensure_ascii=False),flush=True)
    except BaseException as exc:
        write(out/'FAILED.json',dict(error=repr(exc),elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        signal.alarm(0)
        write(out/'EXECUTION-EXIT.json',dict(status='CPU_WORK_FINISHED_NO_CHILDREN',pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0)),GPU_used=False,NVML_used=False,SSH_used=False,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);main(p.parse_args().out)

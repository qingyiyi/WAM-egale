"""CPU integrity audit and acceptance/trace analysis, with no speed claim."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from experiment import plan, write, stamp
from paths import EXP, load


def summarize(trees):
    counts = Counter(int(r['actual_depth']) for r in trees)
    n = len(trees)
    if not n:
        raise ValueError('nonempty measured verifier group required')
    return dict(trees=n, depth0=counts[0], depth1=counts[1], depth2=counts[2],
        first_accept_rate=(counts[1]+counts[2])/n, full_accept_rate=counts[2]/n,
        updates_per_tree=(counts[0]+2*counts[1]+3*counts[2])/n,
        recoverable_vs_global_trees=sum(r['maximum_admissible_depth']>r['global_depth'] for r in trees),
        recoverable_vs_global_updates=sum(r['maximum_admissible_depth']-r['global_depth'] for r in trees))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    a = ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only audit required')
    job = a.job.resolve()
    p = plan(job)
    collection = json.loads((job/'GPU-COLLECTION.json').read_text())
    assert collection['status']=='PASS_GPU_FULL_EPISODE_COLLECTION'
    assert collection['completed_episodes']==4 and collection['model_launches']==5
    assert collection['owned_roots_exited'] and collection['cleanup']['memory_used_bytes']==0
    assert not collection['cleanup']['compute'] and not collection['cleanup']['graphics']
    assert collection['aggregate_GPU_seconds']<18000 and collection['aggregate_target_intents']<=60000
    import numpy as np
    import torch
    snap = load('accept_offline_snapshot',EXP/'fixed_load_cost/snapshot.py')
    fixed = json.loads((job/'fixed/worker0/WORKER-RESULT.json').read_text())
    assert fixed['chunks']==8 and fixed['raw_vs_shared_O3_all30_KV_output_routes_RNG_exact_both_selectors']
    assert fixed['original_bonus_global_archived_output_exact']
    for init in (3,4):
        for selector in ('GLOBAL','DEPTH'):
            x=torch.load(job/f'fixed/worker0/init{init}-RAW_{selector}-output.pt',map_location='cpu',weights_only=False)
            y=torch.load(job/f'fixed/worker0/init{init}-{selector}-output.pt',map_location='cpu',weights_only=False)
            assert snap.fingerprint(x)==snap.fingerprint(y)
    all_trees={a:[] for a in p['arms']}
    totals={a:dict(successes=0,env_steps=0,infer_chunks=0,target_forwards=0) for a in p['arms']}
    episodes, chunks, archive_pairs=[],[],0
    for init in (3,4):
        paired_meta=[]
        for arm in p['arms']:
            root=job/'campaigns'/arm/'workers'/f'worker{init-3}'
            lines=(root/'client/episodes.jsonl').read_text().splitlines()
            assert len(lines)==1
            r=json.loads(lines[0]); meta=json.loads((root/'client/repro_metadata.json').read_text())
            assert r['status']=='completed' and r['episode_index']==init and r['libero_horizon']==800
            assert r['max_policy_chunks'] is None and not r['truncated_after_max_policy_chunks']
            assert r['success'] or r['horizon_reached']
            assert meta['actual_EGL_initialization_attempts']==1 and meta['chunk_event_logging']
            assert meta['command_count']==r['env_steps']==np.load(root/'client/commands.npy').shape[0]
            paired_meta.append(meta)
            live=json.loads((root/'live_device_egl_audit.json').read_text())
            assert live['nvidia_egl_loaded'] and live['client_process_alive_at_audit']
            c=json.loads((root/'COUNTER-CLOSED.json').read_text())
            assert c['status']=='CLOSED' and c['target_intents']==c['host_returned']
            assert not c['active_forwards'] and not c['request_active'] and not c['failed_forwards']
            phase=c['phase_counters']
            assert phase['committed_scheduler_update_count']==50*r['infer_chunks']
            assert c['shared']['enabled'] and c['shared']['last_request_clean']
            assert c['shared']['requests']==r['infer_chunks'] and c['shared']['candidates']>0
            logs=json.loads((root/'ACCEPTANCE-TREES.json').read_text())
            assert logs['threshold']==0.007839680526450805 and logs['extra_target_forwards']==0
            trees=logs['records']; assert len(trees)==phase['tree_count']
            for depth in (0,1,2):
                assert sum(t['actual_depth']==depth for t in trees)==phase.get('accepted_depth_'+str(depth),0)
            assert phase['bonus_target_scheduler_update_count']==sum(t['actual_depth']==2 for t in trees)
            events=[json.loads(x) for x in (root/'client/chunk_events.jsonl').read_text().splitlines()]
            assert len(events)==r['infer_chunks']
            groups=defaultdict(list)
            for t in trees:
                assert t['maximum_admissible_depth']>=t['global_depth']
                assert t['actual_depth']==t['global_depth'] if arm=='GLOBAL' else t['actual_depth']==t['maximum_admissible_depth']
                groups[int(t['context']['rollout_chunk'])].append(t)
            assert set(groups)==set(range(r['infer_chunks']))
            assert sum(e['env_steps_delta'] for e in events)==r['env_steps']
            for e in events:
                ts=groups[e['rollout_chunk']]; stats=summarize(ts)
                progressed=sum(1+t['actual_depth'] for t in ts)
                # One initial target anchor plus an ordinary tail; 50 always.
                ordinary=50-progressed
                assert 1<=ordinary<=3
                assert ts[0]['context']['env_steps_before']==e['env_steps_before']
                chunks.append(dict(arm=arm,init=init,rollout_chunk=e['rollout_chunk'],
                    env_steps_before=e['env_steps_before'],env_steps_after=e['env_steps_after'],
                    env_steps_delta=e['env_steps_delta'],done=e['done'],
                    ordinary_denoising_updates=ordinary,total_denoising_updates=50,**stats))
            stats=summarize(trees)
            episodes.append(dict(arm=arm,init=init,success=r['success'],env_steps=r['env_steps'],
                infer_chunks=r['infer_chunks'],target_forwards=c['target_intents'],**stats))
            all_trees[arm].extend(trees)
            for k,v in [('successes',int(r['success'])),('env_steps',r['env_steps']),
                        ('infer_chunks',r['infer_chunks']),('target_forwards',c['target_intents'])]:
                totals[arm][k]+=v
            if arm=='GLOBAL':
                old=Path(p['prior_bonus_job'])/'campaigns/BONUS/workers'/f'worker{init-3}'
                goldmeta=json.loads((old/'client/repro_metadata.json').read_text())
                for key in ['model_xml_sha256','init_state_sha256','initial_state_hashes',
                            'final_state_hashes','initial_obs_hashes','initial_rng','final_rng']:
                    assert meta[key]==goldmeta[key],key
                for name in ['commands.npy','initial_state.npz','final_state.npz']:
                    if name.endswith('.npy'):
                        x,y=np.load(root/'client'/name),np.load(old/'client'/name)
                        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                    else:
                        with np.load(root/'client'/name) as x,np.load(old/'client'/name) as y:
                            assert set(x.files)==set(y.files)
                            for k in x.files:
                                assert x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes()
                fresh={x.name:x for x in (root/'server_output').rglob('*.pt')}
                gold={x.name:x for x in (old/'server_output').rglob('*.pt')}
                assert fresh.keys()==gold.keys()
                for name,path in fresh.items():
                    x=torch.load(path,map_location='cpu',weights_only=False)
                    y=torch.load(gold[name],map_location='cpu',weights_only=False)
                    assert snap.fingerprint(x)==snap.fingerprint(y),name
                    archive_pairs+=1
                oldc=json.loads((old/'COUNTER-CLOSED.json').read_text())
                assert phase==oldc['phase_counters'] and c['target_intents']==oldc['target_intents']
        for key in ['model_xml_sha256','init_state_sha256','initial_state_hashes','initial_obs_hashes','initial_rng']:
            assert paired_meta[0][key]==paired_meta[1][key]
    bins={}
    for arm,trees in all_trees.items():
        totals[arm].update(summarize(trees))
        groups=defaultdict(list)
        for t in trees:
            groups['first_chunk' if t['context']['rollout_chunk']==0 else 'later_chunks'].append(t)
            groups['sigma_low' if t['sigma']<1/3 else 'sigma_mid' if t['sigma']<2/3 else 'sigma_high'].append(t)
        bins[arm]={k:summarize(v) for k,v in groups.items()}
    old=json.loads((Path(p['prior_bonus_job'])/'FINAL_RESULT.json').read_text())
    history={a:dict(env_steps=old['totals'][a]['env_steps'],infer_chunks=old['totals'][a]['infer_chunks'],
                   successes=old['totals'][a]['successes']) for a in ['T','OFF','BONUS']}
    acceptance_improved=(totals['DEPTH']['updates_per_tree']>totals['GLOBAL']['updates_per_tree'] and
                         totals['DEPTH']['full_accept_rate']>totals['GLOBAL']['full_accept_rate'])
    success_preserved=totals['DEPTH']['successes']==totals['GLOBAL']['successes']==2
    trace_preserved=totals['DEPTH']['env_steps']<=min(totals['GLOBAL']['env_steps'],history['T']['env_steps'])
    max_trace_preserved=max(r['env_steps'] for r in episodes if r['arm']=='DEPTH')<=300
    promote=acceptance_improved and success_preserved and trace_preserved and max_trace_preserved
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_BONUS_ACCEPTANCE_QUALITY_OFFLINE_AUDIT',at=stamp(),job=str(job),
        collection=collection,episodes=episodes,totals=totals,bins=bins,chunks=chunks,
        historical_quality_references=history,GLOBAL_archived_tensor_pairs_exact=archive_pairs,
        GLOBAL_archived_commands_env_RNG_and_counters_exact=True,numerical_full_chunk_oracle_pairs=4,
        speed_comparison_deferred=True,training=False,threshold_unchanged=True,CUDA_initialized=False,
        promotion=dict(acceptance_improved=acceptance_improved,success_preserved=success_preserved,
                       trace_preserved=trace_preserved,max_trace_preserved=max_trace_preserved,
                       promote_DEPTH=promote,selected_selector='depth_then_cost' if promote else 'global'),
        limitation='Two development initial states of one task; historical baseline used only for quality; instrumented paired models are not clean speed measurements. Same-tree depth guarantee is not a guarantee of shorter closed-loop environment trajectory.')
    write(job/'FINAL_RESULT.json',result,exclusive=True,durable=True)
    print(json.dumps({k:result[k] for k in ['status','totals','promotion']},ensure_ascii=False))


if __name__=='__main__':
    main()

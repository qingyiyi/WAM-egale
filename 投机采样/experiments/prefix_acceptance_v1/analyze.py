"""Offline full-episode quality and true-state correctness audit; no speed claim."""
import argparse
from collections import Counter,defaultdict
import json,os
from pathlib import Path
from experiment import plan,write,stamp
from paths import EXP,load


def stats(trees):
    c=Counter(r['actual_depth'] for r in trees);n=len(trees)
    if not n:raise ValueError('nonempty draft-tree group required')
    return dict(trees=n,depth0=c[0],depth1=c[1],depth2=c[2],
        first_accept_rate=(c[1]+c[2])/n,full_accept_rate=c[2]/n,
        updates_per_tree=(c[0]+2*c[1]+3*c[2])/n,
        effective_parent_opportunities=sum(r['effective_parent_changed'] for r in trees),
        bookkeeping_child_only_changes=sum(r['bookkeeping_child_only_change'] for r in trees))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only analysis boundary')
    job=a.job.resolve();p=plan(job,full=True)
    c=json.loads((job/'GPU-COLLECTION.json').read_text())
    assert c['status']=='PASS_GPU_FULL_EPISODE_COLLECTION' and c['completed_episodes']==6 and c['model_launches']==7
    assert c['owned_roots_exited'] and not c['cleanup']['memory_used_bytes']
    assert not c['cleanup']['compute'] and not c['cleanup']['graphics']
    assert c['aggregate_GPU_seconds']<18000 and c['aggregate_target_intents']<=75000
    import numpy as np
    import torch
    snap=load('prefix_offline_snapshot',EXP/'fixed_load_cost/snapshot.py')
    fixed=json.loads((job/'fixed/worker0/WORKER-RESULT.json').read_text())
    assert fixed['chunks']==8 and fixed['current_DEPTH_archived_output_exact']
    assert fixed['raw_vs_shared_O3_all30_KV_output_routes_RNG_exact_both_selectors']
    for init in (3,4):
        for arm in ('DEPTH','PREFIX'):
            x=torch.load(job/f'fixed/worker0/init{init}-RAW_{arm}-output.pt',map_location='cpu',weights_only=False)
            y=torch.load(job/f'fixed/worker0/init{init}-{arm}-output.pt',map_location='cpu',weights_only=False)
            assert snap.fingerprint(x)==snap.fingerprint(y)
    totals={arm:dict(successes=0,env_steps=0,infer_chunks=0,target_forwards=0) for arm in p['arms']}
    episodes=[];chunks=[];trees_by_arm={arm:[] for arm in ('DEPTH','PREFIX')};pairs=[]
    for init in p['inits']:
        states={};metas={};tree_logs={}
        for arm in p['arms']:
            root=job/'campaigns'/arm/'workers'/f'worker{init-6}'
            lines=(root/'client/episodes.jsonl').read_text().splitlines();assert len(lines)==1
            r=json.loads(lines[0]);meta=json.loads((root/'client/repro_metadata.json').read_text());metas[arm]=meta
            assert r['status']=='completed' and r['episode_index']==init and r['libero_horizon']==800
            assert r['max_policy_chunks'] is None and not r['truncated_after_max_policy_chunks']
            assert r['success'] or r['horizon_reached']
            live=json.loads((root/'live_device_egl_audit.json').read_text())
            assert live['nvidia_egl_loaded'] and live['client_process_alive_at_audit']
            commands=np.load(root/'client/commands.npy')
            assert meta['command_count']==r['env_steps']==commands.shape[0] and meta['actual_EGL_initialization_attempts']==1
            counter=json.loads((root/'COUNTER-CLOSED.json').read_text());phase=counter['phase_counters']
            assert counter['status']=='CLOSED' and counter['target_intents']==counter['host_returned']
            assert not counter['active_forwards'] and not counter['request_active'] and not counter['failed_forwards']
            logs=json.loads((root/'ACCEPTANCE-TREES.json').read_text())
            assert logs['extra_target_forwards']==0 and logs['threshold']==0.007839680526450805
            trees=logs['records'];tree_logs[arm]=trees
            events=[json.loads(x) for x in (root/'client/chunk_events.jsonl').read_text().splitlines()]
            assert len(events)==r['infer_chunks'] and sum(e['env_steps_delta'] for e in events)==r['env_steps']
            if arm=='T':
                assert not trees and not counter['shared']['enabled']
            else:
                assert phase['committed_scheduler_update_count']==50*r['infer_chunks']
                assert len(trees)==phase['tree_count']
                assert counter['shared']['enabled'] and counter['shared']['last_request_clean']
                assert counter['shared']['requests']==r['infer_chunks'] and counter['shared']['candidates']>0
                grouped=defaultdict(list)
                for t in trees:
                    assert t['baseline_depth']==t['maximum_admissible_depth']==t['actual_depth']
                    assert t['candidate_prefix_cost']<=t['baseline_prefix_cost']
                    grouped[t['context']['rollout_chunk']].append(t)
                assert set(grouped)==set(range(r['infer_chunks']))
                for d in (0,1,2):assert sum(t['actual_depth']==d for t in trees)==phase.get('accepted_depth_'+str(d),0)
                assert phase.get('bonus_target_scheduler_update_count',0)==sum(t['actual_depth']==2 for t in trees)
                for e in events:
                    ts=grouped[e['rollout_chunk']];ordinary=50-sum(1+t['actual_depth'] for t in ts)
                    assert 1<=ordinary<=3 and ts[0]['context']['env_steps_before']==e['env_steps_before']
                    chunks.append(dict(arm=arm,init=init,**e,ordinary_denoising_updates=ordinary,**stats(ts)))
                trees_by_arm[arm]+=trees
            states[arm]=dict(commands=commands,phase=phase,root=root)
            row=dict(arm=arm,init=init,success=r['success'],env_steps=r['env_steps'],infer_chunks=r['infer_chunks'],target_forwards=counter['target_intents'])
            if trees:row.update(stats(trees))
            episodes.append(row)
            for k,v in [('successes',int(r['success'])),('env_steps',r['env_steps']),('infer_chunks',r['infer_chunks']),('target_forwards',counter['target_intents'])]:totals[arm][k]+=v
        for key in ['model_xml_sha256','init_state_sha256','initial_state_hashes','initial_obs_hashes','initial_rng']:
            assert metas['T'][key]==metas['DEPTH'][key]==metas['PREFIX'][key],key
        a0,b0=states['DEPTH'],states['PREFIX']
        commands_exact=(a0['commands'].dtype==b0['commands'].dtype and a0['commands'].shape==b0['commands'].shape and a0['commands'].tobytes()==b0['commands'].tobytes())
        final_exact=all(metas['DEPTH'][k]==metas['PREFIX'][k] for k in ['final_state_hashes','final_rng'])
        xs={x.name:x for x in (a0['root']/'server_output').rglob('*.pt')}
        ys={x.name:x for x in (b0['root']/'server_output').rglob('*.pt')}
        outputs_exact=xs.keys()==ys.keys()
        equal_tensors=0
        if outputs_exact:
            for name in xs:
                x=torch.load(xs[name],map_location='cpu',weights_only=False);y=torch.load(ys[name],map_location='cpu',weights_only=False)
                if snap.fingerprint(x)!=snap.fingerprint(y):outputs_exact=False;break
                equal_tensors+=1
        effective=totals['PREFIX'].get('effective_parent_opportunities')
        expected_equal=not any(t['effective_parent_changed'] for t in tree_logs['DEPTH'])
        if expected_equal:
            assert commands_exact and final_exact and outputs_exact and a0['phase']==b0['phase']
        pairs.append(dict(init=init,commands_exact=commands_exact,final_state_RNG_exact=final_exact,
            output_tensors_exact=outputs_exact,equal_tensor_pairs=equal_tensors,phase_counts_exact=a0['phase']==b0['phase'],
            no_effective_change_on_baseline_path=expected_equal))
    bins={}
    for arm,trees in trees_by_arm.items():
        totals[arm].update(stats(trees));g=defaultdict(list)
        for t in trees:g['high' if t['sigma']>=2/3 else 'mid' if t['sigma']>=1/3 else 'low'].append(t)
        bins[arm]={name:stats(rows) for name,rows in g.items()}
    success=all(totals[a]['successes']==2 for a in p['arms'])
    trace=totals['PREFIX']['env_steps']<=totals['DEPTH']['env_steps']
    longest={arm:max(r['env_steps'] for r in episodes if r['arm']==arm) for arm in p['arms']}
    max_trace=longest['PREFIX']<=longest['T']
    adopted=success and trace and max_trace
    exact=all(p['commands_exact'] and p['final_state_RNG_exact'] and p['output_tensors_exact'] for p in pairs)
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_PREFIX_QUALITY_OFFLINE_AUDIT',at=stamp(),job=str(job),collection=c,
        totals=totals,episodes=episodes,chunks=chunks,bins=bins,pairs=pairs,
        numerical_oracle_pairs=4,current_DEPTH_archived_outputs_exact=True,
        quality_protocol=p['quality_protocol'],CUDA_initialized=False,training=False,threshold_unchanged=True,
        speed_comparison_deferred=True,promotion=dict(success_preserved=success,trace_preserved=trace,
            max_trace_preserved=max_trace,adopt_PREFIX=adopted,selected_selector='prefix_then_cost' if adopted else 'depth_then_cost',
            neutral_exact_behavior=exact),
        limitation='Two extra development initial states of task0; parent checkpoint historical exposure not fully excluded. Not formal multi-task noninferiority or a speed measurement.')
    write(job/'FINAL_RESULT.json',result,exclusive=True,durable=True)
    print(json.dumps({k:result[k] for k in ['status','totals','promotion']},ensure_ascii=False))

if __name__=='__main__':main()

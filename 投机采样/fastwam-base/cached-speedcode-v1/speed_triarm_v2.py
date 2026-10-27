"""RTX5090 exclusive real-draft measurement on preregistered canonical DEV inputs."""
from prompt_cache import PromptCache
import argparse
import contextlib
import hashlib
from policy_schedule import policy_role
import json
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from runtime import ROOT, build, inputs, postprocess, write
from adapter import canonical_action
from draft import load as load_draft
from real_speculative import RealSpeculative, choose

P=argparse.ArgumentParser()
P.add_argument('--assets',required=True)
args=P.parse_args()
OUT=ROOT/'results/speed-triarm-attempt01'
OUT.mkdir(parents=True,exist_ok=False)
reg=json.loads((ROOT/'REGISTRATION.json').read_text())
start=time.time()
deadline=min(reg['deadline_epoch']-300,start+reg['RTX5090_gpu_lifecycle_seconds_cap'])
status=dict(phase='LOADING',start=start,deadline=deadline,pid=os.getpid())


def phase(name,**kw):
    status.update(phase=name,updated=time.time(),**kw)
    write(OUT/'STATUS.json',status)


def exclusive():
    assert time.time()<deadline,'speed budget'
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    other=[int(x.strip()) for x in raw.splitlines() if x.strip() and int(x.strip())!=os.getpid()]
    assert not other,('exclusive GPU required',other)


class Clock:
    def __init__(self): self.rows=[]
    @contextlib.contextmanager
    def __call__(self,name):
        torch.cuda.synchronize();t=time.perf_counter()
        yield
        torch.cuda.synchronize();self.rows.append(dict(phase=name,seconds=time.perf_counter()-t))


def summary(rows):
    t=np.array([x['T_seconds'] for x in rows]);s=np.array([x['S_seconds'] for x in rows])
    return dict(N=len(rows),T_total_seconds=float(t.sum()),S_total_seconds=float(s.sum()),
        speedup=float(t.sum()/s.sum()),T_mean_ms=float(t.mean()*1000),S_mean_ms=float(s.mean()*1000),
        T_p50_ms=float(np.median(t)*1000),S_p50_ms=float(np.median(s)*1000),
        T_p90_ms=float(np.quantile(t,.9)*1000),S_p90_ms=float(np.quantile(s,.9)*1000))


signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('speed lifecycle cap')))
signal.alarm(max(1,int(deadline-time.time())))
try:
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    exclusive();phase('LOADING')
    model,processor,cfg,ns=build(args.assets,OUT)
    select=json.loads((ROOT/'training/SELECTED.json').read_text())
    assert hashlib.sha256((ROOT/'training/checkpoint-3000.pt').read_bytes()).hexdigest()==select['checkpoint_sha256']
    spec=RealSpeculative(model,load_draft(ROOT/'training/checkpoint-3000.pt'),select['tau'])
    model.empirical_replay=spec
    cache=PromptCache(model)
    common=json.loads((ROOT/'COMMON-SUCCESS-INPUTS.json').read_text())
    allowed={(r['task'],r['init']) for r in common['original_source_success'] if r['included']}
    assert common['suite']==reg['suite'] and allowed
    fixture_manifest=json.loads((ROOT/'speed-fixtures/MANIFEST.json').read_text())
    fixtures=[f for f in fixture_manifest['inputs'] if (f['identity']['task'],f['identity']['init']) in allowed]
    assert len(fixtures)==3*len(allowed)
    semantic=[]
    loaded=[]
    for f in fixtures:
        with np.load(ROOT/'speed-fixtures'/f['file']) as z:obs={k:z[k].copy() for k in z.files}
        loaded.append((f,obs))
    f,obs=loaded[0]
    with torch.no_grad():
        kwargs=inputs(obs,f['task_description'],model,processor,cfg,ns)
        eager=model.infer_action(**kwargs,compile_action_infer=False)['action'].float()
        t=time.perf_counter();compiled=model.infer_action(**kwargs,compile_action_infer=True)['action'].float()
        torch.cuda.synchronize();compile_T=time.perf_counter()-t
        diff=compiled-eager;relative=float(diff.norm()/eager.norm().clamp_min(1e-6));maxerr=float(diff.abs().max())
        assert relative<=.01 and maxerr<=.05+.02*float(eager.abs().max())
        t=time.perf_counter();model.infer_action_replay(**kwargs,compile_action_infer=True)
        torch.cuda.synchronize();compile_S=time.perf_counter()-t
        spec.validate=True;model.infer_action_replay(**kwargs,compile_action_infer=True);spec.validate=False
        write(OUT/'CONSISTENCY.json',dict(canonical_relative_L2=relative,canonical_max_abs=maxerr,
             compile_T_seconds=compile_T,compile_S_seconds=compile_S,B7=spec.audit_records))
        full,action=[],[]
        full_counts,action_counts=Counter(),Counter()
        phase('PAIRED_FULL_CHUNKS',inputs=len(loaded))
        for i,(f,obs) in enumerate(loaded):
            exclusive()
            # Validate the actual cached and uncached conditions/output/path on
            # every retained input. These calls are outside all warm speed sums.
            kwargs=inputs(obs,f['task_description'],model,processor,cfg,ns)
            raw_context,raw_mask=cache.original(kwargs['prompt'])
            hit_context,hit_mask=model.encode_prompt(kwargs['prompt'])
            assert torch.equal(raw_context,hit_context) and torch.equal(raw_mask,hit_mask)
            outputs={}; paths={}; counters={}
            for label,enabled in [('uncached',False),('cached',True)]:
                cache.enabled=enabled
                outputs[label]={}
                for arm in ['T','S']:
                    value=(model.infer_action if arm=='T' else model.infer_action_replay)(**kwargs,compile_action_infer=True)['action']
                    outputs[label][arm]=value.detach().clone()
                    if arm=='S':
                        paths[label]=[{k:r[k] for k in ['step','kind','depth','parent','child','g'] if k in r} for r in spec.rounds]
                        counters[label]=dict(spec.counts)
            differences={arm:float((outputs['uncached'][arm].float()-outputs['cached'][arm].float()).abs().max()) for arm in ['T','S']}
            rowcheck=dict(input=i,fixture=f,text_context_exact=True,text_mask_exact=True,action_max_abs=differences,
                          paths_exact=paths['uncached']==paths['cached'],counts_exact=counters['uncached']==counters['cached'])
            with (OUT/'TEXT-CACHE-CONSISTENCY.jsonl').open('a') as stream:stream.write(json.dumps(rowcheck)+'\n')
            assert all(v==0 for v in differences.values()) and rowcheck['paths_exact'] and rowcheck['counts_exact'],rowcheck
            semantic.append(rowcheck)
            del outputs,raw_context,raw_mask,hit_context,hit_mask
            # Warm each input once for both arms; compilation/warmup are excluded.
            kwargs=inputs(obs,f['task_description'],model,processor,cfg,ns)
            model.infer_action(**kwargs,compile_action_infer=True)
            model.infer_action_replay(**kwargs,compile_action_infer=True)
            for repeat in range(5):
                row=dict(input=i,fixture=f,repeat=repeat)
                for arm in (('T','S') if (i+repeat)%2==0 else ('S','T')):
                    torch.cuda.synchronize();t=time.perf_counter()
                    kwargs=inputs(obs,f['task_description'],model,processor,cfg,ns)
                    result=(model.infer_action if arm=='T' else model.infer_action_replay)(**kwargs,compile_action_infer=True)
                    postprocess(result['action'],processor,ns)
                    torch.cuda.synchronize();row[arm+'_seconds']=time.perf_counter()-t
                    if arm=='S':row['counts']=dict(spec.counts);row['rounds']=spec.rounds;full_counts.update(spec.counts)
                with (OUT/'FULL-CHUNKS.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
                full.append({k:v for k,v in row.items() if k!='rounds'})
            phase('PAIRED_FULL_CHUNKS',inputs_done=i+1,inputs=len(loaded))
        phase('PAIRED_ACTION')
        for i,(f,obs) in enumerate(loaded):
            exclusive();kwargs=inputs(obs,f['task_description'],model,processor,cfg,ns)
            prepared=model.prepare_action_inputs(**kwargs,compile_action_infer=True)
            for k in ['state','timesteps','deltas']:prepared.pop(k)
            for repeat in range(5):
                row=dict(input=i,repeat=repeat)
                for arm in (('S','T') if (i+repeat)%2==0 else ('T','S')):
                    torch.cuda.synchronize();t=time.perf_counter()
                    state=torch.randn((1,32,7),generator=torch.Generator(device='cpu').manual_seed(42),dtype=torch.float32).to('cuda',dtype=torch.bfloat16)
                    ts,ds=model.infer_action_scheduler.build_inference_schedule(10,device=model.device,dtype=state.dtype,shift_override=5.)
                    output=canonical_action(model,state,ts,ds,**prepared) if arm=='T' else spec.loop(state,ts,ds,**prepared)
                    output[0].detach().to('cpu',dtype=torch.float32)
                    torch.cuda.synchronize();row[arm+'_seconds']=time.perf_counter()-t
                    if arm=='S':row['counts']=dict(spec.counts);action_counts.update(spec.counts)
                with (OUT/'ACTION-CHUNKS.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
                action.append(row)
            del prepared
            phase('PAIRED_ACTION',inputs_done=i+1)
        # Complete policy sequences on registered raw observation traces. This
        # measures inference replay, not simulator stepping or new task success.
        trace_manifest=[g for g in fixture_manifest['traces'] if (g['identity']['task'],g['identity']['init']) in allowed]
        assert trace_manifest and all(all(g['arms'][arm]['success'] for arm in ['T','S','P']) for g in trace_manifest)
        traces=[]
        phase('POLICY_TRACE_REPLAY')
        for mode in ('fixed_target_observations','own_natural_observations'):
            for group_index,group in enumerate(trace_manifest):
                arm_order=('T','S','P')[group_index%3:]+('T','S','P')[:group_index%3]
                for arm in arm_order:
                    exclusive()
                    reference=group['arms']['T' if mode=='fixed_target_observations' else arm]
                    counts=Counter();chunk_rows=[]
                    # File reading is outside the inference boundary for every arm.
                    observations=[]
                    for fp in reference['files']:
                        with np.load(ROOT/'speed-fixtures'/fp) as zz:
                            observations.append({k:zz[k].copy() for k in zz.files})
                    torch.cuda.synchronize();sequence_begin=time.perf_counter()
                    for ordinal,obs in enumerate(observations):
                        role=policy_role(arm,ordinal)
                        torch.cuda.synchronize();chunk_begin=time.perf_counter()
                        kwargs=inputs(obs,group['task_description'],model,processor,cfg,ns)
                        result=(model.infer_action if role=='T' else model.infer_action_replay)(**kwargs,compile_action_infer=True)
                        postprocess(result['action'],processor,ns)
                        torch.cuda.synchronize();seconds=time.perf_counter()-chunk_begin
                        cc=dict(spec.counts) if role=='S' else dict(committed_updates=10,target_calls=10,target_B1_calls=10,target_state_rows=10)
                        assert cc['committed_updates']==10
                        counts.update(cc)
                        chunk_rows.append(dict(ordinal=ordinal,role=role,seconds=seconds,counts=cc))
                    torch.cuda.synchronize();sequence_seconds=time.perf_counter()-sequence_begin
                    assert counts['committed_updates']==10*len(observations)
                    if arm=='P':assert sum(x['role']=='T' for x in chunk_rows)==len(observations)//3
                    row=dict(mode=mode,group=group['identity'],arm=arm,source_arm='T' if mode=='fixed_target_observations' else arm,
                        source_success=reference['success'],source_commands=reference['commands'],
                        chunks=len(observations),seconds=sequence_seconds,counts=dict(counts),chunk_rows=chunk_rows)
                    with (OUT/'POLICY-TRACES.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
                    traces.append({k:v for k,v in row.items() if k!='chunk_rows'})
                    phase('POLICY_TRACE_REPLAY',completed=len(traces),planned=2*len(trace_manifest)*3)
        trace_summary={}
        for mode in ('fixed_target_observations','own_natural_observations'):
            rows=[r for r in traces if r['mode']==mode]
            totals={a:sum(r['seconds'] for r in rows if r['arm']==a) for a in ('T','S','P')}
            paired=[]
            for group in trace_manifest:
                rr={r['arm']:r for r in rows if r['group']==group['identity']}
                paired.append(dict(identity=group['identity'],
                    source_success={a:group['arms'][a]['success'] for a in ('T','S','P')},
                    seconds={a:rr[a]['seconds'] for a in ('T','S','P')},chunks={a:rr[a]['chunks'] for a in ('T','S','P')}))
            conditional={}
            for a in ('S','P'):
                common=[p for p in paired if p['source_success']['T'] and p['source_success'][a]]
                conditional[a]=dict(N=len(common),speedup=sum(p['seconds']['T'] for p in common)/sum(p['seconds'][a] for p in common) if common else None)
            trace_summary[mode]=dict(N=len(paired),total_seconds=totals,speedup={a:totals['T']/totals[a] for a in ('S','P')},
                pairs=paired,common_success_conditional=conditional,
                boundary='raw observations already resident -> repeated complete policy with preprocessing, text/VAE/prefill/action/postprocessing; no simulator stepping')
        # Separate cold semantic diagnostic, never included in speed totals.
        sensitivity=[]
        phase('BATCH_ACCEPTANCE_SENSITIVITY')
        for input_index,(fixture,obs) in enumerate(loaded):
            exclusive();spec.audit_records=[];spec.validate=True
            kwargs=inputs(obs,fixture['task_description'],model,processor,cfg,ns)
            model.infer_action_replay(**kwargs,compile_action_infer=True)
            spec.validate=False
            a=spec.audit_inputs;dirs1=a['draft_first'];dirs2=a['draft_second']
            dd=a['deltas'].to('cuda');scores_by_mode={}
            for label,values in [('B7',a['batched_target'].to('cuda')),('serial_B1',spec.audit_serial_target)]:
                first=(dirs1.float()-values[0:1].float()).flatten(1).norm(dim=1)*dd[0].float().abs()
                second=(dirs2.float()-values[1:3,None].float()).flatten(2).norm(dim=2)*dd[1].float().abs()
                scores_by_mode[label]=torch.cat((first,second.flatten())).tolist()
            choices={k:choose(v,spec.tau) for k,v in scores_by_mode.items()}
            flips=[(a<=spec.tau)!=(b<=spec.tau) for a,b in zip(scores_by_mode['B7'],scores_by_mode['serial_B1'])]
            row=dict(input=input_index,fixture=fixture,step=a['step'],scores=scores_by_mode,choices=choices,
                score_acceptance_flips=flips,path_changed=choices['B7']!=choices['serial_B1'],
                min_tau_margin=min(abs(v-spec.tau) for v in scores_by_mode['B7']),direction_audit=spec.audit_records)
            sensitivity.append(row)
            with (OUT/'ACCEPTANCE-FLOATING-SENSITIVITY.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        write(OUT/'ACCEPTANCE-FLOATING-SENSITIVITY.json',dict(N=len(sensitivity),
            edge_checks=6*len(sensitivity),edge_flips=sum(sum(x['score_acceptance_flips']) for x in sensitivity),
            path_changes=sum(x['path_changed'] for x in sensitivity),tau=spec.tau,
            limitation='First real candidate tree on each retained common-success DEV observation; finite numerical sensitivity diagnostic, not a distributional guarantee. No threshold adjustment.'))
        profiles=[];profile_groups=set()
        phase('COMMON_SUCCESS_STAGE_PROFILES')
        for profile_index,(f,obs) in enumerate(loaded):
            key=(f['identity']['task'],f['identity']['init'])
            if key in profile_groups:continue
            profile_groups.add(key)
            assert f['ordinal']==0
            exclusive();clock=Clock();model.phase_clock=clock
            before_cache=cache.snapshot()
            with clock('full_policy'):
                with clock('observation_preprocess'):
                    kwargs=inputs(obs,f['task_description'],model,processor,cfg,ns)
                profiled=model.infer_action__profiled(**kwargs,compile_action_infer=True)
                with clock('action_postprocess'):
                    postprocess(profiled['action'],processor,ns)
            Tphases=clock.rows.copy();clock.rows=[];spec.profile=True
            with clock('full_policy'):
                with clock('observation_preprocess'):
                    kwargs=inputs(obs,f['task_description'],model,processor,cfg,ns)
                profiled=model.infer_action_replay__profiled(**kwargs,compile_action_infer=True)
                with clock('action_postprocess'):
                    postprocess(profiled['action'],processor,ns)
            denoise=next(r['seconds'] for r in clock.rows if r['phase']=='action_denoise')
            round_total=sum(r['seconds'] for r in spec.rounds);boundary=denoise-round_total
            G=sum(r['g'] for r in spec.rounds)
            assert boundary>=0 and G==10
            profile=dict(input=profile_index,fixture=f,T=Tphases,S=clock.rows,rounds=spec.rounds,
                accounting=dict(G=G,T_denoise=denoise,T_round_sum=round_total,T_boundary=boundary,C_update=denoise/G),
                cache_before=before_cache,cache_after=cache.snapshot(),
                caveat='Separate synchronized diagnostic; loop excludes noise/grid/output; nested timers not summed.')
            assert profile['cache_after']['misses']==before_cache['misses']
            with (OUT/'PROFILE-CHUNKS.jsonl').open('a') as stream:stream.write(json.dumps(profile)+'\n')
            profiles.append(profile);spec.profile=False
        assert profile_groups==allowed
        profile=profiles[0]
        write(OUT/'TEXT-CACHE-CONSISTENCY.json',dict(N=len(semantic),all_contexts_exact=True,all_actions_exact=True,
              all_acceptance_paths_exact=True,cache=cache.snapshot(),source_episode_selection='all three original DEV arms succeeded'))
        assert full_counts['committed_updates']==action_counts['committed_updates']==10*len(full)==10*len(action)
        exclusive()
        result=dict(status='COMPLETE',trained_draft=True,forced_depths=False,tau=select['tau'],
            full_chunk=summary(full),action_denoising=summary(action),policy_trace_replay=trace_summary,full_counts=dict(full_counts),
            action_counts=dict(action_counts),C_update_T_seconds=sum(x['T_seconds'] for x in action)/(10*len(action)),
            C_update_S_seconds=sum(x['S_seconds'] for x in action)/action_counts['committed_updates'],
            separate_synchronized_profile=profile,phase_profiles=profiles,text_cache=cache.snapshot(),common_success_source_groups=len(allowed),quality_claim=False,formal_performance_permission=False,
            limitations=['fixed-input cost, not natural episode E2E','A100 closed-loop cost is concurrent, reported separately'],
            start=start,end=time.time())
        write(OUT/'RESULTS.json',result);phase('COMPLETE')
except BaseException as exc:
    import traceback
    phase('FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    raise
finally:signal.alarm(0)

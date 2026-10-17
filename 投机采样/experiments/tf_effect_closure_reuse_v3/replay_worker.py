"""ON full recorded episode versus original OFF baseline; final full KV exact.

Original two-pass/per-request-KV Attempt stays PARTIAL; no robot quality gate
is changed and its 45-minute absolute cumulative deadline remains in force.
"""
import argparse,os,signal,sys,traceback,types,time
from pathlib import Path
OWN=Path(__file__).resolve().parent;sys.path.insert(0,str(OWN.parent))
from tf_effect_closure_v1.protocol import load,read,write,require,ROOT

def run(s,source,helpers,job,p):
    import numpy as np
    import torch
    from reset_text_bank_v1.bank import TextBank,install as install_bank
    from task4_direction_completion_v1.observer import load_with_existing_cal
    from task4_direction_completion_v1.server_entry import snapshot, fingerprint
    from tf_effect_collector_v1.collector import Collector
    validate,RamAccounting,modules=helpers;validate(s)
    q=p['reference_plan'];ep=Path(p['source_episode']);counter=read(ep/'COUNTER.json')
    initial=torch.load(Path(q['job'])/'arms/TF_CONT/TARGET-INITIAL-RNG.pt',map_location='cpu',weights_only=True)
    require(torch.equal(torch.get_rng_state(),initial['CPU']) and torch.equal(torch.cuda.get_rng_state(s.device),initial['CUDA']),'target-only initialization RNG differs')
    s.action_per_frame=int(s.job_config.action_per_frame)
    install_bank(s,TextBank(q['bank']))
    def no_text(*a,**kw):raise RuntimeError('unexpected text encoder call')
    s._get_t5_prompt_embeds=types.MethodType(no_text,s)
    predictor,_=load_with_existing_cal(q,'TF_CONT',s.device,tuple(range(7)),job/'CAL-REUSE.json')
    s._p0_action_draft_b2_predictor=predictor;s._p0_action_draft_b2_checkpoint=predictor.checkpoint_path
    s._p0_action_draft_mode=q['mode']
    proxy=modules['selection'].install(s,q['selector'])
    modules['buffers'].prepare(s._infer_action_verifier_k2_b2,s._load_p0_shadow_api())
    online=modules['measured_online'].control(s,'O3',allow_cuda=True)
    accounting=RamAccounting(s,cap=p['target_forward_cap'])
    old_save=source.save_async;captures={}
    def capture(value,path,*args,**kwargs):
        name=Path(path).name
        if name.startswith(('actions_','latents_')):
            require(name not in captures,'duplicate saved output')
            captures[name]=value.detach().contiguous().cpu().clone()
    source.save_async=capture
    roots=list((ep/'server_output').glob('real/*'));require(len(roots)==1,'one original output root')
    original_root=roots[0];old_acceptance=counter['acceptance'];baseline=[];collector=None
    with np.load(ep/'client/initial_observations.npz',allow_pickle=False) as z:initial_obs={k:z[k] for k in z.files}
    passes=[]
    try:
        for enabled in (True,):
            name='ON' if enabled else 'OFF';s.save_root=str(job/name/'unused');captures.clear()
            if enabled:
                collector=Collector(s,proxy,job/'teacher',dict(checkpoint=q['selected']['TF_CONT'],target=q['target_checkpoint_id'],
                    data_role='previously_unblinded_DEV_qualification_never_FIT_CAL',storage_cap_bytes=512*2**20))
                collector.install()
            torch.set_rng_state(initial['CPU']);torch.cuda.set_rng_state(initial['CUDA'],s.device)
            request_index=0;obs=initial_obs;action=None;rows=[];start=time.monotonic();accept_offset=0
            start_count=accounting.attempted;phase_before=dict(accounting.phase)
            for req in counter['requests']:
                context=req['trace_context'];kind=req['kind'];frame=req['server_frame_st_id_before']
                before=accounting.attempted;before_records=len(proxy.records)
                if kind=='reset':request=dict(reset=True,prompt=context['prompt'],p0_trace_context=context)
                elif kind=='policy':request=dict(obs=obs,prompt=context['prompt'],p0_trace_context=context)
                else:
                    require(kind=='cache' and action is not None,'known original request kind')
                    obs_list=torch.load(original_root/f'obs_data_{frame}.pt',map_location='cpu',weights_only=False)
                    require(isinstance(obs_list,list) and len(obs_list)>0,'complete actual cache observation list')
                    request=dict(obs=obs_list,state=action,compute_kv_cache=True,imagine=False,p0_trace_context=context)
                if enabled and kind=='policy':collector.begin_chunk(context)
                rng_before=dict(cpu=fingerprint(torch.get_rng_state())['bytes_sha256'],cuda=fingerprint(torch.cuda.get_rng_state(s.device))['bytes_sha256'])
                result=s.infer(request)
                if enabled and kind=='policy':shard=collector.end_chunk()
                count=accounting.attempted-before
                require(count==req['target_forwards'],'original request target count differs')
                actual_records=proxy.records[before_records:]
                if kind=='policy':
                    chunk=context['rollout_chunk'];expected=torch.load(ep/'direction_trace'/f'chunk-{chunk:03d}.pt',map_location='cpu',weights_only=False)
                    require(np.array_equal(result['action'],expected['submitted_server_action']),'recorded physical actions differ')
                    require(rng_before['cpu']==expected['CPU_rng_before'] and rng_before['cuda']==expected['CUDA_rng_before'],'original per-policy input RNG differs')
                    require(fingerprint(torch.get_rng_state())['bytes_sha256']==expected['CPU_rng_after'] and fingerprint(torch.cuda.get_rng_state(s.device))['bytes_sha256']==expected['CUDA_rng_after'],'original per-policy output RNG differs')
                    action=np.array(result['action'],copy=True)
                    for prefix in ('actions','latents'):
                        key=f'{prefix}_{frame}.pt';actual=captures.pop(key)
                        old=torch.load(original_root/key,map_location='cpu',weights_only=True)
                        require(torch.equal(actual,old),'recorded full chunk '+key+' differs')
                    n=len(actual_records)
                    require(actual_records==old_acceptance[accept_offset:accept_offset+n],'original accepted paths/costs differ')
                    accept_offset+=n
                    if enabled:
                        require(shard['trees']==n and shard['rows']==n*3,'collector full tree row coverage')
                        pack=torch.load(shard['path'],map_location='cpu',weights_only=False)
                        cols=pack['columns']
                        trs=[r for r in expected['rounds'] if r['kind']=='speculative_tree']
                        require(len(trs)==n,'original tree coverage differs')
                        for j,tr in enumerate(trs):
                            for node in range(3):
                                k=j*3+node;prev=tr['root_previous_direction'] if node==0 else tr['first_directions'][node-1]
                                require(torch.equal(cols['state'][k],tr['candidate_states'][node].float().reshape(112)),'actual teacher query differs')
                                require(torch.equal(cols['previous'][k],prev.float().reshape(112)),'previous semantics differ')
                                require(torch.equal(cols['target'][k],tr['teacher_target_directions'][node].float().reshape(112)),'actual original teacher differs')
                                reached=node==0 or (tr['accepted_depth']>=1 and node-1==tr['selected_parent'])
                                require(bool(cols['state_reached'][k])==reached,'reached flag differs')
                elif kind=='cache':obs=obs_list[-1]
                state=snapshot(s,empty=(kind=='reset')) if kind=='reset' or request_index==len(counter['requests'])-1 else None
                if kind=='reset':require(state==read(ep/'RESET.json')['state'],'original full reset KV/RNG differs')
                if request_index==len(counter['requests'])-1:require(state==read(ep/'FINAL-SERVER-STATE.json'),'original final all-valid30layer KV/RNG differs')
                item=dict(request_index=request_index,kind=kind,context=context,forwards=count,records=actual_records,
                    full_valid_KV_and_state=state)
                # Original full OFF episode is the frozen baseline. Every policy is
                # checked above; full valid KV is checked at reset and final.
                require(enabled,'this protocol only replays collector ON')
                rows.append(dict(index=request_index,kind=kind,forwards=count,all_valid30layer_KV_RNG_exact=state is not None))
                request_index+=1
                write(job/'status.json',dict(status='FULL_RECORDED_EPISODE_REPLAY',collector=name,
                    requests=request_index,policies=sum(r['kind']=='policy' for r in rows),planned_policies=50,
                    target_forwards=accounting.attempted,elapsed_pass_sec=time.monotonic()-start),replace=True)
            require(accept_offset==len(old_acceptance) and accounting.attempted-start_count==counter['target_forwards'],'full original target/path conservation')
            phases={k:v-phase_before.get(k,0) for k,v in accounting.phase.items()}
            require(phases==counter['phase_counters'],'full original phase counters differ')
            require(sum(x['kind']=='policy' for x in rows)==50,'all50 original chunks required')
            passes.append(dict(collector=name,requests=len(rows),policy_chunks=50,target_forwards=accounting.attempted-start_count,
                accepted_trees=accept_offset,all_original_actions_latents_and_commands_exact=True,
                original_final_valid_KV_RNG_exact=True,original_per_policy_RNG_exact=True,OFF_baseline_reused=True,elapsed_sec=time.monotonic()-start))
            write(job/f'{name}-REQUEST-PROOF.json',rows)
            if collector:collector.close();collector=None
        require(s._p0_trace is None and s._p0_state_teacher is None,'production trace/teacher must remain disabled')
        events=modules['buffers'].EVENTS
        require(len(events)==50 and all(x['enabled'] and x['ok'] and x['closed'] for x in events),'O3 did not remain enabled for all50 chunks')
        closed=accounting.close()
        write(job/'WORKER-RESULT.json',dict(status='PASS_FULL_RECORDED_EPISODE_COLLECTOR_EQUIVALENCE_BASELINE_REUSED',
            passes=passes,recorded_full_episode_replay=True,new_robot_episodes=0,original_episode_was_failed=True,
            final_all_valid_KV_RNG_exact=True,per_policy_original_RNG_routes_counts_exact=True,all_original_actions_video_commands_exact=True,
            OFF_baseline='original complete recorded M11 episode; not a new OFF replay',per_request_all_KV_comparison=False,
            baseline_reused=True,chunk_RNG_exact=True,reset_and_final_full_valid_KV_RNG_exact=True,
            extra_teacher_forwards=0,actual_o3_preserved=True,teacher_rows=904*3,
            target_accounting=closed,permissions=p['permissions'],performance_claim=False))
    finally:
        if collector:collector.close()
        online.close();source.save_async=old_save

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);args=a.parse_args();p=load(args.job)
    require(os.getenv('CUDA_VISIBLE_DEVICES')=='0' and (args.job/'RUN-STARTED.json').exists(),'bounded supervisor required')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('qualification work cap')))
    signal.alarm(p['max_run_wall_sec']-p['cleanup_reserve_sec']-30)
    original_save=None
    try:
        from tf_usable_quality_v1.legacy import server_helpers
        helpers=server_helpers()
        from wan_va import wan_va_server as source
        original_save=source.save_async
        def serve(s,rank,host,port):
            require(rank==0,'one GPU worker')
            with torch.no_grad():run(s,source,helpers,args.job,p)
        source.run_async_server_mode=serve
        sys.argv=[str(__file__),'--config-name','libero','--port',str(p['port']),'--save_root',str(args.job/'unused')]
        source.init_logger();source.main()
    except BaseException as exc:
        write(args.job/'WORKER-FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        signal.alarm(0)
        if original_save:original_save.__globals__['executor'].shutdown(wait=True)
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':main()

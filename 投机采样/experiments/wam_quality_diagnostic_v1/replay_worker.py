"""Explicit GPU worker: ten exact baseline chunks, then ten controlled replays.

Importing this module performs no replay and never initializes CUDA. The original
dense B14 source first has to reproduce the recorded O3 actions/video/route. A
mismatch stops the job; no tolerance or synthetic substitute is available.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types

OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN.parent))
from wam_quality_diagnostic_v1.replay_protocol import ROOT,FRESH,load,read,write,require

def run_server(server,p,job,source,helpers):
    import numpy as np
    import torch
    from reset_text_bank_v1.bank import TextBank,install as install_bank
    from target_trace_draft_finetune_v1.adapter import load_checkpoint
    validate_server,RamAccounting,modules=helpers
    validate_server(server)
    require(server._p0_action_draft_mode=='verifier_k2_b2_bonus','fixed K2/B2 bonus mode required')
    require(server._p0_state_teacher is None and server._p0_trace is None,'no trace/teacher mode mutation')
    rng=torch.load(p['input_manifest']['actual_initial_target_rng']['path'],map_location='cpu',weights_only=True)
    require(torch.equal(torch.get_rng_state(),rng['CPU']) and torch.equal(torch.cuda.get_rng_state(server.device),rng['CUDA']),
        'new target initialization differs from actual original target RNG')
    channels=tuple(int(x) for x in server.job_config.used_action_channel_ids)
    require(channels==tuple(range(7)),'original7 action channels required')
    cp=p['input_manifest']['current_checkpoint']
    predictor=load_checkpoint(cp['checkpoint'],architecture='transformer64',device=server.device,
        expected_channels=channels,expected_sha256=cp['checkpoint_sha256'])
    server._p0_action_draft_b2_predictor=predictor
    server._p0_action_draft_b2_checkpoint=predictor.checkpoint_path
    server.action_per_frame=int(server.job_config.action_per_frame)
    proxy=modules['selection'].install(server,'prefix_then_cost')
    # Deliberately use the original dense RAW body. No online O3 patch is
    # installed; exact reproduction of the real O3 recording is the first gate.
    bank=TextBank(p['reference_plan']['bank']);install_bank(server,bank)
    def no_text(*args,**kwargs):raise RuntimeError('validated bank only; unexpected T5 encoding')
    server._get_t5_prompt_embeds=types.MethodType(no_text,server)
    accounting=RamAccounting(server,cap=p['total_target_forward_cap'])
    started=time.monotonic();active=dict(pass_index=0,chunk=-1,tree_index=0,extra=0,pending=None)
    chosen={(x['chunk'],x['tree_index']):x for x in p['query_selection']}
    numeric=[];chunk_rows=[];action_wall=[]
    original_candidate=server._target_action_candidate_directions
    original_score=proxy.original.score_k2_b2_tree
    original_action=server._infer_action_verifier_k2_b2
    source_counter=read(FRESH/'episodes/FINETUNE_TF64-t7-i24/COUNTER.json')
    expected_trees={i:[x for x in source_counter['acceptance'] if x['context']['rollout_chunk']==i] for i in range(10)}
    source_arrays=p['input_manifest']['chunks'][:10]
    def guard():
        require(time.monotonic()-started<p['active_wall_sec'],'active replay deadline reached')
        require(not (job/'STOP_REQUESTED').exists(),'STOP_REQUESTED')
        require(active['extra']<=p['extra_target_forward_cap'] and len(numeric)<=p['packed_query_cap'],'query budget exceeded')
    def snapshot():
        # Valid slots only, one query at a time in host RAM. Never save 30-layer
        # KV to disk, never inspect undefined unmasked cache allocation.
        result=[]
        for block in server.transformer.blocks:
            c=block.attn1.attn_caches[server.cache_name];slots=c['mask'].nonzero(as_tuple=False).flatten()
            result.append(dict(mask=c['mask'].detach().cpu().clone(),id=c['id'].detach().cpu().clone(),
                is_pred=c['is_pred'].detach().cpu().clone(),slots=slots.detach().cpu(),
                k=c['k'][:,slots].detach().cpu().clone(),v=c['v'][:,slots].detach().cpu().clone()))
        return result
    def unchanged(before):
        for block,b in zip(server.transformer.blocks,before):
            c=block.attn1.attn_caches[server.cache_name]
            require(all(torch.equal(c[k].detach().cpu(),b[k]) for k in ('mask','id','is_pred')),'counterfactual mutated KV metadata')
            slots=b['slots'].to(c['mask'].device)
            require(torch.equal(c['k'][:,slots].detach().cpu(),b['k']) and torch.equal(c['v'][:,slots].detach().cpu(),b['v']),
                    'counterfactual mutated valid KV values')
    def numbers(a,b):
        d=(a.float()-b.float()).abs()
        return dict(bitwise=torch.equal(a,b),max_abs=float(d.max()),mean_abs=float(d.mean()),different_elements=int(torch.count_nonzero(d)))
    def candidates(_s,states,times,frame):
        guard();index=active['tree_index'];active['tree_index']+=1
        base=original_candidate(states,times,frame)
        key=(active['chunk'],index)
        if active['pass_index']!=1 or key not in chosen:return base
        require(active['pending'] is None,'query score not consumed')
        require(active['extra']+8<=p['extra_target_forward_cap'],'reserve all8 comparison forwards before execution')
        before=snapshot();cr=torch.get_rng_state().clone();gr=torch.cuda.get_rng_state(server.device).clone()
        active['extra']+=1
        packed=original_candidate(states,times,frame)
        require(torch.equal(base,packed),'same-input dense B14 repeat is not bitwise deterministic')
        seq=[]
        for i in range(7):
            guard();active['extra']+=1
            seq.append(original_candidate(states[i:i+1],times[i:i+1],frame))
        sequential=torch.cat(seq,dim=0)
        unchanged(before);del before
        require(torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(server.device)),
                'counterfactual changed CPU/CUDA RNG')
        active['pending']=dict(key=key,selection=chosen[key],packed=packed,sequential=sequential,
            states=states.detach().cpu().clone(),times=torch.as_tensor(times,dtype=torch.float32,device='cpu').clone(),comparison=numbers(packed,sequential))
        return base
    server._target_action_candidate_directions=types.MethodType(candidates,server)
    def score(tree,target):
        result=original_score(tree,target)
        q=active['pending']
        if q is None:return result
        seq_compact=torch.stack([server._p0_action_compact(x) for x in q['sequential']])
        other=original_score(tree,seq_compact)
        a=modules['selection'].choose(proxy.original,result,p['reference_plan']['threshold'])
        b=modules['selection'].choose(proxy.original,other,p['reference_plan']['threshold'])
        depth=modules['selection'].depth
        da=depth(a.first_cost,a.second_cost,p['reference_plan']['threshold']);db=depth(b.first_cost,b.second_cost,p['reference_plan']['threshold'])
        chunk,index=q['key'];record=expected_trees[chunk][index]
        require((a.parent,a.child,da)==(record['actual_parent'],record['actual_child'],record['actual_depth']),
                'probe baseline route differs from qualified actual route')
        ident=f'chunk{chunk:02d}-tree{index:02d}'
        teacher=[]
        for node in range(3):
            state=tree.root_state if node==0 else tree.first_states[node-1]
            previous=tree.root_previous_direction if node==0 else tree.first_directions[node-1]
            state_cpu=state.detach().cpu().clone()
            require(torch.equal(state_cpu,q['states'][node][list(channels)].reshape_as(state_cpu)),
                    'actual target candidate state differs from recorded predictor input')
            teacher.append(dict(node=node,state=state_cpu,previous=previous.detach().cpu().clone(),
                target_packed=target[node].detach().cpu().clone(),target_sequential=seq_compact[node].detach().cpu().clone(),
                sigma=tree.sigma_path[0 if node==0 else 1],next_sigma=tree.sigma_path[1 if node==0 else 2],
                timestep=tree.target_timesteps[0 if node==0 else 1],
                selected_branch=node==0 or node-1==a.parent,
                draft_edge_committed=(node==0 and da>=1) or (node>0 and node-1==a.parent and da==2),
                teacher_label='actual_same_query_target_cfg_direction',data_role='DEV_DIAGNOSTIC_NOT_FIT'))
        torch.save(dict(schema='actual-current-TF-DEV-query-numeric-replay/v1',context=record['context'],
            checkpoint_sha256=cp['checkpoint_sha256'],candidate_states=q['states'],candidate_timesteps=q['times'],
            packed_target=q['packed'].detach().cpu(),sequential_target=q['sequential'].detach().cpu(),teacher_rows=teacher,
            KV_saved=False,requires_original_observation_history=True,data_role='DEV_DIAGNOSTIC_NOT_TRAINING'),job/'queries'/f'{ident}.pt')
        numeric.append(dict(id=ident,chunk=chunk,tree_index=index,sigma=record['sigma'],selection=q['selection'],
            directions=q['comparison'],packed_first_costs=list(result.first_costs),
            packed_second_costs=[list(x) for x in result.second_costs],sequential_first_costs=list(other.first_costs),
            sequential_second_costs=[list(x) for x in other.second_costs],packed_route=[a.parent,a.child,da],
            sequential_route=[b.parent,b.child,db],route_or_depth_flip=(a.parent,a.child,da)!=(b.parent,b.child,db),
            depth_flip=da!=db,actual_KV_unchanged=True,actual_CPU_CUDA_RNG_unchanged=True,
            root_same_query_target_error=numbers(target[0],seq_compact[0])))
        active['pending']=None
        return result
    proxy.score_k2_b2_tree=score
    def action(_s,*args,**kwargs):
        torch.cuda.synchronize(server.device);begin=time.monotonic()
        try:return original_action(*args,**kwargs)
        finally:
            torch.cuda.synchronize(server.device)
            action_wall.append(dict(pass_index=active['pass_index'],chunk=active['chunk'],wall_sec=time.monotonic()-begin,
                includes_capture_and_probe_overhead=True,performance_claim=False))
    server._infer_action_verifier_k2_b2=types.MethodType(action,server)
    with np.load(p['input_manifest']['initial_observations']['path'],allow_pickle=False) as z:initial_obs={k:z[k] for k in z.files}
    prompt=source_arrays[0]['policy_context']['prompt']
    def drain():source.save_async.__globals__['executor'].submit(lambda:None).result(timeout=60)
    def actual_tensor(path):return torch.load(path,map_location='cpu',weights_only=True)
    for pass_index in (0,1):
        if pass_index==1:require((job/'BASELINE-REPRODUCTION.json').is_file(),'all10 complete baseline chunks must qualify before probes')
        active['pass_index']=pass_index
        server.save_root=str(job/f'pass{pass_index}')
        torch.set_rng_state(rng['CPU']);torch.cuda.set_rng_state(rng['CUDA'],server.device)
        # Metadata keeps the original actual episode identity; it never enters the model.
        context=dict(source_arrays[0]['policy_context']);context.pop('rollout_chunk',None)
        context.pop('env_steps_before',None);context.pop('simulator_timestep_before',None)
        server.infer(dict(reset=True,prompt=prompt,p0_trace_context=context))
        require(server.frame_st_id==0 and torch.equal(torch.get_rng_state(),rng['CPU'])
                and torch.equal(torch.cuda.get_rng_state(server.device),rng['CUDA']),'reset does not restore original RNG/empty frame')
        obs=initial_obs
        for item in source_arrays:
            guard();chunk=item['chunk'];active.update(chunk=chunk,tree_index=0)
            before=len(proxy.records);ph=dict(accounting.phase);calls=accounting.attempted
            require(server.frame_st_id==item['frame_st_id'],'original policy frame desynchronized')
            result=server.infer(dict(obs=obs,prompt=prompt,p0_trace_context=item['policy_context']))
            drain()
            folder=Path(server.exp_save_root)
            for prefix in ('actions','latents'):
                name=f'{prefix}_{item["frame_st_id"]}.pt'
                actual=actual_tensor(folder/name);expected=actual_tensor(item['files'][name]['path'])
                require(torch.equal(actual,expected) and bool(torch.isfinite(actual).all()),
                        f'original {prefix} bitwise reproduction failed pass={pass_index} chunk={chunk}')
            observed=proxy.records[before:]
            require(observed==expected_trees[chunk],'full original acceptance path/cost/context reproduction failed')
            require(active['pending'] is None,'selected query score did not close')
            obs_path=item['files'][f'obs_data_{item["frame_st_id"]}.pt']['path']
            frames=torch.load(obs_path,map_location='cpu',weights_only=False)
            require(isinstance(frames,list) and frames,'actual recorded observation frame list required')
            server.infer(dict(obs=frames,compute_kv_cache=True,imagine=False,state=result['action'],p0_trace_context=item['cache_context']))
            drain();require(server.frame_st_id==item['cache']['server_frame_st_id_after'],'complete original cache handoff frame differs')
            phase={k:int(v-ph.get(k,0)) for k,v in accounting.phase.items() if v!=ph.get(k,0)}
            chunk_rows.append(dict(pass_index=pass_index,chunk=chunk,action_video_bitwise=True,acceptance_path_bitwise=True,
                complete_policy_and_cache=True,target_calls=accounting.attempted-calls,phase_counters=phase,
                tree_fallback_target_scheduler_steps=sum(x['actual_depth']<2 for x in observed),
                tree_fallback_semantics='one already-verified target scheduler transition per non-full tree, not an extra target forward',
                all_action_wall_includes_capture=True,missing_round_timing=['T_draft','pack','verify','accept','extra'],
                performance_claim=False))
            write(job/f'PASS{pass_index}-CHUNK{chunk:02d}.json',chunk_rows[-1])
            obs=frames[-1]
        if pass_index==0:
            write(job/'BASELINE-REPRODUCTION.json',dict(status='PASS_10_COMPLETE_ORIGINAL_ACTION_VIDEO_PATH_CHUNKS',
                complete_policy_cache_chunks=10,RAW_dense_B14_matches_recorded_O3=True,numerical_comparisons_started=False,
                original_failed_status_unchanged=True,no_EGL=True,no_new_sampling=True))
    require(len(numeric)==32 and active['extra']==256 and len(chunk_rows)==20,'complete fixed replay/query coverage required')
    closed=accounting.close()
    write(job/'WORKER-RESULT.json',dict(status='COMPLETE_Q02_CONTROLLED_NUMERIC_REPLAY_PENDING_OWNED_CLEANUP',
        same_query_qualified=True,complete_policy_cache_chunks=20,unique_original_chunks=10,
        packed_tree_queries=32,real_candidate_queries=224,three_predictor_query_rows=96,
        extra_target_forwards=256,total_target_forwards=closed['target_intents'],numeric=numeric,
        chunk_rows=chunk_rows,complete_action_function_wall_diagnostic=action_wall,
        direction_mismatch_queries=sum(not x['directions']['bitwise'] for x in numeric),
        route_or_depth_flips=sum(x['route_or_depth_flip'] for x in numeric),depth_flips=sum(x['depth_flip'] for x in numeric),
        D01_admitted=False,root_cause='UNKNOWN',no_robot_success_counterfactual=True,
        observation_sequence_is_fixed_recording=True,not_closed_loop_quality=True,performance_claim=False,
        all_per_query_KV_and_RNG_unchanged=True,per_query_KV_saved=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);parser.add_argument('--execute-reviewed',action='store_true');a=parser.parse_args()
    require(a.execute_reviewed and os.getenv('CUDA_VISIBLE_DEVICES')=='0','explicit reviewed GPU worker required')
    job=a.job.resolve();p=load(job)
    require((job/'RUN-STARTED.json').is_file(),'independent bounded outer supervisor required')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    def stop(*_):raise TimeoutError('Q02 worker bounded active deadline')
    signal.signal(signal.SIGALRM,stop);signal.alarm(p['active_wall_sec'])
    try:
        from target_trace_draft_fresh_dev_v1.legacy import server_helpers
        helpers=server_helpers()
        from wan_va import wan_va_server as source
        require(Path(source.__file__).resolve()==ROOT/'投机采样/wam-p0-worktree/wan_va/wan_va_server.py','original WAM source origin required')
        def serve(s,local_rank,host,port):
            require(local_rank==0,'single actual target worker only')
            return run_server(s,p,job,source,helpers)
        source.run_async_server_mode=serve
        sys.argv=[str(Path(__file__)),'--config-name','libero','--port','19617','--save_root',str(job/'outputs')]
        source.init_logger();source.main()
        source.save_async.__globals__['executor'].shutdown(wait=True)
    except BaseException as e:
        if not (job/'FAILED.json').exists():write(job/'FAILED.json',dict(status='STOP_Q02_NO_SAME_QUERY_CLAIM',error=repr(e),
            traceback=traceback.format_exc(),same_query_qualified=False,D01_admitted=False,root_cause='UNKNOWN'))
        raise
    finally:
        signal.alarm(0)
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':main()

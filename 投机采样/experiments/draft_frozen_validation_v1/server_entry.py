"""Persistent original WAM with frozen TF/MLP continuations for complete DEV."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import types

EXP=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(EXP))
from draft_frozen_validation_v1.protocol import ARCH,load,write
from tf_usable_quality_v1.legacy import server_helpers


def fingerprint(value):
    import torch
    if isinstance(value,torch.Tensor):
        x=value.detach().contiguous().cpu()
        return dict(shape=list(x.shape),dtype=str(x.dtype),
                    bytes_sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value,dict):
        return {str(k):fingerprint(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [fingerprint(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError('unsupported state snapshot field '+str(type(value)))


def snapshot(s,*,empty):
    import torch
    layers=[]
    for block in s.transformer.blocks:
        cache=block.attn1.attn_caches[s.cache_name]
        mask=cache['mask']
        if empty and (bool(mask.any()) or bool(cache['is_pred'].any()) or not bool((cache['id']==-1).all())):
            raise ValueError('full original reset did not empty every cache layer')
        layers.append(fingerprint(dict(mask=mask,id=cache['id'],is_pred=cache['is_pred'],
                                       k=cache['k'][:,mask],v=cache['v'][:,mask])))
    if empty and (s.frame_st_id!=0 or s.init_latent is not None or any(v is not None for v in s.streaming_vae.feat_cache)):
        raise ValueError('full original reset retained frame/VAE state')
    return dict(frame_st_id=int(s.frame_st_id),use_cfg=bool(s.use_cfg),KV=layers,
                init_latent=fingerprint(s.init_latent),VAE=fingerprint(s.streaming_vae.feat_cache),
                prompt=fingerprint(s.prompt_embeds),negative=fingerprint(s.negative_prompt_embeds),
                action_mask=fingerprint(s.action_mask),q01=fingerprint(s.actions_q01),q99=fingerprint(s.actions_q99),
                action_norm_method=s.action_norm_method,
                cpu_rng=fingerprint(torch.get_rng_state()),
                cuda_rng=fingerprint(torch.cuda.get_rng_state(s.device)))


def main():
    job=Path(os.environ['DRAFT_FROZEN_JOB']).resolve()
    p=load(job)
    arm=os.environ['DRAFT_FROZEN_ARM']
    if arm not in p['arms'] or os.getenv('CUDA_VISIBLE_DEVICES')!='0' or os.getenv('RESET_TEXT_BANK_ENABLED')!='1':
        raise ValueError('assigned local arm and explicit bank enable required')
    validate_server,RamAccounting,modules=server_helpers()
    from wan_va import wan_va_server as source
    from reset_text_bank_v1.bank import TextBank,install as install_bank
    from tf_usable_train_v1.gpu_actual_cal import actual_checkpoint_cal_gpu_parity as actual_tf_cal
    from tf_usable_mlp_continue_v1.gpu_actual_cal import actual_checkpoint_cal_gpu_parity as actual_mlp_cal
    for name in ('tf_usable_train_v1.gpu_actual_cal',
                 'tf_usable_mlp_continue_v1.gpu_actual_cal'):
        module=sys.modules[name]
        if str(Path(module.__file__).resolve()) not in p['source_sha256']:
            raise ImportError('actual candidate CAL helper source origin differs: '+name)
    if Path(source.__file__).resolve()!=Path(__file__).resolve().parents[2]/'wam-p0-worktree/wan_va/wan_va_server.py':
        raise ImportError('original WAM source origin differs')
    for module_name in ('configs','configs.va_libero_cfg','distributed.fsdp','distributed.util',
                        'modules.utils','modules.model','utils','utils.p0_k3_b2',
                        'utils.p0_trace','utils.p0_state_teacher'):
        module=sys.modules.get(module_name)
        if module is None or not hasattr(module,'__file__') or str(Path(module.__file__).resolve()) not in p['source_sha256']:
            raise ImportError('actual WAM model/config/helper source origin differs: '+module_name)

    def loaded_native_origin():
        native=sys.modules.get('_shared_torch_v1')
        if native is not None and str(Path(native.__file__).resolve()) not in p['source_sha256']:
            raise ImportError('loaded prefix native binary lacks frozen source provenance')
    original_run=source.run_async_server_mode

    def serve(s,local_rank,host,port):
        validate_server(s)
        channels=tuple(int(v) for v in s.job_config.used_action_channel_ids)
        if local_rank!=0 or channels!=tuple(range(7)) or s.job_config.action_per_frame!=4:
            raise ValueError('single-rank original seven-channel action contract required')
        mode='off' if arm=='T' else p['mode']
        if s._p0_action_draft_mode!=mode or s._p0_action_draft_threshold!=p['threshold']:
            raise ValueError('action mode/threshold differ from frozen DEV')
        # Capture target initialization BEFORE any small-model construction or
        # real-CAL execution; restore it after setup and before every episode.
        target_initial_cpu=source.torch.get_rng_state().clone()
        target_initial_cuda=source.torch.cuda.get_rng_state(s.device).clone()
        initial_rng=dict(cpu=fingerprint(target_initial_cpu),cuda=fingerprint(target_initial_cuda))
        source.torch.save(dict(CPU=target_initial_cpu,CUDA=target_initial_cuda),job/'arms'/arm/'TARGET-INITIAL-RNG.pt')
        s.action_per_frame=int(s.job_config.action_per_frame)
        proxy=online=parity=None
        if arm!='T':
            cal_fn=actual_mlp_cal if arm=='MLP_CONT' else actual_tf_cal
            predictor,parity=cal_fn(
                p['selected'][arm]['checkpoint'],ARCH[arm],data_path=p['real_CAL_data']['path'],
                device=s.device,expected_channels=channels,
                expected_sha256=p['selected'][arm]['checkpoint_sha256'],
                out_path=job/'arms'/arm/'FULL-REAL-CAL-PARITY.json')
            if (parity['rows']!=20237 or not parity['same_device_bitwise_exact']
                    or parity['synthetic_queries']!=0 or not parity['complete_CAL']
                    or parity['checkpoint_sha256']!=p['selected'][arm]['checkpoint_sha256']):
                raise ValueError('complete real CAL adapter parity did not pass')
            s._p0_action_draft_b2_predictor=predictor
            s._p0_action_draft_b2_checkpoint=predictor.checkpoint_path
            proxy=modules['selection'].install(s,p['selector'])
            modules['buffers'].prepare(s._infer_action_verifier_k2_b2,s._load_p0_shadow_api())
            online=modules['measured_online'].control(s,'O3',allow_cuda=True)
        bank=TextBank(p['bank'])
        install_bank(s,bank)
        def unexpected_text(_s,*args,**kwargs):
            raise RuntimeError('DEV reset must use validated final embeddings; unexpected T5 forward')
        s._get_t5_prompt_embeds=types.MethodType(unexpected_text,s)
        arm_budget=int(os.environ['DRAFT_FROZEN_ARM_BUDGET'])
        if not 0<arm_budget<=p['job_target_forward_cap']:
            raise ValueError('frozen remaining job target-forward cap required')
        accounting=RamAccounting(s,cap=arm_budget)
        original_reset,original_infer=s._reset,s.infer
        source.torch.set_rng_state(target_initial_cpu)
        source.torch.cuda.set_rng_state(target_initial_cuda,s.device)
        active=dict(key=None,row=None,request_start=0,forward_start=0,returned_start=0,tree_start=0,
                    phase_start={},bank_start=0,completed=set())
        gate=threading.RLock()

        def reset(_s,prompt=None):
            row=active['row']
            if row is None:
                raise ValueError('reset lacks complete DEV assignment')
            before_cpu=source.torch.get_rng_state().clone()
            before_cuda=source.torch.cuda.get_rng_state(s.device).clone()
            source.torch.cuda.synchronize(s.device)
            start=time.monotonic()
            result=original_reset(prompt=prompt)
            source.torch.cuda.synchronize(s.device)
            wall=time.monotonic()-start
            if (not source.torch.equal(before_cpu,source.torch.get_rng_state())
                    or not source.torch.equal(before_cuda,source.torch.cuda.get_rng_state(s.device))):
                raise ValueError('bank reset consumed target RNG')
            for text,value in ((prompt,s.prompt_embeds),('',s.negative_prompt_embeds)):
                if not source.torch.equal(value,bank.get(text,device=s.device,dtype=s.dtype,sequence_length=512)):
                    raise ValueError('actual cached reset embeddings differ')
            state=snapshot(s,empty=True)
            if state['cpu_rng']!=initial_rng['cpu'] or state['cuda_rng']!=initial_rng['cuda']:
                raise ValueError('reset RNG differs from target-only initialization')
            write(job/'episodes'/row['key']/'RESET.json',dict(status='PASS_FULL_RESET_BANK_AND_EMPTY_STATE',
                row=row,state=state,RNG_unchanged=True,reset_wall_sec=wall,performance_claim=False))
            return result
        s._reset=types.MethodType(reset,s)

        def infer(_s,request):
            with gate:
                context=request.get('p0_trace_context') or {}
                key=context.get('parallel_worker_id')
                if request.get('reset'):
                    rows=[r for r in p['cases'] if r['key']==key and r['arm']==arm]
                    if len(rows)!=1 or active['key'] is not None or key in active['completed']:
                        raise ValueError('one fresh reset after complete prior episode required')
                    row=rows[0]
                    if context.get('task_index')!=row['task'] or context.get('episode_index')!=row['init']:
                        raise ValueError('actual task/init differs from frozen DEV')
                    source.torch.set_rng_state(target_initial_cpu)
                    source.torch.cuda.set_rng_state(target_initial_cuda,s.device)
                    active.update(key=key,row=row,request_start=len(accounting.requests),
                        forward_start=accounting.attempted,returned_start=accounting.returned,
                        tree_start=0 if proxy is None else len(proxy.records),
                        phase_start=dict(accounting.phase),bank_start=bank.hits)
                    accounting.cap=min(arm_budget,accounting.attempted+p['per_episode_target_forward_cap'])
                    s.save_root=str(job/'episodes'/key/'server_output')
                elif (active['key'] is None or key!=active['key']
                      or context.get('task_index')!=active['row']['task'] or context.get('episode_index')!=active['row']['init']):
                    raise ValueError('policy/cache request does not belong to active full DEV episode')
                # The native server creates frame_st_id inside its first
                # _reset. Record the absent pre-reset field honestly; every
                # policy/cache request must still have an initialized frame.
                before_frame=(getattr(s,'frame_st_id',None) if request.get('reset')
                              else int(s.frame_st_id))
                result=original_infer(request)
                item=accounting.requests[-1]
                if item['trace_context']!=request.get('p0_trace_context'):
                    raise ValueError('RAM request sequence/context changed')
                names=[] if item['kind']=='reset' else ([f'actions_{before_frame}.pt',f'latents_{before_frame}.pt']
                    if item['kind']=='policy' else [f'obs_data_{before_frame}.pt'])
                item.update(server_frame_st_id_before=before_frame,server_frame_st_id_after=int(s.frame_st_id),
                            expected_saved_outputs=names)
                return result
        s.infer=types.MethodType(infer,s)

        def monitor():
            try:
                while True:
                    time.sleep(.2)
                    with gate:
                        key=active['key']
                        if key is None:
                            continue
                        out=job/'episodes'/key
                        if not (out/'FINISH_REQUESTED').exists():
                            continue
                        executor=source.save_async.__globals__['executor']
                        if executor._max_workers!=1:
                            raise ValueError('persistent async drain requires original single-worker executor')
                        executor.submit(lambda:None).result(timeout=90)
                        source.torch.cuda.synchronize(s.device)
                        loaded_native_origin()
                        if (accounting.failed or accounting.active or accounting.attempted!=accounting.returned
                                or (proxy is not None and proxy.pending is not None)):
                            raise ValueError('unfinished target/selection accounting')
                        phases={k:v-active['phase_start'].get(k,0) for k,v in accounting.phase.items()}
                        acceptance=[] if proxy is None else proxy.records[active['tree_start']:]
                        if proxy is not None and len(acceptance)!=phases.get('tree_count',0):
                            raise ValueError('acceptance tree coverage differs from actual counter')
                        write(out/'FINAL-SERVER-STATE.json',snapshot(s,empty=False))
                        write(out/'COUNTER.json',dict(arm=arm,key=key,
                            target_forwards=accounting.attempted-active['forward_start'],
                            returned_forwards=accounting.returned-active['returned_start'],
                            requests=accounting.requests[active['request_start']:],phase_counters=phases,
                            acceptance=acceptance,bank_hits=bank.hits-active['bank_start'],
                            persistent_model=True,performance_claim=False))
                        active['completed'].add(key)
                        active['key']=None
                        active['row']=None
                        write(out/'FINISH_READY.json',dict(status='PASS',async_drained=True))
            except BaseException as exc:
                write(job/f'{arm}-MONITOR-FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        threading.Thread(target=monitor,daemon=True).start()
        source.torch.cuda.synchronize(s.device)
        loaded_native_origin()
        if (fingerprint(source.torch.get_rng_state())!=initial_rng['cpu']
                or fingerprint(source.torch.cuda.get_rng_state(s.device))!=initial_rng['cuda']):
            raise ValueError('setup RNG restoration failed before READY')
        write(job/f'{arm}-READY.json',dict(status='READY',arm=arm,pid=os.getpid(),port=port,
            target_initial_rng=initial_rng,ready_rng=initial_rng,
            target_RNG_captured_before_draft_and_CAL=True,setup_RNG_restored=True,
            model_persistent=True,performance_claim=False,bank_explicitly_enabled=True,
            full_real_CAL_rows=0 if parity is None else parity['rows'],
            predictor_checkpoint=None if arm=='T' else p['selected'][arm]['checkpoint'],
            selector='none' if arm=='T' else p['selector'],O1_O2_O3=arm!='T',synthetic_queries=0,
            arm_target_budget=arm_budget,per_episode_target_cap=p['per_episode_target_forward_cap']))
        return original_run(s,local_rank,host,port)

    source.run_async_server_mode=serve
    source.init_logger()
    source.main()


if __name__=='__main__':
    main()

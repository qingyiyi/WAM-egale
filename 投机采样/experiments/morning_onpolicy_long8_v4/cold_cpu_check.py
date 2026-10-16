"""Actual public VA_Server constructor -> install -> original infer(reset).

Only external model/resource factories are CPU tiny fixtures. Constructor,
reset, installed candidate and prediction methods are the real source methods.
No action_per_frame assignment/deletion on the server is used to fake coldness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
from paths import ROOT, OWN, BASE, load

CANDIDATE=Path('/var/tmp/cxz-wam-quality-20261014/p0-draft-expanded-data-v1-20261014/B_MIXED_HISTORICAL_C-seed20260922/checkpoint.pt')
CANDIDATE_SHA='03e1354fc6bf4e9838607590466e3de70efc8b10a959f96665502bae68d3de49'
CURRENT=Path('/var/tmp/cxz-wam-quality-20261014/p0-draft-root-retention-v1-20261014/dataset.pt')
CURRENT_SHA='51c2947fa9af27d3b65beafd03a4c6a4d542907b44819f82d50b3a95d24ee044'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or os.sched_getaffinity(0)!={8}:
        raise ValueError('CPU8 single core CUDA hidden')
    start=time.monotonic()
    f=load('test_cpu',BASE/'test_cpu.py');t=f.torch;t.set_num_threads(1);t.set_num_interop_threads(1)
    from production_fixture import loaded_model
    from candidate_predictor import install
    old=load('coldfix_original_candidate_reference',OWN.parent/'overnight_onpolicy_v1/candidate_predictor.py')
    source=f.public_server
    ctor=source.VA_Server.__init__;reset=source.VA_Server._reset;infer=source.VA_Server.infer
    if hashlib.sha256(CURRENT.read_bytes()).hexdigest()!=CURRENT_SHA:
        raise ValueError('current TRAIN fixture changed')
    cache=t.load(CURRENT,map_location='cpu',weights_only=False)
    model=loaded_model(dtype=t.bfloat16)
    vae=t.nn.Linear(1,1);text=t.nn.Linear(1,1)
    vae.config=SimpleNamespace(latents_mean=[0.]*48,latents_std=[1.]*48,z_dim=48)
    class TinyStreamingResource:
        def __init__(self,vae):self.vae=vae;self.feat_cache=[]
        def clear_cache(self):self.feat_cache=[]
        def encode_chunk(self,video):
            return t.zeros(video.shape[0],96,video.shape[2],video.shape[3]//16,video.shape[4]//16,dtype=video.dtype)
    plan=dict(new_checkpoint=str(CANDIDATE),new_checkpoint_sha256=CANDIDATE_SHA)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='coldfix-actual-ctor-') as tmp:
        cfg=SimpleNamespace(save_root=tmp,param_dtype=t.bfloat16,local_rank=0,
            enable_offload=True,vae_device='cpu',infer_mode='i2va',world_size=1,
            snr_shift=3.,action_snr_shift=3.,wan22_pretrained_model_name_or_path='CPU-resource-fixture',
            env_type='libero',guidance_scale=5,action_guidance_scale=1,
            used_action_channel_ids=list(range(7)),frame_chunk_size=4,action_per_frame=4,
            height=32,width=16,obs_cam_keys=['a','b'],patch_size=(1,2,2),attn_window=30,
            num_inference_steps=20,action_num_inference_steps=50,video_exec_step=-1,
            action_dim=30,norm_stat={'q01':[-1.]*30,'q99':[1.]*30},action_norm_method='quantiles')
        env=dict(WAM_P0_ACTION_DRAFT_MODE='verifier_k2_b2_bonus',
            WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=str(CANDIDATE),WAM_P0_ACTION_DRAFT_SELECTOR='global',
            WAM_P0_ACTION_DRAFT_THRESHOLD='0.007839680526450805',WAM_P0_PHASE_PROFILE='0',
            WAM_P0_SHADOW_CANDIDATE_DEPTH='0',WAM_P0_STATE_TEACHER_DIR='',WAM_P0_TRACE_DIR='',
            LINGBOT_CACHE_MODE='original',LINGBOT_CACHE_VALIDATE='0',LINGBOT_CHUNK_TIMING='0',
            LINGBOT_NSYS_PROFILE='off',LINGBOT_ATTN_MODE='torch',LINGBOT_DISABLE_FSDP='1',WAM_VAE_DEVICE='cpu')
        with patch.dict(os.environ,env), \
             patch.object(source,'load_vae',return_value=vae), \
             patch.object(source,'WanVAEStreamingWrapper',side_effect=TinyStreamingResource), \
             patch.object(source,'load_tokenizer',return_value=SimpleNamespace()), \
             patch.object(source,'load_text_encoder',return_value=text), \
             patch.object(source,'load_transformer',return_value=model), \
             patch.object(source,'_configure_model',side_effect=lambda model,**kwargs:model), \
             patch.object(t.cuda,'empty_cache',return_value=None):
            # The real ctor still chooses its declared CUDA device object,
            # without allocation; CPU placement is explicit fixture scope.
            cold=source.VA_Server(cfg)
            assert cold.__class__.__init__ is ctor and not hasattr(cold,'action_per_frame')
            cold.device=t.device('cpu')
            torch_rng=t.get_rng_state().clone()
            candidate=install(cold,plan,allow_device_transfer=False)
            assert not hasattr(cold,'action_per_frame') and t.equal(torch_rng,t.get_rng_state())
            assert candidate.layout_status()['status']=='AWAITING_ORIGINAL_RESET_NO_GEOMETRY_CLAIM'
            q=t.zeros(112,dtype=t.bfloat16)
            try:candidate.predict(q,q,.5,.4,500.);raise AssertionError('prediction before original reset accepted')
            except RuntimeError as e:assert str(e)=='candidate prediction requires original server reset first'
            assert not hasattr(cold,'action_per_frame') and t.equal(torch_rng,t.get_rng_state())
            assert cold._reset.__func__ is reset and cold.infer.__func__ is infer
            cold.infer(dict(reset=True,prompt=None))
            assert cold.action_per_frame==cfg.action_per_frame==4
            assert candidate.validate_layout()==dict(status='VALIDATED_ORIGINAL_RESET_GEOMETRY',action_per_frame=4,frame_chunk_size=4)
            assert all(b.attn1.attn_caches['pos']['k'].dtype==t.bfloat16 for b in model.blocks)
            checks+=['real_original_ctor_has_no_action_per_frame','cold_install_does_not_create_guess_geometry',
                'premature_predict_rejected_without_input_or_RNG_mutation','real_public_infer_reset_creates_geometry',
                'all30_original_reset_BF16_cache_layout']
            warm=source.VA_Server(cfg);warm.device=t.device('cpu')
            warm.infer(dict(reset=True,prompt=None))
            torch_rng=t.get_rng_state().clone()
            reference=old.install(warm,plan,allow_device_transfer=False)
            assert t.equal(torch_rng,t.get_rng_state())
            comparisons=[]
            for row in cache['train_rows'][:32]:
                state=row['state'].bfloat16();previous=row['previous'].bfloat16()
                before=(state.clone(),previous.clone());rng=t.get_rng_state().clone()
                got=candidate.predict(state,previous,row['sigma'],row['next_sigma'],row['timestep'])
                gold=reference.predict(state,previous,row['sigma'],row['next_sigma'],row['timestep'])
                raw=candidate.predictor.predict(state,previous,row['sigma'],row['next_sigma'],row['timestep'])
                assert got.shape==(2,112) and got.dtype==t.bfloat16 and got.device.type=='cpu' and not got.requires_grad
                assert t.equal(got.contiguous().view(t.uint8),gold.contiguous().view(t.uint8))
                assert t.equal(got.contiguous().view(t.uint8),raw.contiguous().view(t.uint8))
                assert t.equal(state,before[0]) and t.equal(previous,before[1]) and t.equal(rng,t.get_rng_state())
                comparisons.append(hashlib.sha256(got.contiguous().view(t.uint8).numpy().tobytes()).hexdigest())
            checks+=['32_actual_TRAIN_queries_cold_vs_warm_vs_original_predict_BF16_bytes_exact',
                'original_CPU_math_norm_heads_and_RNG_unchanged']
            # Invalid actual geometry is created by the original reset, not a
            # private field assignment, and rejected before candidate math.
            cfg.action_per_frame=8;cold.infer(dict(reset=True,prompt=None))
            assert cold.action_per_frame==8
            rng=t.get_rng_state().clone()
            try:candidate.predict(q,q,.5,.4,500.);raise AssertionError('invalid real reset geometry accepted')
            except ValueError as e:assert str(e)=='original reset must establish frozen K2/B2 compact geometry'
            assert t.equal(rng,t.get_rng_state())
            cfg.action_per_frame=4;cold.infer(dict(reset=True,prompt=None))
            assert cold.action_per_frame==4 and cold._reset.__func__ is reset
            got=candidate.predict(state,previous,row['sigma'],row['next_sigma'],row['timestep'])
            assert t.equal(got.contiguous().view(t.uint8),gold.contiguous().view(t.uint8))
            checks+=['invalid_original_reset_layout_rejected_before_math','repeat_original_reset_then_predict_exact']
            # Exercise the real public first-policy path after repeat original
            # resets, including original observation preparation/normalization.
            import numpy as np
            from collections import Counter
            from selection import install as install_selection
            from measured_online import control
            from collector import Collector
            import buffers
            proxy=install_selection(cold,'prefix_then_cost')
            buffers.prepare(cold._infer_action_verifier_k2_b2,cold._load_p0_shadow_api())
            context=dict(campaign_id='CPU-cold-public',parallel_worker_id='cpu',episode_attempt_id='cold-reset-first',
                task_index=0,episode_index=17,rollout_chunk=0)
            request=dict(obs=[dict(a=np.zeros((32,16,3),dtype=np.uint8),b=np.zeros((32,16,3),dtype=np.uint8))],p0_trace_context=context)
            public=[]
            for enabled in (False,True):
                cold._p0_state_teacher=None;cold.infer(dict(reset=True,prompt=None))
                cold.prompt_embeds=t.ones(1,3,8,dtype=t.bfloat16);cold.negative_prompt_embeds=t.zeros(1,3,8,dtype=t.bfloat16)
                proxy.records.clear();proxy.pending=None
                counter=Counter();cold._phase_profile_counter=lambda name,value=1:counter.update({name:value})
                collector=None
                if enabled:
                    collector=Collector(Path(tmp)/'public-teacher',dict(task=0,init=17,split='train',cpu_fixture=True,threshold=.007839680526450805))
                    cold._p0_state_teacher=collector
                t.manual_seed(7717)
                controller=control(cold,'O3',allow_cuda=False,cpu_reference_enabled=True)
                try:
                    result=cold.infer(request)
                    action=t.as_tensor(result['action']).clone()
                    assert action.shape==(7,4,4) and t.isfinite(action).all()
                    assert counter['committed_scheduler_update_count']==50 and len(proxy.records)==counter['tree_count']
                    assert controller.last_clean and controller.control is None
                    if enabled:assert not collector.errors and collector.total==counter['tree_count']
                    public.append(dict(action=action,cache=f.cache_state(model),phase=dict(counter),
                        routes=list(proxy.records),rng=t.get_rng_state().clone()))
                finally:controller.close()
            a,b=public
            assert t.equal(a['action'].contiguous().view(t.uint8),b['action'].contiguous().view(t.uint8)) and f.same_state(a['cache'],b['cache'])
            assert a['phase']==b['phase'] and a['routes']==b['routes'] and t.equal(a['rng'],b['rng'])
            source.save_async.__globals__['executor'].shutdown(wait=True)
            public_policy=dict(complete_first_policy_chunks=2,action_shape=[7,4,4],action_dtype=str(a['action'].dtype),
                committed_updates_per_chunk=50,teacher_OFF_ON_output_routes_phase_all30_KV_RNG_exact=True,
                actual_public_infer_and_original_encode_obs=True,
                tiny_VAE_resource_outputs_are_not_real_camera_posterior=True)
            checks+=['two_real_public_reset_first_policy_full50_chunks_teacher_OFF_ON_all30_RNG_exact']
    assert not t.cuda.is_initialized() and not f.CUDA_ATTEMPTS
    out=dict(status='PASS_REAL_COLD_CTOR_INSTALL_ORIGINAL_RESET_PREDICT',
        checks=checks,actual_original_ctor_file=source.__file__,constructor_not_new_or_already_reset_fixture=True,
        external_factories_are_tiny_CPU_resources=True,explicit_CPU_device_fixture_placement=True,
        server_geometry_never_privately_assigned_or_deleted=True,original_reset_and_infer_methods_unchanged=True,
        current_TRAIN32_output_sha256=comparisons,candidate_sha256=CANDIDATE_SHA,
        first_public_policy=public_policy,
        CPU_affinity=[8],CUDA_initialized=False,CUDA_attempts=[],GPU_started=False,
        new_native_four_full_OFF_ON_chunks_required=True,previous_native_oracles_not_current_proof=True,
        elapsed_seconds=time.monotonic()-start)
    with args.out.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=out['status'],checks=len(checks),seconds=out['elapsed_seconds'])))


if __name__=='__main__':main()

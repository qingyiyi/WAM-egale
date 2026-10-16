"""CPU-only regressions importing real public Wan/VA_Server interfaces.

Tiny randomly initialized CPU weights/history are interface fixtures, not
the trained target, LIBERO episodes or GPU numerical/performance acceptance.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest as U
from unittest.mock import patch

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('Set CUDA_VISIBLE_DEVICES="" for this CPU-only test')
from contract import WORKTREE, MODEL_SOURCE, MODEL_SHA256, ROWS, Budget, digest, job_path, cpu_job_metadata, enabled_from_env, read_proposal
sys.path.insert(0, str(WORKTREE))
import torch

CUDA_ATTEMPTS = []
def forbid_cuda(*args, **kwargs):
    CUDA_ATTEMPTS.append('forbidden CUDA initialization')
    raise AssertionError('CPU task must not initialize CUDA')
torch.cuda._lazy_init = forbid_cuda
torch.cuda.init = forbid_cuda
torch.set_num_threads(2)
from wan_va.modules.model import WanAttention, WanTransformer3DModel, custom_sdpa
from wan_va import wan_va_server as public_server
from adapter import SingleWorkerCHC, LeaseError
from capture import capture_queries


def load_snapshot():
    path = WORKTREE.parent / 'experiments/fixed_load_cost/snapshot.py'
    spec = importlib.util.spec_from_file_location('chc_real_snapshot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SNAPSHOT = load_snapshot()
ORIGINAL_CLASS_FORWARD = WanAttention.forward


def model(layers=1, dtype=torch.float32, history=6, capacity=48):
    torch.random.default_generator.manual_seed(417)
    m = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=4,
        action_dim=30, text_dim=8, freq_dim=4, ffn_dim=16, num_layers=layers).to(dtype=dtype).eval()
    for block in m.blocks:
        a = block.attn1
        a.init_kv_cache('pos', capacity, 2, 4, 'cpu', dtype, 2)
        c = a.attn_caches['pos']
        c['k'].fill_(float('nan')); c['v'].fill_(float('nan'))
        slots = torch.arange(history) * 2 + 1 if history * 2 <= capacity else torch.arange(history)
        c['mask'][slots] = True
        c['id'][slots] = torch.arange(history)
        c['k'][:, slots] = torch.randn(2, history, 2, 4, dtype=dtype)
        c['v'][:, slots] = torch.randn(2, history, 2, 4, dtype=dtype)
    return m


def inputs(index=0, batch=14, dtype=torch.float32):
    return dict(noisy_latents=torch.randn(batch, 30, 4, 4, 1, dtype=dtype) + index * .01,
                text_emb=torch.randn(batch, 3, 8, dtype=dtype),
                grid_id=torch.zeros(batch, 3, 16), timesteps=torch.full((batch, 4), 50. - index))


def candidate(m, inp, rows):
    return m(inp, update_cache=0, cache_name='pos', action_mode=True, candidate_cache_rows=rows)


def cache_state(m):
    return [SNAPSHOT.capture_cache(b.attn1.attn_caches['pos']) for b in m.blocks]


def same_state(a, b):
    return len(a) == len(b) and all(x.keys() == y.keys() and all(
        torch.equal(x[k], y[k]) if isinstance(x[k], torch.Tensor) else x[k] == y[k]
        for k in x) for x, y in zip(a, b))


def reset_server(m, tmp):
    s = public_server.VA_Server.__new__(public_server.VA_Server)
    s.device = torch.device('cpu'); s.vae_device = torch.device('cpu'); s.dtype = torch.float32
    s.cache_name = 'pos'; s.env_type = 'libero'; s.save_root = tmp
    s._phase_profile_enabled = False; s._p0_shadow_api = None
    s._p0_action_draft_b2_predictor = None; s._p0_action_draft_mode = 'off'
    s._p0_action_draft_b2_checkpoint = read_proposal()['frozen_A_off']
    s.transformer = m; s.streaming_vae = NS(clear_cache=lambda: None)
    s.job_config = NS(guidance_scale=5, action_guidance_scale=1, used_action_channel_ids=list(range(7)),
        frame_chunk_size=4, action_per_frame=4, height=128, width=128, obs_cam_keys=['a','b'], patch_size=(1,2,2),
        attn_window=30, action_dim=30, norm_stat={'q01':[-1.]*30,'q99':[1.]*30}, action_norm_method='quantiles')
    with patch.object(torch.cuda, 'empty_cache', return_value=None):
        s.infer({'reset': True, 'prompt': None})
    s.prompt_embeds = torch.ones(1, 3, 8); s.negative_prompt_embeds = torch.zeros(1, 3, 8)
    return s


class Boundaries(U.TestCase):
    def test_already_modified_backend_refused_without_patches(self):
        m=model();a=m.blocks[0].attn1;original=a.forward
        a.attn_op=lambda q,k,v:custom_sdpa(q,k,v)
        with self.assertRaisesRegex(LeaseError,'original torch SDPA backend'):
            SingleWorkerCHC(m,enabled=True,expected_layers=1).install()
        self.assertEqual(a.forward,original);self.assertNotIn('forward',m.__dict__)

    def test_nonCPU_input_refused_before_public_model_math(self):
        m=model();x=inputs();x['noisy_latents']=x['noisy_latents'].to('meta')
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
            with c.lease(request='meta-not-CUDA',arm='HC') as token:
                with patch.object(m.action_embedder,'forward',side_effect=AssertionError('public math entered')):
                    with self.assertRaisesRegex(LeaseError,'CUDA not authorized'):
                        candidate(m,x,token.rows)
                self.assertTrue(token.revoked);self.assertFalse(c.entries)

    def test_default_off_changes_no_instance_methods(self):
        m = model(); methods = dict(m.__dict__); original = m.blocks[0].attn1.forward
        with SingleWorkerCHC(m) as c:
            with c.lease(request='unused', arm='HC') as token:
                self.assertIsNone(token)
            self.assertFalse(c.installed)
        self.assertEqual(set(methods), set(m.__dict__))
        self.assertEqual(m.blocks[0].attn1.forward, original)
        self.assertIs(WanAttention.forward, ORIGINAL_CLASS_FORWARD)

    def test_default_off_numeric_original(self):
        m = model(); x = inputs(); rows = torch.tensor(ROWS)
        with torch.no_grad():
            gold = candidate(m, x, rows)
            with SingleWorkerCHC(m):
                self.assertTrue(torch.equal(gold, candidate(m, x, rows)))

    def test_old_flags_do_not_enable(self):
        with patch.dict(os.environ, {'P0_KV_ATTR_ENABLED':'1','P0_KV_ATTR_GPU_APPROVED':'1'}, clear=True):
            self.assertFalse(enabled_from_env())

    def test_new_flag_is_explicit_exact_one(self):
        for value in ['0', 'true', 'yes', '']:
            with self.subTest(value=value), patch.dict(os.environ, {'P0_SINGLE_WORKER_CHC_ENABLED':value}):
                self.assertFalse(enabled_from_env())
        with patch.dict(os.environ, {'P0_SINGLE_WORKER_CHC_ENABLED':'1'}):
            self.assertTrue(enabled_from_env())

    def test_cuda_unauthorized_without_calling_cuda(self):
        c = SingleWorkerCHC(model(), enabled=True, expected_layers=1)
        with self.assertRaisesRegex(LeaseError, 'CUDA not authorized'):
            c._device('cuda:0')
        self.assertFalse(torch.cuda.is_initialized())

    def test_source_pin_refuses_changes(self):
        m = model()
        with self.assertRaisesRegex(LeaseError, 'source SHA'):
            SingleWorkerCHC(m, enabled=True, expected_layers=1, expected_source_sha256='bad').install()
        self.assertNotIn('forward', m.__dict__)

    def test_layer_count_guard(self):
        with self.assertRaisesRegex(LeaseError, 'all-layer'):
            SingleWorkerCHC(model(), enabled=True).install()

    def test_eval_only(self):
        m = model().train()
        with self.assertRaisesRegex(LeaseError, 'eval original'):
            SingleWorkerCHC(m, enabled=True, expected_layers=1).install()

    def test_duplicate_install_guard(self):
        m = model()
        with SingleWorkerCHC(m, enabled=True, expected_layers=1):
            with self.assertRaisesRegex(LeaseError, 'already installed'):
                SingleWorkerCHC(m, enabled=True, expected_layers=1).install()

    def test_atomic_partial_install_failure_restores(self):
        m = model(2); before = [b.attn1.forward for b in m.blocks]
        import adapter
        original = adapter._private_forward
        count = [0]
        def fault(*a):
            count[0] += 1
            if count[0] == 2: raise RuntimeError('installation fault')
            return original(*a)
        with patch.object(adapter, '_private_forward', fault):
            with self.assertRaisesRegex(RuntimeError, 'installation fault'):
                SingleWorkerCHC(m, enabled=True, expected_layers=2).install()
        self.assertEqual([b.attn1.forward for b in m.blocks], before)
        with SingleWorkerCHC(m, enabled=True, expected_layers=2): pass

    def test_cross_attention_and_public_class_untouched(self):
        m = model(); a2 = m.blocks[0].attn2; f2 = a2.forward; op2 = a2.attn_op
        with SingleWorkerCHC(m, enabled=True, expected_layers=1):
            self.assertEqual(a2.forward, f2); self.assertIs(a2.attn_op, op2)
            self.assertIs(WanAttention.forward, ORIGINAL_CLASS_FORWARD)
        self.assertEqual(digest(MODEL_SOURCE), MODEL_SHA256)

    def test_attention_signature_preserved(self):
        m = model(); sig = inspect.signature(m.blocks[0].attn1.forward)
        with SingleWorkerCHC(m, enabled=True, expected_layers=1):
            self.assertEqual(inspect.signature(m.blocks[0].attn1.forward), sig)

    def test_other_thread_refused(self):
        m = model(); errors = []
        with SingleWorkerCHC(m, enabled=True, expected_layers=1) as c:
            def other():
                try: c.invalidate('foreign thread')
                except Exception as e: errors.append(e)
            t = threading.Thread(target=other); t.start(); t.join()
        self.assertEqual(len(errors), 1); self.assertIn('another thread', str(errors[0]))

    def test_leases_not_nested(self):
        with SingleWorkerCHC(model(), enabled=True, expected_layers=1) as c:
            with c.lease(request='a', arm='C'):
                with self.assertRaisesRegex(LeaseError, 'nested'):
                    with c.lease(request='b', arm='HC'): pass

    def test_lease_requires_install(self):
        c = SingleWorkerCHC(model(), enabled=True, expected_layers=1)
        with self.assertRaisesRegex(LeaseError, 'install first'):
            with c.lease(request='a', arm='C'): pass

    def test_invalid_request_arm_rows(self):
        with SingleWorkerCHC(model(), enabled=True, expected_layers=1) as c:
            for kwargs in [dict(request='',arm='C'),dict(request='a',arm='H'),dict(request='a',arm='C',row_values=(0,)*14)]:
                with self.subTest(kwargs=kwargs), self.assertRaises(LeaseError):
                    with c.lease(**kwargs): pass


class Numerics(U.TestCase):
    def test_real_all30_changing29_queries_both_arms_bitwise(self):
        m = model(30); xs = [inputs(i) for i in range(29)]; rows = torch.tensor(ROWS)
        before = cache_state(m); rng = torch.get_rng_state().clone()
        with torch.no_grad(): gold = [candidate(m, x, rows) for x in xs]
        with SingleWorkerCHC(m, enabled=True, diagnostic=True) as c:
            for arm in ['C','HC']:
                with self.subTest(arm=arm), c.lease(request='all30changing29',arm=arm) as token, torch.no_grad():
                    outputs = [candidate(m, x, token.rows) for x in xs]
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(gold,outputs)))
                    self.assertEqual(len(c.entries), 30)
                    for ops in c.ops.values():
                        self.assertEqual(ops['gather'],1); self.assertEqual(ops['expand'],1)
                        self.assertEqual(ops['cat'],29); self.assertEqual(ops['SDPA'],29)
                        self.assertEqual(ops['metadata'],30 if arm=='C' else 1)
                        self.assertEqual(ops['row_check'],30 if arm=='C' else 1)
                self.assertFalse(c.entries); self.assertEqual(c.payload_bytes,0)
        self.assertTrue(same_state(before,cache_state(m)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIs(type(m),WanTransformer3DModel)

    def test_real_all30_bfloat16_bitwise(self):
        m = model(30,torch.bfloat16); x=inputs(dtype=torch.bfloat16)
        with torch.no_grad():gold=candidate(m,x,torch.tensor(ROWS))
        with SingleWorkerCHC(m,enabled=True) as c:
            for arm in ['C','HC']:
                with self.subTest(arm=arm),c.lease(request='bf16',arm=arm) as token,torch.no_grad():
                    self.assertTrue(torch.equal(gold,candidate(m,x,token.rows)))

    def test_all30_one_row_perturbation_other13_bitwise(self):
        m=model(30);x=inputs(); changed={k:v.clone() for k,v in x.items()};changed['noisy_latents'][0].add_(.9)
        with SingleWorkerCHC(m,enabled=True) as c:
            for arm in ['R','C','HC']:
                with self.subTest(arm=arm),c.lease(request='isolation',arm=arm) as token,torch.no_grad():
                    base=candidate(m,x,token.rows);out=candidate(m,changed,token.rows)
                    self.assertTrue(torch.equal(base[1:],out[1:]));self.assertFalse(torch.equal(base[0],out[0]))

    def test_real2144_history_attention_interface(self):
        m=model(history=2144,capacity=2200);x=inputs()
        with torch.no_grad():gold=candidate(m,x,torch.tensor(ROWS))
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
            for arm in ['C','HC']:
                with self.subTest(arm=arm),c.lease(request='2144',arm=arm) as token,torch.no_grad():
                    self.assertTrue(torch.equal(gold,candidate(m,x,token.rows)))
                    self.assertEqual(c.entries[0].k.shape[1],2144)

    def test_empty_history_valid_current_only(self):
        m=model(history=0);x=inputs()
        with torch.no_grad():gold=candidate(m,x,torch.tensor(ROWS))
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
            with c.lease(request='empty',arm='HC') as token,torch.no_grad():
                self.assertTrue(torch.equal(gold,candidate(m,x,token.rows)));self.assertEqual(c.payload_bytes,0)

    def test_disabled_diagnostic_no_hot_counters(self):
        m=model();x=inputs()
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
            with c.lease(request='off',arm='HC') as token,torch.no_grad():candidate(m,x,token.rows)
            self.assertEqual(c.ops,{})

    def test_owned_rows_have_versions_under_outer_inference_mode(self):
        m=model();x=inputs()
        with torch.no_grad():gold=candidate(m,x,torch.tensor(ROWS))
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c,torch.inference_mode():
            with c.lease(request='outer-inference',arm='HC') as token:
                self.assertEqual(token.rows._version,0)
                self.assertTrue(torch.equal(gold,candidate(m,x,token.rows)))

    def test_outside_lease_original_candidate_delegate(self):
        m=model();x=inputs();rows=torch.tensor(ROWS)
        with torch.no_grad():gold=candidate(m,x,rows)
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c,torch.no_grad():
            self.assertTrue(torch.equal(gold,candidate(m,x,rows)));self.assertFalse(c.entries)

    def test_positional_model_interface(self):
        m=model();x=inputs()
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
            with c.lease(request='positional',arm='HC') as token,torch.no_grad():
                self.assertEqual(m(x,0,'pos',True,False,token.rows).shape,(14,16,30))

    def test_same_query_reuses_storage_but_current_changed(self):
        m=model();x=inputs();y=inputs(1)
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
            with c.lease(request='reuse',arm='HC') as token,torch.no_grad():
                out1=candidate(m,x,token.rows);ptr=c.entries[0].k.data_ptr()
                out2=candidate(m,y,token.rows)
                self.assertEqual(c.entries[0].k.data_ptr(),ptr);self.assertFalse(torch.equal(out1,out2))


class Invalidations(U.TestCase):
    def setUp(self):
        self.m=model();self.a=self.m.blocks[0].attn1;self.cache=self.a.attn_caches['pos'];self.x=inputs()
        self.ctrl=SingleWorkerCHC(self.m,enabled=True,expected_layers=1).install()
        self.addCleanup(self.ctrl.close)

    def verify_mutation(self, mutate, *, arm='HC'):
        with self.ctrl.lease(request='before',arm=arm) as token,torch.no_grad():
            candidate(self.m,self.x,token.rows);self.assertTrue(self.ctrl.entries)
            mutate(token)
            with self.assertRaises(LeaseError):candidate(self.m,self.x,token.rows)
            self.assertFalse(self.ctrl.entries);self.assertTrue(token.revoked)
        with torch.no_grad():gold=candidate(self.m,self.x,torch.tensor(ROWS))
        with self.ctrl.lease(request='after',arm=arm) as token,torch.no_grad():
            self.assertTrue(torch.equal(gold,candidate(self.m,self.x,token.rows)))

    def test_prediction_commit_update_cache(self):
        self.verify_mutation(lambda _:self.a.update_cache('pos',torch.ones(2,2,2,4),torch.ones(2,2,2,4),True))

    def test_real_terminal_model_forward(self):
        self.verify_mutation(lambda _:self.m(inputs(batch=2),update_cache=1,cache_name='pos',action_mode=True))

    def test_real_observation_update2(self):
        self.verify_mutation(lambda _:self.m(inputs(batch=2),update_cache=2,cache_name='pos',action_mode=True))

    def test_clear_prediction(self):
        self.cache['is_pred'][1]=True
        self.verify_mutation(lambda _:self.m.clear_pred_cache('pos'))

    def test_reset_recreate_cache(self):
        def change(_):
            self.m.clear_cache('pos');self.m.create_empty_cache('pos',4,8,16,device='cpu',dtype=torch.float32,batch_size=2)
        self.verify_mutation(change)

    def test_original_snapshot_same_address_same_values(self):
        saved=SNAPSHOT.capture_cache(self.cache);address=self.cache['k'].data_ptr()
        self.verify_mutation(lambda _:self.ctrl.external_call(SNAPSHOT.restore_cache,self.cache,saved))
        self.assertEqual(address,self.cache['k'].data_ptr())

    def test_same_mask_length_inplace_K_value_change(self):
        self.verify_mutation(lambda _:self.cache['k'][:,1].add_(.7))

    def test_same_mask_length_inplace_V_value_change(self):
        self.verify_mutation(lambda _:self.cache['v'][:,1].add_(.7),arm='C')

    def test_same_length_slot_replacement(self):
        def change(_):
            self.cache['k'][:,0]=1.;self.cache['v'][:,0]=2.;self.cache['mask'][1]=False;self.cache['mask'][0]=True
        self.verify_mutation(change)

    def test_id_only_changed_still_invalidates(self):
        self.verify_mutation(lambda _:self.cache['id'][1].add_(1))

    def test_full_capacity_temporary_forward_eviction_not_restored(self):
        self.cache['mask'].fill_(True);self.cache['id'].copy_(torch.arange(48))
        self.cache['k'].fill_(1.);self.cache['v'].fill_(2.)
        before=int(self.cache['mask'].sum())
        self.verify_mutation(lambda _:self.m(inputs(batch=2),update_cache=0,cache_name='pos',action_mode=True))
        self.assertLess(int(self.cache['mask'].sum()),before)

    def test_update0_rollback_even_same_visible_slots(self):
        mask=self.cache['mask'].clone()
        self.verify_mutation(lambda _:self.m(inputs(batch=2),update_cache=0,cache_name='pos',action_mode=True))
        self.assertTrue(torch.equal(mask,self.cache['mask']))

    def test_row_values_mutated_inplace(self):
        self.verify_mutation(lambda token:token.rows.__setitem__(0,1))

    def test_equivalent_unowned_row_tensor_refused(self):
        with self.ctrl.lease(request='rowidentity',arm='HC') as token,torch.no_grad():
            candidate(self.m,self.x,token.rows)
            with self.assertRaisesRegex(LeaseError,'context changed'):
                candidate(self.m,self.x,token.rows.clone())

    def test_new_request_is_cold_not_shared(self):
        with self.ctrl.lease(request='a',arm='HC') as token,torch.no_grad():
            candidate(self.m,self.x,token.rows);entry=self.ctrl.entries[0]
        with self.ctrl.lease(request='b',arm='HC') as token,torch.no_grad():
            candidate(self.m,self.x,token.rows);self.assertIsNot(self.ctrl.entries[0],entry)
        self.assertFalse(self.ctrl.entries)

    def test_cache_name_change_refused(self):
        with self.ctrl.lease(request='names',arm='HC') as token,torch.no_grad():
            with self.assertRaisesRegex(LeaseError,'context changed'):
                self.m(self.x,action_mode=True,cache_name='other',candidate_cache_rows=token.rows)

    def test_backend_change_refused_before_projection(self):
        op=self.a.attn_op
        with self.ctrl.lease(request='backend',arm='HC') as token,torch.no_grad():
            candidate(self.m,self.x,token.rows);self.a.attn_op=lambda q,k,v:op(q,k,v)
            with self.assertRaisesRegex(LeaseError,'backend identity'):
                candidate(self.m,self.x,token.rows)
        self.a.attn_op=op

    def test_load_state_dict_generation(self):
        self.verify_mutation(lambda _:self.m.load_state_dict(self.m.state_dict()))

    def test_direct_parameter_change_detected(self):
        self.verify_mutation(lambda _:next(self.m.parameters()).add_(.001))

    def test_model_apply_dtype_invalidates_lease(self):
        with self.ctrl.lease(request='dtype',arm='HC') as token,torch.no_grad():
            candidate(self.m,self.x,token.rows);self.m.to(dtype=torch.bfloat16)
            self.assertTrue(token.revoked);self.assertFalse(self.ctrl.entries)
            with self.assertRaises(LeaseError):candidate(self.m,inputs(dtype=torch.bfloat16),token.rows)

    def test_model_train_invalidates_lease(self):
        with self.ctrl.lease(request='train',arm='HC') as token,torch.no_grad():
            candidate(self.m,self.x,token.rows);self.m.train()
            self.assertTrue(token.revoked);self.assertFalse(self.ctrl.entries)

    def test_pool_reallocated_equal_values_refused(self):
        self.verify_mutation(lambda _:self.cache.__setitem__('k',self.cache['k'].clone()))

    def test_same_shape_strided_pool_replacement(self):
        def replace(_):
            old=self.cache['k'];b,n,h,d=old.shape
            backing=torch.empty(b,n,h,d*2,dtype=old.dtype)
            new=backing[...,::2];new.copy_(old);self.cache['k']=new
        self.verify_mutation(replace)

    def test_metadata_dtype_and_shape_refused_even_cold(self):
        originals=dict(self.cache)
        for field,new in [('id',self.cache['id'].float()),('is_pred',self.cache['is_pred'].long()),
                          ('mask',self.cache['mask'][:-1]),('id',self.cache['id'][:-1])]:
            with self.subTest(field=field),self.ctrl.lease(request='schema',arm='HC') as token,torch.no_grad():
                self.cache[field]=new
                with self.assertRaisesRegex(LeaseError,'pool dtype/device/CFG geometry'):
                    candidate(self.m,self.x,token.rows)
            self.cache.update(originals)

    def test_external_restore_fault_leaves_revoked(self):
        with self.ctrl.lease(request='fault',arm='HC') as token,torch.no_grad():
            candidate(self.m,self.x,token.rows)
            with self.assertRaisesRegex(RuntimeError,'restorefault'):
                self.ctrl.external_call(lambda:(_ for _ in ()).throw(RuntimeError('restorefault')))
            self.assertTrue(token.revoked);self.assertFalse(self.ctrl.entries)

    def test_cache_mode_change_not_supported(self):
        with self.ctrl.lease(request='cachemode',arm='HC') as token,torch.no_grad():
            self.a.cache_mode='contiguous'
            with self.assertRaises(LeaseError):candidate(self.m,self.x,token.rows)
        self.a.cache_mode='original'

    def test_inference_history_without_version_refused(self):
        with torch.inference_mode():new=self.cache['k'].clone()
        self.cache['k']=new
        with self.ctrl.lease(request='unversioned',arm='HC') as token,torch.no_grad():
            with self.assertRaisesRegex(LeaseError,'no mutation version'):
                candidate(self.m,self.x,token.rows)


class FailureCleanup(U.TestCase):
    def test_cache_payload_budget_failure_releases_all_layers(self):
        m=model(2);x=inputs()
        with SingleWorkerCHC(m,enabled=True,expected_layers=2,max_cache_bytes=1) as c:
            with c.lease(request='budget',arm='HC') as token,torch.no_grad():
                with self.assertRaisesRegex(LeaseError,'payload cap'):
                    candidate(m,x,token.rows)
                self.assertFalse(c.entries);self.assertEqual(c.payload_bytes,0);self.assertTrue(token.revoked)

    def test_failure_after_first_layer_releases_partial_entries(self):
        m=model(2);x=inputs();need=2*14*6*2*4*4
        with SingleWorkerCHC(m,enabled=True,expected_layers=2,max_cache_bytes=need) as c:
            with c.lease(request='layer2budget',arm='HC') as token,torch.no_grad():
                with self.assertRaisesRegex(LeaseError,'payload cap'):
                    candidate(m,x,token.rows)
                self.assertFalse(c.entries);self.assertEqual(c.payload_bytes,0)

    def test_exception_restores_instance_methods(self):
        m=model();original=m.blocks[0].attn1.forward
        with self.assertRaisesRegex(RuntimeError,'callerfault'):
            with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
                with c.lease(request='exception',arm='HC') as token,torch.no_grad():
                    candidate(m,inputs(),token.rows);raise RuntimeError('callerfault')
        self.assertEqual(m.blocks[0].attn1.forward,original);self.assertFalse(c.entries)

    def test_lease_close_releases_retained_tensor_references(self):
        import weakref,gc
        m=model();x=inputs()
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
            with c.lease(request='release',arm='HC') as token,torch.no_grad():
                candidate(m,x,token.rows);ref=weakref.ref(c.entries[0].k)
            gc.collect();self.assertIsNone(ref());self.assertFalse(c.entries)

    def test_attention_exception_drops_existing_entries(self):
        m=model();x=inputs()
        with SingleWorkerCHC(m,enabled=True,expected_layers=1) as c:
            with c.lease(request='badshape',arm='HC') as token,torch.no_grad():
                candidate(m,x,token.rows)
                with self.assertRaises(LeaseError):
                    candidate(m,inputs(batch=13),token.rows)
                self.assertFalse(c.entries)


class ServerInterfaces(U.TestCase):
    def test_public_imports_are_actual_files(self):
        self.assertEqual(Path(public_server.__file__).resolve(),WORKTREE/'wan_va/wan_va_server.py')
        self.assertEqual(WanAttention.__module__,'wan_va.modules.model')
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_server_CFG_pack_binding_and_candidate_path(self):
        m=model(30)
        with tempfile.TemporaryDirectory() as tmp:
            s=reset_server(m,tmp)
            inp=s._prepare_action_candidate_input(torch.randn(7,30,4,4,1),torch.arange(7.),56)
            packed,original_rows=s._repeat_action_candidates_for_cfg(inp)
            self.assertEqual(tuple(original_rows.tolist()),ROWS)
            with torch.no_grad():gold=m(packed,action_mode=True,candidate_cache_rows=original_rows)
            with SingleWorkerCHC(m,enabled=True) as c:
                with c.lease(request='serverrows',arm='HC') as token,torch.no_grad():
                    self.assertTrue(torch.equal(gold,m(packed,action_mode=True,candidate_cache_rows=token.rows)))

    def test_real_server_reset_invalidates_all30(self):
        m=model(30)
        with tempfile.TemporaryDirectory() as tmp:
            s=reset_server(m,tmp)
            with SingleWorkerCHC(m,enabled=True) as c:
                with c.lease(request='reset',arm='HC') as token,torch.no_grad():
                    candidate(m,inputs(),token.rows)
                    with patch.object(torch.cuda,'empty_cache',return_value=None):s.infer({'reset':True,'prompt':None})
                    self.assertTrue(token.revoked);self.assertFalse(c.entries)

    def test_real_reset_then_frozen_Aoff_loader_on_CPU(self):
        m=model()
        with tempfile.TemporaryDirectory() as tmp:
            s=reset_server(m,tmp);s._p0_action_draft_mode='verifier_k2_b2'
            p=s._load_p0_action_draft_b2_predictor();p.model.eval()
            weights=torch.load(read_proposal()['frozen_A_off'],map_location='cpu',weights_only=False)['model_state_dict']
            self.assertTrue(all(torch.equal(v,weights[k]) for k,v in p.model.state_dict().items()))
            self.assertFalse(torch.cuda.is_initialized())


class CaptureInterfaces(U.TestCase):
    def server(self,tmp,*,repeat=False,trees=29,fault=None,history_change=None,between_call=None,candidate_write=False):
        class DriverModel(WanTransformer3DModel):
            def __init__(self):
                # Keep the public reset/cache API as well as real attention.
                # Only the CPU ledger driver below replaces whole-model math.
                super().__init__(num_attention_heads=2, attention_head_dim=4,
                    action_dim=30, text_dim=8, freq_dim=4, ffn_dim=16, num_layers=30)
                self.eval();self.calls=0
            def forward(self,inp,update_cache=0,cache_name='pos',action_mode=False,train_mode=False,candidate_cache_rows=None):
                self.calls+=1
                if candidate_cache_rows is None:return torch.zeros(2,16,30)
                out=super().forward(inp,update_cache,cache_name,action_mode,train_mode,candidate_cache_rows)
                if candidate_write:self.blocks[-1].attn1.attn_caches['pos']['k'][:,0].add_(.1)
                return out
        m=DriverModel();s=reset_server(m,tmp);after=[]
        s._phase_profile_counter=lambda *a:None
        def infer_cpu(obs,frame_st_id):
            base=torch.ones(7,30,4,4,1)
            for i in range(21):m(dict(noisy_latents=torch.ones(2,30,4,4,1)),action_mode=False,update_cache=1 if i==20 else 0)
            m(dict(noisy_latents=torch.ones(2,30,4,4,1)),action_mode=True)
            # Synthetic driver ledger, but actual all-30 model candidate math;
            # history must be captured after video/anchor, not from reset.
            for block in m.blocks:
                c=block.attn1.attn_caches['pos']
                c['k'][:,0]=1.;c['v'][:,0]=2.;c['mask'][0]=True;c['id'][0]=56
            for i in range(trees):
                if fault==i:raise RuntimeError('capturefault')
                if history_change==i:m.blocks[-1].attn1.attn_caches['pos']['v'][:,0].add_(.1)
                if between_call==i:m(dict(noisy_latents=torch.ones(2,30,4,4,1)),action_mode=True)
                s._target_action_candidate_directions(base if repeat else base+i*.02,
                    torch.ones(7)*(50 if repeat else 49-i),56)
            if trees==29:m(dict(noisy_latents=torch.ones(2,30,4,4,1)),action_mode=True)
            m(dict(noisy_latents=torch.ones(2,30,4,4,1)),action_mode=True,update_cache=1)
            after.append('policy must not finish');return None,None
        s._infer=infer_cpu
        return s,m,after

    def test_capture_default_disabled_no_infer(self):
        s=NS(infer=lambda *a:(_ for _ in ()).throw(AssertionError('must not infer')))
        self.assertEqual(capture_queries(s,{})['status'],'DISABLED')

    def test_actual_infer_and_candidate_entry_29_queries_stop_before_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,after=self.server(tmp);original=m.forward
            with torch.no_grad():r=capture_queries(s,{},enabled=True)
            self.assertEqual(m.calls,52);self.assertEqual(len(r['queries']),29);self.assertFalse(after)
            self.assertEqual(m.forward,original);self.assertEqual(r['complete_policy_chunks'],0)
            self.assertEqual(len(r['history_before_candidates']),30)
            self.assertTrue(r['history_constant_across_all29']);self.assertTrue(r['history_snapshot_in_RAM_only'])
            self.assertTrue(torch.equal(r['history_before_candidates'][-1]['slots'],torch.tensor([0])))
            q=r['queries'][7]
            with torch.no_grad():self.assertTrue(torch.equal(q.output,q.replay(m,rows=torch.tensor(ROWS))))

    def test_history_change_between_queries_refused_and_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,_=self.server(tmp,history_change=2);original=m.forward
            with torch.no_grad(),self.assertRaisesRegex(ValueError,'history changed'):
                capture_queries(s,{},enabled=True)
            self.assertEqual(m.forward,original)

    def test_non_candidate_between_queries_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,_=self.server(tmp,between_call=2);original=m.forward
            with torch.no_grad(),self.assertRaisesRegex(ValueError,'non-candidate call inside'):
                capture_queries(s,{},enabled=True)
            self.assertEqual(m.forward,original)

    def test_candidate_history_write_refused_and_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,_=self.server(tmp,candidate_write=True);original=m.forward
            with torch.no_grad(),self.assertRaisesRegex(ValueError,'candidate forward wrote'):
                capture_queries(s,{},enabled=True)
            self.assertEqual(m.forward,original)

    def test_saved_first_candidate_history_restores_after_tail_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,_=self.server(tmp)
            with torch.no_grad():
                r=capture_queries(s,{},enabled=True);q=r['queries'][7]
                m.blocks[0].attn1.attn_caches['pos']['v'][:,0].add_(100)
                for block,saved in zip(m.blocks,r['history_before_candidates']):
                    SNAPSHOT.restore_cache(block.attn1.attn_caches['pos'],saved)
                self.assertTrue(torch.equal(q.output,q.replay(m,rows=torch.tensor(ROWS))))

    def test_duplicate_static_query_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,_=self.server(tmp,repeat=True);original=m.forward
            with torch.no_grad(),self.assertRaisesRegex(ValueError,'same query'):
                capture_queries(s,{},enabled=True)
            self.assertEqual(m.forward,original)

    def test_early_terminal_rejected_and_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,_=self.server(tmp,trees=28);original=m.forward
            with torch.no_grad(),self.assertRaisesRegex(ValueError,'terminal arrived'):
                capture_queries(s,{},enabled=True)
            self.assertEqual(m.forward,original)

    def test_capture_exception_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,_=self.server(tmp,fault=2);original=m.forward
            with torch.no_grad(),self.assertRaisesRegex(RuntimeError,'capturefault'):
                capture_queries(s,{},enabled=True)
            self.assertEqual(m.forward,original)

    def test_budget_reserved_before_failed_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,m,_=self.server(tmp);budget=Budget(cap=1)
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                capture_queries(s,{},enabled=True,budget=budget)
            self.assertEqual(m.calls,1);self.assertEqual(budget.used,1)


class Contracts(U.TestCase):
    def test_frozen_proposal_GPU_still_not_approved(self):
        p=read_proposal();self.assertFalse(p['gpu_run_approved']);self.assertEqual(p['hard_target_forward_cap'],900)

    def test_forward_exceptions_not_refunded(self):
        b=Budget(cap=1);b.reserve()
        with self.assertRaisesRegex(RuntimeError,'budget exhausted'):b.reserve()
        self.assertEqual(b.used,1)

    def test_deadline_reservation(self):
        now=[0.];b=Budget(seconds=2,clock=lambda:now[0]);now[0]=2.
        with self.assertRaises(TimeoutError):b.reserve()
        self.assertEqual(b.used,0)

    def test_reject_budget_growth(self):
        with self.assertRaises(ValueError):Budget(cap=901)
        with self.assertRaises(ValueError):Budget(seconds=2701)

    def test_job_unicode_explicit_relocation(self):
        with tempfile.TemporaryDirectory(prefix='投机_') as tmp:
            p=job_path(tmp,'preflight/actual-child.json')
            self.assertTrue(p.is_relative_to(Path(tmp).resolve()))
            self.assertFalse(cpu_job_metadata(tmp)['GPU_run_approved'])

    def test_job_escape_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryDirectory() as outside:
            Path(tmp,'link').symlink_to(outside,target_is_directory=True)
            for s in ['../oldjob/result.json',str(Path(outside)/'bad'),'link/bad']:
                with self.subTest(s=s),self.assertRaises(ValueError):job_path(tmp,s)

    def test_no_training_GPU_or_driver_initialization(self):
        self.assertFalse(torch.cuda.is_initialized());self.assertEqual(CUDA_ATTEMPTS,[])
        self.assertEqual(digest(MODEL_SOURCE),MODEL_SHA256)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    suite=U.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    class RecordingResult(U.TextTestResult):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.executed_tests=[]
        def startTest(self,test):
            self.executed_tests.append(test.id());super().startTest(test)
    result=U.TextTestRunner(verbosity=2,resultclass=RecordingResult).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() and not CUDA_ATTEMPTS else 'FAIL',
        tests_run=result.testsRun,executed_tests=result.executed_tests,
        tests_by_class={name:sum('.'+name+'.' in t for t in result.executed_tests)
            for name in ['Boundaries','Numerics','Invalidations','FailureCleanup','ServerInterfaces','CaptureInterfaces','Contracts']},
        failures=[{'test':str(t),'trace':s} for t,s in result.failures],
        errors=[{'test':str(t),'trace':s} for t,s in result.errors],skipped=result.skipped,
        CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),forbidden_CUDA_attempts=CUDA_ATTEMPTS,
        public_model_sha256=digest(MODEL_SOURCE),real_public_modules=True,
        real_all30_tiny_model_and_changing29_queries=True,actual_target_weights_loaded=False,
        real_Aoff_loader_CPU=True,GPU_correctness='NOT_RUN',GPU_performance='NOT_RUN',
        LIBERO_episodes=0,independent_auditor=False)
    out=Path(args.out)
    with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    raise SystemExit(0 if report['status']=='PASS' else 1)

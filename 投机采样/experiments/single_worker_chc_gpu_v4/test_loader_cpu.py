"""Targeted positive/negative production-loader and pre-capture regressions."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest as U
from unittest.mock import patch

if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise SystemExit('CPU-only regression requires empty CUDA visibility')
import test_cpu as fixture
from production_fixture import loaded_model
from adapter import SingleWorkerCHC, LeaseError
from capture import Query
from contract import WORKTREE, MODEL_SOURCE, MODEL_SHA256, digest, ROWS
from task_contract import OWN, enabled_plan, read_cpu_plan
from worker import Runtime
t=fixture.torch
CLI_ENTRY_NEGATIVE_PROOFS = []


def legacy_only_cli_environment():
    # Negative cases must not inherit any newer revision's live approval.
    env = {k: v for k, v in os.environ.items() if not k.startswith('P0_CHC_')}
    env.update(CUDA_VISIBLE_DEVICES='', P0_SINGLE_WORKER_CHC_ENABLED='1',
               P0_SINGLE_WORKER_CHC_GPU_APPROVED='1')
    return env


def rejected_entry_before_torch(env, *, cpu_probe=False):
    # A fresh interpreter, not a mocked loader: catch the actual CLI gate and
    # inspect sys.modules before importing model/Torch/CUDA can be possible.
    args = [str(OWN/'worker.py'), '--job',
            str(WORKTREE.parent/'results/runtime/p0-single-worker-chc-20261012-085017')]
    if cpu_probe:
        args.append('--cpu-probe')
    bootstrap = (
        'import json,runpy,sys\n'
        f'sys.path.insert(0,{str(OWN)!r})\n'
        f'sys.argv={args!r}\n'
        'try:\n'
        ' runpy.run_path(sys.argv[0],run_name="__main__")\n'
        'except ValueError as exc:\n'
        ' print(json.dumps({"error":str(exc),"torch_imported":"torch" in sys.modules,'
        '"public_server_imported":"wan_va.wan_va_server" in sys.modules}))\n'
        ' sys.exit(73)\n'
        'raise AssertionError("old job admitted")\n'
    )
    result = subprocess.run([sys.executable, '-c', bootstrap], env=env,
                            capture_output=True, text=True, timeout=30)
    if result.returncode != 73:
        raise AssertionError((result.returncode, result.stdout, result.stderr))
    proof = json.loads(result.stdout.strip())
    if proof['torch_imported'] or proof['public_server_imported']:
        raise AssertionError('CLI rejection occurred after model/Torch import')
    proof['cpu_probe'] = cpu_probe
    CLI_ENTRY_NEGATIVE_PROOFS.append(proof)
    return proof


class ProductionLoader(U.TestCase):
    def test_actual_loader_binding_from_pretrained_all30(self):
        m=loaded_model()
        self.assertEqual(type(m).__module__,'modules.model')
        self.assertIs(type(m),fixture.public_server.load_transformer.__globals__['WanTransformer3DModel'])
        self.assertEqual(len(m.blocks),30)
        self.assertFalse(m.training)
        self.assertTrue(all(not p.requires_grad for p in m.parameters()))
        self.assertEqual(SingleWorkerCHC(m,enabled=True).admit()['module'],'modules.model')

    def test_production_admit_is_read_only_no_forward_no_patch(self):
        m=loaded_model();old=fixture.cache_state(m);rng=t.random.get_rng_state().clone()
        model_keys=set(m.__dict__);attention_keys=[set(b.attn1.__dict__) for b in m.blocks]
        with patch.object(m,'forward',side_effect=AssertionError('model math entered')):
            info=SingleWorkerCHC(m,enabled=True).admit()
        self.assertEqual(info['target_forwards'],0)
        self.assertEqual(info['layers'],30)
        self.assertFalse(info['patched'])
        self.assertFalse(info['KV_pools_required'])
        self.assertEqual(model_keys,set(m.__dict__))
        self.assertEqual(attention_keys,[set(b.attn1.__dict__) for b in m.blocks])
        self.assertTrue(fixture.same_state(old,fixture.cache_state(m)))
        self.assertTrue(t.equal(rng,t.random.get_rng_state()))

    def test_admission_without_initialized_KV(self):
        m=loaded_model()
        for b in m.blocks:b.attn1.attn_caches={}
        self.assertEqual(SingleWorkerCHC(m,enabled=True).admit()['status'],'PASS')

    def test_both_aliases_admitted_with_their_own_SDPA_identity(self):
        canonical=fixture.model(30);production=loaded_model()
        self.assertIsNot(type(canonical),type(production))
        self.assertIsNot(canonical.blocks[0].attn1.attn_op,production.blocks[0].attn1.attn_op)
        for m in (canonical,production):
            with self.subTest(module=type(m).__module__):
                self.assertEqual(SingleWorkerCHC(m,enabled=True).admit()['status'],'PASS')

    def test_default_off_production_unchanged(self):
        m=loaded_model();keys=set(m.__dict__);before=[b.attn1.forward for b in m.blocks]
        with SingleWorkerCHC(m) as control:
            self.assertEqual(control.admit()['status'],'DISABLED')
            self.assertFalse(control.installed)
        self.assertEqual(keys,set(m.__dict__))
        self.assertEqual(before,[b.attn1.forward for b in m.blocks])

    def test_all30_production_FP32_BF16_changing29_R_C_HC_bitwise(self):
        for dtype in (t.float32,t.bfloat16):
            with self.subTest(dtype=dtype):
                m=loaded_model(dtype);queries=[]
                with t.no_grad():
                    for i in range(29):
                        x=fixture.inputs(i,dtype=dtype)
                        out=fixture.candidate(m,x,t.tensor(ROWS))
                        queries.append(Query(x,dict(action_mode=True,update_cache=0,cache_name='pos',train_mode=False),out))
                with tempfile.TemporaryDirectory() as tmp:
                    s=fixture.NS(transformer=m,device=t.device('cpu'),cache_name='pos')
                    rt=Runtime(s,None,tmp,{},CPU_probe=True);rt.queries=queries;rt.history=fixture.cache_state(m)
                    rt.install_control()
                    try:
                        for arm in ('R','C','HC'):
                            rt.ledger.begin('warmup_'+arm)
                            row=rt.sequence(arm)
                            self.assertEqual(row['status'],'PASS')
                        self.assertEqual(rt.ledger.attempted,87)
                        self.assertEqual(rt.ledger.completed,87)
                    finally:rt.control.close()

    def test_server_first_import_in_fresh_CPU_process(self):
        code=f'''import sys,os,json
sys.path.insert(0,{str(WORKTREE)!r});sys.path.insert(0,{str(OWN)!r})
from wan_va import wan_va_server
from production_fixture import loaded_model
from adapter import SingleWorkerCHC
import torch
m=loaded_model();info=SingleWorkerCHC(m,enabled=True).admit()
assert info['module']=='modules.model' and not torch.cuda.is_initialized()
print('PRODUCTION_SERVER_FIRST_PASS')'''
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PRODUCTION_SERVER_FIRST_PASS',result.stdout)


class InvalidInterfaces(U.TestCase):
    def setUp(self):
        self.m=loaded_model();self.c=SingleWorkerCHC(self.m,enabled=True)

    def rejected(self,pattern):
        before=[b.attn1.forward for b in self.m.blocks];keys=set(self.m.__dict__)
        with self.assertRaisesRegex(LeaseError,pattern):self.c.install()
        self.assertFalse(self.c.installed)
        self.assertEqual(self.c.patches,[])
        self.assertEqual(self.c.entries,{})
        self.assertEqual(before,[b.attn1.forward for b in self.m.blocks])
        self.assertEqual(keys,set(self.m.__dict__))

    def test_arbitrary_module_alias_rejected(self):
        with patch.object(type(self.m),'__module__','other_project.model'):
            self.rejected('pinned legal')

    def test_source_SHA_mismatch_rejected(self):
        self.c.expected_source_sha256='0'*64;self.rejected('source SHA')

    def test_module_file_mismatch_rejected(self):
        module=sys.modules[type(self.m).__module__]
        with patch.object(module,'__file__','/tmp/unrelated-model.py'):self.rejected('pinned legal')

    def test_module_spec_origin_mismatch_rejected(self):
        module=sys.modules[type(self.m).__module__]
        with patch.object(module.__spec__,'origin','/tmp/unrelated-model.py'):self.rejected('pinned legal')

    def test_exported_model_class_mismatch_rejected(self):
        module=sys.modules[type(self.m).__module__]
        with patch.object(module,'WanTransformer3DModel',object):self.rejected('pinned legal')

    def test_wrong_attention_class_rejected(self):
        a=self.m.blocks[-1].attn1
        fake=type('WanAttention',(type(a),),{'__module__':type(a).__module__})
        self.m.blocks[-1].attn1=fake(8,heads=2,dim_head=4).eval()
        self.rejected('WanAttention interface')

    def test_forged_exported_attention_subclass_rejected(self):
        a=self.m.blocks[-1].attn1;module=sys.modules[type(self.m).__module__]
        fake=type('WanAttention',(type(a),),{'__module__':type(a).__module__})
        self.m.blocks[-1].attn1=fake(8,heads=2,dim_head=4).eval()
        with patch.object(module,'WanAttention',fake):self.rejected('WanAttention interface')

    def test_mixed_alias_attention_rejected(self):
        self.m.blocks[-1].attn1=fixture.model().blocks[0].attn1
        self.rejected('WanAttention interface')

    def test_instance_forward_replaced_rejected(self):
        a=self.m.blocks[-1].attn1
        a.forward=types.MethodType(lambda self,*args,**kwargs:None,a)
        self.rejected('WanAttention interface')

    def test_class_forward_replaced_rejected(self):
        cls=type(self.m.blocks[-1].attn1)
        with patch.object(cls,'forward',lambda self,*args,**kwargs:None):self.rejected('WanAttention interface')

    def test_forged_source_filename_forward_rejected_by_code(self):
        cls=type(self.m.blocks[-1].attn1)
        fn=lambda self,*args,**kwargs:None
        fake=types.FunctionType(fn.__code__.replace(co_filename=str(MODEL_SOURCE)),sys.modules[type(self.m).__module__].__dict__)
        with patch.object(cls,'forward',fake):self.rejected('WanAttention interface')

    def test_changed_forward_defaults_rejected(self):
        fn=type(self.m.blocks[-1].attn1).forward
        with patch.object(fn,'__defaults__',(1,'pos',None)):self.rejected('WanAttention interface')

    def test_backend_replaced_rejected(self):
        self.m.blocks[-1].attn1.attn_op=lambda q,k,v:q
        self.rejected('original torch SDPA backend')

    def test_cross_alias_backend_rejected(self):
        self.m.blocks[-1].attn1.attn_op=fixture.custom_sdpa
        self.rejected('original torch SDPA backend')

    def test_backend_and_module_symbol_changed_together_rejected(self):
        module=sys.modules[type(self.m).__module__]
        fake=types.FunctionType((lambda q,k,v:q).__code__.replace(co_filename=str(MODEL_SOURCE)),module.__dict__)
        self.m.blocks[-1].attn1.attn_op=fake
        with patch.object(module,'custom_sdpa',fake):self.rejected('original torch SDPA backend')

    def test_train_mode_rejected(self):
        self.m.train();self.rejected('eval original')

    def test_last_layer_train_mode_rejected(self):
        self.m.blocks[-1].attn1.train();self.rejected('eval original')

    def test_last_layer_nonoriginal_cache_rejected(self):
        self.m.blocks[-1].attn1.cache_mode='contiguous';self.rejected('WanAttention interface')

    def test_wrong_layer_count_rejected(self):
        self.c.expected_layers=29;self.rejected('all-layer')

    def test_nonCPU_parameters_rejected_without_CUDA(self):
        self.m.to('meta');self.rejected('CUDA not authorized')


class AdmissionOrder(U.TestCase):
    def test_bad_interface_stops_Runtime_before_ready_or_forward(self):
        m=loaded_model();m.blocks[-1].attn1.attn_op=lambda q,k,v:q
        original=m.forward;keys=set(m.__dict__)
        s=fixture.NS(transformer=m,device=t.device('cpu'),cache_name='pos')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(LeaseError,'original torch SDPA backend'):
                Runtime(s,None,tmp,{},CPU_probe=True)
            self.assertFalse((Path(tmp)/'preflight/cpu-worker/ready.json').exists())
            self.assertFalse((Path(tmp)/'preflight/cpu-worker/INTERFACE-ADMISSION.json').exists())
        self.assertEqual(m.forward,original);self.assertEqual(set(m.__dict__),keys)

    def runtime(self,tmp):
        m=loaded_model();s=fixture.NS(transformer=m,device=t.device('cpu'),cache_name='pos')
        return Runtime(s,None,tmp,{},CPU_probe=True)

    def test_changed_after_ready_rejected_before_load_reset_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=self.runtime(tmp);rt.m.blocks[-1].attn1.attn_op=lambda q,k,v:q
            with patch.object(t,'load',side_effect=AssertionError('fixture loaded before admission')) as load:
                with patch('capture.capture_queries',side_effect=AssertionError('capture entered')) as cap:
                    with self.assertRaisesRegex(LeaseError,'original torch SDPA backend'):rt.capture()
            self.assertFalse(load.called);self.assertFalse(cap.called)
            self.assertEqual(rt.ledger.attempted,0);self.assertIsNone(rt.capture_data)

    def test_valid_admission_precedes_fixture_load_and_does_not_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=self.runtime(tmp);rt.p={'snapshot_reference':{'path':'CPU-test-not-a-real-fixture'}}
            before=[b.attn1.forward for b in rt.m.blocks]
            with patch.object(t,'load',side_effect=RuntimeError('AFTER_ADMISSION')) as load:
                with self.assertRaisesRegex(RuntimeError,'AFTER_ADMISSION'):rt.capture()
            self.assertTrue(load.called)
            self.assertEqual(rt.admission['module'],'modules.model')
            self.assertEqual(before,[b.attn1.forward for b in rt.m.blocks])
            self.assertEqual(rt.ledger.attempted,0)

    def test_ready_contains_pre_capture_admission_zero_forwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt=self.runtime(tmp);base=Path(tmp)/'preflight/cpu-worker'
            self.assertTrue((base/'ready.json').exists())
            info=json.loads((base/'INTERFACE-ADMISSION.json').read_text())
            self.assertEqual(info['status'],'PASS');self.assertEqual(info['target_forwards'],0)
            self.assertEqual(rt.ledger.attempted,0)

    def test_GPU_Runtime_rejected_even_with_old_flags(self):
        s=fixture.NS(transformer=None,device=t.device('cpu'))
        with patch.dict(os.environ,{'P0_SINGLE_WORKER_CHC_ENABLED':'1','P0_SINGLE_WORKER_CHC_GPU_APPROVED':'1'},clear=True):
            with self.assertRaisesRegex(ValueError,'default-disabled'):Runtime(s,None,'/not-created',{})

    def test_CPU_Runtime_rejects_nonCPU_device_without_initialization(self):
        s=fixture.NS(device=t.device('cuda:0'))
        with self.assertRaisesRegex(ValueError,'CPU probe device required'):Runtime(s,None,'/not-created',{},CPU_probe=True)


class CPUOnlyEntry(U.TestCase):
    def test_GPU_plan_disabled_even_with_old_approval(self):
        with patch.dict(os.environ,{'P0_SINGLE_WORKER_CHC_ENABLED':'1','P0_SINGLE_WORKER_CHC_GPU_APPROVED':'1'},clear=True):
            with self.assertRaisesRegex(ValueError,'default-disabled'):enabled_plan('/old/job',cpu_gate=True)

    def test_old_job_cannot_be_CPU_probe_output_path(self):
        with self.assertRaisesRegex(ValueError,'^new private CPU fixture required$'):read_cpu_plan(WORKTREE.parent/'results/runtime/p0-single-worker-chc-20261012-085017')

    def test_CLI_cannot_reuse_stopped_GPU_job(self):
        env=legacy_only_cli_environment()
        result=subprocess.run([sys.executable,str(OWN/'worker.py'),'--job',str(WORKTREE.parent/'results/runtime/p0-single-worker-chc-20261012-085017')],
            env=env,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('default-disabled',result.stderr)

    def test_legacy_approval_CLI_rejection_precedes_Torch_and_model_import(self):
        for cpu_probe in (False, True):
            with self.subTest(cpu_probe=cpu_probe):
                proof=rejected_entry_before_torch(legacy_only_cli_environment(),cpu_probe=cpu_probe)
                self.assertIn('default-disabled',proof['error'])

    def test_current_approval_cannot_reuse_old_job_before_Torch_import(self):
        from task_contract import ENABLED, APPROVED
        env=legacy_only_cli_environment();env[ENABLED]=env[APPROVED]='1'
        for cpu_probe in (False, True):
            with self.subTest(cpu_probe=cpu_probe):
                proof=rejected_entry_before_torch(env,cpu_probe=cpu_probe)
                expected='new private CPU fixture required' if cpu_probe else 'scope mismatch: execution_revision'
                self.assertEqual(proof['error'],expected)

    def test_no_CUDA_or_public_file_change(self):
        self.assertFalse(t.cuda.is_initialized());self.assertEqual(fixture.CUDA_ATTEMPTS,[])
        self.assertEqual(digest(MODEL_SOURCE),MODEL_SHA256)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    suite=U.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    class RecordingResult(U.TextTestResult):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.executed=[]
        def startTest(self,test):self.executed.append(test.id());super().startTest(test)
    result=U.TextTestRunner(verbosity=2,resultclass=RecordingResult).run(suite)
    row=dict(status='PASS' if result.wasSuccessful() and not fixture.CUDA_ATTEMPTS else 'FAIL',
        tests_run=result.testsRun,executed_tests=result.executed,
        tests_by_class={name:sum('.'+name+'.' in s for s in result.executed) for name in ('ProductionLoader','InvalidInterfaces','AdmissionOrder','CPUOnlyEntry')},
        failures=[{'test':str(x),'trace':y} for x,y in result.failures],errors=[{'test':str(x),'trace':y} for x,y in result.errors],
        CPU_only=True,CUDA_initialized=t.cuda.is_initialized(),forbidden_CUDA_attempts=fixture.CUDA_ATTEMPTS,
        fresh_CLI_entry_negative_proofs=CLI_ENTRY_NEGATIVE_PROOFS,
        actual_production_loader_used=True,loader_checkpoint='temporary RANDOM tiny CPU checkpoint, not target',
        FP32_BF16_all30_changing29=True,GPU_correctness='NOT_RUN',GPU_performance='NOT_RUN',training=False)
    with Path(args.out).open('x') as f:json.dump(row,f,ensure_ascii=False,indent=2);f.write('\n')
    raise SystemExit(0 if row['status']=='PASS' else 1)

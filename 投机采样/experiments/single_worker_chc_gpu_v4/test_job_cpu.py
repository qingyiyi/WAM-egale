"""CPU-only real process/contract/safety regression for this NEW job."""
import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest as U
from unittest.mock import patch

from task_contract import STAGES, OWN, REVISION, ENABLED, APPROVED, Ledger, validate, enabled_plan, gpu_entry_gate, digest
from resource_guard import Memory, MEMORY_VERSION, MEMORY_ACCOUNTING, memory_fields, Idle, NVML
from supervisor import Supervisor
import ownership

AP=argparse.ArgumentParser();AP.add_argument('--job',required=True);AP.add_argument('--out',required=True)
A=AP.parse_args();JOB=Path(A.job);PLAN=json.loads((JOB/'plan.json').read_text())
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise SystemExit('CPU mask required')


class ContractTests(U.TestCase):
    def test_explicit_single_worker_original_contract(self):self.assertEqual(validate(PLAN),PLAN)
    def test_budget_arithmetic895_not900extra(self):self.assertEqual(sum(STAGES.values()),895)
    def test_no_old_job_or_dual_scope(self):
        for key,value in [('GPU_model_worker_cap',2),('GPU_attempt_cap',2),('hard_target_forward_cap',901),
                          ('GPU_hard_limit_minutes',46),('EGL_initialization_cap',1),('training',True),
                          ('complete_policy_chunk_cap',1),('bonus',True),('layers',1)]:
            p=copy.deepcopy(PLAN);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate(p)
    def test_modified_order_or_row_map_refused(self):
        p=copy.deepcopy(PLAN);p['rows']=[0]*14
        with self.assertRaises(ValueError):validate(p)
        p=copy.deepcopy(PLAN);p['main_orders']=p['main_orders'][:-1]
        with self.assertRaises(ValueError):validate(p)
    def test_stage_growth_refused(self):
        p=copy.deepcopy(PLAN);p['stage_caps']['capture']=53
        with self.assertRaises(ValueError):validate(p)
    def test_old_flags_not_new_authorization(self):
        with patch.dict(os.environ,{'P0_KV_ATTR_ENABLED':'1','P0_KV_ATTR_GPU_APPROVED':'1'},clear=True):
            with self.assertRaisesRegex(ValueError,'default-disabled'):enabled_plan(JOB)
    def test_CPU_gate_cannot_be_skipped(self):
        with self.assertRaises(FileNotFoundError):enabled_plan(JOB,cpu_gate=True)
    def test_unicode_paths_explicit_not_old_string_replace(self):
        self.assertEqual(str(JOB),PLAN['job']);self.assertIn('论文探索',PLAN['job'])
        self.assertEqual(Path(PLAN['worktree']).name,'wam-p0-worktree')


class AutomaticHandoffTests(U.TestCase):
    def test_CPU_failure_never_hands_off_to_GPU(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Supervisor.__new__(Supervisor);s.job=Path(tmp)
            with patch.object(s,'cpu',side_effect=RuntimeError('CPU gate failed')),patch.object(s,'gpu') as gpu:
                with self.assertRaisesRegex(RuntimeError,'CPU gate failed'):s.execute_phases('auto')
                gpu.assert_not_called()
            self.assertFalse((s.job/'GPU-CONTROLLER.json').exists())

    def test_automatic_handoff_order_and_CPU_only_no_GPU(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Supervisor.__new__(Supervisor);s.job=Path(tmp);order=[]
            def cpu():order.append('CPU')
            def gpu():
                self.assertTrue((s.job/'GPU-CONTROLLER.json').exists())
                order.append('GPU');return {'status':'PASS'}
            with patch.object(s,'cpu',side_effect=cpu),patch.object(s,'gpu',side_effect=gpu):
                self.assertEqual(s.execute_phases('auto'),{'status':'PASS'})
                self.assertEqual(order,['CPU','GPU'])
            with patch.object(s,'cpu'),patch.object(s,'gpu') as g:
                self.assertIsNone(s.execute_phases('cpu'));g.assert_not_called()


class LedgerTests(U.TestCase):
    def test_one_stage_reserve_before_math(self):
        l=Ledger();l.begin('capture');self.assertEqual(l.reserved,52)
        l.before_forward();self.assertEqual(l.attempted,1);self.assertEqual(l.completed,0)
        l.after_forward();self.assertEqual(l.completed,1)
    def test_failed_attempt_not_refunded(self):
        l=Ledger();l.begin('resource_probe');l.before_forward()
        with self.assertRaises(RuntimeError):l.before_forward()
        self.assertEqual(l.attempted,1);self.assertEqual(l.completed,0)
    def test_no_retry_after_stage_error(self):
        l=Ledger();l.begin('capture')
        with self.assertRaises(ValueError):l.begin('capture')
    def test_unknown_command_not_reserved(self):
        l=Ledger()
        with self.assertRaises(ValueError):l.begin('extra')
        self.assertEqual(l.reserved,0)
    def test_forward_global_hard_cap(self):
        l=Ledger();l.begin('capture');l.attempted=900
        with self.assertRaises(RuntimeError):l.before_forward()
        self.assertEqual(l.attempted,900)
    def test_absolute_monotonic_deadline(self):
        now=[0.];l=Ledger(clock=lambda:now[0]);l.deadline=2.;l.begin('capture');now[0]=2.
        with self.assertRaises(TimeoutError):l.before_forward()
        self.assertEqual(l.attempted,0)


class ResourceTests(U.TestCase):
    def sample(self,u=0):return dict(memory_accounting=MEMORY_ACCOUNTING,memory_used_bytes=0,compute=[],graphics=[],utilization_percent=u)
    def test_memory_reserved_not_allocated_ABI(self):
        m=Memory();m.version=MEMORY_VERSION;m.total=40*2**30;m.reserved=512*2**20;m.free=m.total-m.reserved;m.used=0
        self.assertEqual(memory_fields(m)['memory_used_bytes'],0)
        self.assertEqual(MEMORY_VERSION,ctypes.sizeof(Memory)|(2<<24))
    def test_ABI_accounting_failure_refused(self):
        with self.assertRaises(RuntimeError):memory_fields(Memory())
    def test_idle_requires15_spaced_samples(self):
        idle=Idle()
        self.assertFalse(idle.add(self.sample(25),0))
        for i in range(14):self.assertFalse(idle.add(self.sample(),i+1))
        self.assertTrue(idle.add(self.sample(),15))
    def test_idle_rapid_samples_do_not_qualify(self):
        idle=Idle()
        for i in range(15):self.assertFalse(idle.add(self.sample(),i*.001))
    def test_foreign_memory_or_PID_resets_idle(self):
        idle=Idle(samples=2);idle.add(self.sample(),0);s=self.sample();s['compute']=[{'pid':123}]
        self.assertFalse(idle.add(s,1));self.assertEqual(idle.samples,[])
    def test_real_NVML_v2_read_only_sample(self):
        nv=NVML()
        try:sample=nv.sample()
        finally:nv.close()
        self.assertEqual(sample['memory_accounting'],MEMORY_ACCOUNTING)
        self.assertEqual(sample['memory_total_bytes'],40*2**30)
        (JOB/'preflight/real-NVML-v2.json').write_text(json.dumps(sample,indent=2)+'\n')


class ProcessTests(U.TestCase):
    def child(self):
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True,
                           stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.addCleanup(lambda:p.kill() if p.poll() is None else None);self.addCleanup(lambda:None)
        return p
    def test_live_root_pinning_not_just_PGID(self):
        p=self.child();pins={}
        try:
            ownership.pin_roots([dict(pid=p.pid,process_group=p.pid,command=p.args)],pins)
            self.assertTrue(ownership.is_owned(ownership.read_process(p.pid),pins))
            root=pins[p.pid]
            from dataclasses import replace
            self.assertFalse(ownership.is_owned(replace(root,start_ticks=root.start_ticks+1),pins))
        finally:p.kill();p.wait()
    def test_proc_command_mismatch_not_allowlisted(self):
        p=self.child()
        try:
            with self.assertRaises(RuntimeError):ownership.pin_roots([dict(pid=p.pid,process_group=p.pid,command=['wrong'])],{})
        finally:p.kill();p.wait()
    def test_foreign_cpu_fixture_not_owned(self):
        p=self.child()
        try:self.assertEqual(ownership.foreign_pids(str(p.pid)+',0',{}),[p.pid])
        finally:p.kill();p.wait()
    def test_real_cleanup_only_registered_process(self):
        own=self.child();other=self.child();s=Supervisor.__new__(Supervisor);s.children=[own];s.pins={};s.lock=threading.RLock()
        try:
            ownership.pin_roots([dict(pid=own.pid,process_group=own.pid,command=own.args)],s.pins)
            s.cleanup();self.assertIsNotNone(own.poll());self.assertIsNone(other.poll())
        finally:
            for p in [own,other]:
                if p.poll() is None:p.kill()
                p.wait()
    def test_expiry_writes_stop_and_cleans_even_no_wait_input(self):
        own=self.child();s=Supervisor.__new__(Supervisor);s.children=[own];s.pins={};s.lock=threading.RLock()
        s.failure_reason=None;s.timed_out=False;s.stop=threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            s.job=Path(tmp);ownership.pin_roots([dict(pid=own.pid,process_group=own.pid,command=own.args)],s.pins)
            s.expire('fake absolute deadline');self.assertTrue(s.stop.is_set());self.assertTrue(s.timed_out)
            self.assertTrue((s.job/'STOP_REQUESTED').exists());self.assertIsNotNone(own.poll())
    def test_supervisor_environment_clean_CPU_GPU_split(self):
        s=Supervisor(JOB)
        with patch.dict(os.environ,{'WAM_P0_TRACE_DIR':'bad','P0_KV_ATTR_ENABLED':'1','CUDA_LAUNCH_BLOCKING':'1'}):
            cpu=s.environment(cpu=True);gpu=s.environment(cpu=False)
        self.assertEqual(cpu['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(gpu['CUDA_VISIBLE_DEVICES'],'0')
        self.assertNotIn('WAM_P0_TRACE_DIR',gpu);self.assertNotIn('P0_KV_ATTR_ENABLED',gpu)
        self.assertNotIn('CUDA_LAUNCH_BLOCKING',gpu);self.assertEqual(gpu['WORLD_SIZE'],'1')
        self.assertEqual(gpu['HF_HUB_OFFLINE'],'1')
    def test_actual_worker_negative_entry_rejects_GPU_mask_for_CPU_probe(self):
        s=Supervisor(JOB);env=s.environment(cpu=False)
        r=subprocess.run([PLAN['model_python'],str(OWN/'worker.py'),'--job',str(JOB),'--cpu-probe'],
            env=env,text=True,capture_output=True,timeout=30)
        self.assertNotEqual(r.returncode,0);self.assertIn('CPU visibility',r.stderr)


class CompareTests(U.TestCase):
    def rows(self):
        return [dict(status='PASS',profile_OFF=True,query_count=29,cold_setup_staging_keys_release_included=True,
            bitwise_all_outputs=True,wall_sec={'R':5,'C':4.5,'HC':4}[a],arm=a,block=b,CUDA_event_ms=5000.,extra_peak_bytes=0)
            for b in range(6) for a in ['R','C','HC']]
    def test_stable_six_block_positive_pilot_only(self):
        from gpu_compare import off_analysis
        r=off_analysis(self.rows());self.assertEqual(r['status'],'PASS');self.assertFalse(r['success_environment_steps_E2E_proven'])
        self.assertTrue(all(x['signal']=='full_verifier_positive_pilot' for x in r['effects']))
    def test_unstable_samples_retained_FAIL_no_retry(self):
        from gpu_compare import off_analysis
        rows=self.rows()
        for v in rows:
            if v['arm']=='R':v['wall_sec']=1 if v['block']%2 else 9
        r=off_analysis(rows);self.assertEqual(r['status'],'FAIL');self.assertEqual(len(r['metrics']['R']['raw_sec']),6)
    def test_missing_or_duplicate_samples_refused(self):
        from gpu_compare import off_analysis
        with self.assertRaises(ValueError):off_analysis(self.rows()[:-1])
        with self.assertRaises(ValueError):off_analysis(self.rows()+[self.rows()[0]])
    def test_CPU_GPU_readiness_paths_separate(self):
        import ast
        source=(OWN/'worker.py').read_text();ast.parse(source)
        self.assertIn("'preflight/cpu-worker' if CPU_probe else 'worker'",source)
        self.assertFalse((JOB/'worker/ready.json').exists());self.assertFalse((JOB/'GPU-ATTEMPT.json').exists())
    def test_no_CUDA_initialization(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())


class NewEntryTests(U.TestCase):
    def test_revision_and_implementation_identity_required(self):
        for key,value in [('execution_revision','old'),('resource_poll_interval_sec',179),('resource_wait_explicitly_authorized',False)]:
            p=copy.deepcopy(PLAN);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate(p)
        p=copy.deepcopy(PLAN);p['implementation_sha256'].pop(next(iter(p['implementation_sha256'])))
        with self.assertRaisesRegex(ValueError,'implementation identity'):validate(p)

    def test_old_job_cannot_be_enabled_by_new_flags(self):
        old=OWN.parent.parent/'results/runtime/p0-single-worker-chc-20261012-085017'
        with self.assertRaisesRegex(ValueError,'scope mismatch|new private job'):enabled_plan(old)

    def test_fresh_GPU_entry_requires_new_CPU_gate_before_import(self):
        s=Supervisor(JOB)
        r=subprocess.run([PLAN['model_python'],str(OWN/'worker.py'),'--job',str(JOB)],env=s.environment(cpu=False),text=True,capture_output=True,timeout=30)
        self.assertNotEqual(r.returncode,0);self.assertIn('CPU-GATE.json',r.stderr)
        self.assertFalse((JOB/'worker/GPU-ENTRY.json').exists())

    def fixture_entry(self,tmp):
        import task_contract as T
        job=Path(tmp);p=copy.deepcopy(PLAN);p['job']=str(job)
        (job/'plan.json').write_text(json.dumps(p))
        now=time.monotonic()
        (job/'GPU-ATTEMPT.json').write_text(json.dumps(dict(job=str(job),execution_revision=REVISION,attempt=1,
                monotonic_start=now,monotonic_deadline=now+2700)))
        return T,job,p

    def test_one_shot_claim_not_repeatable_without_GPU(self):
        with tempfile.TemporaryDirectory() as tmp:
            T,job,p=self.fixture_entry(tmp)
            with patch.object(T,'enabled_plan',return_value=p),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                self.assertEqual(T.gpu_entry_gate(job,claim=True),p)
                self.assertEqual(T.gpu_entry_gate(job),p)
                with self.assertRaises(FileExistsError):T.gpu_entry_gate(job,claim=True)

    def test_terminal_attempt_cannot_reenter(self):
        with tempfile.TemporaryDirectory() as tmp:
            T,job,p=self.fixture_entry(tmp);(job/'FAILED.json').write_text('{}')
            with patch.object(T,'enabled_plan',return_value=p),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                with self.assertRaisesRegex(ValueError,'terminal/stopped'):T.gpu_entry_gate(job,claim=True)
            self.assertFalse((job/'worker/GPU-ENTRY.json').exists())

    def test_expired_deadline_refused_before_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            T,job,p=self.fixture_entry(tmp);a=json.loads((job/'GPU-ATTEMPT.json').read_text());a['monotonic_start']-=2701;a['monotonic_deadline']-=2701
            (job/'GPU-ATTEMPT.json').write_text(json.dumps(a))
            with patch.object(T,'enabled_plan',return_value=p),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                with self.assertRaisesRegex(TimeoutError,'cleanup-reserved deadline'):T.gpu_entry_gate(job,claim=True)

    def test_CPU_gate_hash_cannot_be_borrowed(self):
        import task_contract as T
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);p=copy.deepcopy(PLAN);p['job']=str(job);(job/'plan.json').write_text(json.dumps(p));(job/'preflight').mkdir()
            (job/'preflight/CPU-GATE.json').write_text(json.dumps(dict(status='PASS',plan_sha256='0'*64,implementation_sha256=p['implementation_sha256'])))
            with patch.object(T,'validate',return_value=p):
                with self.assertRaisesRegex(ValueError,'fresh CPU gate missing/changed'):T.enabled_plan(job,cpu_gate=True)


class CounterIntegrationTests(U.TestCase):
    def test_persistence_or_system_mode_change_cannot_be_requested(self):
        for key,value in [('counter_persistence_enabled',False),('GPU_compute_mode_changed',True)]:
            p=copy.deepcopy(PLAN);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate(p)

    def test_CPU_fault_fields_cannot_leak_into_GPU_plan(self):
        for key in ('CPU_only','CPU_test_fault','CPU_exit_after_one_command'):
            p=copy.deepcopy(PLAN);p[key]=True
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'CPU fixture'):validate(p)

    def test_GPU_counter_requires_admitted_context_before_creating_file(self):
        from counter_journal import PersistentLedger
        from task_contract import COUNTER_JOB
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'counter.jsonl'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0');env.pop(COUNTER_JOB,None)
            with patch.dict(os.environ,env,clear=True):
                with self.assertRaisesRegex(ValueError,'admitted GPU counter'):
                    PersistentLedger(path,{'job':str(JOB)},enabled=True)
            self.assertFalse(path.exists())

    def test_missing_counter_never_inherits_parent_zero(self):
        from counter_recovery import recover_owned,budget_from_recovery
        with tempfile.TemporaryDirectory() as tmp:
            parent=Ledger().summary();r=recover_owned(Path(tmp)/'absent',{'job':'CPU'},parent_last_ack=parent)
            b=budget_from_recovery(parent,r)
            self.assertIsNone(b['target_attempted']);self.assertIsNone(b['target_completed'])
            self.assertFalse(r['whole_stage_success_inferred']);self.assertEqual(r['parent_last_ack_budget'],parent)

    def test_writer_must_match_pinned_actual_CPU_process(self):
        from counter_journal import PersistentLedger,JournalError
        from counter_recovery import recover_owned
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal';identity={'job':tmp};l=PersistentLedger(path,identity,enabled=True)
            try:
                owner=l.writer.owner
                self.assertEqual(recover_owned(path,identity,expected_owner=owner)['ledger']['target_attempted'],0)
                with self.assertRaisesRegex(JournalError,'pinned worker'):
                    recover_owned(path,identity,expected_owner={**owner,'start_ticks':owner['start_ticks']+1})
            finally:l.close()

    def test_supervisor_unknown_count_is_null_not_zero(self):
        s=Supervisor.__new__(Supervisor);s.gpu_start=0.;s.ledger=Ledger()
        s.read_counter=lambda:dict(status='UNKNOWN_NO_JOURNAL',ledger=None)
        b=s.budget_summary();self.assertIsNone(b['target_attempted']);self.assertIsNone(b['target_completed'])

    def test_supervisor_uses_verified_pending_intent_not_stale_ack(self):
        s=Supervisor.__new__(Supervisor);s.gpu_start=0.;s.ledger=Ledger()
        s.read_counter=lambda:dict(status='INTERRUPTED_VERIFIED_PREFIX',ledger={
            **s.ledger.summary(),'target_attempted':2,'target_completed':1,'target_inflight':1})
        b=s.budget_summary();self.assertEqual((b['target_attempted'],b['target_completed'],b['target_inflight']),(2,1,1))
        self.assertFalse(b['CUDA_device_completion_proven'])

    def test_original_stage_caps_and_Ledger_AST_preserved(self):
        import ast
        old=OWN.parent/'single_worker_chc_gpu_v2/task_contract.py'
        trees=[ast.parse(p.read_text()) for p in (old,OWN/'task_contract.py',OWN/'ledger_core.py')]
        def ledger(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Ledger')
        self.assertEqual(ast.dump(ledger(trees[0]),include_attributes=False),ast.dump(ledger(trees[1]),include_attributes=False))
        self.assertEqual(ast.dump(ledger(trees[0]),include_attributes=False),ast.dump(ledger(trees[2]),include_attributes=False))


class AuthorizedResourceQueue(U.TestCase):
    def setup_queue(self,tmp,*,stop_after_busy=False):
        s=Supervisor.__new__(Supervisor);s.job=Path(tmp);s.p=copy.deepcopy(PLAN);s.p['ports']=[0]
        s.gpu_start=None;s.gpu_end=None;s.ledger=Ledger();s.children=[];s.pins={};s.failure_reason=None
        s.stage='RESOURCE';s.passed=['CPU'];s.worker=None;s.nvml=None;s.events=[];s.statuses=[]
        s.status=lambda **kw:s.statuses.append(s.stage)
        s.event=lambda name,row:s.events.append((name,row))
        now=[0.];waits=[]
        class Stop:
            flag=False
            def is_set(self):return self.flag
            def wait(self,seconds):
                waits.append(seconds);now[0]+=seconds
                if stop_after_busy:self.flag=True
                return self.flag
        s.stop=Stop()
        quiet=dict(uuid='CPU_FIXTURE_ONLY',memory_accounting=MEMORY_ACCOUNTING,memory_total_bytes=40*2**30,
                   memory_used_bytes=0,memory_reserved_bytes=512*2**20,memory_free_bytes=40*2**30-512*2**20,
                   compute=[],graphics=[],utilization_percent=0)
        busy={**quiet,'memory_used_bytes':25*2**30,'compute':[dict(pid=987654321,memory_bytes=25*2**30)]}
        class FakeNVML:
            calls=0
            def sample(self):
                self.calls+=1;return busy if self.calls==1 or stop_after_busy else quiet
        return s,now,waits,FakeNVML

    def test_busy_poll180_then_stable_empty_before_any_GPU(self):
        import supervisor as M
        with tempfile.TemporaryDirectory() as tmp:
            s,now,waits,FakeNVML=self.setup_queue(tmp)
            with patch.object(M,'NVML',FakeNVML),patch.object(M.time,'monotonic',side_effect=lambda:now[0]),patch.object(M,'freeze'):
                s.resource()
            self.assertEqual(waits[0],180);self.assertEqual(waits[1:],[1]*14)
            self.assertEqual([x[0] for x in s.events],['WAIT'])
            self.assertEqual(s.events[0][1]['poll_interval_sec'],180)
            self.assertIsNone(s.gpu_start);self.assertIsNone(s.worker);self.assertEqual(s.ledger.attempted,0)
            self.assertFalse((s.job/'GPU-ATTEMPT.json').exists());self.assertTrue((s.job/'RESOURCE-GATE.json').exists())

    def test_queued_stop_interrupts_without_model_or_foreign_signal(self):
        import supervisor as M
        with tempfile.TemporaryDirectory() as tmp:
            s,now,waits,FakeNVML=self.setup_queue(tmp,stop_after_busy=True)
            with patch.object(M,'NVML',FakeNVML),patch.object(M.time,'monotonic',side_effect=lambda:now[0]):
                with self.assertRaisesRegex(RuntimeError,'owned stop'):s.resource()
            self.assertEqual(waits,[180]);self.assertIsNone(s.gpu_start);self.assertEqual(s.children,[])
            self.assertFalse((s.job/'GPU-ATTEMPT.json').exists())


if __name__=='__main__':
    r=U.TextTestRunner(verbosity=2).run(U.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    import torch
    report=dict(status='PASS' if r.wasSuccessful() and not torch.cuda.is_initialized() else 'FAIL',tests_run=r.testsRun,
        failures=[{'test':str(t),'trace':s} for t,s in r.failures],errors=[{'test':str(t),'trace':s} for t,s in r.errors],
        CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),actual_process_identity_cleanup=True,new_GPU_started=False,training=False)
    with Path(A.out).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)

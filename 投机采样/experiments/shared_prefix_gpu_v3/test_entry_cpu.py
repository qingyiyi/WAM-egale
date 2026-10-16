"""Fresh job entry, budgets and a REAL killed CPU worker counter regression."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest as U

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise SystemExit('CPU only')
from task_contract import OWN, BASE, STAGES, Ledger, ENABLED, APPROVED, validate, enabled_plan, digest, write, counter_identity
from counter_journal import PersistentLedger, recover, JournalError
from prefix_compare import off_analysis, aggregate_strict

JOB=None;PLAN=None


class Contract(U.TestCase):
    def test_plan_bound_to_fresh_revision(self):
        self.assertEqual(enabled_plan(JOB),PLAN)

    def test_prior_consumption_not_reset(self):
        for k,v in (('aggregate_prior_target_intents',0),('aggregate_prior_GPU_seconds',0),('GPU_hard_limit_minutes',45)):
            p=copy.deepcopy(PLAN);p[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):validate(p)

    def test_new_job_path_required(self):
        p=copy.deepcopy(PLAN);p['job']=str(JOB.parent/'p0-single-worker-chc-v4-old')
        with self.assertRaisesRegex(ValueError,'new private job'):validate(p)

    def test_diagnostic_requires_separate_authorization(self):
        p=copy.deepcopy(PLAN);p['diagnostic_nonbitwise_approved']=True;p['diagnostic_authorization_text']=''
        with self.assertRaisesRegex(ValueError,'explicit diagnostic'):validate(p)

    def test_stage_consumption_and_retries(self):
        c=Ledger()
        for stage,cap in STAGES.items():
            c.begin(stage)
            for _ in range(cap):c.before_forward();c.after_forward()
            with self.assertRaises(RuntimeError):c.before_forward()
            with self.assertRaises(ValueError):c.begin(stage)
        self.assertEqual(c.attempted,585);self.assertEqual(c.completed,585)
        self.assertLessEqual(c.attempted+183,900)

    def test_deadline_blocks_before_intent(self):
        c=Ledger(clock=lambda:5);c.deadline=5;c.begin('warmup_R')
        with self.assertRaises(TimeoutError):c.before_forward()
        self.assertEqual(c.attempted,0)

    def test_extra_stages_no_refund_permission(self):
        with self.assertRaises(ValueError):Ledger().begin('unallocated130')

    def test_disabled_and_old_flags_fail_before_torch(self):
        code='''import runpy,sys
sys.argv=sys.argv[1:]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except Exception as e:
 print(type(e).__name__,str(e)); print('torch_imported', 'torch' in sys.modules)
 print('server_imported', 'wan_va.wan_va_server' in sys.modules)
 raise SystemExit(37)
'''
        for flags in ({},{'P0_CHC_V4_ENABLED':'1','P0_CHC_V4_GPU_APPROVED':'1'},{'P0_SHARED_PREFIX_V1_ENABLED':'1','P0_SHARED_PREFIX_V1_GPU_APPROVED':'1'}):
            env={k:v for k,v in os.environ.items() if not k.startswith('P0_')}
            env.update(flags,CUDA_VISIBLE_DEVICES='0')
            r=subprocess.run([sys.executable,'-c',code,str(OWN/'worker.py'),'--job',str(JOB)],env=env,text=True,capture_output=True,timeout=20)
            self.assertEqual(r.returncode,37,r.stderr)
            self.assertIn('default-disabled',r.stdout);self.assertIn('torch_imported False',r.stdout);self.assertIn('server_imported False',r.stdout)

    def test_missing_cpu_gate_refuses_gpu_before_import(self):
        code='''import runpy,sys
sys.argv=sys.argv[1:]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except Exception as e:
 print(type(e).__name__,str(e)); print('torch_imported', 'torch' in sys.modules)
 raise SystemExit(37)
'''
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0');env[ENABLED]=env[APPROVED]='1'
        r=subprocess.run([sys.executable,'-c',code,str(OWN/'worker.py'),'--job',str(JOB)],env=env,text=True,capture_output=True,timeout=20)
        self.assertEqual(r.returncode,37,r.stderr);self.assertIn('CPU-GATE.json',r.stdout);self.assertIn('torch_imported False',r.stdout)


class Persistence(U.TestCase):
    def test_valid_closed_journal_and_tamper_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ledger.jsonl';ident={'job':'CPU-test-closed'}
            c=PersistentLedger(path,ident,enabled=True);c.begin('warmup_S');c.before_forward();c.after_forward();c.close('test')
            got=recover(path,ident);self.assertEqual(got['status'],'CLOSED');self.assertEqual(got['ledger']['target_attempted'],1)
            old=path.read_bytes();path.write_bytes(old.replace(b'"RETURN"',b'"BEFORE"'))
            with self.assertRaises(JournalError):recover(path,ident)

    def test_real_worker_killed_after_before_preserves_intent(self):
        child=JOB/'workers/case-kill-after-before'
        child.mkdir(parents=True)
        p={**PLAN,'job':str(child),'CPU_only':True,'GPU_run_approved':False,'gpu_run_approved':False,
           'parent_plan_sha256':digest(JOB/'plan.json'),'CPU_test_fault':'after_BEFORE_2'}
        write(child/'cpu-plan.json',p,exclusive=True)
        base=child/'preflight/cpu-worker';(base/'inbox').mkdir(parents=True);(base/'outbox').mkdir()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env[ENABLED]=env[APPROVED]='1'
        cmd=[sys.executable,str(OWN/'worker.py'),'--job',str(child),'--cpu-serve']
        with (child/'CPU-child.log').open('w') as log:
            proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+90
            while not (base/'SERVING.json').exists():
                if proc.poll() is not None:raise AssertionError('real CPU worker exited '+(child/'CPU-child.log').read_text())
                if time.monotonic()>deadline:raise TimeoutError('CPU worker ready')
                time.sleep(.05)
            write(base/'inbox/0000.json',dict(id=0,job=str(child),stage='warmup_S',op='sequence',arm='S',timed=False),exclusive=True)
            while not (base/'PAUSE.json').exists():
                if proc.poll() is not None:raise AssertionError((child/'CPU-child.log').read_text())
                if time.monotonic()>deadline:raise TimeoutError('actual worker counter pause')
                time.sleep(.05)
            pause=json.loads((base/'PAUSE.json').read_text());self.assertEqual(pause['boundary'],'after_BEFORE_2')
            self.assertEqual(pause['ledger']['target_attempted'],2)
            # The process is our Popen child, not a foreign PID or stale group.
            proc.kill();proc.wait(timeout=10)
            rec=recover(base/'counter.jsonl',counter_identity(child,CPU=True))
            self.assertEqual(rec['ledger']['target_attempted'],2);self.assertEqual(rec['ledger']['target_completed'],1)
            self.assertEqual(rec['ledger']['target_inflight'],1);self.assertFalse(rec['writer_same_process_alive'])
            self.assertFalse((child/'GPU-ATTEMPT.json').exists());self.assertFalse((child/'worker/ready.json').exists())
            write(child/'CPU-RECOVERED.json',rec,exclusive=True)
        finally:
            if proc.poll() is None:proc.kill();proc.wait(timeout=10)


class Comparator(U.TestCase):
    def test_final_strict_includes_later_OFF_and_finish(self):
        for a,b,c in ((True,False,True),(True,True,False),(False,True,True)):
            self.assertFalse(aggregate_strict(a,{'strict_bitwise_all':b},{'strict_all_S_numerical_checks':c}))
        self.assertTrue(aggregate_strict(True,{'strict_bitwise_all':True},{'strict_all_S_numerical_checks':True}))

    def test_micro_has_no_performance_without_diagnostic_authorization(self):
        from types import SimpleNamespace
        import tempfile
        sys.path.append(str(BASE))
        import test_cpu as fixture
        import prefix_micro
        from unittest.mock import patch
        t=fixture.torch
        with tempfile.TemporaryDirectory() as tmp:
            rt=SimpleNamespace(t=t,job=Path(tmp),device='cpu',
                p={'direct_attention_micro_call_cap':256,'diagnostic_nonbitwise_approved':False},
                check_stop=lambda:None,sync=lambda:None)
            (rt.job/'worker').mkdir()
            def reference(*args,**kwargs):
                from prefix_attention import cpu_reference
                return cpu_reference(*args)
            # Real micro CONTROL FLOW on CPU with explicit reference math;
            # no CUDA kernel proof. Timers are forbidden on this branch.
            with patch('prefix_attention.shared_attention',side_effect=reference),patch.object(t.cuda,'Event',side_effect=AssertionError('unauthorized timer')):
                out=prefix_micro.run_micro(rt)
            self.assertEqual(out['bench'],[])
            self.assertTrue(out['performance_skipped_without_diagnostic_authorization'])
            self.assertLessEqual(out['direct_attention_micro_calls'],256)

    def samples(self):
        return [dict(status='PASS',profile_OFF=True,query_count=29,cold_setup_staging_keys_release_included=True,
                     arm=a,block=i,wall_sec=2 if a=='R' else 1.5,CUDA_event_ms=1,
                     extra_peak_bytes=0,bitwise_all_outputs=True) for i in range(6) for a in ('R','S')]
    def test_strict_speed_result(self):
        row=off_analysis(self.samples());self.assertEqual(row['status'],'PASS');self.assertEqual(row['reduction_percent'],25)
    def test_nonbitwise_no_silent_acceptance(self):
        rows=self.samples();rows[1]['bitwise_all_outputs']=False
        with self.assertRaises(ValueError):off_analysis(rows)
        self.assertEqual(off_analysis(rows,diagnostic_allowed=True)['status'],'DIAGNOSTIC_ONLY_NOT_ACCEPTED')
    def test_missing_block_not_discarded(self):
        with self.assertRaises(ValueError):off_analysis(self.samples()[:-1])
    def test_unstable_not_filtered(self):
        rows=self.samples()
        for row in rows:
            if row['arm']=='S':row['wall_sec']=.5 if row['block']<3 else 3
        self.assertEqual(off_analysis(rows)['status'],'FAIL')


def main():
    global JOB,PLAN
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    JOB=Path(a.job);PLAN=enabled_plan(JOB)
    result=U.TextTestRunner(verbosity=2).run(U.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    write(a.out,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),CUDA_initialized=False,CUDA_attempts=[],
        same_actual_worker_kill_recovery=True,target_weights_loaded=False,new_GPU_started=False))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()

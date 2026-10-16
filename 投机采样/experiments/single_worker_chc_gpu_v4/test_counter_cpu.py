"""WAL faults and termination of the actual Runtime/serve/public CPU model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest as U
from unittest.mock import patch

from counter_journal import PersistentLedger, JournalError, recover, canonical, MAX_BYTES
from task_contract import OWN, FLAG, write, digest, read_cpu_plan, gpu_entry_gate
from ledger_core import Ledger, STAGES
import ownership

if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise SystemExit('CPU visibility required')
JOB = None
CASE_RESULTS = []


class Journal(U.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='chc-counter-unit-')
        self.path = Path(self.tmp.name) / 'counter.jsonl'
        self.identity = dict(job=self.tmp.name, plan_sha256='a'*64, kind='CPU-only-counter-unit')
        self.l = None

    def tearDown(self):
        if self.l is not None: self.l.close('unit_cleanup')
        self.tmp.cleanup()

    def ledger(self):
        self.l = PersistentLedger(self.path, self.identity, enabled=True); return self.l

    def recovery(self): return recover(self.path, self.identity)

    def rewrite(self, change):
        rows=[json.loads(s) for s in self.path.read_text().splitlines()]; change(rows)
        previous='0'*64; out=[]
        for row in rows:
            row.pop('checksum');row['previous']=previous
            previous=hashlib.sha256(canonical(row)).hexdigest()
            out.append(canonical({**row,'checksum':previous})+b'\n')
        self.path.write_bytes(b''.join(out))

    def test_default_off_no_file_or_directory(self):
        p=self.path.parent/'absent/ledger'
        with patch.dict(os.environ,{},clear=True):
            l=PersistentLedger(p,self.identity)
            l.begin('warmup_R');l.before_forward();l.after_forward();l.close()
        self.assertFalse(p.parent.exists())

    def test_old_GPU_flags_cannot_enable(self):
        with patch.dict(os.environ,{'P0_CHC_V2_ENABLED':'1','P0_CHC_V2_GPU_APPROVED':'1','CUDA_VISIBLE_DEVICES':''},clear=True):
            with self.assertRaisesRegex(ValueError,'default-disabled'):self.ledger()
        self.assertFalse(self.path.exists())

    def test_nonCPU_visibility_refused(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'CPU visibility'):self.ledger()
        self.assertFalse(self.path.exists())

    def test_missing_is_unknown_not_zero(self):
        r=self.recovery();self.assertEqual(r['status'],'UNKNOWN_NO_JOURNAL');self.assertIsNone(r['ledger'])

    def test_empty_is_unknown_not_zero(self):
        self.path.touch();self.assertIsNone(self.recovery()['ledger'])

    def test_torn_header_is_unknown(self):
        self.path.write_bytes(b'{"schema":1');self.assertIsNone(self.recovery()['ledger'])

    def test_valid_OPEN_has_genuine_zero(self):
        self.ledger();r=self.recovery()
        self.assertEqual(r['status'],'LIVE_VERIFIED_PREFIX');self.assertEqual(r['ledger']['target_attempted'],0)

    def test_before_is_persisted_without_command_ack(self):
        l=self.ledger();l.begin('warmup_R');l.before_forward();r=self.recovery()
        self.assertEqual(r['ledger']['target_attempted'],1);self.assertEqual(r['ledger']['target_completed'],0)
        self.assertEqual(r['unconfirmed_call_upper_bound'],1);self.assertFalse(r['exact_math_counts'])

    def test_return_is_persisted_without_command_ack(self):
        l=self.ledger();l.begin('warmup_R');l.before_forward();l.after_forward()
        self.assertEqual(self.recovery()['ledger']['target_completed'],1)

    def test_raise_consumes_budget_no_refund(self):
        l=self.ledger();l.begin('warmup_R');l.before_forward();l.forward_failed(ValueError())
        s=self.recovery()['ledger'];self.assertEqual((s['target_attempted'],s['target_completed'],s['target_failed']),(1,0,1))

    def test_stage_cap_before_math(self):
        l=self.ledger();l.begin('warmup_R')
        for _ in range(29):l.before_forward();l.after_forward()
        with self.assertRaisesRegex(RuntimeError,'cap'):l.before_forward()
        self.assertEqual(self.recovery()['ledger']['target_attempted'],29)

    def test_full_895_budget_and_all_stages(self):
        l=self.ledger()
        for key,n in STAGES.items():
            l.begin(key)
            for _ in range(n):l.before_forward();l.after_forward()
        self.assertEqual(self.recovery()['ledger']['target_attempted'],895)
        self.assertLess(self.path.stat().st_size,MAX_BYTES)

    def test_900_hard_guard_retained(self):
        l=Ledger();l.begin('capture');l.attempted=900
        with self.assertRaisesRegex(RuntimeError,'cap'):l.before_forward()

    def test_deadline_refuses_before_new_intent(self):
        l=self.ledger();l.begin('warmup_R');l.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError):l.before_forward()
        self.assertEqual(self.recovery()['ledger']['target_attempted'],0)

    def test_duplicate_stage_refused(self):
        l=self.ledger();l.begin('warmup_R')
        with self.assertRaises(ValueError):l.begin('warmup_R')

    def test_second_outstanding_refused(self):
        l=self.ledger();l.begin('warmup_R');l.before_forward()
        with self.assertRaises(JournalError):l.before_forward()
        self.assertEqual(self.recovery()['ledger']['target_attempted'],1)

    def test_return_without_intent_refused(self):
        l=self.ledger()
        with self.assertRaises(JournalError):l.after_forward()

    def test_raise_without_intent_refused(self):
        l=self.ledger()
        with self.assertRaises(JournalError):l.forward_failed(ValueError())

    def test_journal_cannot_be_reopened_or_resumed(self):
        self.ledger()
        with self.assertRaises(FileExistsError):PersistentLedger(self.path,self.identity,enabled=True)

    def test_writer_symlink_refused(self):
        target=self.path.parent/'target';target.write_text('untouched');self.path.symlink_to(target)
        with self.assertRaises(OSError):self.ledger()
        self.assertEqual(target.read_text(),'untouched')

    def test_reader_symlink_refused(self):
        target=self.path.parent/'target';target.touch();self.path.symlink_to(target)
        with self.assertRaises(OSError):self.recovery()

    def test_identity_mismatch_refused(self):
        self.ledger()
        with self.assertRaisesRegex(JournalError,'identity'):recover(self.path,{**self.identity,'job':'other'})

    def test_wrong_plan_hash_refused(self):
        self.ledger()
        with self.assertRaisesRegex(JournalError,'identity'):recover(self.path,{**self.identity,'plan_sha256':'b'*64})

    def test_partial_final_record_preserves_verified_prefix(self):
        l=self.ledger();l.begin('warmup_R');l.before_forward()
        with self.path.open('ab') as f:f.write(b'{"schema":1')
        r=self.recovery();self.assertEqual(r['ledger']['target_attempted'],1)
        self.assertGreater(r['uncommitted_tail_bytes'],0)
        # Avoid appending CLOSE after the deliberately torn record.
        l.writer.poisoned=True

    def test_complete_bad_checksum_is_not_ignored_tail(self):
        self.ledger();b=self.path.read_bytes();self.path.write_bytes(b.replace(b'"seq":1',b'"seq":9'))
        with self.assertRaisesRegex(JournalError,'checksum|sequence'):self.recovery()
        self.l.writer.poisoned=True

    def test_middle_record_deletion_refused(self):
        l=self.ledger();l.begin('warmup_R');l.before_forward();l.after_forward()
        rows=self.path.read_bytes().splitlines(keepends=True);self.path.write_bytes(b''.join(rows[:1]+rows[2:]))
        with self.assertRaises(JournalError):self.recovery()
        l.writer.poisoned=True

    def test_forged_snapshot_refused_even_with_valid_checksum(self):
        self.ledger();self.rewrite(lambda rows:rows[0]['state'].update(target_attempted=1))
        with self.assertRaisesRegex(JournalError,'snapshot'):self.recovery()
        self.l.writer.poisoned=True

    def test_boolean_count_not_accepted_as_int(self):
        self.ledger();self.rewrite(lambda rows:rows[0]['state'].update(target_attempted=False))
        with self.assertRaisesRegex(JournalError,'snapshot'):self.recovery()
        self.l.writer.poisoned=True

    def test_invalid_owner_refused(self):
        self.ledger();self.rewrite(lambda rows:rows[0]['owner'].update(start_ticks=None))
        with self.assertRaisesRegex(JournalError,'owner'):self.recovery()
        self.l.writer.poisoned=True

    def test_short_writes_completed_before_math(self):
        original=os.write
        with patch('counter_journal.os.write',side_effect=lambda fd,b:original(fd,b[:19])):
            l=self.ledger();l.begin('warmup_R');l.before_forward();l.after_forward()
        self.assertEqual(self.recovery()['ledger']['target_completed'],1)

    def test_fsync_failure_stops_before_math(self):
        l=self.ledger();l.begin('warmup_R');entered=[]
        with patch('counter_journal.os.fsync',side_effect=OSError('injected disk fault')):
            with self.assertRaises(OSError):
                l.before_forward();entered.append('math')
        self.assertEqual(entered,[]);self.assertTrue(l.writer.poisoned)
        with self.assertRaises(JournalError):l.before_forward()
        r=self.recovery();self.assertEqual(r['ledger']['target_completed'],0)
        self.assertFalse(r['exact_math_counts'])

    def test_close_is_idempotent_and_terminal(self):
        l=self.ledger();l.close();h=digest(self.path);l.close()
        self.assertEqual(h,digest(self.path));self.assertEqual(self.recovery()['status'],'CLOSED')
        with self.assertRaises(JournalError):l.before_forward()

    def test_missing_old_GPU_counter_not_replaced_by_parent_zero(self):
        old=OWN.parent.parent/'results/runtime/p0-single-worker-chc-v2-20261012-102720'
        r=recover(old/'worker/counter.jsonl',dict(job=str(old),plan_sha256='old',kind='CPU-only-counter-fix'))
        self.assertIsNone(r['ledger']);self.assertEqual(r['status'],'UNKNOWN_NO_JOURNAL')

    def test_GPU_entry_refused_even_with_new_and_old_flags(self):
        with patch.dict(os.environ,{'P0_CHC_V2_GPU_APPROVED':'1','P0_SINGLE_WORKER_CHC_GPU_APPROVED':'1'}):
            with self.assertRaisesRegex(FileNotFoundError,'CPU-GATE.json'):gpu_entry_gate(JOB)


def case_plan(name, **updates):
    job=JOB/'workers'/('case-'+name);job.mkdir(parents=True,exist_ok=False)
    p=read_cpu_plan(JOB);p={**p,'job':str(job),'CPU_exit_after_one_command':True,
        'CPU_only':True,'GPU_run_approved':False,'gpu_run_approved':False,
        'parent_plan_sha256':digest(JOB/'plan.json'),**updates}
    write(job/'cpu-plan.json',p,exclusive=True)
    base=job/'preflight/cpu-worker';(base/'inbox').mkdir(parents=True);(base/'outbox').mkdir()
    return job,p,base


class ActualWorker(U.TestCase):
    def wait_file(self,path,child,log,timeout=100):
        until=time.monotonic()+timeout
        while not path.exists():
            if child.poll() is not None:self.fail('actual worker exited: '+log.read_text()[-5000:])
            if time.monotonic()>until:self.fail('actual worker boundary timeout: '+str(path))
            time.sleep(.02)

    def run_case(self,name,fault=None,kind='KILL',expected=(2,1,1)):
        job,p,base=case_plan(name,CPU_test_fault=fault)
        cmd=[sys.executable,str(OWN/'worker.py'),'--job',str(job),'--cpu-serve']
        log=job/'worker.log';pins={};child=None
        with log.open('xb') as out:
            child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
                                   stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            ownership.pin_roots([dict(pid=child.pid,process_group=child.pid,command=cmd)],pins)
            def send(sig):
                info=ownership.read_process(child.pid);root=pins[child.pid]
                self.assertEqual((info.start_ticks,info.uid,info.pgid,info.argv),
                                 (root.start_ticks,root.uid,root.pgid,root.argv))
                os.killpg(child.pid,sig)
            self.wait_file(base/'SERVING.json',child,log)
            ready=json.loads((base/'ready.json').read_text());self.assertEqual(ready['device'],'cpu')
            self.assertFalse(ready['CUDA_initialized']);self.assertTrue(ready['CPU_probe'])
            parent_ack=dict(target_attempted=0,target_completed=0)
            if kind!='IDLE_TERM':
                write(base/'inbox/0000.json',dict(id=0,job=str(job),stage='warmup_R',op='sequence',arm='R',timed=True),exclusive=True)
                if fault:self.wait_file(base/'PAUSE.json',child,log)
            if kind in ('TERM','IDLE_TERM'):
                send(signal.SIGTERM)
                if fault:time.sleep(.05);(base/'RELEASE').write_text('release owned CPU worker\n')
            elif kind=='TERM_KILL':
                send(signal.SIGTERM);time.sleep(.05);self.assertIsNone(child.poll());send(signal.SIGKILL)
            elif kind=='KILL':send(signal.SIGKILL)
            code=child.wait(timeout=30)
            self.assertEqual(code,0 if kind in ('TERM','IDLE_TERM','NORMAL') else -signal.SIGKILL,log.read_text()[-5000:])
            identity=dict(job=str(job),plan_sha256=digest(job/'cpu-plan.json'),kind='CPU-only-counter-fix')
            recovered=recover(base/'counter.jsonl',identity);s=recovered['ledger']
            from counter_recovery import recover_owned,budget_from_recovery
            pin=pins[child.pid]
            owned=recover_owned(base/'counter.jsonl',identity,
                expected_owner=dict(pid=pin.pid,uid=pin.uid,start_ticks=pin.start_ticks),parent_last_ack=parent_ack)
            derived=budget_from_recovery(parent_ack,owned)
            self.assertEqual(derived['target_attempted'],s['target_attempted'])
            self.assertEqual(derived['target_completed'],s['target_completed'])
            self.assertFalse(owned['whole_stage_success_inferred']);self.assertFalse(owned['raw_journal_modified'])
            self.assertEqual((s['target_attempted'],s['target_completed'],s['target_inflight']),expected)
            self.assertFalse(recovered['writer_same_process_alive']);self.assertFalse(recovered['CUDA_completion_proven'])
            self.assertLessEqual(recovered['unconfirmed_call_upper_bound'],1)
            if kind in ('KILL','TERM_KILL'):
                self.assertFalse((base/'outbox/0000.json').exists());self.assertFalse((base/'EXIT.json').exists())
                self.assertEqual(recovered['status'],'INTERRUPTED_VERIFIED_PREFIX')
            else:
                self.assertTrue((base/'EXIT.json').exists());self.assertEqual(recovered['status'],'CLOSED')
                exit=json.loads((base/'EXIT.json').read_text())
                self.assertEqual(exit['stop_signal'],signal.SIGTERM if kind!='NORMAL' else 0)
            if kind=='NORMAL':
                row=json.loads((base/'outbox/0000.json').read_text());self.assertEqual(row['status'],'PASS')
                self.assertTrue(row['bitwise_all_outputs'] and row['all30_history_metadata_RNG_unchanged'])
            result=dict(case=name,status='PASS',actual_Runtime_serve=True,actual_production_loader_module='modules.model',
                        layers=30,batch=14,real_public_model_math=True,CPU_only=True,trained_target_weights=False,
                        exit_code=code,parent_last_ack=parent_ack,recovered_counter=recovered,
                        outbox_ack=(base/'outbox/0000.json').exists(),EXIT_present=(base/'EXIT.json').exists(),
                        ownership_verified_signals_only=True,all_owned_processes_gone=True,
                        GPU_started=False,training=False)
            write(job/'RESULT.json',result,exclusive=True);CASE_RESULTS.append(result)
        finally:
            if child.poll() is None:
                info=ownership.read_process(child.pid);root=pins[child.pid]
                if (info.start_ticks,info.uid,info.pgid,info.argv)==(root.start_ticks,root.uid,root.pgid,root.argv):
                    os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=5)

    def test_normal_real_worker_all29(self):self.run_case('normal',kind='NORMAL',expected=(29,29,0))
    def test_SIGTERM_idle_genuine_zero(self):self.run_case('term-idle',kind='IDLE_TERM',expected=(0,0,0))
    def test_SIGTERM_inside_second_real_forward(self):self.run_case('term-mid','mid_forward_2','TERM',(2,2,0))
    def test_SIGKILL_inside_second_real_forward(self):self.run_case('kill-mid','mid_forward_2')
    def test_SIGKILL_after_BEGIN_before_math(self):self.run_case('kill-begin','after_BEGIN',expected=(0,0,0))
    def test_SIGKILL_intent_committed_before_second_math(self):self.run_case('kill-before','after_BEFORE_2')
    def test_SIGKILL_second_return_before_RETURN_commit(self):self.run_case('kill-return-gap','after_model_return_2')
    def test_SIGKILL_after29_before_stage_ack(self):self.run_case('kill-pre-ack','pre_ack',expected=(29,29,0))
    def test_SIGTERM_then_SIGKILL_escalation(self):self.run_case('term-kill','mid_forward_2','TERM_KILL')

    def test_all30_BF16_29changing_R_C_HC_with_persistence(self):
        job,p,base=case_plan('bf16',CPU_fixture_dtype='bfloat16')
        result=subprocess.run([sys.executable,str(OWN/'worker.py'),'--job',str(job),'--cpu-probe'],
                              env=os.environ,capture_output=True,text=True,timeout=120)
        (job/'probe.log').write_text(result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,result.stderr)
        row=json.loads((job/'preflight/worker-CPU-probe.json').read_text())
        self.assertEqual(row['status'],'PASS');self.assertFalse(row['CUDA_initialized'])
        self.assertTrue(all(v['bitwise_all_outputs'] and v['all30_history_metadata_RNG_unchanged'] for v in row['results']))
        self.assertEqual(row['recovered_counter']['ledger']['target_attempted'],87)
        self.assertEqual(row['recovered_counter']['ledger']['target_completed'],87)
        self.assertEqual(row['recovered_counter']['status'],'CLOSED')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    JOB=Path(a.job).resolve();read_cpu_plan(JOB)
    class Recording(U.TextTestResult):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.executed=[]
        def startTest(self,test):self.executed.append(test.id());super().startTest(test)
    r=U.TextTestRunner(verbosity=2,resultclass=Recording).run(U.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    import test_cpu as fixture
    status='PASS' if r.wasSuccessful() and not fixture.CUDA_ATTEMPTS and not fixture.torch.cuda.is_initialized() else 'FAIL'
    write(a.out,dict(status=status,tests_run=r.testsRun,executed_tests=r.executed,
        tests_by_class={k:sum('.'+k+'.' in v for v in r.executed) for k in ('Journal','ActualWorker')},
        failures=[dict(test=str(x),trace=y) for x,y in r.failures],errors=[dict(test=str(x),trace=y) for x,y in r.errors],
        actual_worker_termination_cases=CASE_RESULTS,CPU_only=True,CUDA_initialized=fixture.torch.cuda.is_initialized(),
        forbidden_CUDA_attempts=fixture.CUDA_ATTEMPTS,GPU_started=False,training=False),exclusive=True)
    raise SystemExit(0 if status=='PASS' else 1)

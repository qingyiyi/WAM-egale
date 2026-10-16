"""CPU production-loader integration and actual client/budget guards.

Tiny random CPU weights validate interfaces/lifecycle only. GPU full chunk
bitwise acceptance is a separate prerequisite to formal E2E collection.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from types import SimpleNamespace as NS

from task_contract import OWN, BASE, WORKTREE, digest, write

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicit CPU-only gate required')
    # Avoid this new file shadowing the pinned historical CPU fixture module.
    path = BASE/'test_cpu.py'
    spec = importlib.util.spec_from_file_location('test_cpu',path)
    f = importlib.util.module_from_spec(spec)
    sys.modules['test_cpu'] = f
    spec.loader.exec_module(f)
    from production_fixture import loaded_model
    from prefix_adapter import SharedPrefix, LeaseError
    from online_shared import OnlineShared
    from runtime import Runtime
    from resource_guard import Idle, MEMORY_ACCOUNTING
    from client_entry import install as client_install
    from supervisor import charged_reservations
    from launch_identity import pin as pin_launch
    import ownership
    t = f.torch
    m = loaded_model()
    x = [f.inputs(i,batch=7) for i in range(3)]
    ordinary = {k:v[:2] for k,v in f.inputs(batch=2).items()}
    base_state = f.cache_state(m)
    negative = t.zeros(1,3,8)

    class Server:
        _repeat_action_candidates_for_cfg = f.public_server.VA_Server._repeat_action_candidates_for_cfg
        def __init__(self):
            self.transformer=m
            self.use_cfg=True
            self.negative_prompt_embeds=negative
            self.device=t.device('cpu')
            self.dtype=t.float32
            self.cache_name='pos'
            self.frame_st_id=4
        def _phase_profile_counter(self,name,value=1): return None
        def _infer_action_verifier_k2_b2(self, *, mutate=False, fail=False, bad_thread=False):
            # Actual ordinary forward precedes candidates, just as original K2.
            self.transformer(ordinary,action_mode=True,cache_name='pos',update_cache=0)
            outputs=[]
            for i,inp in enumerate(x):
                packed,rows=self._repeat_action_candidates_for_cfg(inp)
                outputs.append(f.candidate(self.transformer,packed,rows))
                if fail: raise RuntimeError('probe failure after first candidate')
                if i==0 and mutate:
                    # Original ordinary target forward revokes even at update=0.
                    self.transformer(ordinary,action_mode=True,cache_name='pos',update_cache=0)
            # Include a final ordinary forward before cleanup (terminal boundary).
            self.transformer(ordinary,action_mode=True,cache_name='pos',update_cache=0)
            return outputs
        def infer(self,request):
            return self.transformer(ordinary,action_mode=True,cache_name='pos',update_cache=0)

    def direct_shared(server):
        result=[]
        with t.no_grad():
            m(ordinary,action_mode=True,cache_name='pos',update_cache=0)
        with t.no_grad(),SharedPrefix(m,enabled=True,cpu_reference_enabled=True) as c:
            with c.lease(request='CPU-explicit-S0',arm='S') as token:
                for inp in x:
                    packed,original_rows=server._repeat_action_candidates_for_cfg(inp)
                    assert t.equal(original_rows,token.rows)
                    result.append(f.candidate(m,packed,token.rows))
        with t.no_grad():
            m(ordinary,action_mode=True,cache_name='pos',update_cache=0)
        return result

    class Checks(unittest.TestCase):
        def tearDown(self):
            for block,saved in zip(m.blocks,base_state):
                f.SNAPSHOT.restore_cache(block.attn1.attn_caches['pos'],saved)
            self.assertFalse(t.cuda.is_initialized())
            self.assertFalse(f.CUDA_ATTEMPTS)

        def test_actual_production_loader_30_layers(self):
            self.assertEqual(type(m).__module__,'modules.model')
            self.assertEqual(len(m.blocks),30)
            self.assertEqual(digest(WORKTREE/'wan_va/modules/model.py'),f.MODEL_SHA256)

        def test_accepted_S0_wrappers_are_byte_identical(self):
            for n in ('prefix_adapter.py','prefix_attention.py','prefix_kernel.py'):
                self.assertEqual(digest(OWN/n),digest(OWN.parent/'shared_prefix_torch_lengths_v1'/n))

        def test_default_disabled_original_output_and_no_patch(self):
            s=Server();before=dict(vars(s));old=m.blocks[0].attn1.forward
            control=OnlineShared(s).install()
            with t.no_grad(): output=s._infer_action_verifier_k2_b2()
            control.close()
            self.assertEqual(vars(s),before)
            self.assertEqual(m.blocks[0].attn1.forward,old)
            self.assertEqual(len(output),3)

        def test_online_matches_explicit_accepted_CPU_S0_bitwise(self):
            s=Server();gold=direct_shared(s)
            expected_state=f.cache_state(m);rng=t.get_rng_state().clone()
            for block,saved in zip(m.blocks,base_state):
                f.SNAPSHOT.restore_cache(block.attn1.attn_caches['pos'],saved)
            c=OnlineShared(s,enabled=True,cpu_reference_enabled=True).install()
            try:
                with t.no_grad(): actual=s._infer_action_verifier_k2_b2()
                self.assertTrue(all(t.equal(a,b) for a,b in zip(gold,actual)))
                self.assertEqual(c.leases,1);self.assertEqual(c.candidates,3)
                self.assertTrue(c.last_clean);self.assertIsNone(c.control)
                self.assertTrue(f.same_state(expected_state,f.cache_state(m)))
                self.assertTrue(t.equal(rng,t.get_rng_state()))
            finally: c.close()

        def test_ordinary_forward_revocation_reopens_for_next_candidate(self):
            s=Server();gold=direct_shared(s)
            c=OnlineShared(s,enabled=True,cpu_reference_enabled=True).install()
            try:
                with t.no_grad(): actual=s._infer_action_verifier_k2_b2(mutate=True)
                self.assertEqual(c.leases,2);self.assertEqual(c.revocations,1)
                self.assertTrue(all(t.equal(a,b) for a,b in zip(gold,actual)))
                self.assertTrue(c.last_clean)
            finally: c.close()

        def test_two_chunks_use_fresh_requests_and_restore_all_layers(self):
            s=Server();original=[b.attn1.forward for b in m.blocks]
            c=OnlineShared(s,enabled=True,cpu_reference_enabled=True).install()
            try:
                with t.no_grad():
                    a=s._infer_action_verifier_k2_b2();b=s._infer_action_verifier_k2_b2()
                self.assertEqual(c.requests,2);self.assertEqual(c.leases,2)
                self.assertTrue(all(t.equal(v,w) for v,w in zip(a,b)))
                self.assertTrue(all(block.attn1.forward==old for block,old in zip(m.blocks,original)))
            finally: c.close()

        def test_exception_cleans_then_next_request_can_run(self):
            s=Server();original=[b.attn1.forward for b in m.blocks]
            c=OnlineShared(s,enabled=True,cpu_reference_enabled=True).install()
            try:
                with t.no_grad(),self.assertRaisesRegex(RuntimeError,'probe failure'):
                    s._infer_action_verifier_k2_b2(fail=True)
                self.assertTrue(c.last_clean);self.assertIsNone(c.control)
                self.assertTrue(all(block.attn1.forward==old for block,old in zip(m.blocks,original)))
                with t.no_grad(): s._infer_action_verifier_k2_b2()
                self.assertEqual(c.requests,2)
            finally: c.close()

        def test_actual_wrong_CFG_rows_fail_closed(self):
            s=Server();original=s._repeat_action_candidates_for_cfg
            def bad(inp):
                packed,rows=original(inp);return packed,rows.flip(0)
            s._repeat_action_candidates_for_cfg=bad
            c=OnlineShared(s,enabled=True,cpu_reference_enabled=True).install()
            try:
                with t.no_grad(),self.assertRaisesRegex(LeaseError,'row values changed'):
                    s._infer_action_verifier_k2_b2()
                self.assertTrue(c.last_clean)
            finally:c.close()

        def test_outside_request_candidates_refused(self):
            s=Server();c=OnlineShared(s,enabled=True,cpu_reference_enabled=True).install()
            try:
                with self.assertRaisesRegex(LeaseError,'outside owned action thread'):
                    s._repeat_action_candidates_for_cfg(x[0])
            finally:c.close()

        def test_adapter_constructed_in_actual_inference_thread(self):
            s=Server();c=OnlineShared(s,enabled=True,cpu_reference_enabled=True).install()
            errors=[]
            def run():
                try:
                    with t.no_grad():s._infer_action_verifier_k2_b2()
                except BaseException as e:errors.append(repr(e))
            worker=threading.Thread(target=run);worker.start();worker.join(30)
            try:
                self.assertFalse(worker.is_alive());self.assertFalse(errors)
                self.assertTrue(c.last_clean)
            finally:c.close()

        def test_hook_counter_cap_before_attempt_no_per_forward_disk(self):
            s=Server()
            with tempfile.TemporaryDirectory() as tmp:
                rt=Runtime(s,Path(tmp),2)
                try:
                    with t.no_grad():s.infer({});s.infer({})
                    self.assertEqual((rt.attempted,rt.returned),(2,2))
                    with t.no_grad(),self.assertRaisesRegex(RuntimeError,'finite target forward'):
                        s.infer({})
                    self.assertEqual(rt.attempted,2);self.assertEqual(rt.active,0)
                    rt.close()
                    result=json.loads((Path(tmp)/'COUNTER-CLOSED.json').read_text())
                    self.assertFalse(result['per_forward_filesystem_IO'])
                    self.assertEqual(result['target_intents'],2)
                finally:
                    for h in rt.hooks:h.remove()
                    rt.shared.close()

        def test_idle_requires_both_process_kinds_and_memory_util(self):
            sample=dict(memory_accounting=MEMORY_ACCOUNTING,memory_used_bytes=0,
                        compute=[],graphics=[],utilization_percent=0)
            idle=Idle(samples=3)
            self.assertFalse(idle.add(sample,1));self.assertFalse(idle.add(sample,2))
            self.assertFalse(idle.add({**sample,'graphics':[{'pid':99}]},3))
            for ts in (4,5):self.assertFalse(idle.add(sample,ts))
            self.assertTrue(idle.add(sample,6))

        def test_client_full_horizon_source_and_no_hidden_construction_retries(self):
            import inspect
            import client_entry
            text=inspect.getsource(client_entry.install)
            self.assertIn('return cls(**args)',text)
            self.assertNotIn('range(5)',text)
            self.assertIn("kwargs['chunk_events_path'] = None",text)
            base=(WORKTREE.parent/'scripts/p0_libero_multichunk_client.py').read_text()
            self.assertIn('while _simulator_timestep(env, simulator_timestep_start + env_steps) < LIBERO_HORIZON:',base)
            self.assertIn('"truncated_after_max_policy_chunks": False',base)

        def test_all_new_Python_compiles_without_CUDA(self):
            for path in OWN.glob('*.py'):
                compile(path.read_text(),str(path),'exec')

        def test_unclosed_or_crashed_worker_charges_entire_reservation(self):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                write(out/'counter-state.json',dict(target_intents=3))
                v=charged_reservations(out,[dict(out=str(out),cap=4000)])
                self.assertEqual(v['charged_target_intents'],4000)
                self.assertFalse(v['reservations'][0]['exact_closed_count'])

        def test_clean_durable_reservation_uses_actual_intents(self):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                write(out/'COUNTER-CLOSED.json',dict(status='CLOSED',cap=4000,target_intents=72,
                     host_returned=72,failed_forwards=0,active_forwards=0,request_active=False))
                v=charged_reservations(out,[dict(out=str(out),cap=4000)])
                self.assertEqual(v['charged_target_intents'],72)
                self.assertTrue(v['reservations'][0]['exact_closed_count'])

        def test_corrupt_terminal_forward_accounting_is_rejected(self):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                write(out/'COUNTER-CLOSED.json',dict(status='CLOSED',cap=4000,target_intents=73,
                     host_returned=72,failed_forwards=0,active_forwards=0,request_active=False))
                with self.assertRaisesRegex(ValueError,'invalid closed reservation'):
                    charged_reservations(out,[dict(out=str(out),cap=4000)])

        def test_launch_cmdline_transient_empty_waits_for_same_PID(self):
            info=ownership.Process(91,92,91,100,os.geteuid(),('python','task'))
            transient=ownership.Process(91,92,91,100,os.geteuid(),())
            sequence=iter((transient,info,info));ticks=[0.]
            def sleep(v):ticks[0]+=v
            actual=pin_launch(NS(pid=91,poll=lambda:None),info.argv,parent_pid=92,
                    lookup=lambda pid:next(sequence),clock=lambda:ticks[0],sleep=sleep)
            self.assertEqual(actual,info);self.assertGreater(ticks[0],0)

        def test_launch_PID_reuse_while_waiting_is_refused(self):
            a=ownership.Process(91,92,91,100,os.geteuid(),())
            b=ownership.Process(91,92,91,101,os.geteuid(),('python','task'))
            sequence=iter((a,b));ticks=[0.]
            def sleep(v):ticks[0]+=v
            with self.assertRaisesRegex(RuntimeError,'PID identity changed'):
                pin_launch(NS(pid=91,poll=lambda:None),b.argv,parent_pid=92,
                    lookup=lambda pid:next(sequence),clock=lambda:ticks[0],sleep=sleep)

        def test_launch_failure_cleans_tracked_new_child_even_before_full_pin(self):
            import supervisor
            campaign=supervisor.Campaign.__new__(supervisor.Campaign)
            campaign.guard=lambda:None;campaign.roots=[];campaign.pins={};campaign.boots={}
            campaign.model_launches=0
            old=supervisor.pin_launch
            def fail(p,cmd,on_boot):
                on_boot(ownership.read_process(p.pid))
                raise RuntimeError('injected pin failure')
            supervisor.pin_launch=fail
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    campaign.job=Path(tmp)
                    with self.assertRaisesRegex(RuntimeError,'injected pin failure'):
                        campaign.spawn([sys.executable,'-c','import time; time.sleep(60)'],
                                       Path(tmp)/'child.log',dict(os.environ))
                    self.assertEqual(len(campaign.roots),1)
                    self.assertIsNotNone(campaign.roots[0].poll())
                    self.assertFalse(campaign.pins)
                    self.assertTrue(list(Path(tmp).glob('LAUNCH-PIN-FAILED-*.json')))
            finally:supervisor.pin_launch=old

        def test_actual_five_CPU_launches_pin_exact_UID_start_argv(self):
            for _ in range(5):
                cmd=[sys.executable,'-c','import time; time.sleep(1)']
                proc=subprocess.Popen(cmd,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
                try:
                    info=pin_launch(proc,cmd)
                    self.assertEqual(info.argv,tuple(cmd));self.assertEqual(info.pgid,proc.pid)
                    self.assertEqual(info.ppid,os.getpid())
                finally:
                    if proc.poll() is None:proc.terminate()
                    proc.wait(timeout=3)

        def test_finite_disk_scope_entry_floor_and_output_limit(self):
            import disk_guard
            p=dict(disk_initial_min_bytes=16*2**30,disk_runtime_floor_bytes=12*2**30,
                   new_output_bytes_cap=2**30)
            disk_guard.entry(39*2**30,p);disk_guard.runtime(15*2**30,2**30,p)
            with self.assertRaisesRegex(RuntimeError,'initial free disk'):
                disk_guard.entry(15*2**30,p)
            with self.assertRaisesRegex(RuntimeError,'runtime floor'):
                disk_guard.runtime(11*2**30,0,p)
            with self.assertRaisesRegex(RuntimeError,'output byte cap'):
                disk_guard.runtime(39*2**30,2**30+1,p)

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    write(args.out,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
          failures=[str(v) for v in result.failures],errors=[str(v) for v in result.errors],
          production_loader_all30=True,CUDA_initialized=t.cuda.is_initialized(),
          CUDA_attempts=f.CUDA_ATTEMPTS,claim='CPU interface/lifecycle only; separate GPU bitwise gate'),exclusive=True)
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()

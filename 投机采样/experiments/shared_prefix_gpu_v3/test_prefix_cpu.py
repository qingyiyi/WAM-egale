"""CPU mathematical oracle, real public interface and zero-copy regressions."""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest as U
from unittest.mock import patch

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('CPU preflight needs empty CUDA visibility')
BASE = Path(__file__).resolve().parent.parent / 'single_worker_chc_gpu_v4'
sys.path.append(str(BASE))
import test_cpu as f
from production_fixture import loaded_model
from prefix_adapter import SharedPrefix
from adapter import LeaseError
from prefix_attention import cpu_reference, shared_attention
torch = f.torch


def tensors(length=65, dtype=torch.float64, strided=False):
    g = torch.Generator().manual_seed(99)
    q = torch.randn((14, 16, 2, 4), dtype=dtype, generator=g)
    k = torch.randn(q.shape, dtype=dtype, generator=g)
    v = torch.randn(q.shape, dtype=dtype, generator=g)
    capacity = 2*length+3
    pk = torch.full((2, capacity, 2, 4), float('nan'), dtype=dtype)
    pv = torch.full_like(pk, float('nan'))
    slots = torch.arange(length, dtype=torch.long)*2+1
    pk[:, slots] = torch.randn((2, length, 2, 4), dtype=dtype, generator=g)
    pv[:, slots] = torch.randn((2, length, 2, 4), dtype=dtype, generator=g)
    if strided:
        q, k, v, pk, pv = (x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v, pk, pv))
    return q, pk, pv, k, v, slots, torch.tensor(f.ROWS)


def oracle(q, pk, pv, k, v, slots, rows):
    # Fourteen-row expansion is allowed ONLY in the independent baseline.
    kk = torch.cat((pk[:, slots].index_select(0, rows), k), 1)
    vv = torch.cat((pv[:, slots].index_select(0, rows), v), 1)
    return torch.nn.functional.scaled_dot_product_attention(
        q.double().transpose(1, 2), kk.double().transpose(1, 2), vv.double().transpose(1, 2)
    ).transpose(1, 2).to(q.dtype)


class Math(U.TestCase):
    def test_lengths_empty_tail_short_long(self):
        for l in (0, 1, 17, 63, 65, 2016, 2144):
            with self.subTest(l=l):
                x = tensors(l)
                torch.testing.assert_close(cpu_reference(*x), oracle(*x), rtol=1e-12, atol=1e-12)

    def test_joint_not_sum_of_two_softmaxes(self):
        q, pk, pv, k, v, slots, rows = tensors(48)
        q.zero_(); pk[:, slots] = 0; k.zero_(); pv[:, slots] = 1; v.zero_()
        out = cpu_reference(q, pk, pv, k, v, slots, rows)
        self.assertTrue(torch.equal(out, torch.full_like(out, .75)))

    def test_unused_nan_slots_not_read(self):
        x = tensors(); self.assertTrue(torch.isfinite(cpu_reference(*x)).all())

    def test_noncontiguous_shapes(self):
        x = tensors(strided=True)
        self.assertFalse(x[1].is_contiguous())
        torch.testing.assert_close(cpu_reference(*x), oracle(*x), rtol=1e-12, atol=1e-12)

    def test_candidate_current_isolation(self):
        x = tensors(); a = cpu_reference(*x)
        x[3][0].add_(3); x[4][0].add_(4)
        b = cpu_reference(*x)
        self.assertTrue(torch.equal(a[1:], b[1:])); self.assertFalse(torch.equal(a[0], b[0]))

    def test_CFG_not_shared(self):
        x = tensors(); a = cpu_reference(*x)
        x[2][0, x[5]] += 1
        b = cpu_reference(*x)
        self.assertTrue(torch.equal(a[7:], b[7:])); self.assertFalse(torch.equal(a[:7], b[:7]))

    def test_dtype_fp32_bf16(self):
        for dt in (torch.float32, torch.bfloat16):
            with self.subTest(dt=dt):
                x = tensors(65, dt)
                self.assertTrue(torch.equal(cpu_reference(*x), oracle(*x)))

    def test_default_off_before_kernel_import(self):
        with self.assertRaisesRegex(ValueError, 'default disabled'):
            shared_attention(*tensors())
        self.assertNotIn('prefix_kernel', sys.modules)

    def test_explicit_CPU_reference_required(self):
        with self.assertRaisesRegex(ValueError, 'explicit CPU'):
            shared_attention(*tensors(), enabled=True)

    def test_bad_row_map(self):
        x = list(tensors()); x[-1][0] = 1
        with self.assertRaisesRegex(ValueError, 'canonical'):
            cpu_reference(*x)

    def test_bad_slots(self):
        for mode in ('negative', 'overflow', 'duplicate', 'unordered'):
            x = list(tensors())
            if mode == 'negative': x[-2][0] = -1
            elif mode == 'overflow': x[-2][0] = x[1].shape[1]
            elif mode == 'duplicate': x[-2][1] = x[-2][0]
            else: x[-2] = x[-2].flip(0)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'sorted unique'):
                cpu_reference(*x)

    def test_prefix_one_row_refused(self):
        x = list(tensors()); x[1] = x[1][:1]; x[2] = x[2][:1]
        with self.assertRaisesRegex(ValueError, 'two CFG'):
            cpu_reference(*x)


class Interface(U.TestCase):
    def test_frozen_HC_baseline_bitwise_and_mode_release(self):
        m=f.model();x=f.inputs()
        with torch.no_grad():
            gold=f.candidate(m,x,torch.tensor(f.ROWS))
            with SharedPrefix(m,enabled=True,expected_layers=1,cpu_reference_enabled=True) as c:
                with c.lease(request='CPU-frozen-HC',arm='HC') as token:
                    self.assertTrue(torch.equal(gold,f.candidate(m,x,token.rows)))
                    self.assertEqual(c.entries[0].k.shape[0],14)
                self.assertIsNone(c.current_arm)
                with c.lease(request='CPU-S-after-HC',arm='S') as token:
                    f.candidate(m,x,token.rows)
                    self.assertEqual(c.entries[0].k.shape[0],2)

    def test_old_C_arm_not_silently_accepted(self):
        m=f.model()
        with SharedPrefix(m,enabled=True,expected_layers=1,cpu_reference_enabled=True) as c:
            with self.assertRaisesRegex(LeaseError,'R, frozen HC and S'):
                with c.lease(request='CPU-C-forbidden',arm='C'): pass

    def test_default_off_retains_original(self):
        m = f.model(); x = f.inputs(); orig = m.blocks[0].attn1.forward
        with torch.no_grad():
            a = f.candidate(m, x, torch.tensor(f.ROWS))
            with SharedPrefix(m):
                self.assertEqual(m.blocks[0].attn1.forward, orig)
                self.assertTrue(torch.equal(a, f.candidate(m, x, torch.tensor(f.ROWS))))

    def test_zero_copy_no_expand_no_fullcat(self):
        m = f.model(); x = f.inputs()
        with torch.no_grad(), SharedPrefix(m, enabled=True, expected_layers=1, cpu_reference_enabled=True, diagnostic=True) as c:
            with c.lease(request='CPU-zero-copy') as token:
                f.candidate(m, x, token.rows); e = c.entries[0]
                pool = m.blocks[0].attn1.attn_caches['pos']
                self.assertIs(e.k, pool['k']); self.assertIs(e.v, pool['v'])
                self.assertEqual(e.k.shape[0], 2)
                self.assertEqual(c.payload_bytes, 6*8)
                f.candidate(m, x, token.rows)
                self.assertEqual(c.ops[0], dict(metadata=1, row_check=1, shared_attention=2))
            self.assertFalse(c.entries); self.assertEqual(c.payload_bytes, 0)

    def test_inference_lease_and_private_slot_mutation_refused(self):
        m=f.model();x=f.inputs()
        with SharedPrefix(m,enabled=True,expected_layers=1,cpu_reference_enabled=True) as c,torch.inference_mode():
            with c.lease(request='CPU-inference-slots') as token:
                f.candidate(m,x,token.rows)
                self.assertIsInstance(c.entries[0].slots._version,int)
                c.entries[0].slots[0]=3
                with self.assertRaisesRegex(LeaseError,'valid-slot index changed'):
                    f.candidate(m,x,token.rows)
                self.assertFalse(c.entries);self.assertTrue(token.revoked)

    def test_all30_real_production_loader_changing_inputs(self):
        m = loaded_model(); xs = [f.inputs(i) for i in range(3)]
        history = f.cache_state(m); rng = torch.random.get_rng_state().clone()
        with torch.no_grad():
            gold = [f.candidate(m, x, torch.tensor(f.ROWS)) for x in xs]
            with SharedPrefix(m, enabled=True, cpu_reference_enabled=True) as c:
                with c.lease(request='CPU-real-loader-all30') as token:
                    out = [f.candidate(m, x, token.rows) for x in xs]
                    self.assertEqual(len(c.entries), 30)
                    self.assertTrue(all(e.k.shape[0] == 2 for e in c.entries.values()))
                    self.assertEqual(c.payload_bytes, 30*6*8)
        # Mathematical replacement changes floating-point reduction order;
        # this is an INTERFACE test, not the old GPU bitwise acceptance gate.
        for a, b in zip(out, gold):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
        self.assertTrue(f.same_state(history, f.cache_state(m)))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_live_history_write_refused(self):
        m = f.model(); x = f.inputs()
        with torch.no_grad(), SharedPrefix(m, enabled=True, expected_layers=1, cpu_reference_enabled=True) as c:
            with c.lease(request='CPU-mutation') as token:
                f.candidate(m, x, token.rows)
                m.blocks[0].attn1.attn_caches['pos']['v'][:, 1].add_(1)
                with self.assertRaisesRegex(LeaseError, 'history/backend/model key'):
                    f.candidate(m, x, token.rows)
                self.assertTrue(token.revoked); self.assertFalse(c.entries)

    def test_row_map_mutation_refused(self):
        m = f.model(); x = f.inputs()
        with torch.no_grad(), SharedPrefix(m, enabled=True, expected_layers=1, cpu_reference_enabled=True) as c:
            with c.lease(request='CPU-rows') as token:
                f.candidate(m, x, token.rows); token.rows[0] = 1
                with self.assertRaises(LeaseError): f.candidate(m, x, token.rows)
                self.assertTrue(token.revoked); self.assertFalse(c.entries)

    def test_external_restore_generation_refused(self):
        m = f.model(); x = f.inputs()
        with torch.no_grad(), SharedPrefix(m, enabled=True, expected_layers=1, cpu_reference_enabled=True) as c:
            with c.lease(request='CPU-generation') as token:
                f.candidate(m, x, token.rows); c.invalidate('restore')
                with self.assertRaises(LeaseError): f.candidate(m, x, token.rows)
            with c.lease(request='CPU-new-generation') as token:
                f.candidate(m, x, token.rows)
                self.assertEqual(len(c.entries), 1)

    def test_history_inference_versionless_refused(self):
        m = f.model(); x = f.inputs()
        with torch.inference_mode():
            m.blocks[0].attn1.attn_caches['pos']['k'] = m.blocks[0].attn1.attn_caches['pos']['k'].clone()
        with torch.no_grad(), SharedPrefix(m, enabled=True, expected_layers=1, cpu_reference_enabled=True) as c:
            with c.lease(request='CPU-no-version') as token:
                with self.assertRaisesRegex(LeaseError, 'no mutation version'):
                    f.candidate(m, x, token.rows)

    def test_modified_backend_refused_before_patch(self):
        m = f.model(); m.blocks[0].attn1.attn_op = lambda *x: None
        with self.assertRaisesRegex(LeaseError, 'original torch SDPA'):
            SharedPrefix(m, enabled=True, expected_layers=1, cpu_reference_enabled=True).install()
        self.assertNotIn('forward', m.__dict__)

    def test_cuda_boundary_without_initialization(self):
        c = SharedPrefix(f.model(), enabled=True, expected_layers=1)
        with self.assertRaisesRegex(LeaseError, 'CUDA not authorized'): c._device('cuda')

    def test_other13_full_model_isolation(self):
        m = f.model(2); x = f.inputs(); y = {k: v.clone() for k, v in x.items()}; y['noisy_latents'][0].add_(.25)
        with torch.no_grad(), SharedPrefix(m, enabled=True, expected_layers=2, cpu_reference_enabled=True) as c:
            with c.lease(request='CPU-isolation') as token:
                a = f.candidate(m, x, token.rows); b = f.candidate(m, y, token.rows)
        self.assertTrue(torch.equal(a[1:], b[1:])); self.assertFalse(torch.equal(a[0], b[0]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    result = U.TextTestRunner(verbosity=2).run(U.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    output = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), CUDA_initialized=torch.cuda.is_initialized(),
                  CUDA_attempts=f.CUDA_ATTEMPTS, actual_public_loader_all30=True,
                  zero_copy_CFG2_no_history14_no_fullKV14=True, training=False, new_GPU_started=False,
                  CPU_math_does_not_prove_CUDA_kernel_or_GPU_bitwise=True)
    Path(a.out).write_text(json.dumps(output, indent=2)+'\n')
    if not result.wasSuccessful() or output['CUDA_initialized'] or f.CUDA_ATTEMPTS:
        raise SystemExit(1)


if __name__ == '__main__': main()

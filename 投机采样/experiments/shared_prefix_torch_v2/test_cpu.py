"""Source math preservation, CPU-only native admission, and persisted budgets."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from task_contract import OWN, digest, enabled_plan, write


class Checks(unittest.TestCase):
    def test_original_math_and_vendor(self):
        from prepare_source import generated
        original = (OWN / 'csrc/vendor/flash_attn/flash_fwd_kernel.h').read_text()
        shared, edits, _, _ = generated(original)
        self.assertEqual(shared,(OWN / 'csrc/shared_flash_fwd_kernel.h').read_text())
        self.assertEqual(len(edits),6)
        provenance = json.loads((OWN / 'SOURCE-PROVENANCE.json').read_text())
        for file,h in provenance['vendor_files'].items():
            self.assertEqual(digest(OWN / file),h,file)
        self.assertTrue(provenance['arithmetic_and_synchronization_byte_preserved'])

    def test_native_rejects_CPU_without_CUDA(self):
        import torch
        import _shared_torch_v1 as native
        self.assertEqual(torch.__version__,'2.9.0+cu126')
        self.assertEqual(torch.version.git_version,native.pytorch_commit)
        x = torch.zeros(14,16,24,128,dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError,'CUDA BF16'):
            native.dense(x,x,x)
        self.assertFalse(torch.cuda.is_initialized())

    def test_default_off(self):
        from task_contract import ENABLED
        old = os.environ.pop(ENABLED,None)
        try:
            with self.assertRaisesRegex(ValueError,'default off'):
                enabled_plan('/nonexistent')
        finally:
            if old is not None: os.environ[ENABLED] = old

    def test_counter_cap_and_recovery(self):
        from direct_counter import Counter, recover
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'counter.jsonl'; identity = dict(cpu=True)
            c = Counter(p,identity,cap=1)
            self.assertEqual(c.call('one',lambda: 123),123)
            with self.assertRaisesRegex(RuntimeError,'intent cap'):
                c.call('forbidden',lambda: self.fail('cap entered math'))
            c.close(); r = recover(p,identity,1)
            self.assertEqual(r['counter'],dict(attempted=1,returned=1,failed=0,cap=1))
            self.assertEqual(r['status'],'CLOSED')

    def test_failed_intent_never_refunded(self):
        from direct_counter import Counter, recover
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'counter.jsonl'; identity = dict(cpu=True)
            c = Counter(p,identity,cap=1)
            with self.assertRaises(ZeroDivisionError):
                c.call('failure',lambda: 1/0)
            c.close(); r = recover(p,identity,1)
            self.assertEqual(r['counter'],dict(attempted=1,returned=0,failed=1,cap=1))
            bad = p.read_bytes().replace(b'failure',b'changed'); p.write_bytes(bad)
            with self.assertRaises(ValueError): recover(p,identity,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job',required=True)
    job = Path(ap.parse_args().job)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU only')
    enabled_plan(job)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful(): raise SystemExit(1)
    import torch
    if torch.cuda.is_initialized(): raise ValueError('preflight initialized CUDA')
    write(job / 'CPU-GATE.json',dict(status='PASS',tests=result.testsRun,
        plan_sha256=digest(job / 'plan.json'),CUDA_initialized=False),exclusive=True)


if __name__ == '__main__': main()

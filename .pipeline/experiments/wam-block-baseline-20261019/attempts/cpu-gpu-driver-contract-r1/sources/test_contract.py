"""CPU semantic/negative tests; synthetic controls are NOT real-data evidence."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from .block_baseline import SequentialBlock, aligned_attention
from .benchmark_block import benchmark_plan, measure_pair, paired_order, summary
from .debug_taps import first_observed_divergence, original_taps
from .engines import shared_history_attention
from .fixtures import BlockFixture, decode, visible_snapshot
from .metrics import byte_equal, compare, state_equal
from .reference import reference_namespace, REVIEW_COMMIT
from .runtime import cpu_only, require_gpu_authorization

REPOSITORY = Path(os.environ["WAM_FROZEN_REPOSITORY"])


class LogicalComparisonTests(unittest.TestCase):
    def test_signed_zero_is_not_byte_equal(self):
        a, b = torch.tensor([0.0]), torch.tensor([-0.0])
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(byte_equal(a, b))
        self.assertFalse(compare(a, b)["bitwise_pass"])

    def test_same_nan_cannot_pass(self):
        value = torch.tensor([float("nan")])
        self.assertTrue(byte_equal(value, value))
        self.assertFalse(compare(value, value)["finite_pass"])
        self.assertFalse(compare(value, value)["bitwise_pass"])

    def test_logical_elements_not_padding(self):
        storage = torch.arange(20.).reshape(4, 5)
        view = storage[:, 1:4]
        self.assertTrue(byte_equal(view, view.clone()))
        self.assertFalse(view.is_contiguous())

    def test_shape_and_dtype_mismatch_rejected(self):
        self.assertFalse(byte_equal(torch.zeros(4), torch.zeros(2, 2)))
        self.assertFalse(compare(torch.zeros(4), torch.zeros(4, dtype=torch.float64))["bitwise_pass"])


class GuardTests(unittest.TestCase):
    def test_gpu_default_denied_before_native_import(self):
        with self.assertRaises(PermissionError):
            with aligned_attention(None):
                self.fail("must not enter")
        self.assertFalse(torch.cuda.is_initialized())

    def test_gpu_auth_missing_false_and_expired(self):
        with self.assertRaises(PermissionError):
            require_gpu_authorization(None, "performance")
        for record in ({}, dict(allow_new_gpu_jobs=False),
                       dict(allow_new_gpu_jobs=True, review_commit=REVIEW_COMMIT,
                            allowed_phases=["performance"], cleanup_deadline="2000-01-01T00:00:00+08:00")):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "auth.json"
                path.write_text(json.dumps(record))
                with self.assertRaises(PermissionError):
                    require_gpu_authorization(path, "performance")
        self.assertFalse(torch.cuda.is_initialized())

    def test_unknown_encoding_and_inconsistent_shape(self):
        with self.assertRaises(ValueError):
            decode(dict(encoding="unknown", value=torch.zeros(1), shape=[1]))
        with self.assertRaises(ValueError):
            decode(dict(encoding="tensor", value=torch.zeros(2), shape=[1]))

    def test_explicit_cfg_map_and_repeat_reference(self):
        encoded = dict(encoding="two_CFG_groups_repeated7", value=torch.tensor([0, 1]), shape=[14])
        decoded = decode([encoded, encoded])
        self.assertEqual(decoded[0].tolist(), [0] * 7 + [1] * 7)
        self.assertIs(decoded[0], decoded[1])

    def test_b14_readonly_override_allowed_but_write_rejected(self):
        fixture = BlockFixture(Path("SYNTHETIC_CONTROL"), dict(
            identity=["short", 14, "action", "first", 0], block_args=(),
            block_kwargs=dict(update_cache=0)))
        self.assertEqual(fixture.arguments(update_cache=0)[1]["update_cache"], 0)
        with self.assertRaises(ValueError):
            fixture.arguments(update_cache=1)
        with self.assertRaises(ValueError):
            fixture.arguments(update_cache=2)

    def test_shared_native_import_denied(self):
        with self.assertRaises(PermissionError):
            with shared_history_attention(None, REPOSITORY, None, None):
                self.fail("native adapter must not enter")
        self.assertFalse(torch.cuda.is_initialized())

    def test_benchmark_keeps_independent_rounds_and_pending_graph(self):
        self.assertEqual(paired_order(0), ("A", "B", "B", "A"))
        self.assertEqual(paired_order(1), ("B", "A", "A", "B"))
        plan = benchmark_plan()
        self.assertFalse(plan["formal_robot_performance_permission"])
        self.assertIn("NOT_IMPLEMENTED", plan["full_block_Graph"])
        values = [dict(A_event_ms=2, B_event_ms=value) for value in (1., 2., 4.)]
        self.assertAlmostEqual(summary(values)["ratio_geomean"], 1.)
        self.assertEqual(summary(values)["independent_rounds"], 3)

    def test_gpu_matrix_missing_authorization_cannot_probe_cuda_or_spawn(self):
        from types import SimpleNamespace
        from .gpu_matrix import execute
        with self.assertRaises(PermissionError):
            execute(SimpleNamespace(authorization=None))
        self.assertFalse(torch.cuda.is_initialized())

    def test_timer_protocol_restores_the_selected_arm_without_real_cuda(self):
        # SYNTHETIC_TIMER_PROTOCOL_ONLY: no actual time, kernel or GPU evidence.
        from types import SimpleNamespace
        class Event:
            def __init__(self, **kwargs): pass
            def record(self): pass
            def synchronize(self): pass
            def elapsed_time(self, end): return 1.0
        fake = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: None))
        restored, called = [], []
        def operation(name):
            def forward():
                self.assertEqual(restored[-1], name)
                called.append(name)
            return forward
        output = measure_pair(fake, operation("A"), operation("B"), restored.append,
                              rounds=3, repeats=1, warmup=0)
        self.assertEqual(called, list("ABBABAABABBA"))
        self.assertEqual(output["summary"]["independent_rounds"], 3)
        self.assertEqual(len(output["raw_samples"]), 12)


class SequentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = reference_namespace(REPOSITORY)

    def block_and_inputs(self):
        block = self.ns["WanTransformerBlock"](dim=8, ffn_dim=16, num_heads=2,
                   cross_attn_norm=True, attn_mode="torch").to(dtype=torch.bfloat16).eval()
        block.requires_grad_(False)
        inputs = (torch.randn(2, 2, 8, dtype=torch.bfloat16),
                  torch.randn(2, 3, 8, dtype=torch.bfloat16),
                  torch.randn(2, 2, 6, 8, dtype=torch.bfloat16),
                  torch.polar(torch.ones(2, 2, 1, 2), torch.randn(2, 2, 1, 2)))
        return block, inputs

    def test_independent_class_compilation_and_parameter_identity(self):
        block, inputs = self.block_and_inputs()
        sequential = SequentialBlock(block)
        self.assertIs(sequential.original.attn1.to_q.weight, block.attn1.to_q.weight)
        self.assertIn(REVIEW_COMMIT, block.forward.__code__.co_filename)
        with torch.inference_mode():
            self.assertTrue(compare(block(*inputs), sequential(*inputs))["bitwise_pass"])

    def test_original_cache_readonly_and_legal_write(self):
        block, inputs = self.block_and_inputs()
        sequential = SequentialBlock(block)
        for mode in (0, 1, 2):
            initial = self.ns["WanAttention"]._original_cache(4, 2, 4, "cpu", torch.bfloat16, 2)
            # A real branch control: full cache forces the original eviction rule.
            initial["mask"][:] = True
            initial["id"][:] = torch.arange(4)
            initial["k"].normal_()
            initial["v"].normal_()
            values, states = [], []
            for operation in (block, sequential):
                block.attn1.attn_caches["pos"] = {key: value.clone() for key, value in initial.items()}
                with torch.inference_mode():
                    values.append(operation(*inputs, update_cache=mode))
                states.append(visible_snapshot(block.attn1.attn_caches["pos"]))
            self.assertTrue(compare(*values)["bitwise_pass"])
            self.assertTrue(state_equal(*states)["pass_all"])

    def test_private_candidates_readonly_do_not_cross_attend(self):
        block, inputs = self.block_and_inputs()
        inputs = tuple(value.repeat_interleave(7, 0) for value in inputs)
        cache = self.ns["WanAttention"]._original_cache(4, 2, 4, "cpu", torch.bfloat16, 2)
        cache["mask"][:2] = True
        cache["k"].normal_()
        cache["v"].normal_()
        block.attn1.attn_caches["pos"] = cache
        before = visible_snapshot(cache)
        rows = torch.tensor([0] * 7 + [1] * 7)
        sequential = SequentialBlock(block)
        with torch.inference_mode():
            reference = block(*inputs, candidate_cache_rows=rows)
            candidate = sequential(*inputs, candidate_cache_rows=rows)
        self.assertTrue(compare(reference, candidate)["bitwise_pass"])
        self.assertTrue(state_equal(before, visible_snapshot(cache))["pass_all"])
        with self.assertRaises((RuntimeError, ValueError)):
            sequential(*inputs, candidate_cache_rows=rows, update_cache=1)

    def test_taps_are_observational_and_output_is_self_propagated(self):
        block, inputs = self.block_and_inputs()
        sequential = SequentialBlock(block)
        seen = {}
        with torch.inference_mode():
            reference = sequential(*inputs)
            candidate = sequential(*inputs, tap=lambda name, value: seen.update({name: value.detach().clone()}))
        self.assertTrue(compare(reference, candidate)["bitwise_pass"])
        self.assertTrue({"self.modulated_input", "self.sdpa", "self.residual",
                         "cross.sdpa", "cross.residual", "ffn.expansion",
                         "ffn.activation", "ffn.contraction", "block.output"}.issubset(seen))
        self.assertTrue(byte_equal(seen["block.output"], candidate))

    def test_reference_hooks_never_replace_outputs_and_report_common_boundaries(self):
        block, inputs = self.block_and_inputs()
        reference_seen, candidate_seen = {}, {}
        def sink(name, value):
            reference_seen[name] = value.detach().clone()
            return value + 1  # Hook must ignore even a non-None callback result.
        with torch.inference_mode():
            uninstrumented = block(*inputs)
            with original_taps(block, sink):
                reference = block(*inputs)
            candidate = SequentialBlock(block)(*inputs, tap=lambda name, value: candidate_seen.update({name: value.detach().clone()}))
        self.assertTrue(compare(uninstrumented, reference)["bitwise_pass"])
        self.assertTrue(compare(reference, candidate)["bitwise_pass"])
        report = first_observed_divergence(reference_seen, candidate_seen)
        self.assertIsNone(report["first_observed_boundary"])
        candidate_seen["ffn.expansion"] = candidate_seen["ffn.expansion"].clone()
        candidate_seen["ffn.expansion"][0, 0, 0] += 1
        self.assertEqual(first_observed_divergence(reference_seen, candidate_seen)["first_observed_boundary"], "ffn.expansion")

    def test_unsupported_cache_or_activation_not_silently_emulated(self):
        block, inputs = self.block_and_inputs()
        sequential = SequentialBlock(block)
        block.attn1.cache_mode = "lifecycle"
        with self.assertRaises(NotImplementedError):
            sequential(*inputs)
        block.ffn.net[0].approximate = "none"
        with self.assertRaises(TypeError):
            SequentialBlock(block)

    def test_state_comparison_detects_mutation(self):
        cache = self.ns["WanAttention"]._original_cache(4, 2, 4, "cpu", torch.bfloat16, 2)
        cache["mask"][:2] = True
        cache["k"].normal_()
        cache["v"].normal_()
        before = visible_snapshot(cache)
        cache["k"][0, 0, 0, 0] += 1
        self.assertFalse(state_equal(before, visible_snapshot(cache))["pass_all"])

    def test_same_nonfinite_valid_cache_never_passes(self):
        cache = self.ns["WanAttention"]._original_cache(4, 2, 4, "cpu", torch.bfloat16, 2)
        cache["mask"][:2] = True
        cache["k"].zero_()
        cache["v"].zero_()
        cache["k"][0, 0, 0, 0] = float("nan")
        before = visible_snapshot(cache)
        self.assertFalse(state_equal(before, before)["pass_all"])


if __name__ == "__main__":
    cpu_only()
    unittest.main(verbosity=2)

"""Isolated comparison adapters. Native operations require explicit GPU opt-in.

The static projection adapter is a block-only diagnostic analogue of the
frozen OriginalShapePack. It is not a production invalidation protocol and
ordinary caching is not a MegaTransformer contribution.
"""
import ast
from contextlib import contextmanager
import copy
from pathlib import Path
import sys
import types

import torch

from .metrics import byte_equal
from .reference import frozen_source, MODEL_SOURCE


@contextmanager
def static_context_projections(block, context):
    """Cache original-shape BF16 K/V projections for one immutable fixture.

    Preparation, byte admission and saved-output checks are outside timing.
    Both comparison arms use this same context manager for the cached stratum.
    No CFG collapse or candidate slicing is performed.
    """
    if context.ndim != 3 or context.shape[1:] != (512, 3072) or context.dtype != torch.bfloat16:
        raise ValueError("Only the registered full context projection geometry is supported")
    targets = (block.attn2.to_k, block.attn2.to_v)
    original = [target.forward for target in targets]
    versions = [(param, param._version) for target in targets for param in target.parameters()]
    with torch.inference_mode(False), torch.no_grad():
        original_context = context.detach().clone()
        saved = [operation(context).detach().clone() for operation in original]
    output_versions = [(value, value._version) for value in saved]
    if not byte_equal(original_context, context) or not all(bool(torch.isfinite(x).all()) for x in saved):
        raise RuntimeError("Immutable context admission failed")
    previous = [(target, "forward" in vars(target), vars(target).get("forward")) for target in targets]
    try:
        for target, output in zip(targets, saved):
            def hit(value, *, _output=output):
                if value is not context:
                    raise RuntimeError("Static diagnostic cannot reuse a different context object")
                return _output
            target.forward = hit
        yield dict(outputs=saved, scope="STATIC_IMMUTABLE_BLOCK_REPLAY_ONLY",
                   cached_projection_count=2, original_M=context.shape[0] * context.shape[1])
    finally:
        for target, had, prior in reversed(previous):
            if had:
                target.forward = prior
            else:
                del target.forward
        if (not byte_equal(original_context, context)
                or any(param._version != version for param, version in versions)
                or any(value._version != version for value, version in output_versions)):
            raise RuntimeError("Static diagnostic context/weights mutated")


@contextmanager
def shared_history_attention(block, repository: Path, valid_slots, candidate_rows, *, gpu_authorized=False):
    """Reuse the existing guarded candidate kernel through the pinned AST splice.

    All arithmetic outside the private candidate-history fragment is original.
    A new combined baseline still requires its own fixture numerical gate.
    """
    if not gpu_authorized:
        raise PermissionError("NOT_RUN_NO_GPU_AUTHORIZATION")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "wam_candidate_tile_v1"))
    import _wam_candidate_tile as native

    source = frozen_source(repository, MODEL_SOURCE)
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "WanAttention")
    original_fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    fn = copy.deepcopy(original_fn)
    candidate = next(node for node in fn.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "candidate_cache_rows is not None")
    if len(candidate.body) != 7:
        raise RuntimeError("Pinned private-candidate fragment contract changed")
    old_body = copy.deepcopy(candidate.body)
    candidate.body = old_body[:2] + ast.parse(
        "hidden_states = __shared(self, query, key, value, kv_cache, candidate_cache_rows)").body
    restored = copy.deepcopy(fn)
    restored_candidate = next(node for node in restored.body if isinstance(node, ast.If)
                             and ast.unparse(node.test) == "candidate_cache_rows is not None")
    restored_candidate.body = old_body
    if ast.dump(restored, include_attributes=False) != ast.dump(original_fn, include_attributes=False):
        raise RuntimeError("Shared adapter modified unrelated original arithmetic")
    expected_rows = torch.tensor([0] * 7 + [1] * 7, device=valid_slots.device)
    if not byte_equal(candidate_rows, expected_rows):
        raise RuntimeError("Candidate mapping failed static admission")
    # Frozen guarded-v2 geometry rule, not selection from noisy measured results.
    prefix = valid_slots.numel()
    group = 2 if (prefix + 32 + 63) // 64 == (prefix + 16 + 63) // 64 else 1

    def shared(attention, q, k, v, cache, rows):
        if rows is not candidate_rows:
            raise RuntimeError("Candidate mapping changed outside admitted static fixture")
        return native.grouped(q, cache["k"], cache["v"], k, v, valid_slots, group)

    original = block.attn1.forward
    namespace = dict(original.__func__.__globals__, __shared=shared)
    fn.name = "_block_replay_guarded_forward"
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 "<pinned-block-private-history-splice>", "exec"), namespace)
    had, prior = "forward" in vars(block.attn1), vars(block.attn1).get("forward")
    block.attn1.forward = types.MethodType(namespace[fn.name], block.attn1)
    try:
        yield dict(group=group, scope="guarded B14 self; independent Q16 and shared historical prefix")
    finally:
        if had:
            block.attn1.forward = prior
        else:
            del block.attn1.forward

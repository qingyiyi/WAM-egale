"""CPU stage/layout audit on two of the already registered real core fixtures."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import inspect
import json
from pathlib import Path
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .block_baseline import SequentialBlock
from .debug_taps import first_observed_divergence, linear_shape_ledger, original_taps
from .engines import static_context_projections
from .fixtures import BlockFixture, tensor_description, visible_snapshot
from .metrics import compare, state_equal
from .reference import load_block, REVIEW_COMMIT
from .runtime import cpu_only


@contextmanager
def cpu_linear_dispatch(block, output):
    """Observe ATen CPU matrix geometry, without guessing CUDA GEMM dispatch."""
    stack, handles = [], []
    for name, module in block.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        def pre(module, args, *, _name=name):
            stack.append(_name)
        def post(module, args, value):
            stack.pop()
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
    class Ledger(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            if stack and str(function) in ("aten.linear.default", "aten.mm.default", "aten.addmm.default", "aten.bmm.default"):
                output.append(dict(module=stack[-1], operator=str(function),
                                   tensor_arguments=[tensor_description(value) for value in args if isinstance(value, torch.Tensor)],
                                   device_scope="CPU_FRONTEND_OBSERVATION_NOT_CUDA_GEMM_KERNEL_PROOF"))
            return function(*args, **(kwargs or {}))
    try:
        with Ledger():
            yield
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("both", "short-B2", "long-B14"), default="both")
    args = parser.parse_args()
    cpu_only()
    start = time.monotonic()
    result = dict(status="RUNNING_CPU_STAGE_AUDIT", review_commit=REVIEW_COMMIT,
                  started_at=datetime.now().astimezone().isoformat(), rows=[],
                  scope="CPU normalized fixture contract; no CUDA performance/quality conclusion")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        block, _ = load_block(args.repository, args.checkpoint)
        sequential = SequentialBlock(block).eval()
        selected_cases = (("short", 2), ("long", 14)) if args.case == "both" else (
            (("short", 2),) if args.case == "short-B2" else (("long", 14),))
        for length, batch in selected_cases:
            fixture = BlockFixture.load(args.fixtures, length, batch)
            inputs, kwargs = fixture.arguments()
            rng = torch.random.get_rng_state()
            reference_tensors, candidate_tensors, linear_rows, dispatch = {}, {}, [], []
            fixture.install_cache(block)
            with torch.inference_mode():
                uninstrumented = block(*inputs, **kwargs)
            fixture.install_cache(block)
            torch.random.set_rng_state(rng)
            with original_taps(block, lambda name, value: reference_tensors.update({name: value.detach().clone()})), \
                    linear_shape_ledger(block, linear_rows), cpu_linear_dispatch(block, dispatch), torch.inference_mode():
                reference = block(*inputs, **kwargs)
            reference_state = visible_snapshot(block.attn1.attn_caches["pos"])
            fixture.install_cache(block)
            torch.random.set_rng_state(rng)
            with torch.inference_mode():
                candidate = sequential(*inputs, **kwargs, tap=lambda name, value: candidate_tensors.update({name: value.detach().clone()}))
            candidate_state = visible_snapshot(block.attn1.attn_caches["pos"])
            row = dict(case_id=fixture.case_id, complete_block=compare(reference, candidate),
                       instrumentation_self_check=compare(uninstrumented, reference),
                       stages=first_observed_divergence(reference_tensors, candidate_tensors),
                       state=state_equal(reference_state, candidate_state),
                       linear_module_geometry=linear_rows, ATen_CPU_matrix_dispatch=dispatch,
                       cached_controls={})
            for arm, operation in (("R0", block), ("B0", sequential)):
                fixture.install_cache(block)
                torch.random.set_rng_state(rng)
                prepare = time.monotonic()
                with static_context_projections(block, inputs[1]) as cache:
                    prepared_wall = time.monotonic() - prepare
                    with torch.inference_mode():
                        cached = operation(*inputs, **kwargs)
                    row["cached_controls"][arm] = dict(
                        uncached_output=compare(reference, cached),
                        state=state_equal(reference_state, visible_snapshot(block.attn1.attn_caches["pos"])),
                        static_replay_scope=cache["scope"], original_M=cache["original_M"],
                        CPU_preparation_resource_seconds=prepared_wall,
                        GPU_performance="NOT_RUN")
            row["pass_all"] = (row["instrumentation_self_check"]["bitwise_pass"]
                               and row["complete_block"]["bitwise_pass"] and row["state"]["pass_all"]
                               and row["stages"]["first_observed_boundary"] is None
                               and all(value["uncached_output"]["bitwise_pass"] and value["state"]["pass_all"]
                                       for value in row["cached_controls"].values()))
            result["rows"].append(row)
            if not row["pass_all"]:
                raise RuntimeError("Observed stage/static-cache divergence; stop")
        from diffusers.models.activations import GELU
        from diffusers.models.attention import FeedForward
        from diffusers.models.normalization import FP32LayerNorm
        result["installed_dependencies"] = {cls.__name__: dict(path=inspect.getfile(cls), source=inspect.getsource(cls))
                                             for cls in (GELU, FeedForward, FP32LayerNorm)}
        result["backend_observations"] = dict(torch_version=torch.__version__,
            torch_git=torch.version.git_version, cuda_build=torch.version.cuda,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
            math_sdp_enabled=torch.backends.cuda.math_sdp_enabled(),
            mem_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
            autocast_cpu=torch.is_autocast_enabled("cpu"), autocast_cuda=torch.is_autocast_enabled("cuda"),
            observations_only_no_flags_changed=True)
        result["status"] = "COMPLETE_CPU_STAGE_AND_STATIC_REPLAY_PASS_GPU_PENDING"
    except Exception as error:
        result["status"] = "FAILED_CPU_STAGE_AUDIT"
        result["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result["finished_at"] = datetime.now().astimezone().isoformat()
        result["lifecycle_wall_seconds"] = time.monotonic() - start
        result["cuda_initialized"] = torch.cuda.is_initialized()
        (args.output / "CPU-STAGE-AND-SHAPE-RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()

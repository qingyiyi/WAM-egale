"""Prepare a bounded paired benchmark; execution requires a new authorization.

CUDA Graph for the unmodified full block is not silently enabled: original
nonzero/tensor-to-Python cache validation is not a capturable fixed-address DAG.
Only matched eager block comparisons are implemented here. Existing SDPA Graph
core evidence remains separate from whole-block performance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time

from .fixtures import CORE_CASES
from .reference import REVIEW_COMMIT
from .runtime import require_gpu_authorization


def paired_order(round_index):
    return ("A", "B", "B", "A") if round_index % 2 == 0 else ("B", "A", "A", "B")


def benchmark_plan():
    return dict(schema="wam-block-benchmark-plan/v1", review_commit=REVIEW_COMMIT,
                status="PREPARED_NOT_RUN_NO_GPU_AUTHORIZATION",
                fixtures=[dict(length=length, batch=batch, layer=0, kind="action", role="first")
                          for length, batch in CORE_CASES],
                engines=["R0_TORCH", "R1_ALIGNED_ATTENTION", "B0_TORCH", "B0_ALIGNED_ATTENTION",
                         "R_STRONG_GUARDED_B14"],
                cache=["uncached", "static_original_shape_context_projections"],
                order=[list(paired_order(i)) for i in range(3)],
                independent_sample="one interleaved execution round; not each replay or tensor element",
                minimum_independent_rounds=3, warmup_calls_per_arm=5,
                A_A_noise_first=True, profile_in_timing=False,
                scopes=["prepared_eager_block_stream_elapsed", "integration_with_state_restore_wall"],
                full_block_Graph="NOT_IMPLEMENTED_UNMODIFIED_CACHE_BRANCHES_NOT_CAPTURABLE",
                strongest_combination="requires new joint numeric/state gate; no certificate inheritance",
                AA_gate=dict(max_abs_paired_ratio_deviation=0.025, engineering_only=True),
                B0_screen=dict(core_paired_geomean_ratio_max=1.05, reproducible_stratum_ratio_max=1.10),
                formal_robot_performance_permission=False)


def summary(samples):
    ratios = [sample["B_event_ms"] / sample["A_event_ms"] for sample in samples]
    if not ratios or not all(math.isfinite(value) and value > 0 for value in ratios):
        raise ValueError("Positive finite raw samples required")
    return dict(independent_rounds=len(ratios), paired_ratios=ratios,
                ratio_median=statistics.median(ratios), ratio_min=min(ratios), ratio_max=max(ratios),
                ratio_geomean=math.exp(statistics.mean(math.log(value) for value in ratios)),
                descriptive_only=len(ratios) < 3)


def measure_pair(torch, operation_a, operation_b, restore, *, rounds=3,
                 repeats=3, warmup=5, guard=lambda: None):
    """No hooks/profiler/checks inside the timed block; immutable addresses.

    restore() copies into existing buffers. Every measured call is a whole
    block. State restore and preparation are outside prepared event timing,
    but included in the separate integration wall observation.
    """
    if rounds < 3:
        raise ValueError("Need at least three independent execution rounds")
    operations = {"A": operation_a, "B": operation_b}
    for name in ("A", "B"):
        for _ in range(warmup):
            guard()
            restore(name)
            operations[name]()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    raw, rows = [], []
    for index in range(rounds):
        round_samples = {"A": [], "B": []}
        for position, name in enumerate(paired_order(index)):
            for sample_index in range(repeats):
                guard()
                torch.cuda.synchronize()
                integrated_start = time.perf_counter()
                restore(name)
                torch.cuda.synchronize()
                prepared_start = time.perf_counter()
                begin.record()
                operations[name]()
                end.record()
                end.synchronize()
                finished = time.perf_counter()
                value = begin.elapsed_time(end)
                raw.append(dict(round=index, position=position, arm=name, repeat=sample_index,
                                event_ms=value, prepared_wall_ms=(finished - prepared_start) * 1000,
                                integration_wall_ms=(finished - integrated_start) * 1000))
                round_samples[name].append(value)
        rows.append(dict(round=index, A_event_ms=statistics.median(round_samples["A"]),
                         B_event_ms=statistics.median(round_samples["B"])))
    return dict(raw_samples=raw, rounds=rows, summary=summary(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    if args.execute:
        # No CUDA query or native import can occur before this gate.
        require_gpu_authorization(args.authorization, "performance")
        if not all((args.repository, args.checkpoint, args.fixtures)):
            raise ValueError("Explicit frozen repository/checkpoint/fixture paths are required")
        from .gpu_matrix import execute
        execute(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "GPU-BENCHMARK-PLAN.json").write_text(
        json.dumps(benchmark_plan(), ensure_ascii=False, indent=2) + "\n")
    with (args.output / "PERFORMANCE-RESULT.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "engine", "cache", "scope", "status", "event_ms"])
        writer.writeheader()
        for length, batch in CORE_CASES:
            for engine in benchmark_plan()["engines"]:
                for cache in benchmark_plan()["cache"]:
                    if batch == 2 and engine == "R_STRONG_GUARDED_B14":
                        continue
                    writer.writerow(dict(case=f"{length}-B{batch}-action-layer0-first", engine=engine,
                                         cache=cache, scope="complete_block", status="NOT_RUN_NO_GPU_AUTHORIZATION"))
    print("PREPARED_NOT_RUN; GPU rows contain no timing values")


if __name__ == "__main__":
    main()

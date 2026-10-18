"""Finite CPU full-block replay of six saved real inputs, plus state controls.

No CUDA/native import or robot loop. This establishes a CPU logical-contract
result only; every GPU stage stays PENDING in the output.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

import torch

from .block_baseline import SequentialBlock
from .fixtures import BlockFixture, CORE_CASES, tensors, visible_snapshot
from .metrics import byte_equal, compare, state_equal
from .reference import load_block, REVIEW_COMMIT
from .runtime import cpu_only


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def run_one(block, sequential, fixture, update_cache=0):
    args, kwargs = fixture.arguments(update_cache=update_cache)
    inputs_before = {name: tensor.clone() for name, tensor in tensors([args, kwargs])}
    rng = torch.random.get_rng_state().clone()
    versions = {name: (param.data_ptr(), param._version) for name, param in block.named_parameters()}
    outputs, states, contract = {}, {}, {}
    for arm, operation in (("R0_A", block), ("R0_B", block), ("B0", sequential)):
        cache = fixture.install_cache(block)
        before = visible_snapshot(cache)
        torch.random.set_rng_state(rng)
        with torch.inference_mode():
            outputs[arm] = operation(*args, **kwargs)
        states[arm] = visible_snapshot(cache)
        inputs_unchanged = all(byte_equal(inputs_before[name], value)
                               for name, value in tensors([args, kwargs]))
        # R0's original temporary B2 writes can alter INVALID slot metadata.
        # Valid history must be preserved; full post-state must match B0/R0.
        valid_before = before["slots"]
        history_unchanged = all(byte_equal(before[key], cache[key][:, valid_before])
                                for key in ("k", "v"))
        valid_metadata_unchanged = all(
            byte_equal(before[key][valid_before], cache[key][valid_before])
            for key in ("id", "is_pred"))
        readonly_metadata = (all(byte_equal(before[key], cache[key])
                                  for key in ("mask", "id", "is_pred"))
                             if fixture.data["identity"][1] == 14 else
                             byte_equal(before["mask"], cache["mask"]))
        contract[arm] = dict(inputs_unchanged=inputs_unchanged,
                            rng_unchanged=byte_equal(rng, torch.random.get_rng_state()),
                            history_kv_unchanged=history_unchanged,
                            history_metadata_unchanged=valid_metadata_unchanged,
                            readonly_metadata_unchanged=(readonly_metadata if update_cache == 0 else None),
                            valid_tokens_after=cache["mask"].sum().item())
        if not inputs_unchanged:
            raise RuntimeError(f"{arm} changed an immutable input; do not contaminate next arm")
    weight_versions_unchanged = all(
        versions[name] == (param.data_ptr(), param._version) for name, param in block.named_parameters())
    reference = compare(outputs["R0_A"], outputs["R0_B"])
    result = compare(outputs["R0_A"], outputs["B0"])
    state = state_equal(states["R0_A"], states["B0"])
    reference_state = state_equal(states["R0_A"], states["R0_B"])
    contracts_pass = all(item["inputs_unchanged"] and item["rng_unchanged"]
                         and item["history_kv_unchanged"] and item["history_metadata_unchanged"]
                         and (item["readonly_metadata_unchanged"] is not False)
                         for item in contract.values())
    status = ("CPU_LOGICAL_CONTRACT_PASS" if reference["bitwise_pass"] and result["bitwise_pass"]
              and state["pass_all"] and reference_state["pass_all"] and contracts_pass
              and weight_versions_unchanged else "CPU_CONTRACT_FAILED")
    return dict(case_id=fixture.case_id, update_cache=update_cache,
                input_role=("REAL_CAPTURED_QUERY" if update_cache == 0 else
                            "DERIVED_LEGAL_WRITE_STATE_CONTROL_NOT_ROBOT"),
                status=status, reference_self_reproduction=reference,
                complete_block=result, post_state=state, reference_post_state=reference_state,
                contracts=contract, weight_references_and_versions_unchanged=weight_versions_unchanged,
                historical_gpu_gold_cpu_comparison=compare(fixture.data["block_gold"], outputs["R0_A"]),
                historical_gpu_gold_interpretation="diagnostic only; CPU/GPU are different numeric environments",
                first_divergence=(None if result["bitwise_pass"] else "NOT_LOCALIZED_STOP_BEFORE_GPU"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", default="4,5")
    parser.add_argument("--wall-cap-seconds", type=int, default=1800)
    parser.add_argument("--legal-write-controls", action="store_true")
    args = parser.parse_args()
    if not 0 < args.wall_cap_seconds <= 7200:
        raise ValueError("CPU replay is limited to at most two hours")
    start = time.monotonic()
    result = dict(schema="wam-sequential-block-cpu/v1", review_commit=REVIEW_COMMIT,
                  started_at=datetime.now().astimezone().isoformat(),
                  status="RUNNING_CPU_ONLY", collection_complete=False, rows=[],
                  scope="six normalized real action block0 inputs on CPU; not original CUDA layout",
                  gpu_validation="NOT_RUN_NO_GPU_AUTHORIZATION", performance_status="NOT_TIMED",
                  engineering_quality_screen=False, independent_test_complete=False,
                  noninferiority_established=False, formal_performance_permission=False)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "CPU-ACCURACY-RESULT.json"
    code = 1
    try:
        result["resource"] = cpu_only(tuple(int(x) for x in args.cpus.split(",")))
        block, weight_manifest = load_block(args.repository, args.checkpoint)
        atomic_json(args.output / "BLOCK-PARAMETERS.json", weight_manifest)
        sequential = SequentialBlock(block).eval()
        fixture_manifest = []
        for length, batch in CORE_CASES:
            if time.monotonic() - start > args.wall_cap_seconds - 120:
                raise TimeoutError("Preserve CPU replay cleanup margin; no next fixture")
            fixture = BlockFixture.load(args.fixtures, length, batch)
            fixture_manifest.append(fixture.manifest())
            atomic_json(args.output / "FIXTURE-MANIFEST.json", fixture_manifest)
            modes = (0, 1, 2) if args.legal_write_controls and batch == 2 else (0,)
            for update_cache in modes:
                if time.monotonic() - start > args.wall_cap_seconds - 60:
                    raise TimeoutError("CPU lifecycle cap before next full-block control")
                row = run_one(block, sequential, fixture, update_cache)
                result["rows"].append(row)
                atomic_json(output, result)
                print(row["case_id"], "update", update_cache, row["status"], flush=True)
                if row["status"] != "CPU_LOGICAL_CONTRACT_PASS":
                    raise RuntimeError("First full-block/contract divergence; stop the matrix")
        atomic_json(args.output / "FIXTURE-MANIFEST.json", fixture_manifest)
        result["collection_complete"] = True
        result["status"] = "COMPLETE_CPU_LOGICAL_CONTRACT_PASS_GPU_PENDING"
        code = 0
    except Exception as error:
        result["status"] = "PARTIAL_CPU_REPLAY_FAILED"
        result["failure"] = dict(type=type(error).__name__, message=str(error),
                                 traceback=traceback.format_exc())
    finally:
        result["finished_at"] = datetime.now().astimezone().isoformat()
        result["lifecycle_wall_seconds"] = time.monotonic() - start
        result["cuda_initialized"] = torch.cuda.is_initialized()
        atomic_json(output, result)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

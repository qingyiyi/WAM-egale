"""Six real block fixtures, one module/cache, original versus copied classes."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from experiments.block_sequential_baseline_v1.benchmark_block import measure_pair
from experiments.block_sequential_baseline_v1.cpu_job import write
from experiments.block_sequential_baseline_v1.fixtures import BlockFixture, CORE_CASES, tensors, visible_snapshot
from experiments.block_sequential_baseline_v1.metrics import compare, byte_equal, state_equal
from experiments.block_sequential_baseline_v1.reference import load_block
from experiments.block_sequential_baseline_v1.runtime import require_gpu_authorization
from .binding import copied_block


def device_tree(value, device, memo=None):
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        output = value.to(device)
    elif isinstance(value, dict):
        output = {key: device_tree(item, device, memo) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        output = type(value)(device_tree(item, device, memo) for item in value)
    else:
        output = value
    memo[id(value)] = output
    return output


def resource():
    def query(argument):
        return subprocess.check_output(["nvidia-smi", argument, "--format=csv,noheader,nounits"],
                                       text=True, timeout=10).strip()
    return dict(gpu=query("--query-gpu=uuid,memory.free,utilization.gpu"),
                pids=query("--query-compute-apps=pid"))


def execute(args):
    started = time.monotonic()
    auth = require_gpu_authorization(args.authorization, "accuracy")
    manifest = json.loads(Path(auth["implementation_manifest"]).read_text())
    root = Path(__file__).resolve().parents[3]
    for entry in manifest["source_entries"]:
        if hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("Registered source changed")
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a fresh worker")
    before = resource()
    if before["pids"]:
        raise RuntimeError("GPU_BUSY_NO_CONTEXT_CREATED_NO_RETRY")
    fields = [value.strip() for value in before["gpu"].split(",")]
    if len(fields) != 3 or fields[0] != auth["expected_gpu_uuid"] or float(fields[1]) < 8000:
        raise RuntimeError("GPU_IDENTITY_OR_MEMORY_GATE_FAILED")
    os.sched_setaffinity(0, {4, 5})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(exist_ok=False, parents=True)
    result = dict(status="RUNNING", started_at=datetime.now().astimezone().isoformat(),
                  numerical=[], timings=[], resources=[before], block_attempted=0, block_returned=0,
                  formal_performance_permission=False, engineering_quality_screen=False,
                  independent_test_complete=False, noninferiority_established=False,
                  scope="six normalized saved action block0 fixtures; Torch-only uncached; not full model or episode",
                  old_AA_failure_unchanged=True, performance_certificate=False)
    deadline = datetime.fromisoformat(auth["cleanup_deadline"]).timestamp()
    cap = min(auth["gpu_lifecycle_seconds_cap"], deadline - time.time())

    def guard():
        if time.monotonic() - started > cap - 120 or time.time() > deadline - 120:
            raise TimeoutError("Cleanup reserve reached")
        if result["block_attempted"] >= min(auth["block_attempt_cap"], 294):
            raise RuntimeError("Block-call cap reached")

    def checkpoint():
        write(args.output / "RESULT.json", result)

    try:
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability(0) != (8, 0):
            raise RuntimeError("Only registered SM80 GPU supported")
        block, _ = load_block(args.repository, args.checkpoint)
        block = block.to("cuda").eval()
        parameters = {name: (id(value), value.data_ptr()) for name, value in block.named_parameters()}
        for length, batch in CORE_CASES:
            guard()
            if cap - (time.monotonic() - started) < 300:
                raise TimeoutError("Insufficient whole-fixture and cleanup reserve")
            observation = resource()
            result["resources"].append(observation)
            if set(observation["pids"].splitlines()) - {str(os.getpid())}:
                raise RuntimeError("Other GPU process appeared; no foreign signals")
            fixture = BlockFixture.load(args.fixtures, length, batch)
            fixture = BlockFixture(fixture.path, device_tree(fixture.data, "cuda"))
            inputs, kwargs = fixture.arguments()
            cache = fixture.install_cache(block)
            initial = visible_snapshot(cache)
            immutable = {name: value.clone() for name, value in tensors([inputs, kwargs])}
            cpu_rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            binding = None

            def restore(arm):
                nonlocal binding
                if binding is not None:
                    binding.__exit__(None, None, None)
                    binding = None
                if arm == "B":
                    binding = copied_block(block)
                    binding.__enter__()
                for name in ("id", "mask", "is_pred"):
                    cache[name].copy_(initial[name])
                for name in ("k", "v"):
                    cache[name].index_copy_(1, initial["slots"], initial[name])

            def forward():
                result["block_attempted"] += 1
                with torch.inference_mode():
                    output = block(*inputs, **kwargs)
                result["block_returned"] += 1
                return output

            try:
                oracle = oracle_state = None
                for arm in ("A", "B", "A"):
                    guard()
                    restore(arm)
                    output = forward()
                    torch.cuda.synchronize()
                    state = visible_snapshot(cache)
                    if oracle is None:
                        oracle, oracle_state = output.clone(), state
                    row = dict(case=fixture.case_id, arm=arm, output=compare(oracle, output),
                               historical_GPU_gold=compare(fixture.data["block_gold"], output),
                               state=state_equal(oracle_state, state),
                               inputs_unchanged=all(byte_equal(immutable[name], value)
                                                    for name, value in tensors([inputs, kwargs])),
                               RNG_unchanged=byte_equal(cpu_rng, torch.random.get_rng_state())
                                   and byte_equal(gpu_rng, torch.cuda.get_rng_state()))
                    result["numerical"].append(row)
                    checkpoint()
                    if not (row["output"]["bitwise_pass"] and row["historical_GPU_gold"]["bitwise_pass"]
                            and row["state"]["pass_all"] and row["inputs_unchanged"] and row["RNG_unchanged"]):
                        raise RuntimeError("Strict numerical/state gate failed")
                measurement = measure_pair(torch, forward, forward, restore,
                                           rounds=3, repeats=3, warmup=5, guard=guard)
                post = dict(state=state_equal(oracle_state, visible_snapshot(cache)),
                            inputs_unchanged=all(byte_equal(immutable[name], value)
                                                 for name, value in tensors([inputs, kwargs])),
                            RNG_unchanged=byte_equal(cpu_rng, torch.random.get_rng_state())
                                and byte_equal(gpu_rng, torch.cuda.get_rng_state()),
                            parameter_addresses_unchanged=parameters == {
                                name: (id(value), value.data_ptr()) for name, value in block.named_parameters()})
                result["timings"].append(dict(case=fixture.case_id, measurement=measurement,
                                               post_timing=post, same_module_and_cache_addresses=True))
                checkpoint()
                if not (post["state"]["pass_all"] and post["inputs_unchanged"]
                        and post["RNG_unchanged"] and post["parameter_addresses_unchanged"]):
                    raise RuntimeError("Post-timing state gate failed")
            finally:
                if binding is not None:
                    binding.__exit__(None, None, None)
        result["status"] = "COMPLETE_FINITE_COPY_REPRODUCTION_WITH_DIAGNOSTIC_TIMING"
    except Exception as error:
        result.update(status="FAILED_OR_PARTIAL", error=repr(error))
        raise
    finally:
        result.update(finished_at=datetime.now().astimezone().isoformat(),
                      worker_lifecycle_seconds=time.monotonic() - started,
                      cleanup="binding restored; dedicated CUDA worker exits")
        checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("authorization", "repository", "checkpoint", "fixtures", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()

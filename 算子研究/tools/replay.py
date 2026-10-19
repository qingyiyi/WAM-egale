"""同一模型、参数、输入与cache，对照原版/baseline/optimized。"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from wam_ops.install import install_block, install_model
from .fixtures import BlockFixture, tensors, visible_snapshot
from .metrics import compare, byte_equal, state_equal

ROOT = Path(__file__).resolve().parents[2]
ENGINES = ("original", "baseline", "optimized")


def write(path, data):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def device_tree(value, device="cuda", memo=None):
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
    return dict(gpu=query("--query-gpu=uuid,memory.used,memory.free,utilization.gpu"),
                pids=query("--query-compute-apps=pid"))


def history_cache(saved):
    metadata = {key: saved[key].clone() for key in ("id", "mask", "is_pred")}
    metadata.update(k=torch.empty(saved["pool_shape"], dtype=torch.bfloat16, device="cuda"),
                    v=torch.empty(saved["pool_shape"], dtype=torch.bfloat16, device="cuda"))
    for field in ("k", "v"):
        metadata[field][:, saved["slots"]] = saved[field + "_valid"]
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    rounds = registration.get("timing_rounds", 3)
    warmups = registration.get("warmup_calls_per_engine", 0)
    if registration.get("user_authorization") is None:
        raise PermissionError("缺少当前任务授权登记")
    deadline = datetime.fromisoformat(registration["absolute_cleanup_deadline"]).timestamp()
    if time.time() + 300 >= deadline:
        raise TimeoutError("没有完整作业与清理余量")
    before = resource()
    fields = [field.strip() for field in before["gpu"].split(",")]
    if before["pids"] or fields[0] != registration["physical_gpu_uuid"] or float(fields[2]) < 30000:
        raise RuntimeError("GPU_BUSY_OR_IDENTITY_MISMATCH_NO_CONTEXT_CREATED")
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {4, 5})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    result = dict(status="RUNNING", started_at=datetime.now().astimezone().isoformat(),
                  numeric=[], timing=[], resources=[before], full_model_forwards=0, block_forwards=0,
                  engineering_quality_screen=False, independent_test_complete=False,
                  noninferiority_established=False, formal_performance_permission=False,
                  model_dtype="same from_pretrained(torch_dtype=BF16) loading as canonical server",
                  B2_full_model_source="DERIVED_REAL_VERIFIER_ROOT_CFG_ROWS_0_7_NOT_CANONICAL_SEQUENTIAL_TRACE",
                  timing_protocol=f"{rounds} cyclic interleaved rounds; {warmups} extra warmups/engine; separate processes")

    def save():
        write(args.output / "RESULT.json", result)

    def guard():
        if time.time() > deadline - 90 or time.monotonic() - started > 510:
            raise TimeoutError("worker清理余量")
        if (result["full_model_forwards"] >= registration.get("full_model_per_process_cap", 75)
                or result["block_forwards"] >= registration.get("block_per_process_cap", 250)):
            raise RuntimeError("worker调用预算")

    def check_resource():
        observation = resource()
        result["resources"].append(observation)
        if set(observation["pids"].splitlines()) - {str(os.getpid())}:
            raise RuntimeError("其他GPU进程进入，停止正式计时")

    def evaluate(scope, case, operation, installation, restore, state, inputs, historical=None, timing=True):
        oracle = oracle_state = None
        immutable = {name: value.clone() for name, value in tensors(inputs)}
        cpu_rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()

        def call():
            guard()
            result["full_model_forwards" if scope == "full30" else "block_forwards"] += 1
            with torch.inference_mode():
                return operation()

        def contract(output, engine, oracle, oracle_state):
            observed = dict(output=compare(oracle, output), state=[
                state_equal(a, b) for a, b in zip(oracle_state, state())],
                inputs_unchanged=all(byte_equal(immutable[name], value) for name, value in tensors(inputs)),
                RNG_unchanged=byte_equal(cpu_rng, torch.random.get_rng_state())
                              and byte_equal(gpu_rng, torch.cuda.get_rng_state()))
            if historical is not None:
                observed["historical_GPU_gold"] = compare(historical, output)
            passed = observed["output"]["bitwise_pass"] and all(row["pass_all"] for row in observed["state"])
            passed &= observed["inputs_unchanged"] and observed["RNG_unchanged"]
            if historical is not None:
                passed &= observed["historical_GPU_gold"]["bitwise_pass"]
            result["numeric"].append(dict(scope=scope, case=case, engine=engine, **observed))
            save()
            if not passed:
                raise RuntimeError(f"{scope}/{case}/{engine}逐位或状态门失败")

        for engine in ENGINES:
            restore()
            with installation(engine):
                output = call()
            torch.cuda.synchronize()
            if oracle is None:
                oracle, oracle_state = output.clone(), state()
            contract(output, engine, oracle, oracle_state)

        if not timing:
            return
        event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        raw = []
        timed_engines = tuple(args.timing_engines)
        for engine in timed_engines:
            for _ in range(warmups):
                restore()
                with installation(engine):
                    output = call()
                torch.cuda.synchronize()
                if not byte_equal(oracle, output):
                    raise RuntimeError("预热返回值逐位不一致")
        for round_index in range(rounds):
            offset = round_index % len(timed_engines)
            order = timed_engines[offset:] + timed_engines[:offset]
            for engine in order:
                restore()
                with installation(engine):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    event_start.record()
                    output = call()
                    event_end.record()
                    event_end.synchronize()
                    wall_ms = (time.perf_counter() - begin) * 1000
                raw.append(dict(round=round_index, engine=engine,
                                event_ms=event_start.elapsed_time(event_end), wall_ms=wall_ms))
                if not byte_equal(oracle, output):
                    raise RuntimeError("计时返回值逐位不一致")
        contract(output, "post_timing", oracle, oracle_state)
        medians = {engine: statistics.median(row["event_ms"] for row in raw if row["engine"] == engine)
                   for engine in timed_engines}
        result["timing"].append(dict(scope=scope, case=case, raw=raw, median_ms=medians,
            baseline_over_original=(medians["baseline"] / medians["original"] if "baseline" in medians else None),
            optimized_over_original=(medians["optimized"] / medians["original"] if "optimized" in medians else None)))
        save()

    try:
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.90)
        sys.path.insert(0, str(ROOT / "投机采样/wam-p0-worktree"))
        from wan_va.modules.model import WanTransformer3DModel
        checkpoint = ROOT / "models/lingbot-va-posttrain-libero-long-modelscope-20260922/transformer"
        model = WanTransformer3DModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
                    attn_mode="torch", local_files_only=True).to("cuda").eval().requires_grad_(False)
        if len(model.blocks) != 30:
            raise RuntimeError("不是实际30层模型")
        result["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        original_classes = [type(block) for block in model.blocks]
        parameter_addresses = {name: (id(value), value.data_ptr()) for name, value in model.named_parameters()}
        block_fixtures = ROOT / "投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r3/process1"
        full_fixtures = ROOT / "投机采样/results/runtime/p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4"
        for length in ("short", "middle", "long"):
            check_resource()
            for batch in (2, 14):
                fixture = BlockFixture.load(block_fixtures, length, batch)
                fixture = BlockFixture(fixture.path, device_tree(fixture.data))
                arguments, kwargs = fixture.arguments()
                block = model.blocks[0]
                cache = fixture.install_cache(block)
                initial = visible_snapshot(cache)

                def restore_block():
                    for field in ("id", "mask", "is_pred"):
                        cache[field].copy_(initial[field])
                    for field in ("k", "v"):
                        cache[field].index_copy_(1, initial["slots"], initial[field])

                evaluate("block", f"{length}-B{batch}", lambda: block(*arguments, **kwargs),
                    lambda engine: nullcontext() if engine == "original" else install_block(block, engine),
                    restore_block, lambda: [visible_snapshot(cache)], [arguments, kwargs], fixture.data["block_gold"])
                if length == "short" and batch == 2:
                    for update in (1, 2):
                        kwargs["update_cache"] = update
                        evaluate("block", f"short-B2-write{update}", lambda: block(*arguments, **kwargs),
                            lambda engine: nullcontext() if engine == "original" else install_block(block, engine),
                            restore_block, lambda: [visible_snapshot(cache)], [arguments, kwargs], timing=False)
                    kwargs["update_cache"] = 0
                del fixture, cache, initial
                block.attn1.attn_caches = {}
            frozen = torch.load(full_fixtures / length / "real-verifier-fixture.pt",
                                map_location="cpu", mmap=True, weights_only=False)
            history = device_tree(frozen["history"])
            caches = [history_cache(saved) for saved in history]
            for block, cache in zip(model.blocks, caches):
                block.attn1.attn_caches = {"pos": cache}

            def restore_model():
                for cache, saved in zip(caches, history):
                    for field in ("id", "mask", "is_pred"):
                        cache[field].copy_(saved[field])
                    for field in ("k", "v"):
                        cache[field].index_copy_(1, saved["slots"], saved[field + "_valid"])

            for batch in (2, 14):
                query = device_tree(frozen["queries"][0])
                inputs, kwargs = query["input"], query["kwargs"]
                kwargs.pop("input_dict", None)
                gold = query["output"] if batch == 14 else None
                if batch == 2:
                    indices = torch.tensor([0, 7], device="cuda")
                    inputs = {name: value.index_select(0, indices) for name, value in inputs.items()}
                    kwargs.pop("candidate_cache_rows")
                evaluate("full30", f"{length}-B{batch}", lambda: model(inputs, **kwargs),
                    lambda engine: nullcontext() if engine == "original" else install_model(model, engine),
                    restore_model, lambda: [visible_snapshot(cache) for cache in caches], [inputs, kwargs], gold)
                if [type(block) for block in model.blocks] != original_classes:
                    raise RuntimeError("模型类未还原")
                if parameter_addresses != {name: (id(value), value.data_ptr()) for name, value in model.named_parameters()}:
                    raise RuntimeError("模型参数对象/地址改变")
                result.setdefault("restoration", []).append(dict(
                    case=f"{length}-B{batch}", classes_restored=True, parameter_addresses_unchanged=True))
                save()
            for block in model.blocks:
                block.attn1.attn_caches = {}
            del history, caches, frozen
        check_resource()
        result.update(status="COMPLETE_BLOCK_AND_ACTUAL30_FINITE_PARITY_AND_TIMING",
                      all_parameters_and_classes_restored=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status="FAILED_OR_PARTIAL", error=repr(error))
        raise
    finally:
        result.update(finished_at=datetime.now().astimezone().isoformat(),
                      worker_lifecycle_seconds=time.monotonic() - started)
        save()


if __name__ == "__main__":
    main()

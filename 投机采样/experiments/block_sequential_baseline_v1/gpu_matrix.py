"""Future GPU block matrix, default denied; not executed during CPU preparation.

Only one real block is loaded. Each arm owns mutable cache/module instances;
all arms share the exact same immutable Parameter objects and logical inputs.
No production patch, robot loop, full-model load, Graph or profiler is used.
"""
from contextlib import ExitStack
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from .benchmark_block import measure_pair
from .block_baseline import SequentialBlock, aligned_attention
from .engines import shared_history_attention, static_context_projections
from .fixtures import BlockFixture, CORE_CASES, tensors, visible_snapshot
from .metrics import byte_equal, compare, state_equal
from .reference import load_block, REVIEW_COMMIT
from .runtime import require_gpu_authorization


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _device_tree(value, device, memo=None):
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        output = value.to(device)
    elif isinstance(value, dict):
        output = {key: _device_tree(item, device, memo) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        output = type(value)(_device_tree(item, device, memo) for item in value)
    else:
        output = value
    memo[id(value)] = output
    return output


def _resource_snapshot(index):
    def query(*arguments):
        return subprocess.check_output(["nvidia-smi", "-i", str(index), *arguments], timeout=10).decode().strip()
    return dict(gpu=query("--query-gpu=uuid,name,memory.free,utilization.gpu,clocks.sm", "--format=csv,noheader,nounits"),
                process_pids=query("--query-compute-apps=pid", "--format=csv,noheader,nounits"))


def independent_module_arms(base, names):
    """Independent mutable modules, exactly shared immutable Parameters."""
    memo = {id(param): param for param in base.parameters()}
    return {name: copy.deepcopy(base, memo.copy()) for name in names}


def execute(args):
    start = time.monotonic()
    # Must precede all native imports, CUDA probes and output-directory writes.
    auth = require_gpu_authorization(args.authorization, "performance")
    if "accuracy" not in auth.get("allowed_phases", []):
        raise PermissionError("Block performance requires an explicitly authorized numerical gate too")
    expected_cases = [f"{length}-B{batch}-action-layer0-first" for length, batch in CORE_CASES]
    if auth.get("input_manifest") != expected_cases:
        raise PermissionError("Only the exact preregistered six-fixture matrix is accepted")
    if not 0 < auth.get("block_attempt_cap", 0) <= 5000:
        raise PermissionError("An explicit bounded block-call budget is required")
    if not 0 < auth.get("storage_cap_bytes", 0) <= 2**30:
        raise PermissionError("A positive storage cap no larger than one GiB is required")
    manifest = json.loads(Path(auth["implementation_manifest"]).read_text())
    if manifest["base_commit"] != REVIEW_COMMIT or manifest["patch_sha256"] != auth.get("implementation_patch_sha256"):
        raise PermissionError("Implementation identity differs from the reviewed authorization")
    local_root = Path(__file__).resolve().parents[3]
    for entry in manifest["source_entries"]:
        source = local_root / entry["path"]
        if source.stat().st_size > 2 * 2**20 or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise PermissionError("Reviewed small source changed before GPU execution")
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a fresh dedicated GPU worker")
    index = int(auth.get("gpu_index", 0))
    if index != 0 or os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "0":
        raise PermissionError("This local one-GPU job requires an unambiguous physical GPU0 mapping")
    before_gpu = _resource_snapshot(index)
    other_pids = [line for line in before_gpu["process_pids"].splitlines() if line.strip()]
    if other_pids:
        raise RuntimeError("PERF_RESOURCE_UNKNOWN: another GPU process is present; no signals or wait")
    fields = [value.strip() for value in before_gpu["gpu"].split(",")]
    if (len(fields) == 5 and (fields[0] != auth.get("expected_gpu_uuid")
                              or "A100-SXM4" not in fields[1])):
        raise PermissionError("This timing authorization must bind the exact local A100-SXM4 UUID")
    if len(fields) != 5 or float(fields[3]) > 5:
        raise RuntimeError("PERF_RESOURCE_UNKNOWN: idle utilization anomaly; no CUDA context created")
    if float(fields[2]) < 8000:
        raise RuntimeError("Insufficient free GPU memory for the bounded block matrix")
    os.sched_setaffinity(0, {4, 5})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    absolute_cleanup = datetime.fromisoformat(auth["cleanup_deadline"])
    cap = min(auth["gpu_lifecycle_seconds_cap"],
              (absolute_cleanup - datetime.now().astimezone()).total_seconds())
    if cap < 300:
        raise RuntimeError("Not enough lifecycle budget to initialize and clean up")
    pid = os.getpid()
    birth = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    _json(args.output / "OWNED-PROCESS.json", dict(pid=pid, starttime_ticks=birth, uid=os.getuid()))
    _json(args.output / "REGISTRATION.json", dict(auth, started_at=datetime.now().astimezone().isoformat(),
                                                   resource_preflight=before_gpu, scope="normalized real block0 replay only"))
    result = dict(status="RUNNING_AUTHORIZED_BLOCK_MATRIX", rows=[], numeric=[], block_attempted=0,
                  block_returned=0, new_robot_episodes=0, full_target_model_forwards=0,
                  raw_samples_are_not_independent=True,
                  engineering_quality_screen=False, independent_test_complete=False,
                  noninferiority_established=False, formal_performance_permission=False)
    def guard():
        if time.monotonic() - start > cap - 120 or (args.output / "STOP_REQUESTED").exists():
            raise TimeoutError("Preserve registered cleanup margin")
        if result["block_attempted"] >= auth["block_attempt_cap"]:
            raise RuntimeError("Registered block-call cap reached")
        if sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) > min(auth.get("storage_cap_bytes", 0), 2**30):
            raise RuntimeError("Registered storage cap reached")
    def resource_between_windows(label):
        snapshot = _resource_snapshot(index)
        result.setdefault("resource_outside_timing", []).append(dict(label=label, **snapshot))
        pids = {int(value) for value in snapshot["process_pids"].splitlines() if value.strip()}
        if pids - {pid}:
            raise RuntimeError("PERF_RESOURCE_UNKNOWN: outside-window observation found another GPU process")
    def operation(module, inputs, kwargs):
        def forward():
            result["block_attempted"] += 1
            with torch.inference_mode():
                output = module(*inputs, **kwargs)
            result["block_returned"] += 1
            return output
        return forward
    try:
        torch.cuda.set_device(index)
        if torch.cuda.get_device_capability(index) != (8, 0):
            raise RuntimeError("This reviewed matrix is restricted to SM80")
        device = torch.device("cuda", index)
        base, _ = load_block(args.repository, args.checkpoint)
        base = base.to(device).eval()
        for length, batch in CORE_CASES:
            guard()
            resource_between_windows(f"before-{length}-B{batch}")
            if cap - (time.monotonic() - start) < 300:
                raise TimeoutError("Do not start another whole fixture without cleanup reserve")
            fixture = BlockFixture.load(args.fixtures, length, batch)
            fixture = BlockFixture(fixture.path, _device_tree(fixture.data, device))
            inputs, kwargs = fixture.arguments()
            saved_inputs = {name: value.clone() for name, value in tensors([inputs, kwargs])}
            for cache_mode in ("uncached", "static_original_shape_context_projections"):
                guard()
                # Each numerical/performance arm owns mutable module/cache state.
                # Parameter memo prevents copying any model weights.
                base.attn1.attn_caches = {}
                names = ["R0", "R0_TWIN", "R1", "B0_TORCH", "B0_ALIGNED"]
                if batch == 14:
                    names.append("R_STRONG_GUARDED_B14")
                modules = independent_module_arms(base, names)
                caches = {name: fixture.install_cache(module) for name, module in modules.items()}
                initial = visible_snapshot(caches["R0"])
                rng_cpu = torch.random.get_rng_state().clone()
                rng_cuda = torch.cuda.get_rng_state(index).clone()
                functions = {}
                def restore(name):
                    cache = caches[name]
                    for field in ("mask", "id", "is_pred"):
                        cache[field].copy_(initial[field])
                    for field in ("k", "v"):
                        cache[field].index_copy_(1, initial["slots"], initial[field])
                with ExitStack() as stack:
                    for name, module in modules.items():
                        if cache_mode != "uncached":
                            stack.enter_context(static_context_projections(module, inputs[1]))
                        if name in ("R1", "B0_ALIGNED", "R_STRONG_GUARDED_B14"):
                            stack.enter_context(aligned_attention(module, gpu_authorized=True))
                        if name == "R_STRONG_GUARDED_B14":
                            stack.enter_context(shared_history_attention(module, args.repository, initial["slots"],
                                                                         kwargs["candidate_cache_rows"], gpu_authorized=True))
                        forward_module = SequentialBlock(module).eval() if name.startswith("B0_") else module
                        functions[name] = operation(forward_module, inputs, kwargs)
                    oracle, oracle_state = None, None
                    for name in names:
                        guard()
                        restore(name)
                        torch.random.set_rng_state(rng_cpu)
                        torch.cuda.set_rng_state(rng_cuda, index)
                        output = functions[name]()
                        torch.cuda.synchronize()
                        state = visible_snapshot(caches[name])
                        if oracle is None:
                            oracle, oracle_state = output.detach().cpu(), state
                        numerical = compare(oracle, output.detach().cpu())
                        state_comparison = state_equal(oracle_state, state)
                        immutable = all(byte_equal(saved_inputs[key], value) for key, value in tensors([inputs, kwargs]))
                        rng_same = byte_equal(rng_cpu, torch.random.get_rng_state()) and byte_equal(rng_cuda, torch.cuda.get_rng_state(index))
                        historical = compare(fixture.data["block_gold"].cpu(), output.detach().cpu())
                        row = dict(case=fixture.case_id, cache=cache_mode, engine=name, complete_block=numerical,
                                   state=state_comparison, inputs_unchanged=immutable, RNG_unchanged=rng_same,
                                   historical_GPU_gold=historical,
                                   scope="normalized serialized input GPU replay; original stride/alias not recovered")
                        result["numeric"].append(row)
                        _json(args.output / "RESULT.json", result)
                        if not (numerical["bitwise_pass"] and state_comparison["pass_all"] and immutable and rng_same and historical["bitwise_pass"]):
                            raise RuntimeError("Strict fixture/block/state gate failed; no timing or tolerance relaxation")
                    def measure(a, b):
                        arm_map = {"A": a, "B": b}
                        return measure_pair(torch, functions[a], functions[b], lambda arm: restore(arm_map[arm]),
                                            rounds=3, repeats=3, warmup=5, guard=guard)
                    AA = measure("R0", "R0_TWIN")
                    result["rows"].append(dict(case=fixture.case_id, cache=cache_mode, engine="A_A", measurement=AA))
                    if any(abs(value - 1) > 0.025 for value in AA["summary"]["paired_ratios"]):
                        raise RuntimeError("PERF_PENDING_AA_NOISE: stop matrix, preserve all raw samples")
                    for name in names[2:]:
                        measured = measure("R0", name)
                        result["rows"].append(dict(case=fixture.case_id, cache=cache_mode, engine=name,
                                                   scope="prepared block stream elapsed and fixture integration with restore",
                                                   measurement=measured))
                        _json(args.output / "RESULT.json", result)
                    post_state = {name: state_equal(oracle_state, visible_snapshot(caches[name]))
                                  for name in names}
                    inputs_ok = all(byte_equal(saved_inputs[key], value) for key, value in tensors([inputs, kwargs]))
                    rng_ok = byte_equal(rng_cpu, torch.random.get_rng_state()) and byte_equal(rng_cuda, torch.cuda.get_rng_state(index))
                    result.setdefault("post_timing_state", []).append(dict(
                        case=fixture.case_id, cache=cache_mode, fields=post_state,
                        inputs_unchanged=inputs_ok, RNG_unchanged=rng_ok))
                    if not inputs_ok or not rng_ok or not all(value["pass_all"] for value in post_state.values()):
                        raise RuntimeError("Post-timing state/input/RNG contract failed")
                # Preserve aliases/weights, release only this fixture's owned mutable buffers.
                functions.clear()
                modules.clear()
                caches.clear()
            _json(args.output / "RESULT.json", result)
            resource_between_windows(f"after-{length}-B{batch}")
        b0 = [row for row in result["rows"] if row["engine"] == "B0_TORCH"]
        ratios = [row["measurement"]["summary"]["ratio_geomean"] for row in b0]
        result["B0_engineering_screen"] = dict(
            scope="block only; neither robot noninferiority nor universal performance equality",
            matched_point_count=len(ratios), paired_geomean=math.exp(sum(map(math.log, ratios)) / len(ratios)),
            reproducible_over_10pct=[row["case"] for row in b0 if all(value > 1.10 for value in row["measurement"]["summary"]["paired_ratios"])])
        screen = result["B0_engineering_screen"]
        screen["unresolved_point_variation"] = [row["case"] for row in b0 if
            any(value > 1.10 for value in row["measurement"]["summary"]["paired_ratios"])
            and not all(value > 1.10 for value in row["measurement"]["summary"]["paired_ratios"])]
        screen["bounded_screen_passed"] = (len(ratios) == 12 and screen["paired_geomean"] <= 1.05
            and not screen["reproducible_over_10pct"] and not screen["unresolved_point_variation"])
        result["status"] = "COMPLETE_FINITE_BLOCK_MATRIX_NOT_ROBOT_QUALITY"
    except Exception as error:
        result.update(status="PARTIAL_OR_FAILED_BLOCK_MATRIX", reason=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        result["finished_at"] = datetime.now().astimezone().isoformat()
        result["lifecycle_wall_seconds"] = time.monotonic() - start
        result["cleanup"] = "private adapters restored by ExitStack; CUDA context released on dedicated worker exit"
        _json(args.output / "RESULT.json", result)
        with (args.output / "PERFORMANCE-RESULT.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case", "cache", "engine", "status", "independent_rounds", "paired_geomean"])
            writer.writeheader()
            for row in result["rows"]:
                writer.writerow(dict(case=row["case"], cache=row["cache"], engine=row["engine"],
                                     status=result["status"], independent_rounds=row["measurement"]["summary"]["independent_rounds"],
                                     paired_geomean=row["measurement"]["summary"]["ratio_geomean"]))
        with (args.output / "RAW-TIMING-SAMPLES.csv").open("w", newline="") as stream:
            fields = ["case", "cache", "engine", "round", "position", "arm", "repeat", "event_ms", "prepared_wall_ms", "integration_wall_ms"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in result["rows"]:
                for sample in row["measurement"]["raw_samples"]:
                    writer.writerow(dict(case=row["case"], cache=row["cache"], engine=row["engine"], **sample))

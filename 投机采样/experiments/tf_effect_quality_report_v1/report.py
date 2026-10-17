"""Read-only CPU audit of the fixed, complete96 TF-effect DEV campaign.

This module never launches a simulator, training worker, GPU process or retry.
It writes only a separately registered report directory, never the campaign.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from functools import lru_cache
from types import SimpleNamespace

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
RT = ROOT / "投机采样/results/runtime"
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXP))
COMMIT = "b0dfac3a02f65de33a1217246b83003615166640"
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
ARMS = ("T", "TF_CONT", "TF_ON", "MLP_ON")
CAMERAS = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
LIMIT = 128 * 2**20
READ_LIMIT = 64 * 2**20
UNKNOWN = "UNKNOWN"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    require(path.stat().st_size <= READ_LIMIT, "oversized small evidence: " + str(path))
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= LIMIT, "individual report exceeds storage budget")
    with path.open("xb") as stream:
        stream.write(raw)


def dist(values):
    import numpy as np
    require(all(math.isfinite(x) for x in values), "nonfinite metric")
    return dict(n=len(values), total=sum(values), mean=float(np.mean(values)) if values else None,
        p50=float(np.quantile(values, .5)) if values else None,
        p90=float(np.quantile(values, .9)) if values else None,
        min=min(values) if values else None, max=max(values) if values else None)


def runtime():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU report must hide CUDA")
    require(len(os.sched_getaffinity(0)) <= 2, "pin CPU report to at most two cores")


def markers(job):
    names = {p for p in job.rglob("*") if p.is_file() and
        ("FAILED" in p.name or p.name in ("STOP_REQUESTED", "CLOCK-STOP.json"))}
    return {str(p.relative_to(job)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(names)}


def prepare(job, output):
    birth = time.time()
    runtime()
    job, output = Path(job).resolve(strict=True), Path(output).resolve()
    require(output.parent == RT and not output.exists() and output != job,
        "new independent runtime report directory required")
    from tf_effect_campaign_v1.protocol import load
    p = load(job)
    require(p["collect"] is False and p["kind"] == "DEV" and len(p["cases"]) == 96,
        "only the preregistered96 complete DEV campaign is supported")
    now = birth
    global_hard = datetime.fromisoformat(p["global_cleanup_deadline"]).timestamp()
    deadline = min(now + 900, global_hard)
    require(time.time() < deadline, "unchanged global12h report budget already exhausted")
    reg = dict(schema="tf-effect-quality-report-registration/v1", source_commit=COMMIT,
        goal="CPU audit all96 terminal DEV episodes, strict initial pairing and fixed TF-on gates",
        input_campaign=str(job), input_manifest=str(job / "plan.json"),
        input_manifest_sha256=sha(job / "plan.json"),
        input_registration_sha256=sha(job / "PREREGISTRATION.json"),
        created_at=datetime.now(timezone.utc).isoformat(), birth_epoch=now,
        deadline_epoch=deadline, wall_clock_cap_seconds=deadline - now,
        requested_report_cap_seconds=900, global_cleanup_deadline=p["global_cleanup_deadline"],
        global_12h_budget_not_reset=True,
        CPU_cores_max=2, GPU=False, storage_cap_bytes=LIMIT,
        source_sha256={str(OWN / "report.py"): sha(OWN / "report.py")},
        terminal_status="REGISTERED_NOT_STARTED", permissions=PERMISSIONS,
        no_training=True, no_collection=True, no_retry=True, no_120=True, no_TEST=True,
        cleanup="no children or CUDA; restore signal/affinity, exit; never signal campaign processes",
        evidence_path=str(output))
    output.mkdir(exist_ok=False)
    write(output / "PREREGISTRATION.json", reg)
    write(output / "PLAN.json", dict(reg, preregistration_sha256=sha(output / "PREREGISTRATION.json")))
    return reg


def array_hash(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rng_fingerprint(value):
    import torch
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
        and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
        "actual saved RNG must be a nonempty CPU uint8 state vector")
    x = value.detach().contiguous()
    return dict(shape=list(x.shape), dtype=str(x.dtype),
        bytes_sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())


def arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}


def exact_arrays(a, b):
    return set(a) == set(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
        and a[k].tobytes() == b[k].tobytes() for k in a)


def exact_value(a, b):
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(exact_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact_value(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def identity(context, case, attempt):
    require(context["task_index"] == case["task"] and context["episode_index"] == case["init"]
        and context["parallel_worker_id"] == case["key"] and context["episode_attempt_id"] == attempt,
        "request/acceptance identity differs")


@lru_cache(maxsize=1)
def original_cpu_postprocessor():
    """Extract the frozen original CPU method; do not reimplement its math.

    The real server explicitly moves actions to CPU before postprocessing and
    constructs FP32 quantiles on CPU. This is the same device/arithmetic path.
    """
    import torch
    server = ROOT / "投机采样/wam-p0-worktree/wan_va/wan_va_server.py"
    config = ROOT / "投机采样/wam-p0-worktree/wan_va/configs/va_libero_cfg.py"
    methods = [n for n in ast.walk(ast.parse(server.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "postprocess_action"]
    require(len(methods) == 1 and not methods[0].decorator_list, "original CPU postprocess method differs")
    values = [n.value for n in ast.parse(config.read_text()).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "norm_stat" for t in n.targets)]
    require(len(values) == 1, "unique frozen norm_stat required")
    allowed = (ast.Dict, ast.List, ast.Constant, ast.Load, ast.BinOp, ast.Add, ast.Mult, ast.UnaryOp, ast.USub)
    require(all(isinstance(n, allowed) for n in ast.walk(values[0])), "norm_stat is not literal arithmetic")
    norm = eval(compile(ast.Expression(values[0]), str(config), "eval"), {"__builtins__": {}})
    scope = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(server), "exec"), scope)
    obj = SimpleNamespace(actions_q01=torch.tensor(norm["q01"], dtype=torch.float32).reshape(-1, 1, 1),
        actions_q99=torch.tensor(norm["q99"], dtype=torch.float32).reshape(-1, 1, 1),
        action_norm_method="quantiles", job_config=SimpleNamespace(used_action_channel_ids=list(range(7))))
    return scope["postprocess_action"], obj


def output_coverage(out, record, counter, mapping):
    import numpy as np
    import torch
    expected, policy, cache = [], [], []
    for item in counter["requests"]:
        kind, frame = item["kind"], item["server_frame_st_id_before"]
        if kind == "reset":
            require(item["expected_saved_outputs"] == [] and item["server_frame_st_id_after"] == 0,
                "reset save/frame differs")
            continue
        chunk = item["trace_context"]["rollout_chunk"]
        require(type(chunk) is int and frame == 4 * chunk, "frame/chunk differs")
        if kind == "policy":
            policy.append(chunk)
            names = [f"actions_{frame}.pt", f"latents_{frame}.pt"]
            require(item["server_frame_st_id_after"] == frame and item["expected_saved_outputs"] == names,
                "policy save/frame differs")
            expected += names
        elif kind == "cache":
            cache.append(chunk)
            require(item["server_frame_st_id_after"] == frame + 4 and item["expected_saved_outputs"] == []
                and item["suppressed_saved_outputs"] == [f"obs_data_{frame}.pt"],
                "registered observation suppression differs")
        else:
            raise ValueError("unknown model request kind")
    require(policy == list(range(record["infer_chunks"])) and cache == list(range(record["cache_infer_calls"])),
        "policy/cache sequence gaps")
    actual = list((out / "server_output").rglob("*.pt"))
    require(len(actual) == len(expected) == len(set(expected))
        and {x.name for x in actual} == set(expected), "saved action/latent inventory differs")
    saved = counter["save_async"]
    require(saved["original_executor_drained"] is True and saved["obs_suppressed"] == len(cache)
        and saved["kept"] == len(expected)
        and {Path(x["path"]).resolve() for x in saved["kept_artifacts"]} == {x.resolve() for x in actual},
        "original asynchronous queue/save coverage differs")
    chunk_map = {x["chunk"]: x for x in mapping["chunks"]}
    require(set(chunk_map) == set(policy) and len(chunk_map) == len(mapping["chunks"]),
        "transported chunk sequence differs")
    for file in actual:
        value = torch.load(file, map_location="cpu", weights_only=True)
        require(isinstance(value, torch.Tensor) and value.numel() > 0
            and bool(torch.isfinite(value).all()), "nonfinite original saved tensor")
        entry = next(x for x in saved["kept_artifacts"] if Path(x["path"]).resolve() == file.resolve())
        require(entry["bytes"] == file.stat().st_size and entry["suppressed"] is False,
            "saved artifact byte receipt differs")
        if file.name.startswith("actions_"):
            chunk = int(file.stem.split("_")[1]) // 4
            got = chunk_map[chunk]
            require(got["dtype"] == "float32" and got["shape"] == [7, 4, 4], "transport action geometry differs")
            # The original save precedes physical postprocessing and is a full
            # normalized BF16 tensor. Invoke the exact extracted original CPU
            # postprocessor, preserving all casts and arithmetic boundaries.
            expected_action = np.asarray(got["received_action"], dtype=np.float32)
            postprocess, obj = original_cpu_postprocessor()
            cpu_action = postprocess(obj, value)
            require(cpu_action.shape == expected_action.shape and cpu_action.tobytes() == expected_action.tobytes(),
                "saved physical action differs from transported action")
    return dict(files=len(actual), action_files=len(policy), latent_files=len(policy),
        observation_files=0, observations_suppressed=len(cache), all_saved_tensors_finite=True,
        original_executor_drained=True, saved_physical_to_transport_bitwise_equal=True)


def episode(job, case, ready, plan):
    import numpy as np
    out = job / "episodes" / case["key"]
    folder = out / "client"
    records = [json.loads(x) for x in (folder / "episodes.jsonl").read_text().splitlines() if x.strip()]
    require(len(records) == 1, "one complete actual client episode required")
    r = records[0]
    require(r.get("status") == "completed" and r.get("benchmark") == "libero_10"
        and r.get("task_index") == case["task"] and r.get("episode_index") == case["init"]
        and r.get("libero_horizon") == 800 and r.get("max_policy_chunks") is None
        and r.get("truncated_after_max_policy_chunks") is False
        and type(r["success"]) is bool and (r["success"] or r.get("horizon_reached") is True)
        and type(r["env_steps"]) is int and 0 < r["env_steps"] <= 800 and r["infer_chunks"] > 0,
        "complete original terminal episode required")
    require(len(r["action_shapes"]) == r["infer_chunks"] and all(s == [7, 4, 4] for s in r["action_shapes"]),
        "full action shapes differ")
    meta = read(folder / "repro_metadata.json")
    require(meta["seed"] == plan["seed"] and meta["no_policy_chunk_truncation"] is True,
        "metadata seed/truncation differs")
    images = arrays(folder / "initial_observations.npz")
    require(set(images) == set(meta["initial_obs_hashes"]) == set(CAMERAS), "both initial cameras required")
    for name, value in images.items():
        require(value.shape == (128, 128, 3) and value.dtype == np.uint8
            and array_hash(value) == meta["initial_obs_hashes"][name], "actual camera hash differs")
    for when in ("initial", "final"):
        values = arrays(folder / f"{when}_state.npz")
        require(set(values) == set(meta[f"{when}_state_hashes"]), "sim/controller field inventory differs")
        for name, value in values.items():
            require(bool(np.isfinite(value).all()) and array_hash(value) == meta[f"{when}_state_hashes"][name],
                "actual sim/controller array differs from receipt")
    commands = np.load(folder / "commands.npy", allow_pickle=False)
    require(commands.dtype == np.float32 and commands.shape == (r["env_steps"], 7)
        and bool(np.isfinite(commands).all()) and meta["command_count"] == len(commands), "full command coverage differs")
    mapping = read(folder / "COMMAND-MAPPING.json")
    require(mapping["schema"] == "physical-action-to-environment-command/v1" and mapping["complete_episode"] is True
        and mapping["success"] == r["success"] and mapping["env_steps"] == r["env_steps"]
        and len(mapping["submitted"]) == len(mapping["environment"]) == len(commands)
        and mapping["termination_reason"] == ("success" if r["success"] else "full_horizon"),
        "full submitted/environment mapping differs")
    chunks = {x["chunk"]: x for x in mapping["chunks"]}
    for k, (a, b) in enumerate(zip(mapping["submitted"], mapping["environment"])):
        require(a["env_index"] == b["env_index"] == k and a["chunk"] == b["chunk"]
            and a["command"] == b["command"], "environment command changed")
        submitted = np.asarray(a["command"], dtype=np.float32)
        require(submitted.tobytes() == commands[k].tobytes(), "command log differs from submitted command")
        chunk = chunks[a["chunk"]]
        identity(chunk["context"], case, r["episode_attempt_id"])
        received = np.asarray(chunk["received_action"], dtype=np.float32)
        require(submitted.tobytes() == received[:, a["frame"], a["subaction"]].tobytes(),
            "transported physical action differs from submitted command")
    reset = read(out / "RESET.json")
    require(reset["status"] == "PASS_FULL_RESET_BANK_AND_EMPTY_STATE" and reset["row"] == case
        and reset["RNG_unchanged"] is True and reset["state"]["frame_st_id"] == 0
        and len(reset["state"]["KV"]) == 30 and reset["state"]["init_latent"] is None
        and reset["state"]["cpu_rng"] == ready["target_initial_rng"]["cpu"]
        and reset["state"]["cuda_rng"] == ready["target_initial_rng"]["cuda"], "full reset/RNG differs")
    c = read(out / "COUNTER.json")
    require(c["row"] == case and c["arm"] == case["arm"] and c["key"] == case["key"]
        and c["target_forwards"] == c["returned_forwards"] and 0 < c["target_forwards"] <= plan["per_episode_target_forward_cap"]
        and c["permissions"] == PERMISSIONS and c["performance_claim"] is False
        and c["extra_teacher_forwards"] == 0, "closed frozen counter differs")
    teacher = read(out / "TEACHER-INDEX.json")
    require(teacher == c["teacher"] and teacher["enabled"] is False and teacher["shards"] == []
        and teacher["rows"] == teacher["reached_rows"] == teacher["trees"] == teacher["extra_teacher_forwards"] == 0,
        "DEV must not capture a new training teacher dataset")
    require(read(out / "CLIENT-CLEANUP.json")["all_roots_reaped"] is True, "client cleanup incomplete")
    for name in ("FINISH_SERVER.json", "FINISH_READY.json"):
        finish = read(out / name)
        require(finish["status"] == "PASS" and finish["full_episode_terminal"] is True
            and finish["success"] == r["success"] and finish["env_steps"] == r["env_steps"]
            and finish["teacher_complete"] is False and finish["extra_teacher_forwards"] == 0
            and finish["permissions"] == PERMISSIONS, "full terminal server proof differs")
    requests = c["requests"]
    counts = Counter(x["kind"] for x in requests)
    require(all(x["ok"] for x in requests) and counts["reset"] == 1
        and counts["policy"] == r["infer_chunks"] and counts["cache"] == r["cache_infer_calls"]
        and sum(x["target_forwards"] for x in requests) == c["target_forwards"], "request coverage differs")
    for x in requests:
        identity(x["trace_context"], case, r["episode_attempt_id"])
    depths = Counter()
    for tree in c["acceptance"]:
        identity(tree["context"], case, r["episode_attempt_id"])
        depth = tree["actual_depth"]
        require(depth in (0, 1, 2) and tree["threshold"] == plan["threshold"]
            and tree["effective_selector"] == plan["selector"], "registered tree gate/selector differs")
        first, second = float(tree["first_costs"][tree["actual_parent"]]), float(tree["second_costs"][tree["actual_parent"]][tree["actual_child"]])
        require(math.isfinite(first) and math.isfinite(second) and first >= 0 and second >= 0
            and depth == (0 if first > plan["threshold"] else 1 if second > plan["threshold"] else 2),
            "accepted path violates saved local threshold")
        costs = [float(x) for x in tree["first_costs"]] + [float(x) for row in tree["second_costs"] for x in row]
        require(len(tree["first_costs"]) == len(tree["second_costs"]) == 2
            and all(len(row) == 2 for row in tree["second_costs"])
            and all(math.isfinite(x) and x >= 0 for x in costs), "full saved local cost geometry/finiteness differs")
        possible = [0 if tree["first_costs"][a] > plan["threshold"] else
            1 if tree["second_costs"][a][b] > plan["threshold"] else 2 for a in (0, 1) for b in (0, 1)]
        require(depth == max(possible) == tree["maximum_admissible_depth"], "actual route is not maximum legal prefix")
        depths[depth] += 1
    speculative = case["arm"] != "T"
    trees = sum(depths.values())
    require((trees > 0 and trees == c["phase_counters"].get("tree_count", 0)) if speculative else trees == 0,
        "speculative tree count differs")
    require(len(c["chunks"]) == r["infer_chunks"], "per-chunk counter coverage differs")
    if speculative:
        require(c["phase_counters"].get("committed_scheduler_update_count") == 50 * r["infer_chunks"]
            and len(c["O3_events"]) == r["infer_chunks"]
            and all(x.get("enabled") and x.get("ok") and x.get("closed") for x in c["O3_events"]),
            "actual legal updates/O3 episode coverage differs")
    for chunk in c["chunks"]:
        identity(chunk["context"], case, r["episode_attempt_id"])
        require(not chunk.get("teacher_shard"), "DEV unexpectedly saved teacher shards")
        if speculative:
            require(chunk["phase_counters"].get("committed_scheduler_update_count") == 50,
                "complete speculative chunk legal scheduler update count differs")
    final = read(out / "FINAL-SERVER-STATE.json")
    require(final["frame_st_id"] >= 0 and len(final["KV"]) == 30
        and all(set(x) == {"mask", "id", "is_pred", "k", "v"} for x in final["KV"])
        and {"VAE", "init_latent", "cpu_rng", "cuda_rng", "prompt", "negative", "action_mask", "q01", "q99", "action_norm_method"} <= set(final),
        "full final30-layer state receipt differs")
    _, normalizer = original_cpu_postprocessor()
    for name, value in (("q01", normalizer.actions_q01), ("q99", normalizer.actions_q99)):
        require(final[name] == dict(shape=list(value.shape), dtype=str(value.dtype), bytes_sha256=array_hash(value.numpy())),
            "original physical postprocess quantiles differ from final actual server")
    saved = output_coverage(out, r, c, mapping)
    return dict(arm=case["arm"], role=case["role"], task=case["task"], init=case["init"], key=case["key"],
        episode_attempt_id=r["episode_attempt_id"], success=r["success"], horizon_reached=r["horizon_reached"],
        termination_reason=mapping["termination_reason"], env_steps=r["env_steps"], failure800_penalty=r["env_steps"] if r["success"] else 800,
        policy_chunks=r["infer_chunks"], target_forwards=c["target_forwards"], trees=trees,
        accepted_depth={str(k): depths[k] for k in (0, 1, 2)}, accepted_depth_sum=sum(k * v for k, v in depths.items()),
        legal_scheduler_updates_g=c["phase_counters"].get("committed_scheduler_update_count", UNKNOWN),
        target_contract_scheduler_updates=50 * r["infer_chunks"],
        sum_T_round_sec=UNKNOWN, cost_per_legal_commit=UNKNOWN,
        complete_episode_wall_sec=float(r["elapsed_sec"]), reset_wall_sec=reset["reset_wall_sec"],
        request_wall_sec={k: sum(x["wall_sec"] for x in requests if x["kind"] == k) for k in counts},
        phase_counters=c["phase_counters"], initial_pairing={k: meta[k] for k in
            ("seed", "model_xml_sha256", "init_state_sha256", "initial_state_hashes", "initial_obs_hashes", "initial_rng")},
        saved_output_audit=saved, teacher_enabled=False, extra_teacher_forwards=0,
        performance_claim=False, timing_scope="shared-resource complete episode diagnostic; not formal performance")


def summarize(rows, expected):
    trees = sum(x["trees"] for x in rows)
    known_g = all(type(x["legal_scheduler_updates_g"]) is int for x in rows)
    return dict(planned_episodes=expected, audited_complete_episodes=len(rows),
        successes=sum(x["success"] for x in rows), failures=[{k: x[k] for k in
            ("task", "init", "key", "env_steps", "termination_reason")} for x in rows if not x["success"]],
        success_rate_on_audited_only=sum(x["success"] for x in rows) / len(rows) if rows else None,
        complete_registered_arm=len(rows) == expected,
        actual_env_steps=dist([x["env_steps"] for x in rows]),
        failure800_penalty=dist([x["failure800_penalty"] for x in rows]),
        target_forwards=sum(x["target_forwards"] for x in rows), policy_chunks=sum(x["policy_chunks"] for x in rows),
        accepted_depth={str(k): sum(x["accepted_depth"][str(k)] for x in rows) for k in (0, 1, 2)},
        trees=trees, mean_accepted_depth=sum(x["accepted_depth_sum"] for x in rows) / trees if trees else None,
        legal_scheduler_updates_g=sum(x["legal_scheduler_updates_g"] for x in rows) if rows and known_g else UNKNOWN,
        sum_T_round_sec=UNKNOWN, cost_per_legal_commit=UNKNOWN,
        complete_episode_wall_sec=dist([x["complete_episode_wall_sec"] for x in rows]), performance_claim=False)


def paired(rows, candidate, reference, expected):
    by = {(x["task"], x["init"], x["arm"]): x for x in rows}
    keys = sorted({(x["task"], x["init"]) for x in rows if x["arm"] in (candidate, reference)})
    pairs, missing = [], []
    for task, init in keys:
        if any((task, init, a) not in by for a in (candidate, reference)):
            missing.append(dict(task=task, init=init))
            continue
        a, b = by[task, init, candidate], by[task, init, reference]
        pairs.append(dict(task=task, init=init, candidate_key=a["key"], reference_key=b["key"],
            candidate_success=a["success"], reference_success=b["success"], candidate_steps=a["env_steps"], reference_steps=b["env_steps"],
            both_success_steps_ratio=a["env_steps"] / b["env_steps"] if a["success"] and b["success"] else None))
    n = len(pairs)
    return dict(candidate=candidate, reference=reference, planned_pairs=expected, audited_pairs=n,
        complete_registered_comparison=n == expected and not missing, missing=missing,
        four_cells={f"reference_{r}_candidate_{c}": sum(x["reference_success"] == bool(r) and x["candidate_success"] == bool(c) for x in pairs)
            for r in (0, 1) for c in (0, 1)},
        success_difference_pp=100 * sum(int(x["candidate_success"]) - int(x["reference_success"]) for x in pairs) / n if n else None,
        both_success_steps_ratio=dist([x["both_success_steps_ratio"] for x in pairs if x["both_success_steps_ratio"] is not None]),
        rows=pairs, confidence_interval=UNKNOWN, noninferiority_established=False,
        limitation="fixed engineering DEV/known-risk groups; neither independent TEST nor noninferiority evidence")


def pairing(job, base, new, rng_exact):
    import numpy as np
    bf, nf = [job / "episodes" / r["key"] / "client" for r in (base, new)]
    x, y = dict(base["initial_pairing"]), dict(new["initial_pairing"])
    x.pop("initial_obs_hashes"); y.pop("initial_obs_hashes")
    physical_arrays = exact_arrays(arrays(bf / "initial_state.npz"), arrays(nf / "initial_state.npz"))
    reset_equal = read(bf.parent / "RESET.json")["state"] == read(nf.parent / "RESET.json")["state"]
    cameras = {}
    a, b = arrays(bf / "initial_observations.npz"), arrays(nf / "initial_observations.npz")
    for k in CAMERAS:
        diff = np.abs(a[k].astype(np.int16) - b[k].astype(np.int16))
        exact = a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes()
        cameras[k] = dict(exact=exact, changed_channels=int(np.count_nonzero(diff)),
            changed_pixels=int(np.count_nonzero(np.any(diff != 0, axis=2))), max_abs=int(diff.max()))
    physical = x == y and physical_arrays and reset_equal and rng_exact
    return dict(role=new["role"], task=new["task"], init=new["init"], reference=base["arm"], candidate=new["arm"],
        reference_key=base["key"], candidate_key=new["key"],
        initial_metadata_without_cameras_equal=x == y, actual_sim_controller_arrays_exact=physical_arrays,
        actual_target_initial_RNG_equal=rng_exact, full_reset_state_equal=reset_equal,
        physical_RNG_equal=physical, camera_pixels=cameras,
        all_exact=physical and all(v["exact"] for v in cameras.values()),
        scope="saved initial sim/controller, reset and CPU/CUDA RNG; unsaved simulator state UNKNOWN")


def training_audit(p):
    import torch
    dirs = {p["selected"][a]["training_dir"] for a in ("TF_ON", "MLP_ON")}
    require(len(dirs) == 1, "TF-on and matched MLP must share registered training input")
    folder = Path(dirs.pop()).resolve(strict=True)
    plan, final, selection = [read(folder / k) for k in ("PLAN.json", "FINAL-RESULT.json", "selected.json")]
    require(plan["source_commit"] == COMMIT and final["status"] == "PASS_FIXED_FINAL_TF_EFFECT_TRAINING"
        and final["conditions"] == ["TF_CONT", "TF_ON", "MLP_ON"] and final["complete_runs"] == 3
        and final["permissions"] == PERMISSIONS and final["CUDA_initialized"] is False
        and plan["initial_total_updates"] == 2500 and plan["continuation_updates"] == 500 and plan["total_updates"] == 3000
        and plan["seed"] == 20260923 and plan["normalization_refit"] is False
        and plan["batch"] == 512 and plan["target_half"] == plan["same_query_half"] == 256
        and plan["learning_rate"] == 3e-5 and plan["weight_decay"] == .01
        and plan["checkpoint_selection"] == "final update500 only; no CAL or seed selection"
        and not (folder / "FAILED.json").exists() and not (folder / "STOP_REQUESTED").exists(),
        "finite fixed fair500-update training contract differs")
    for file, digest in plan["source_sha256"].items():
        require(sha(file) == digest, "training frozen source changed")
    for file, digest in plan["evidence_sha256"].items():
        require(sha(file) == digest, "training source evidence changed")
    old = torch.load(p["selected"]["TF_CONT"]["checkpoint"], map_location="cpu", weights_only=True)
    records = {}
    for a in ("TF_CONT", "TF_ON", "MLP_ON"):
        record = read(folder / f"{a}-seed20260923/RESULT.json")
        require(record["status"] == "PASS_FIXED_FINAL_CELL" and record["condition"] == a
            and record["continuation_updates"] == 500 and record["initial_total_updates"] == 2500
            and record["total_updates"] == 3000 and record["seed"] == 20260923
            and record["normalization_exact_parent"] is True and record["permissions"] == PERMISSIONS,
            "fixed final training cell differs")
        new = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
        parent = torch.load(record["parent_checkpoint"], map_location="cpu", weights_only=True)
        require(sha(record["parent_checkpoint"]) == record["parent_checkpoint_sha256"]
            and exact_value(new["normalization"], parent["normalization"]), "actual parent/normalizer differs")
        require(sha(record["checkpoint"]) == record["checkpoint_sha256"], "training cell checkpoint changed")
        if a == "TF_CONT":
            exact = exact_value(old["model_state_dict"], new["model_state_dict"]) and exact_value(old["normalization"], new["normalization"])
            require(record["old_TF_CONT_all_model_tensors_bitwise_exact"] is True and exact,
                "deployed oldTF-cont weights are not the equal-budget reproduced control")
        else:
            require(record["checkpoint"] == p["selected"][a]["checkpoint"]
                and record["checkpoint_sha256"] == p["selected"][a]["checkpoint_sha256"]
                and selection["conditions"][a]["checkpoint_sha256"] == record["checkpoint_sha256"],
                "DEV candidate differs from fixed final selected checkpoint")
        records[a] = {k: record[k] for k in ("checkpoint", "checkpoint_sha256", "initial_total_updates",
            "continuation_updates", "total_updates", "parent_checkpoint", "parent_checkpoint_sha256", "architecture")}
    require(records["TF_CONT"]["parent_checkpoint_sha256"] == records["TF_ON"]["parent_checkpoint_sha256"],
        "TF controls did not use same2500 parent")
    return dict(training_dir=str(folder), conditions=records, same_TF_parent=True,
        old_TF_CONT_reproduced_all_model_tensors=True, no_CAL_or_seed_selection=True,
        generator="current TF_CONT3000; differs from common child2500 training start", permissions=PERMISSIONS)


def audit(job):
    import torch
    from tf_effect_campaign_v1.protocol import load
    p = load(job)
    require(p["collect"] is False and p["kind"] == "DEV" and p["arms"] == list(ARMS)
        and len(p["cases"]) == 96 and p["commit"] == COMMIT and p["permissions"] == PERMISSIONS,
        "fixed96 DEV plan differs")
    rows, issues, ready, rng, training = [], [], {}, {}, None
    marker_before = markers(job)
    if marker_before:
        issues.append(dict(scope="campaign", issue="original FAILED/STOP markers preserved", markers=list(marker_before)))
    expected = [x["key"] for a in p["arms"] for x in p["cases"] if x["arm"] == a]
    receipts = {}
    for name in ("RUN-STARTED.json", "CLOCK-GUARD.json", "CLOCK-RESULT.json", "COLLECTION.json",
            "SUPERVISOR-RESULT.json", "FINAL-OWNED-CLEANUP.json"):
        try:
            receipts[name] = read(job / name)
        except (OSError, ValueError) as exc:
            issues.append(dict(scope=name, issue=repr(exc)))
    try:
        col, sup, clean, clock, guard, started = [receipts[k] for k in ("COLLECTION.json", "SUPERVISOR-RESULT.json",
            "FINAL-OWNED-CLEANUP.json", "CLOCK-RESULT.json", "CLOCK-GUARD.json", "RUN-STARTED.json")]
        require(col["status"] == "COMPLETE_REGISTERED_FULL_EPISODES" and col["full_episodes"] == 96
            and col["completed"] == expected and col["collect"] is False and col["permissions"] == PERMISSIONS,
            "not all96 complete original episodes")
        require(sup["status"] == "COMPLETE" and sup["failure"] is None and sup["full_episodes"] == 96
            and sup["completed"] == expected and sup["permissions"] == PERMISSIONS
            and sup["total_target_forwards"] == col["total_target_forwards"] <= p["job_target_forward_cap"],
            "supervisor full96 terminal differs")
        require(clean["all_roots_reaped"] is True and sup["cleanup"] == clean
            and clock["all_observed_owned_live_processes_exited"] is True and clock["error"] is None
            and clock["external_stop_reason"] is None and clock["signals"] == []
            and clock["status"] == "CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS"
            and all(x["observation"]["live"] is False for x in clock["process_observations"])
            and clock["clock"] == guard["clock"] and clock["no_foreign_signals"] is True,
            "owned processes/independent clock not cleanly terminal")
        require(guard["supervisor_pin"]["pid"] == started["pid"]
            and guard["clock"]["plan_sha256"] == sha(job / "plan.json")
            and guard["clock"]["max_seconds"] == p["max_run_wall_sec"]
            and guard["clock"]["absolute_cleanup_deadline"] == p["absolute_cleanup_deadline"],
            "clock actual supervisor/plan/budget binding differs")
        launch = read(job / "ROOT-LAUNCH.json")
        pin = guard["supervisor_pin"]
        require(all(pin[k] == launch["pin"][k] for k in ("pid", "start_ticks", "uid", "pgid", "argv"))
            and launch["start_new_session"] is True and launch["command"] == launch["pin"]["argv"]
            and str(EXP / "tf_effect_campaign_v1/run.py") in pin["argv"] and str(job) in pin["argv"]
            and guard["no_budget_reset"] is True and guard["no_workload_launched"] is True
            and guard["clock"]["cleanup_reserve_seconds"] == p["cleanup_reserve_sec"]
            and guard["clock"]["hard_stop_monotonic"] <= guard["clock"]["process_birth_monotonic"] + p["max_run_wall_sec"]
            and guard["clock"]["work_stop_monotonic"] <= guard["clock"]["hard_stop_monotonic"] - p["cleanup_reserve_sec"]
            and clock["elapsed_quality_process_seconds"] <= guard["clock"]["hard_stop_monotonic"] - guard["clock"]["process_birth_monotonic"] + 2,
            "clock lifetime/argv/actual cumulative cap differs")
    except (KeyError, ValueError, TypeError) as exc:
        issues.append(dict(scope="terminal", issue=repr(exc)))
    try:
        training = training_audit(p)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        issues.append(dict(scope="fair_training", issue=repr(exc)))
    for arm in ARMS:
        try:
            r = read(job / f"{arm}-READY.json")
            require(r["status"] == "READY" and r["arm"] == arm and r["collect"] is False
                and r["collector_installed"] is False and r["setup_RNG_restored"] is True
                and r["target_RNG_captured_before_draft_and_CAL"] is True
                and r["ready_rng"] == r["target_initial_rng"] and r["permissions"] == PERMISSIONS
                and r["bank_explicitly_enabled"] is True and r["synthetic_queries"] == 0
                and read(job / "arms" / arm / "OWNED-CLEANUP.json")["all_roots_reaped"] is True,
                "arm ready/cleanup or no-collector contract differs")
            rng[arm] = torch.load(job / "arms" / arm / "TARGET-INITIAL-RNG.pt", map_location="cpu", weights_only=True)
            require(set(rng[arm]) == {"CPU", "CUDA"}, "actual initial target RNG tensors absent")
            require(all(rng_fingerprint(rng[arm][key]) == r["target_initial_rng"][label]
                for key, label in (("CPU", "cpu"), ("CUDA", "cuda"))),
                "actual initial target RNG tensor bytes/shape/dtype differ from READY/reset receipts")
            if arm == "TF_CONT":
                cal = read(job / "arms" / arm / "CAL-REUSE.json")
                original_cal = read(cal["source"])
                require(r["reused_real_CAL_rows"] == 20237 and r["full_real_CAL_rows"] == 0
                    and r["new_real_CAL_rows"] == 0 and r["predictor_checkpoint"] == p["selected"][arm]["checkpoint"]
                    and cal["status"] == "REUSE_EXISTING_COMPLETE_GPU_CAL_NOT_NEW_CAL_EXECUTION"
                    and cal["reference_rows"] == 20237 and cal["new_CAL_rows"] == 0
                    and cal["model_checkpoint_unchanged"] is True and cal["draft_adapter_source_unchanged"] is True
                    and cal["target_RNG_unchanged"] is True and cal["source_sha256"] == sha(cal["source"])
                    and original_cal["rows"] == 20237 and original_cal["complete_CAL"] is True
                    and original_cal["same_device_bitwise_exact"] is True
                    and original_cal["finite"] is True and original_cal["CPU_rng_unchanged"] is True
                    and original_cal["CUDA_rng_unchanged"] is True and r["O1_O2_O3"] is True
                    and original_cal["checkpoint_sha256"] == p["selected"][arm]["checkpoint_sha256"],
                    "unchanged TF_CONT exact old CAL reuse proof differs")
            elif arm != "T":
                cal = read(Path(p["gpu_cal_jobs"][arm]) / "RESULT.json")
                require(r["full_real_CAL_rows"] == 20237 and r["new_real_CAL_rows"] > 0
                    and r["reused_real_CAL_rows"] == 0 and r["O1_O2_O3"] is True
                    and r["predictor_checkpoint"] == p["selected"][arm]["checkpoint"]
                    and cal["status"] == "PASS_FULL_OLD_AND_NEW_REAL_CAL_SAME_DEVICE_PARITY"
                    and cal["condition"] == arm and cal["checkpoint_sha256"] == p["selected"][arm]["checkpoint_sha256"]
                    and cal["full_old_and_new_CAL"] is True and cal["CPU_rng_unchanged"] is True
                    and cal["CUDA_rng_unchanged"] is True and cal["synthetic_queries"] == 0
                    and cal["sets"]["old_CAL"]["rows"] == 20237
                    and cal["sets"]["new_CAL"]["rows"] == r["new_real_CAL_rows"]
                    and all(x["complete_CAL"] and x["finite"] and x["same_device_bitwise_exact"] for x in cal["sets"].values()),
                    "new candidate full old/new GPU CAL exact parity differs")
            else:
                require(r["predictor_checkpoint"] is None and r["O1_O2_O3"] is False
                    and r["full_real_CAL_rows"] == r["new_real_CAL_rows"] == r["reused_real_CAL_rows"] == 0,
                    "target-only arm unexpectedly has draft/CAL")
            ready[arm] = r
        except (OSError, KeyError, ValueError, TypeError) as exc:
            issues.append(dict(scope=arm, issue=repr(exc)))
    for case in p["cases"]:
        try:
            require(case["arm"] in ready, "valid full arm ready evidence missing")
            rows.append(episode(job, case, ready[case["arm"]], p))
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            issues.append(dict(scope=case["key"], issue=repr(exc)))
    missing = [x["key"] for x in p["cases"] if x["key"] not in {r["key"] for r in rows}]
    pairs = []
    by = {(r["role"], r["task"], r["init"], r["arm"]): r for r in rows}
    for r in rows:
        if r["arm"] == "T":
            continue
        base = by.get((r["role"], r["task"], r["init"], "T"))
        if base is None:
            continue
        actual_rng_equal = r["arm"] in rng and "T" in rng and all(
            torch.equal(rng[r["arm"]][k], rng["T"][k]) for k in ("CPU", "CUDA"))
        pairs.append(pairing(job, base, r, actual_rng_equal))
    fresh = [r for r in rows if r["role"] == "fresh_DEV"]
    risk = [r for r in rows if r["role"] == "risk_DEV"]
    summaries = dict(fresh_DEV={a: summarize([r for r in fresh if r["arm"] == a], 20) for a in ARMS},
        known_risk_DEV={a: summarize([r for r in risk if r["arm"] == a], 8) for a in ("T", "TF_ON")})
    comparisons = {f"{a}_vs_{b}": paired(fresh, a, b, 20) for a, b in
        (("TF_ON", "T"), ("TF_ON", "MLP_ON"), ("TF_CONT", "T"), ("TF_ON", "TF_CONT"))}
    risk_comparison = paired(risk, "TF_ON", "T", 8)
    require(markers(job) == marker_before, "campaign FAILED/STOP markers changed during read-only audit")
    if len(rows) == 96 and "COLLECTION.json" in receipts:
        if sum(r["target_forwards"] for r in rows) != receipts["COLLECTION.json"]["total_target_forwards"]:
            issues.append(dict(scope="total", issue="audited target forward sum differs from collection"))
    complete = len(rows) == 96 and not issues and len(pairs) == 68
    f = summaries["fresh_DEV"]
    ratio = comparisons["TF_ON_vs_T"]["both_success_steps_ratio"]
    checks = dict(TF_ON_fresh_success_at_least_T=f["TF_ON"]["successes"] >= f["T"]["successes"],
        TF_ON_fresh_success_at_least_matched_MLP=f["TF_ON"]["successes"] >= f["MLP_ON"]["successes"],
        nonempty_both_success=ratio["n"] > 0,
        length_P50=ratio["p50"] is not None and ratio["p50"] <= 1.10,
        length_P90=ratio["p90"] is not None and ratio["p90"] <= 1.20,
        length_max=ratio["max"] is not None and ratio["max"] <= 1.30,
        known_risk_no_new_TF_failure=risk_comparison["four_cells"]["reference_1_candidate_0"] == 0)
    numeric = all(checks.values()) if complete else UNKNOWN
    physical = all(x["physical_RNG_equal"] for x in pairs) if len(pairs) == 68 else UNKNOWN
    pixels = all(all(v["exact"] for v in x["camera_pixels"].values()) for x in pairs) if len(pairs) == 68 else UNKNOWN
    strict = complete and numeric is True and physical is True and pixels is True
    status = "COMPLETE96_FIXED_TF_ON_DEV_AUDIT" if complete else "PARTIAL_OR_FAILED_NO_COMPLETE96_DECISION"
    recommendation = ("RECOMMEND_FREEZE_FOR_SEPARATELY_REGISTERED_INDEPENDENT_TEST" if strict else
        "STOP_TF_QUALITY_TUNING_RETAIN_EXPLORATORY_ONLY" if complete else "NO_SELECTION_INCOMPLETE_OR_INVALID_EVIDENCE")
    return dict(schema="tf-effect-quality96-report/v1", source_commit=COMMIT, input_campaign=str(job),
        status=status, complete=complete, planned_full_episodes=96, audited_complete_episodes=len(rows),
        missing_or_invalid_episode_keys=missing, issues=issues, original_failure_markers=marker_before,
        summaries=summaries, comparisons=comparisons, known_risk_TF_ON_vs_T=risk_comparison,
        engineering_numeric_checks=checks if complete else {k: UNKNOWN for k in checks},
        new_DEV_numeric_screen_observed=numeric, physical_RNG_pairing_passed=physical,
        strict_initial_camera_pixels_passed=pixels, strict_current_engineering_screen_observed=strict if complete else UNKNOWN,
        recommendation=recommendation, pairing_comparisons=pairs, pairing_count=len(pairs),
        training_audit=training, episode_rows=rows, total_audited_target_forwards=sum(r["target_forwards"] for r in rows),
        sum_T_round_sec=UNKNOWN, sum_legal_speculative_g=sum(r["legal_scheduler_updates_g"] for r in rows if type(r["legal_scheduler_updates_g"]) is int),
        cost_per_legal_commit=UNKNOWN, timing_scope="shared-resource diagnostic only; no formal speedup claim",
        permissions=PERMISSIONS, global_default_changed=False, independent_TEST_started=False,
        no_120_started=True, report_writes_old_FAILED_or_retention=False, TEST_inits_read=False,
        architecture_superiority_claim=False, performance_claim=False,
        source_sha256_verified=len(p["source_sha256"]), target_model_hash_not_computed=True,
        wall_budget=dict(cap_seconds=p["max_run_wall_sec"], absolute_cleanup_deadline=p["absolute_cleanup_deadline"],
            global_cleanup_deadline=p["global_cleanup_deadline"],
            supervisor_elapsed_sec=receipts.get("SUPERVISOR-RESULT.json", {}).get("elapsed_sec", UNKNOWN),
            actual_CUDA_active_time_sec=UNKNOWN, global_12h_budget_not_reset=True),
        limitations=["fresh20 groups and known-risk8 are distinct DEV strata, never independent TEST or pooled with old T.",
            "All audited failed tasks remain in SR and fixed800 penalty; incomplete tasks are not algorithm failures.",
            "Neither nonsignificance nor a20-pair DEV pass establishes noninferiority or a general quality guarantee.",
            "Even ideal zero newly failed independent pairs at N=20 give1-0.05**(1/20)=13.91pp one-sided95% upper bound, not3pp NI.",
            "Saved initial physics/controller/RNG and camera gates are separate; pixel failures are never removed.",
            "Request and complete-episode timing include shared-resource effects; actual round timing was not collected.",
            "Target legal scheduler update count is UNKNOWN if uninstrumented;50/chunk is separately marked a contract count."])


def prose(result):
    complete = result["complete"]
    preamble = [f"状态：{result['status']}。", "固定基线 commit：`" + COMMIT + "`。",
        "TF_ON是主候选，MLP_ON仅为等预算对照。四个原权限保持false；未启动独立TEST、120局或正式性能。"]
    lines = ["# TF完整DEV结果", ""] + preamble + ["", "|分层|方案|完整/计划|成功|实际步数总和|失败800均值|target forwards|", "|---|---|---:|---:|---:|---:|---:|"]
    for stratum, cells in result["summaries"].items():
        for a, s in cells.items():
            mean = s["failure800_penalty"]["mean"]
            lines.append(f"|{stratum}|{a}|{s['audited_complete_episodes']}/{s['planned_episodes']}|{s['successes']}|{s['actual_env_steps']['total']}|{mean if mean is not None else 'UNKNOWN'}|{s['target_forwards']}|")
    for name, comparison in result["comparisons"].items():
        lines += ["", f"{name}配对：{comparison['four_cells']}；双方成功步数比：{comparison['both_success_steps_ratio']}。"]
    lines += ["", f"known-risk8配对：{result['known_risk_TF_ON_vs_T']['four_cells']}。",
        f"事前数值门：{result['engineering_numeric_checks']}。",
        f"物理/RNG：{result['physical_RNG_pairing_passed']}；严格初始像素：{result['strict_initial_camera_pixels_passed']}。",
        f"建议：{result['recommendation']}。", "", "完整episode耗时仅为共享资源下诊断；ΣT_round/Σg为UNKNOWN，不能报E2E加速。"]
    if not complete:
        lines += ["", "结果不齐或审计失败，不在通过片段上作选择。具体缺失和失败见RESULT.json。"]
    facts = ["# 已观察、未知与建议", ""] + preamble + ["",
        f"已审计完整episode：{result['audited_complete_episodes']}/96；完整终态：{complete}。",
        f"保留FAILED/STOP标记：{len(result['original_failure_markers'])}。",
        f"已核验冻结源码：{result['source_sha256_verified']}项；未运行整套target权重hash。", "",
        "UNKNOWN：未采集的逐round时间、CUDA活跃时间、unsaved simulator内部状态、独立TEST效果、非劣效性及正式性能。", "",
        f"建议：{result['recommendation']}；任何后续TEST须单独登记冻结候选、引擎与样本清单。"]
    facts += ["", "即使20个独立配对中零新增失败，理想单侧95%新增失败率上界仍约13.91pp；本DEV不能证明3pp非劣。配对SR差区间未估，标UNKNOWN。"]
    system = ["# 系统指标边界", "", "本作业仅是固定7候选、现有B14/O3路径的DEV质量实验。", "",
        "|指标|结果|", "|---|---|", f"|实际target forwards（完整已审计局）|{result['total_audited_target_forwards']}|",
        f"|实际合法speculative scheduler提交g|{result['sum_legal_speculative_g']}|", "|ΣT_round|UNKNOWN：未采集逐round时间|",
        "|ΣT_round/Σg|UNKNOWN|", "|1/3/7 × B14/G1/G2 × cached/uncached|本质量作业未执行|",
        "|L2/DRAM/CTA/register/shared memory/occupancy|UNKNOWN：未运行profiler|", "",
        "逐臂target forwards、accepted depth、policy chunks、g与完整episode诊断时间见RESULT.json。不能把forward减少写为task/E2E加速。"]
    claim = ["# Claim与证据", "", "|主张|当前证据|缺口|允许写入论文|", "|---|---|---|---|",
        f"|固定TF-on通过本轮DEV工程门|numeric={result['new_DEV_numeric_screen_observed']}；strict={result['strict_current_engineering_screen_observed']}|独立TEST/非劣|仅按实际DEV状态准确描述|",
        "|TF优于MLP架构|仅固定seed、相同数据预算的DEV对照|独立多seed与架构因果|不允许|",
        "|任务质量保持/非劣|独立TEST未启动|冻结后的配对独立TEST与正确区间|不允许|",
        "|有效提交成本下降|g已记；ΣT_round未记|同等边界逐round计时|不允许|",
        "|MegaTransformer协同/HBM下降/E2E加速|本质量作业没有相关新机制测量|强B14因果矩阵、profile及获准正式性能|不允许|"]
    wall = result["wall_budget"]
    elapsed = wall["supervisor_elapsed_sec"]
    left = max(0, wall["cap_seconds"] - elapsed) if isinstance(elapsed, (int, float)) else UNKNOWN
    stop = ["# 预算与止损", "", f"campaign墙钟上限：{wall['cap_seconds']}秒；supervisor实耗：{elapsed}秒。",
        f"原上限内理论余额：{left}秒；不是续跑/新作业授权。实际CUDA活跃时间UNKNOWN。",
        f"campaign清理硬截止：{wall['absolute_cleanup_deadline']}；全链截止：{wall['global_cleanup_deadline']}。",
        f"FAILED/STOP标记：{result['original_failure_markers']}。", f"审计缺失或错误：{result['issues']}。", "",
        "未完成任务保留PARTIAL；timeout/失败占用原预算，不改名续跑。", "若本轮完整DEV未过门，按登记规则停止TF质量调参，保留TF为探索候选；不改变阈值或旧retention。",
        "本报告CPU作业不启动GPU/训练/采集，不关闭租卡实例，不发送信号或修改旧证据。"]
    return {"QUALITY-RESULT.md": lines, "FACTS.md": facts, "SYSTEM-RESULT.md": system,
        "CLAIM-EVIDENCE-MATRIX.md": claim, "STOP-REPORT.md": stop}


def run(output):
    runtime()
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), "CPU report must not initialize CUDA")
    output = Path(output).resolve(strict=True)
    p = read(output / "PLAN.json")
    require(p["source_commit"] == COMMIT and p["permissions"] == PERMISSIONS
        and sha(output / "PREREGISTRATION.json") == p["preregistration_sha256"]
        and not (output / "RUN-STARTED.json").exists(), "fresh registered report only; no overwrite/retry")
    remaining = p["deadline_epoch"] - time.time()
    if remaining <= 0:
        write(output / "FAILED.json", dict(status="EXPIRED_CPU_REPORT_NOT_RUN", GPU=False,
            error="CPU cumulative/global deadline expired; waiting is charged", permissions=PERMISSIONS))
        raise TimeoutError("CPU cumulative/global deadline expired; waiting is charged")
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU report cumulative/global deadline")))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    start = time.monotonic()
    write(output / "RUN-STARTED.json", dict(pid=os.getpid(), status="CPU_AUDIT_ONLY", started_at=datetime.now(timezone.utc).isoformat(), GPU=False))
    try:
        for name, digest in p["source_sha256"].items():
            require(sha(name) == digest, "report source changed after registration")
        job = Path(p["input_campaign"]).resolve(strict=True)
        require(sha(job / "plan.json") == p["input_manifest_sha256"]
            and sha(job / "PREREGISTRATION.json") == p["input_registration_sha256"], "registered input campaign changed")
        result = audit(job)
        result["CPU_report_elapsed_sec"] = time.monotonic() - start
        result["CPU_report_seconds_since_registration"] = time.time() - p["birth_epoch"]
        result["report_source_sha256"] = sha(__file__)
        write(output / "RESULT.json", result)
        for name, lines in prose(result).items():
            with (output / name).open("x") as stream:
                stream.write("\n".join(lines) + "\n")
        require(not torch.cuda.is_initialized(), "unexpected CUDA initialization")
        require(time.time() <= p["deadline_epoch"], "CPU wall cap exceeded")
        require(sum(x.stat().st_size for x in output.rglob("*") if x.is_file()) <= p["storage_cap_bytes"], "report storage cap exceeded")
        write(output / "FINAL-RESULT.json", dict(status=result["status"], complete=result["complete"],
            elapsed_sec=time.monotonic() - start, GPU=False, CUDA_initialized=False, subprocesses_started=0,
            own_process_cleanup="this single CPU process exits; no campaign process signalled", permissions=PERMISSIONS))
        print(json.dumps(dict(status=result["status"], complete=result["complete"], output=str(output)), ensure_ascii=False))
        return 0 if result["complete"] else 2
    except BaseException as exc:
        write(output / "FAILED.json", dict(status="FAILED_CPU_REPORT_NOT_QUALITY_DECISION", error=repr(exc),
            traceback=traceback.format_exc(), elapsed_sec=time.monotonic() - start, GPU=False,
            campaign_evidence_not_modified=True, permissions=PERMISSIONS))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, help="fixed96 campaign; prepare only")
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        require(args.job is not None, "prepare requires --job")
        prepare(args.job, args.out)
        print(json.dumps(dict(status="REGISTERED_CPU_ONLY_NOT_RUN", out=str(args.out)), ensure_ascii=False))
        return 0
    require(args.job is None, "run uses only frozen PLAN input")
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())

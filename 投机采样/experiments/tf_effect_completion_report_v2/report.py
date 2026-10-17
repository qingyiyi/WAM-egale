"""CPU-only, source-aware audit of one authorized pending-only DEV completion.

Nothing is launched at import or by this program. Old FAILED/PARTIAL records
remain original evidence. A combined result is a new certificate, never an
upgrade of the predecessor namespace or of the four old false permissions.
"""
from __future__ import annotations

import argparse
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

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
RT = ROOT / "投机采样/results/runtime"
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXP))
from tf_effect_quality_report_v1 import report as base
from tf_effect_campaign_v1 import protocol as campaign

COMMIT, PERMISSIONS, ARMS = base.COMMIT, base.PERMISSIONS, base.ARMS
UNKNOWN = "UNKNOWN"
CAP = 128 * 2**20
REPORT_BY = "2026-10-17T09:55:00+08:00"
GPU_BY = "2026-10-17T09:40:00+08:00"
USER_BY = "2026-10-17T10:00:00+08:00"
CONT_SCHEMA = "tf-effect-pending-only-continuation/v2"
CONT_STATUS = "COMPLETE_PENDING_EPISODES_ONLY"
read, sha, require = base.read, base.sha, base.require


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= CAP, "individual report exceeds128MiB")
    with Path(path).open("xb") as stream:
        stream.write(raw)


def cpu_only():
    require(os.getenv("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) <= 2,
        "CPU-only report requires hidden CUDA and at most two pinned cores")


def order(p):
    return [r for arm in ARMS for r in p["cases"] if r["arm"] == arm]


def frozen_inputs(job):
    """Use the unchanged source validator; continuation keeps original96 cases."""
    job = Path(job).resolve(strict=True)
    p = campaign.load(job)
    require(p["commit"] == COMMIT and p["kind"] == "DEV" and p["collect"] is False
        and p["arms"] == list(ARMS) and p["cases"] == campaign.cases(False)
        and p["permissions"] == PERMISSIONS and p["performance_claim"] is False,
        "original fixed96 DEV contract required")
    return p


def lineage(new_job):
    new_job = Path(new_job).resolve(strict=True)
    from tf_effect_campaign_continue_v2 import protocol as continuation
    new = continuation.load(new_job)
    require(new == frozen_inputs(new_job), "extended continuation and original compatibility validators disagree")
    require(new.get("continuation_schema") == CONT_SCHEMA, "not the authorized pending-only continuation")
    old_job = Path(new["predecessor_job"]).resolve(strict=True)
    require(old_job != new_job and old_job.parent == new_job.parent == RT, "two distinct local runtime sources required")
    old = frozen_inputs(old_job)
    require(old.get("continuation_schema") is None, "only one original plus one bounded completion supported")
    for key in ("cases", "arms", "seed", "horizon", "mode", "selector", "threshold", "bank", "selected", "reference_plan", "GPU"):
        require(new[key] == old[key], "scientific contract changed: " + key)
    for file, digest in old["source_sha256"].items():
        require(new["source_sha256"].get(file) == digest, "original production source removed/changed: " + file)
    manifest = Path(new["imported_terminal_manifest"]).resolve(strict=True)
    require(manifest == new_job / "IMPORTED-TERMINALS.json"
        and new["evidence_sha256"].get(str(manifest)) == sha(manifest), "imported terminal manifest is not frozen")
    m = read(manifest)
    require(m["schema"] == "tf-effect-imported-robot-terminals/v2" and isinstance(m["entries"], list),
        "imported robot terminal schema differs")
    budget = new["cumulative_budget"]
    require(budget["original_target_forward_cap"] == old["job_target_forward_cap"]
        and budget["new_target_forward_cap"] == new["job_target_forward_cap"]
        and budget["new_target_forward_cap"] == budget["original_target_forward_cap"] - budget["predecessor_target_forwards_upper_debited"]
        and 0 <= budget["predecessor_target_forwards_lower_known"] <= budget["predecessor_target_forwards_upper_debited"]
        and budget["original_storage_cap_bytes"] == old["storage_cap_bytes"]
        and budget["new_storage_cap_bytes"] == new["storage_cap_bytes"]
        and budget["new_storage_cap_bytes"] == budget["original_storage_cap_bytes"] - budget["predecessor_storage_bytes_actual"]
        and budget["no_budget_reset"] is True, "conservative cumulative resource budget differs")
    require(len(m["entries"]) + len(new["pending_keys"]) == 96
        and m["predecessor_job"] == str(old_job) and m["pending_keys"] == new["pending_keys"],
        "import manifest source/pending counts differ")
    require(datetime.fromisoformat(new["absolute_cleanup_deadline"]) <= datetime.fromisoformat(GPU_BY)
        and datetime.fromisoformat(budget["new_absolute_cleanup_deadline"]) == datetime.fromisoformat(new["absolute_cleanup_deadline"])
        and datetime.fromisoformat(budget["new_absolute_work_stop"]) == datetime.fromisoformat(new["absolute_work_stop"])
        and datetime.fromisoformat(budget["report_by"]) <= datetime.fromisoformat(REPORT_BY)
        and datetime.fromisoformat(budget["user_deadline"]) <= datetime.fromisoformat(USER_BY),
        "authorized completion/report deadlines differ")
    return old_job, old, new_job, new, manifest, m


def prepare(job, out):
    birth = time.time()
    cpu_only()
    old_job, old, new_job, new, manifest, _ = lineage(job)
    out = Path(out).resolve()
    require(out.parent == RT and out not in (old_job, new_job) and not out.exists(), "new independent report namespace required")
    hard = min(datetime.fromisoformat(REPORT_BY).timestamp(), datetime.fromisoformat(new["cumulative_budget"]["report_by"]).timestamp())
    deadline = min(birth + 900, hard)
    require(time.time() < deadline, "authorized CPU reporting window already exhausted")
    p = dict(schema="tf-effect-completion-report-registration/v2", source_commit=COMMIT,
        goal="Audit original fixedDEV96 plus only its authorized missing robot episodes without changing old terminal",
        original_job=str(old_job), completion_job=str(new_job), imported_terminal_manifest=str(manifest),
        input_sha256={str(root / name): sha(root / name) for root in (old_job, new_job)
            for name in ("plan.json", "PREREGISTRATION.json")},
        manifest_sha256=sha(manifest),
        source_sha256={str(OWN / "report.py"): sha(OWN / "report.py"), str(Path(base.__file__).resolve()): sha(base.__file__)},
        birth_epoch=birth, deadline_epoch=deadline, wall_clock_cap_seconds=deadline-birth,
        requested_cap_seconds=900, hard_report_by=REPORT_BY, requested_GPU=False, CPU_cores_max=2,
        storage_cap_bytes=CAP, no_collection=True, no_training=True, no_retry=True, no_TEST=True,
        cleanup="single CPU process, no child or CUDA; exit without signals or source-job writes",
        terminal_status="REGISTERED_NOT_STARTED", evidence_path=str(out), permissions=PERMISSIONS)
    out.mkdir(exist_ok=False)
    write(out / "PREREGISTRATION.json", p)
    write(out / "PLAN.json", dict(p, preregistration_sha256=sha(out / "PREREGISTRATION.json")))
    return p


def robot_terminal(job, case):
    """Freeze any complete robot outcome before classifying pending work.

    COMMAND-MAPPING is written before episodes.jsonl in the frozen client.
    Budget interruption in that gap must not authorize replacing a full failure.
    """
    path = job / "episodes" / case["key"] / "client/episodes.jsonl"
    canonical_path, record_parse_error = path, None
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    except json.JSONDecodeError as exc:
        # Preserve a completed COMMAND-MAPPING/FINISH outcome. The unchanged
        # strict episode auditor still reads this corrupt original file and
        # must reject it as an audit gap, never as an eligible replacement.
        records, record_parse_error = [], repr(exc)
    terminal = [r for r in records if r.get("status") == "completed" and (r.get("success") is True or r.get("horizon_reached") is True)]
    require(len(terminal) <= 1, "duplicate robot terminal: " + case["key"])
    kind, attempt, elapsed = "episodes_jsonl", UNKNOWN, UNKNOWN
    if terminal:
        require(len(records) == 1, "terminal mixed with additional attempts in same source case")
        r = terminal[0]
        require(r.get("benchmark") == "libero_10" and r.get("task_index") == case["task"]
            and r.get("episode_index") == case["init"] and r.get("libero_horizon") == 800
            and r.get("max_policy_chunks") is None and r.get("truncated_after_max_policy_chunks") is False,
            "original full robot outcome contract differs")
        attempt, elapsed = r["episode_attempt_id"], float(r["elapsed_sec"])
    else:
        mapping_path = path.parent / "COMMAND-MAPPING.json"
        if mapping_path.exists():
            r = read(mapping_path)
            if r.get("complete_episode") is not True:
                return None
            require(r.get("schema") == "physical-action-to-environment-command/v1"
                and len(r["submitted"]) == len(r["environment"]) == r["env_steps"]
                and r["termination_reason"] == ("success" if r["success"] else "full_horizon"),
                "complete command mapping terminal is inconsistent")
            contexts = [x["context"] for x in r["chunks"]]
            require(contexts and all(x.get("campaign_id") == job.name
                and x["task_index"] == case["task"] and x["episode_index"] == case["init"]
                and x["parallel_worker_id"] == case["key"] for x in contexts), "terminal mapping identity differs")
            ids = {x["episode_attempt_id"] for x in contexts}
            require(len(ids) == 1, "terminal mapping mixes episode attempts")
            attempt = ids.pop()
            r = dict(r, horizon_reached=not r["success"])
            path, kind = mapping_path, "command_mapping"
        else:
            finish_paths = [path.parent.parent / n for n in ("FINISH_SERVER.json", "FINISH_READY.json")]
            finish = next((f for f in finish_paths if f.exists()), None)
            if finish is None:
                return None
            r = read(finish)
            require(r["status"] == "PASS" and r["full_episode_terminal"] is True
                and r["key"] == case["key"] and r["arm"] == case["arm"] and r["permissions"] == PERMISSIONS,
                "retained finish terminal differs")
            r = dict(r, horizon_reached=not r["success"])
            path, kind = finish, "finish"
            counter = finish.parent / "COUNTER.json"
            if counter.exists():
                c = read(counter)
                ids = {x["trace_context"]["episode_attempt_id"] for x in c["requests"]}
                require(len(ids) == 1, "finish source counter mixes episode identities")
                attempt = ids.pop()
    require(type(r["success"]) is bool and type(r["env_steps"]) is int and 0 < r["env_steps"] <= 800,
        "complete robot terminal outcome lacks actual success/steps")
    return dict(arm=case["arm"], role=case["role"], task=case["task"], init=case["init"], key=case["key"],
        success=r["success"], horizon_reached=r["horizon_reached"], env_steps=r["env_steps"],
        failure800_penalty=r["env_steps"] if r["success"] else 800,
        termination_reason="success" if r["success"] else "full_horizon", episode_attempt_id=attempt,
        source_job=str(job), episode_record=str(path), episode_record_sha256=sha(path),
        terminal_record_kind=kind, complete_episode_wall_sec=elapsed, robot_terminal=True, necessary_audit_complete=False,
        canonical_episode_parse_error=record_parse_error,
        corrupt_canonical_episode_sha256=sha(canonical_path) if record_parse_error is not None else None,
        performance_claim=False)


def small_proofs(job, names):
    return {str(job / n): sha(job / n) for n in names if (job / n).is_file()}


def pin_exited(pin):
    """Read a lifetime only; never signal a source process or a reused PID."""
    try:
        fields = Path(f"/proc/{pin['pid']}/stat").read_text().rpartition(") ")[2].split()
        return int(fields[19]) != pin["start_ticks"] or fields[0] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        return True


def verify_terminal(job, p, *, completion, expected_keys):
    """A predecessor budget stop is retained as a stop, never renamed COMPLETE."""
    names = ("RUN-STARTED.json", "ROOT-LAUNCH.json", "CLOCK-GUARD.json", "CLOCK-RESULT.json",
        "SUPERVISOR-RESULT.json", "FINAL-OWNED-CLEANUP.json")
    docs = {n: read(job / n) for n in names}
    started, launch, guard, clock, sup, clean = [docs[n] for n in names]
    require(clean["all_roots_reaped"] is True and sup["cleanup"] == clean
        and clock["all_observed_owned_live_processes_exited"] is True and clock["error"] is None
        and clock["no_foreign_signals"] is True
        and all(x["observation"]["live"] is False for x in clock["process_observations"]), "source cleanup not proved")
    pin = guard["supervisor_pin"]
    expected_run = EXP / ("tf_effect_campaign_continue_v2/run.py" if completion else "tf_effect_campaign_v1/run.py")
    require(pin["pid"] == started["pid"] and launch["start_new_session"] is True
        and launch["command"] == launch["pin"]["argv"]
        and all(pin[k] == launch["pin"][k] for k in ("pid", "start_ticks", "uid", "pgid", "argv"))
        and str(job) in pin["argv"] and str(expected_run) in pin["argv"] and guard["clock"] == clock["clock"]
        and guard["clock"]["plan_sha256"] == sha(job / "plan.json")
        and guard["clock"]["max_seconds"] == p["max_run_wall_sec"]
        and guard["clock"]["absolute_cleanup_deadline"] == p["absolute_cleanup_deadline"]
        and guard["no_budget_reset"] is True and guard["no_workload_launched"] is True,
        "source actual root/clock/plan binding differs")
    clock_launch = read(job / "CLOCK-LAUNCH.json")
    require(pin_exited(pin) and pin_exited(clock_launch["pin"]), "source supervisor or clock lifetime still live")
    c = guard["clock"]
    require(c["cleanup_reserve_seconds"] == p["cleanup_reserve_sec"]
        and c["hard_stop_monotonic"] <= c["process_birth_monotonic"] + p["max_run_wall_sec"]
        and c["work_stop_monotonic"] <= c["hard_stop_monotonic"] - p["cleanup_reserve_sec"]
        and clock["elapsed_quality_process_seconds"] <= c["hard_stop_monotonic"] - c["process_birth_monotonic"] + 2,
        "source actual cumulative clock cap differs")
    require(sup["permissions"] == PERMISSIONS and sup["performance_claim"] is False, "source permission changed")
    require(type(sup["total_target_forwards"]) is int and 0 <= sup["total_target_forwards"] <= p["job_target_forward_cap"],
        "source completed forward ledger exceeds its own cap")
    completed = sup["completed"]
    require(isinstance(completed, list) and len(completed) == len(set(completed))
        and completed == expected_keys[:len(completed)] and sup["full_episodes"] == len(completed),
        "source completed ledger is not its registered ordered prefix")
    failed_markers = base.markers(job)
    require(sum(read(job / "episodes" / key / "COUNTER.json")["target_forwards"] for key in completed) == sup["total_target_forwards"],
        "source supervisor forward sum differs from its own complete episode ledger")
    if completion and sup["status"] == CONT_STATUS:
        col = read(job / "COLLECTION.json")
        require(sup["failure"] is None and completed == expected_keys
            and col["status"] == CONT_STATUS and col["continuation_only"] is True
            and col["completed"] == completed and col["full_episodes"] == len(expected_keys)
            and col["total_target_forwards"] == sup["total_target_forwards"] <= p["job_target_forward_cap"]
            and clock["external_stop_reason"] is None and not clock["signals"]
            and clock["status"] == "CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS" and not failed_markers,
            "new M-episode completion is not clean and complete")
        classification = "COMPLETE_PENDING_EPISODES_ONLY"
    else:
        require(sup["status"] == "FAILED_OR_PARTIAL" and isinstance(sup["failure"], dict)
            and (job / "FAILED.json").is_file(), "expected retained budget PARTIAL/FAILED terminal")
        error = str(sup["failure"].get("error", ""))
        reason = clock["external_stop_reason"]
        expected_reason = "registered pending-only continuation09:35 work deadline" if completion else "registered complete TF campaign deadline"
        direct = error in ("ValueError('registered work deadline')", 'ValueError("registered work deadline")',
            "RuntimeError('registered work deadline')", 'RuntimeError("registered work deadline")')
        owned = error in ("SystemExit('owned stop')", 'SystemExit("owned stop")')
        requested_stop = error in ("ValueError('STOP_REQUESTED')", 'ValueError("STOP_REQUESTED")')
        budget_reason = reason == expected_reason
        clock_stop = read(job / "CLOCK-STOP.json") if (job / "CLOCK-STOP.json").is_file() else None
        exact_stop_receipt = (clock_stop is not None and clock_stop["reason"] == expected_reason
            and clock_stop["clock"] == c
            and datetime.fromisoformat(clock_stop["at"]) >= datetime.fromisoformat(p["absolute_work_stop"]))
        if reason is not None:
            require(budget_reason and exact_stop_receipt,
                "source clock stop is not the exact registered budget deadline")
        require((direct or (owned and budget_reason) or (requested_stop and exact_stop_receipt))
            and clock["elapsed_quality_process_seconds"] >= c["work_stop_monotonic"] - c["process_birth_monotonic"] - 3,
            "non-budget failure cannot be automatically merged as budget completion")
        require(set(failed_markers) <= {"FAILED.json", "STOP_REQUESTED", "CLOCK-STOP.json"},
            "additional non-budget FAILED markers in source")
        classification = "RETAINED_BUDGET_PARTIAL_FAILED_NOT_UPGRADED"
    return dict(source_job=str(job), status=sup["status"], classification=classification,
        completed_ledger=completed, full_episodes=len(completed), cleanup_passed=True,
        original_markers=failed_markers, supervisor_wall_sec=sup.get("elapsed_sec", UNKNOWN),
        observed_process_wall_sec=clock["elapsed_quality_process_seconds"],
        completed_ledger_target_forwards=sup.get("total_target_forwards", UNKNOWN),
        all_attempt_target_forwards=UNKNOWN, actual_CUDA_active_time_sec=UNKNOWN,
        clock=clock["clock"], proof_sha256=small_proofs(job, names + ("FAILED.json", "COLLECTION.json", "CLOCK-LAUNCH.json")))


def audit_arm(job, p, arm):
    """Verify the actual source arm, not READY/RNG borrowed from another job."""
    import torch
    r = read(job / f"{arm}-READY.json")
    require(r["status"] == "READY" and r["arm"] == arm and r["collect"] is False
        and r["collector_installed"] is False and r["setup_RNG_restored"] is True
        and r["target_RNG_captured_before_draft_and_CAL"] is True
        and r["ready_rng"] == r["target_initial_rng"] and r["permissions"] == PERMISSIONS
        and r["bank_explicitly_enabled"] is True and r["synthetic_queries"] == 0
        and read(job / "arms" / arm / "OWNED-CLEANUP.json")["all_roots_reaped"] is True,
        "actual source arm ready/cleanup contract differs")
    rng_path = job / "arms" / arm / "TARGET-INITIAL-RNG.pt"
    rng = torch.load(rng_path, map_location="cpu", weights_only=True)
    require(set(rng) == {"CPU", "CUDA"} and all(base.rng_fingerprint(rng[k]) == r["target_initial_rng"][label]
        for k, label in (("CPU", "cpu"), ("CUDA", "cuda"))), "actual source RNG bytes/shape/dtype not bound to READY")
    cal_proofs = []
    if arm == "T":
        require(r["predictor_checkpoint"] is None and r["O1_O2_O3"] is False
            and r["full_real_CAL_rows"] == r["new_real_CAL_rows"] == r["reused_real_CAL_rows"] == 0,
            "target-only source unexpectedly uses a draft")
        cal = dict(executed_old_CAL_rows=0, executed_new_CAL_rows=0, reused_old_CAL_rows=0, elapsed_sec=0)
    elif arm == "TF_CONT":
        proof = job / "arms" / arm / "CAL-REUSE.json"
        reuse = read(proof)
        original = read(reuse["source"])
        require(reuse["status"] == "REUSE_EXISTING_COMPLETE_GPU_CAL_NOT_NEW_CAL_EXECUTION"
            and reuse["reference_rows"] == 20237 and reuse["new_CAL_rows"] == 0
            and reuse["model_checkpoint_unchanged"] is True and reuse["draft_adapter_source_unchanged"] is True
            and reuse["target_RNG_unchanged"] is True and sha(reuse["source"]) == reuse["source_sha256"]
            and original["status"] == "PASS_FULL_REAL_CAL_SAME_DEVICE_ADAPTER_PARITY"
            and original["rows"] == 20237 and original["complete_CAL"] is True
            and all(original[k] is True for k in ("same_device_bitwise_exact", "finite", "CPU_rng_unchanged", "CUDA_rng_unchanged"))
            and original["checkpoint_sha256"] == p["selected"][arm]["checkpoint_sha256"]
            and r["reused_real_CAL_rows"] == 20237 and r["full_real_CAL_rows"] == r["new_real_CAL_rows"] == 0,
            "actual unchanged TF-cont GPU CAL reuse proof differs")
        cal_proofs = [proof, Path(reuse["source"])]
        cal = dict(executed_old_CAL_rows=0, executed_new_CAL_rows=0, reused_old_CAL_rows=20237,
            elapsed_sec=0, reuse_reference=str(reuse["source"]))
    else:
        cal_job = Path(p["gpu_cal_jobs"][arm]).resolve(strict=True)
        proof = cal_job / "RESULT.json"
        report = read(proof)
        cp, prep = read(cal_job / "PLAN.json"), read(cal_job / "PREPARED.json")
        require(prep["plan_sha256"] == sha(cal_job / "PLAN.json") and cp["condition"] == arm
            and cp["checkpoint"] == p["selected"][arm]["checkpoint"]
            and cp["checkpoint_sha256"] == p["selected"][arm]["checkpoint_sha256"]
            and report["status"] == "PASS_FULL_OLD_AND_NEW_REAL_CAL_SAME_DEVICE_PARITY"
            and report["condition"] == arm and report["checkpoint_sha256"] == p["selected"][arm]["checkpoint_sha256"]
            and report["full_old_and_new_CAL"] is True and report["CPU_rng_unchanged"] is True
            and report["CUDA_rng_unchanged"] is True and report["synthetic_queries"] == report["target_forwards"] == 0
            and report["sets"]["old_CAL"]["rows"] == r["full_real_CAL_rows"] == 20237
            and report["sets"]["new_CAL"]["rows"] == r["new_real_CAL_rows"] == cp["new_CAL_rows"] > 0
            and r["reused_real_CAL_rows"] == 0
            and all(v["complete_CAL"] and v["finite"] and v["same_device_bitwise_exact"] for v in report["sets"].values())
            and not base.markers(cal_job), "source new candidate own full old/new GPU CAL differs")
        cal_proofs = [proof, cal_job / "PLAN.json", cal_job / "PREPARED.json"]
        cal = dict(executed_old_CAL_rows=20237, executed_new_CAL_rows=cp["new_CAL_rows"],
            reused_old_CAL_rows=0, elapsed_sec=report["elapsed_seconds"])
    if arm != "T":
        require(r["O1_O2_O3"] is True and r["predictor_checkpoint"] == p["selected"][arm]["checkpoint"],
            "actual source draft/O3 differs")
    require(r["CAL_executed_query_rows"] == cal["executed_old_CAL_rows"] + cal["executed_new_CAL_rows"],
        "source READY CAL executed/reused counts disagree")
    proofs = [job / f"{arm}-READY.json", rng_path, job / "arms" / arm / "OWNED-CLEANUP.json"] + cal_proofs
    return dict(ready=r, rng=rng, source_job=str(job), arm=arm, CAL=cal,
        proof_sha256={str(x): sha(x) for x in proofs})


def audit_case(job, p, case, arm):
    """The unchanged original helper reads real source paths and local READY."""
    import torch
    r = base.episode(job, case, arm["ready"], p)
    out = job / "episodes" / case["key"]
    counter, mapping = read(out / "COUNTER.json"), read(out / "client/COMMAND-MAPPING.json")
    contexts = [x["trace_context"] for x in counter["requests"]] + [x["context"] for x in counter["acceptance"]]
    contexts += [x["context"] for x in mapping["chunks"]]
    require(all(x.get("campaign_id") == job.name for x in contexts), "case context borrowed from another source job")
    for path in (out / "server_output").rglob("actions_*.pt"):
        value = torch.load(path, map_location="cpu", weights_only=True)
        require(value.dtype == torch.bfloat16 and tuple(value.shape) == (1, 30, 4, 4, 1),
            "actual production action dtype/geometry differs")
    names = ("COUNTER.json", "FINISH_SERVER.json", "FINISH_READY.json", "CLIENT-CLEANUP.json", "FINAL-SERVER-STATE.json",
        "RESET.json", "TEACHER-INDEX.json", "client/episodes.jsonl", "client/repro_metadata.json", "client/COMMAND-MAPPING.json")
    return dict(r, source_job=str(job), source_plan=str(job / "plan.json"), source_plan_sha256=sha(job / "plan.json"),
        source_arm_proof_sha256=arm["proof_sha256"], source_episode_proof_sha256=small_proofs(out, names),
        necessary_audit_complete=True, normalized_action_dtype="torch.bfloat16", draft_internal_dtype="torch.float32")


def paired_initial(base_job, base_row, new_job, new_row, rng_equal):
    """Same arithmetic as original pairing, with two explicit source roots."""
    import numpy as np
    bf = base_job / "episodes" / base_row["key"] / "client"
    nf = new_job / "episodes" / new_row["key"] / "client"
    x, y = dict(base_row["initial_pairing"]), dict(new_row["initial_pairing"])
    x.pop("initial_obs_hashes"); y.pop("initial_obs_hashes")
    state_equal = base.exact_arrays(base.arrays(bf / "initial_state.npz"), base.arrays(nf / "initial_state.npz"))
    reset_equal = read(bf.parent / "RESET.json")["state"] == read(nf.parent / "RESET.json")["state"]
    aa, bb = base.arrays(bf / "initial_observations.npz"), base.arrays(nf / "initial_observations.npz")
    cameras = {}
    for name in base.CAMERAS:
        d = np.abs(aa[name].astype(np.int16)-bb[name].astype(np.int16))
        cameras[name] = dict(exact=aa[name].dtype == bb[name].dtype and aa[name].shape == bb[name].shape
            and aa[name].tobytes() == bb[name].tobytes(), changed_channels=int(np.count_nonzero(d)),
            changed_pixels=int(np.count_nonzero(np.any(d != 0, axis=2))), max_abs=int(d.max()))
    physical = x == y and state_equal and reset_equal and rng_equal
    return dict(role=new_row["role"], task=new_row["task"], init=new_row["init"], reference="T", candidate=new_row["arm"],
        reference_key=base_row["key"], candidate_key=new_row["key"], reference_source_job=str(base_job), candidate_source_job=str(new_job),
        initial_metadata_without_cameras_equal=x == y, actual_sim_controller_arrays_exact=state_equal,
        actual_target_initial_RNG_equal=rng_equal, full_reset_state_equal=reset_equal,
        physical_RNG_equal=physical, camera_pixels=cameras, all_exact=physical and all(v["exact"] for v in cameras.values()),
        scope="actual saved arrays/RNG/reset/cameras; unsaved simulator internals UNKNOWN")


def outcome_summary(raw, audited, expected):
    return dict(planned_episodes=expected, robot_terminal_episodes=len(raw),
        successes=sum(r["success"] for r in raw),
        failures=[{k: r[k] for k in ("key", "task", "init", "source_job", "env_steps", "necessary_audit_complete")} for r in raw if not r["success"]],
        actual_env_steps=base.dist([r["env_steps"] for r in raw]),
        failure800_penalty=base.dist([r["failure800_penalty"] for r in raw]),
        robot_terminal_wall_sec=base.dist([r["complete_episode_wall_sec"] for r in raw if isinstance(r["complete_episode_wall_sec"], (int, float))]),
        robot_terminal_wall_UNKNOWN=sum(r["complete_episode_wall_sec"] == UNKNOWN for r in raw),
        success_rate_descriptive_terminal_subset=sum(r["success"] for r in raw)/len(raw) if raw else None,
        audit_complete_episodes=len(audited), audited_metrics=base.summarize(audited, expected),
        scope="all robot-terminal outcomes retained; incomplete/gapped audit never gives quality permission")


def all_CAL_work(job, p):
    """Include CAL attempts even when no robot episode followed that setup."""
    rows = []
    for arm in ARMS:
        if arm == "TF_CONT":
            proof = job / "arms" / arm / "CAL-REUSE.json"
            if proof.exists():
                reuse = read(proof)
                rows.append(dict(source_job=str(job), arm=arm, state=reuse["status"],
                    executed_old_CAL_rows=0, executed_new_CAL_rows=0,
                    reused_old_CAL_rows=reuse["reference_rows"], CAL_call_wall_sec=0,
                    reuse_loader_wall_sec=UNKNOWN, proof_sha256={str(proof): sha(proof)}))
            continue
        if arm not in p["gpu_cal_jobs"]:
            continue
        root = Path(p["gpu_cal_jobs"][arm])
        if not root.exists():
            continue
        proof = root / "RESULT.json"
        if proof.exists():
            cal = read(proof)
            rows.append(dict(source_job=str(job), arm=arm, state=cal["status"],
                executed_old_CAL_rows=cal["sets"]["old_CAL"]["rows"],
                executed_new_CAL_rows=cal["sets"]["new_CAL"]["rows"], reused_old_CAL_rows=0,
                CAL_call_wall_sec=cal["elapsed_seconds"], actual_CUDA_active_time_sec=UNKNOWN,
                CPU_prepare_and_model_setup_in_total_source_wall=True,
                proof_sha256=small_proofs(root, ("RESULT.json", "PLAN.json", "PREREGISTRATION.json", "STARTED.json", "FAILED.json"))))
        else:
            started = (root / "STARTED.json").exists()
            fail = read(root / "FAILED.json") if (root / "FAILED.json").exists() else {}
            rows.append(dict(source_job=str(job), arm=arm,
                state="INTERRUPTED_OR_FAILED_CAL" if started else "PREPARED_NO_GPU_CAL_START_RECEIPT",
                executed_old_CAL_rows=UNKNOWN if started else 0, executed_new_CAL_rows=UNKNOWN if started else 0,
                reused_old_CAL_rows=0, CAL_call_wall_sec=fail.get("elapsed_seconds", UNKNOWN),
                actual_CUDA_active_time_sec=UNKNOWN, CPU_prepare_and_model_setup_in_total_source_wall=True,
                missing_terminal_is_not_zero_work=started,
                proof_sha256=small_proofs(root, ("PLAN.json", "PREREGISTRATION.json", "STARTED.json", "FAILED.json"))))
    return rows


def analyze(job):
    import torch
    old_job, old, new_job, new, manifest_path, imported = lineage(job)
    original_cases = order(old)
    cases = {r["key"]: r for r in original_cases}
    old_markers, new_markers = base.markers(old_job), base.markers(new_job)
    issues, terminals = [], []
    old_raw = {r["key"]: value for r in original_cases if (value := robot_terminal(old_job, r)) is not None}
    imported_by = {e["key"]: e for e in imported["entries"]}
    require(len(imported_by) == len(imported["entries"]) and set(imported_by) == set(old_raw), "import manifest omitted/duplicated a full original robot outcome")
    require([e["key"] for e in imported["entries"]] == [r["key"] for r in original_cases if r["key"] in old_raw], "imported outcome order differs")
    for key, entry in imported_by.items():
        raw = old_raw[key]
        require(entry["case"] == cases[key] and Path(entry["source_job"]).resolve() == old_job
            and entry["quality_terminal"] is True and Path(entry["episode_record"]).resolve() == Path(raw["episode_record"])
            and entry["episode_record_sha256"] == raw["episode_record_sha256"]
            and entry.get("terminal_record_kind", "episodes_jsonl") == raw["terminal_record_kind"]
            and entry["success"] == raw["success"] and entry["env_steps"] == raw["env_steps"]
            and entry["audit_status"] in ("COMPLETE_AUDIT", "TERMINAL_QUALITY_WITH_AUDIT_GAP"),
            "imported full robot outcome differs from original raw record")
    pending = [r["key"] for r in original_cases if r["key"] not in old_raw]
    require(new["pending_keys"] == pending and len(pending) == len(set(pending)) > 0,
        "pending set is not exactly original cases without robot terminal; completed failures cannot rerun")
    budget = new["cumulative_budget"]
    debits = budget["unclosed_attempt_debits"]
    debits_by = {d["key"]: d for d in debits}
    require(len(debits_by) == len(debits), "duplicate predecessor unknown-attempt debit")
    known_counters, known_sum = [], 0
    for case in original_cases:
        path = old_job / "episodes" / case["key"] / "COUNTER.json"
        if not path.exists():
            continue
        try:
            count = read(path)
        except json.JSONDecodeError as exc:
            debit = debits_by.get(case["key"])
            require(debit is not None and Path(debit["source_job"]).resolve() == old_job
                and debit["counter_missing"] is True and debit["lower_known"] == 0
                and debit["upper_debited"] == old["per_episode_target_forward_cap"]
                and debit["counter_parse_error"] == repr(exc)
                and debit["quality_terminal"] is (case["key"] in old_raw),
                "invalid JSON predecessor counter lacks its exact conservative debit/parse-error proof")
            if case["key"] in imported_by:
                entry = imported_by[case["key"]]
                require(entry["audit_status"] == "TERMINAL_QUALITY_WITH_AUDIT_GAP"
                    and "COUNTER.json invalid JSON: " + repr(exc) in entry["missing_receipts"],
                    "imported full robot outcome omitted its invalid counter audit gap")
            issues.append(dict(scope="predecessor_counter", key=case["key"], source_job=str(old_job),
                error=repr(exc), counter_sha256=sha(path), actual_target_forwards=UNKNOWN,
                conservative_upper_debit=debit["upper_debited"]))
            continue
        require(count["row"] == case and count["key"] == case["key"] and count["arm"] == case["arm"]
            and type(count["target_forwards"]) is int
            and 0 < count["target_forwards"] == count["returned_forwards"] <= old["per_episode_target_forward_cap"],
            "available predecessor forward counter differs")
        known_sum += count["target_forwards"]
        known_counters.append(case["key"])
    require(len({d["key"] for d in debits}) == len(debits)
        and all(d["key"] in cases and d["key"] not in known_counters and d["counter_missing"] is True
            and d["lower_known"] == 0 and d["upper_debited"] == old["per_episode_target_forward_cap"] for d in debits)
        and known_sum == budget["predecessor_target_forwards_lower_known"]
        and known_sum + sum(d["upper_debited"] for d in debits) == budget["predecessor_target_forwards_upper_debited"],
        "predecessor conservative forward accounting is not based on actual available counters plus bounded missing attempt")
    require(budget["original_created_at"] == old["created_at"]
        and budget["prior_absolute_cleanup_deadline"] == old["absolute_cleanup_deadline"],
        "original process budget origin/deadline was rewritten")
    old_completed = [r["key"] for r in original_cases]
    for source_job, p, is_new, expected in ((old_job, old, False, old_completed), (new_job, new, True, pending)):
        try:
            terminals.append(verify_terminal(source_job, p, completion=is_new, expected_keys=expected))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            issues.append(dict(scope="source_terminal", source_job=str(source_job), error=repr(exc)))
    predecessor_receipt = next((x for x in terminals if x["source_job"] == str(old_job)), None)
    if predecessor_receipt is not None:
        require(budget["predecessor_supervisor_wall_sec"] == predecessor_receipt["supervisor_wall_sec"]
            and budget["predecessor_clock_wall_sec"] == predecessor_receipt["observed_process_wall_sec"],
            "predecessor setup/CAL/cleanup wall consumption differs from cumulative ledger")
    new_raw = {r["key"]: value for r in original_cases if (value := robot_terminal(new_job, r)) is not None}
    require(set(new_raw) <= set(pending) and not (set(new_raw) & set(old_raw)), "completion contains replacement of an old full robot outcome")
    source_rows = {**old_raw, **new_raw}
    require(len(source_rows) == len(old_raw) + len(new_raw), "duplicate logical case outcomes")
    try:
        training = base.training_audit(old)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        training = None
        issues.append(dict(scope="fixed_training", error=repr(exc)))
    arms, audited, audit_gaps = {}, [], []
    for case in original_cases:
        key = case["key"]
        if key not in source_rows:
            continue
        source_job = old_job if key in old_raw else new_job
        p = old if source_job == old_job else new
        arm_key = (str(source_job), case["arm"])
        try:
            if arm_key not in arms:
                arms[arm_key] = audit_arm(source_job, p, case["arm"])
            row = audit_case(source_job, p, case, arms[arm_key])
            require(row["success"] == source_rows[key]["success"] and row["env_steps"] == source_rows[key]["env_steps"], "audited outcome differs from frozen robot terminal")
            audited.append(row)
            source_rows[key]["necessary_audit_complete"] = True
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            audit_gaps.append(dict(key=key, source_job=str(source_job), status="TERMINAL_QUALITY_WITH_AUDIT_GAP",
                success=source_rows[key]["success"], env_steps=source_rows[key]["env_steps"], error=repr(exc)))
    reference_arm = arms.get((str(old_job), "T")) or arms.get((str(new_job), "T"))
    same_arm_proofs = []
    for (source, arm), value in arms.items():
        if reference_arm is None:
            issues.append(dict(scope="actual_initial_RNG", error="no audited target initialization reference"))
            continue
        exact = all(base.exact_value(value["rng"][k], reference_arm["rng"][k]) for k in ("CPU", "CUDA"))
        same_arm_proofs.append(dict(source_job=source, arm=arm, reference_source_job=reference_arm["source_job"], actual_target_initial_RNG_exact=exact))
        if not exact:
            issues.append(dict(scope="actual_initial_RNG", source_job=source, arm=arm, error="actual target initialization differs"))
    pairs = []
    by = {(r["role"], r["task"], r["init"], r["arm"]): r for r in audited}
    for row in audited:
        if row["arm"] == "T":
            continue
        reference = by.get((row["role"], row["task"], row["init"], "T"))
        if reference is None:
            continue
        a = arms[row["source_job"], row["arm"]]
        b = arms[reference["source_job"], "T"]
        exact = all(base.exact_value(a["rng"][k], b["rng"][k]) for k in ("CPU", "CUDA"))
        pairs.append(paired_initial(Path(reference["source_job"]), reference, Path(row["source_job"]), row, exact))
    raw_rows = [source_rows[r["key"]] for r in original_cases if r["key"] in source_rows]
    known_attempts = [r["episode_attempt_id"] for r in raw_rows if r["episode_attempt_id"] != UNKNOWN]
    require(len(set(known_attempts)) == len(known_attempts), "logical cases share an episode-attempt identity")
    fresh = [r for r in raw_rows if r["role"] == "fresh_DEV"]
    risk = [r for r in raw_rows if r["role"] == "risk_DEV"]
    summaries = {"fresh_DEV": {a: outcome_summary([r for r in fresh if r["arm"] == a], [r for r in audited if r["role"] == "fresh_DEV" and r["arm"] == a], 20) for a in ARMS},
        "known_risk_DEV": {a: outcome_summary([r for r in risk if r["arm"] == a], [r for r in audited if r["role"] == "risk_DEV" and r["arm"] == a], 8) for a in ("T", "TF_ON")}}
    comparisons = {f"{a}_vs_{b}": base.paired(fresh, a, b, 20) for a, b in (("TF_ON", "T"), ("TF_ON", "MLP_ON"), ("TF_CONT", "T"), ("TF_ON", "TF_CONT"))}
    risk_comparison = base.paired(risk, "TF_ON", "T", 8)
    complete = len(raw_rows) == len(audited) == 96 and len(pairs) == 68 and not issues and not audit_gaps
    require(base.markers(old_job) == old_markers and base.markers(new_job) == new_markers, "source FAILED/STOP markers changed during audit")
    ratio = comparisons["TF_ON_vs_T"]["both_success_steps_ratio"]
    fs = summaries["fresh_DEV"]
    checks = dict(TF_ON_fresh_success_at_least_T=fs["TF_ON"]["successes"] >= fs["T"]["successes"],
        TF_ON_fresh_success_at_least_matched_MLP=fs["TF_ON"]["successes"] >= fs["MLP_ON"]["successes"],
        nonempty_both_success=ratio["n"] > 0,
        length_P50=ratio["p50"] is not None and ratio["p50"] <= 1.10,
        length_P90=ratio["p90"] is not None and ratio["p90"] <= 1.20,
        length_max=ratio["max"] is not None and ratio["max"] <= 1.30,
        known_risk_no_new_TF_failure=risk_comparison["four_cells"]["reference_1_candidate_0"] == 0)
    numeric = all(checks.values()) if complete else UNKNOWN
    physical = all(r["physical_RNG_equal"] for r in pairs) if complete else UNKNOWN
    pixels = all(all(v["exact"] for v in r["camera_pixels"].values()) for r in pairs) if complete else UNKNOWN
    strict = numeric is True and physical is True and pixels is True if complete else UNKNOWN
    recommendation = ("RECOMMEND_FREEZE_FOR_SEPARATELY_REGISTERED_INDEPENDENT_TEST" if strict is True else
        "STOP_TF_QUALITY_TUNING_RETAIN_EXPLORATORY_ONLY" if complete else "NO_SELECTION_INCOMPLETE_OR_GAPPED_AUDIT")
    source_counts = {str(j): sum(r["target_forwards"] for r in audited if r["source_job"] == str(j)) for j in (old_job, new_job)}
    cal_rows = all_CAL_work(old_job, old) + all_CAL_work(new_job, new)
    return dict(schema="tf-effect-two-namespace-fixedDEV-report/v2", source_commit=COMMIT,
        status="COMPLETE96_ORIGINAL_PLUS_AUTHORIZED_COMPLETION_DEV_AUDIT" if complete else "PARTIAL_OR_GAPPED_FIXED_DEV96_NO_SELECTION",
        original_namespace_upgraded=False, original_job=str(old_job), completion_job=str(new_job),
        logical_fixed_DEV96_complete=complete, complete=complete, planned_full_episodes=96,
        robot_terminal_episodes=len(raw_rows), audited_complete_episodes=len(audited),
        missing_robot_terminal_keys=[r["key"] for r in original_cases if r["key"] not in source_rows],
        original_imported_terminal_keys=list(old_raw), continuation_pending_keys=pending,
        source_terminals=terminals, issues=issues, audit_gaps=audit_gaps,
        original_failure_markers=old_markers, completion_failure_markers=new_markers,
        training_audit=training, same_initial_target_proofs=same_arm_proofs, CAL_work=cal_rows,
        summaries=summaries, comparisons=comparisons, known_risk_TF_ON_vs_T=risk_comparison,
        robot_terminal_outcomes=raw_rows, audited_episode_rows=audited, pairing_comparisons=pairs, pairing_count=len(pairs),
        engineering_numeric_checks=checks if complete else {k: UNKNOWN for k in checks},
        new_DEV_numeric_screen_observed=numeric, physical_RNG_pairing_passed=physical,
        strict_initial_camera_pixels_passed=pixels, strict_current_engineering_screen_observed=strict,
        recommendation=recommendation, permissions=PERMISSIONS, global_default_changed=False,
        independent_TEST_started=False, noninferiority_established=False, formal_performance_permission=False,
        no_120_started=True, TEST_read=False, architecture_superiority_claim=False, performance_claim=False,
        total_audited_selected_target_forwards=sum(source_counts.values()), audited_target_forwards_by_source=source_counts,
        all_attempt_target_forwards=UNKNOWN, predecessor_forward_lower_known=budget["predecessor_target_forwards_lower_known"],
        predecessor_forward_upper_debited=budget["predecessor_target_forwards_upper_debited"],
        interrupted_attempt_upper_debits=budget["unclosed_attempt_debits"],
        sum_legal_speculative_g=sum(r["legal_scheduler_updates_g"] for r in audited if type(r["legal_scheduler_updates_g"]) is int),
        sum_T_round_sec=UNKNOWN, cost_per_legal_commit=UNKNOWN, actual_CUDA_active_time_sec=UNKNOWN,
        cumulative_budget=budget, timing_scope="shared-resource quality diagnostic; all source/attempt/setup/CAL/cleanup wall retained; not formal performance",
        limitations=["Original budget FAILED/PARTIAL remains unchanged; logical union is a separately authorized certificate.",
            "Full original algorithm failures never rerun or disappear when closing audit is incomplete.",
            "Only original fixed96 DEV is combined; fresh20 and known-risk8 stay separate; no oldDEV/TEST is imported.",
            "Selected full-episode forward sum excludes unknown interrupted work; upper debits are not actual measured calls.",
            "Unpaired/gapped records cannot pass engineering gates; initial camera differences are retained.",
            "Small DEV or nonsignificance does not establish3pp noninferiority or Transformer superiority.",
            "No per-round timing or new GPU counters; target-call reduction is not task/E2E speedup."])


def prose(r):
    intro = [f"状态：{r['status']}。", f"机器人完整终态{r['robot_terminal_episodes']}/96，必要审计完整{r['audited_complete_episodes']}/96。",
        "原namespace的FAILED/PARTIAL保留；新结果只描述原固定DEV96的有界补齐，四个原权限始终false。"]
    quality = ["# 固定DEV96：双namespace独立汇总", ""] + intro + ["",
        "|分层|方案|机器人终态/计划|审计齐全|成功|实际步数总和|失败800均值|", "|---|---|---:|---:|---:|---:|---:|"]
    for role, cells in r["summaries"].items():
        for arm, s in cells.items():
            quality.append(f"|{role}|{arm}|{s['robot_terminal_episodes']}/{s['planned_episodes']}|{s['audit_complete_episodes']}|{s['successes']}|{s['actual_env_steps']['total']}|{s['failure800_penalty']['mean']}|")
    quality += ["", "表中保留全部已完整机器人终态，包含原失败和关闭审计缺口；缺口不转成算法失败，也不替换为重跑成功。"]
    for name, row in r["comparisons"].items():
        quality += ["", f"{name}配对：{row['four_cells']}；双方成功步数比：{row['both_success_steps_ratio']}。"]
    quality += ["", f"known-risk8：{r['known_risk_TF_ON_vs_T']['four_cells']}。", f"新工程数值门：{r['engineering_numeric_checks']}。",
        f"物理/RNG：{r['physical_RNG_pairing_passed']}；严格相机：{r['strict_initial_camera_pixels_passed']}。", f"建议：{r['recommendation']}。"]
    facts = ["# 已观察、未知与建议", ""] + intro + ["", f"原证据：{r['original_job']}", f"补齐证据：{r['completion_job']}",
        f"缺机器人终态：{r['missing_robot_terminal_keys']}。", f"审计缺口：{r['audit_gaps']}。", "",
        "UNKNOWN：中断attempt精确forward、CUDA活跃时间、ΣT_round/Σg、独立TEST与非劣性。", f"建议：{r['recommendation']}。",
        "即使20个理想独立配对零新增失败，单侧95%新增失败率上界仍约13.91pp；DEV不能证明3pp非劣。"]
    system = ["# 系统指标边界", "", f"完整已审计局的target forwards：{r['total_audited_selected_target_forwards']}；分来源：{r['audited_target_forwards_by_source']}。",
        f"合法speculative提交g：{r['sum_legal_speculative_g']}。", "ΣT_round与ΣT_round/Σg：UNKNOWN，未采逐round时间。", "",
        "1/3/7 × B14/G1/G2 × cached/uncached、L2/DRAM/CTA/register/shared-memory/occupancy：本质量作业未测，UNKNOWN。",
        "完整episode耗时仅作共享资源诊断；原及补齐的模型setup、CAL、失败中断和cleanup墙钟分别保留，不能报告正式speedup。"]
    claim = ["# Claim与证据", "", "|主张|当前证据|缺口|允许表述|", "|---|---|---|---|",
        f"|本轮固定DEV工程条件|numeric={r['new_DEV_numeric_screen_observed']}；strict={r['strict_current_engineering_screen_observed']}|原报告中的具体gap或独立TEST|仅按实际DEV状态表述|",
        "|原作业通过|原FAILED/PARTIAL完整保留|不能追溯改写|不允许|",
        "|TF优于MLP架构/普遍质量保持|固定seed有限DEV|独立样本/稳定性/正确统计|不允许|",
        "|非劣/正式性能/MegaTransformer协同|四原权限false；无新系统计时|独立TEST及强baseline机制与获准性能|不允许|"]
    stop = ["# 预算、失败与止损", "", f"原失败标记：{r['original_failure_markers']}。", f"补齐失败标记：{r['completion_failure_markers']}。",
        f"两个namespace终态及墙钟：{r['source_terminals']}。", f"GPU CAL实际/复用：{r['CAL_work']}。", "",
        f"原forward已知下界{r['predecessor_forward_lower_known']}，保守扣账上界{r['predecessor_forward_upper_debited']}；中断预留：{r['interrupted_attempt_upper_debits']}。",
        "下界/保守扣账均不能写成所有attempt的精确实测forward。", f"独立累计预算：{r['cumulative_budget']}。", "",
        "无自动重跑、续租、阈值修改、seed/架构搜索或第二环境。完整DEV不合格则按登记停止TF质量调参；不齐则明确缺口，不给PASS。"]
    return {"QUALITY-RESULT.md": quality, "FACTS.md": facts, "SYSTEM-RESULT.md": system,
        "CLAIM-EVIDENCE-MATRIX.md": claim, "STOP-REPORT.md": stop}


def run(out):
    cpu_only()
    out = Path(out).resolve(strict=True)
    p = read(out / "PLAN.json")
    require(p["schema"] == "tf-effect-completion-report-registration/v2" and p["source_commit"] == COMMIT
        and p["permissions"] == PERMISSIONS and sha(out / "PREREGISTRATION.json") == p["preregistration_sha256"]
        and not (out / "RUN-STARTED.json").exists(), "fresh independently registered CPU report only")
    remaining = p["deadline_epoch"]-time.time()
    if remaining <= 0:
        write(out / "FAILED.json", dict(status="EXPIRED_CPU_COMPLETION_REPORT_NOT_RUN", GPU=False, permissions=PERMISSIONS))
        raise TimeoutError("CPU/report09:55 cumulative deadline expired")
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU completion report cumulative deadline")))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    start = time.monotonic()
    write(out / "RUN-STARTED.json", dict(status="CPU_READ_ONLY_TWO_SOURCE_AUDIT", pid=os.getpid(), GPU=False))
    try:
        import torch
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), "no CUDA context allowed")
        for file, digest in {**p["source_sha256"], **p["input_sha256"]}.items():
            require(sha(file) == digest, "registered report source/input changed: " + file)
        require(sha(p["imported_terminal_manifest"]) == p["manifest_sha256"], "import manifest changed")
        result = analyze(Path(p["completion_job"]))
        result["CPU_report_elapsed_sec"] = time.monotonic()-start
        result["CPU_report_seconds_since_registration"] = time.time()-p["birth_epoch"]
        result["report_source_sha256"] = sha(__file__)
        write(out / "RESULT.json", result)
        for name, lines in prose(result).items():
            with (out / name).open("x") as stream:
                stream.write("\n".join(lines)+"\n")
        require(time.time() <= p["deadline_epoch"] and not torch.cuda.is_initialized(), "CPU deadline/CUDA boundary exceeded")
        require(sum(x.stat().st_size for x in out.rglob("*") if x.is_file()) <= CAP, "CPU report storage cap exceeded")
        write(out / "FINAL-RESULT.json", dict(status=result["status"], complete=result["complete"],
            elapsed_sec=time.monotonic()-start, GPU=False, CUDA_initialized=False, children_started=0,
            source_evidence_written=False, own_process_cleanup="single CPU process exits; no source process signalled", permissions=PERMISSIONS))
        print(json.dumps(dict(status=result["status"], complete=result["complete"], out=str(out)), ensure_ascii=False))
        return 0 if result["complete"] else 2
    except BaseException as exc:
        write(out / "FAILED.json", dict(status="FAILED_CPU_COMPLETION_REPORT_NOT_QUALITY_DECISION", error=repr(exc),
            traceback=traceback.format_exc(), GPU=False, elapsed_sec=time.monotonic()-start, source_evidence_written=False, permissions=PERMISSIONS))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--job", type=Path, help="pending-only completion job; prepare only")
    a.add_argument("--out", type=Path, required=True)
    mode = a.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true"); mode.add_argument("--run", action="store_true")
    args = a.parse_args()
    if args.prepare:
        require(args.job is not None, "prepare requires --job")
        prepare(args.job, args.out)
        print(json.dumps(dict(status="REGISTERED_CPU_ONLY_NOT_RUN", out=str(args.out)), ensure_ascii=False))
        return 0
    require(args.job is None, "run uses only the frozen report PLAN")
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU component audit of four already recorded, fixed known-risk episodes.

This entry never runs a policy, simulator, training worker or retry. Its terminal
means that recorded components were inspected, never that a quality gate passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
RT = ROOT / "投机采样/results/runtime"
sys.path.insert(0, str(EXP))
from tf_effect_quality_report_v1 import report as base
from tf_effect_campaign_v1 import protocol as campaign

COMMIT, PERMISSIONS = base.COMMIT, base.PERMISSIONS
SOURCE_JOB = RT / "p0-tf-effect-dev96-v1-20261017-r1"
PAIRS = ((1, 29), (3, 26))
ARMS = ("T", "TF_ON")
HARD = "2026-10-17T08:50:00+08:00"
CAP = 64 * 2**20
UNKNOWN = "UNKNOWN"
GLOBAL_FILES = ("plan.json", "PREREGISTRATION.json", "RUN-STARTED.json", "ROOT-LAUNCH.json",
    "CLOCK-LAUNCH.json", "CLOCK-GUARD.json", "CLOCK-RESULT.json", "SUPERVISOR-RESULT.json",
    "FINAL-OWNED-CLEANUP.json", "FAILED.json", "CLOCK-STOP.json", "STOP_REQUESTED")
require, read = base.require, base.read


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write(out, name, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    used = sum(x.stat().st_size for x in out.iterdir() if x.is_file())
    require(len(raw) + used <= CAP, "64 MiB new diagnostic evidence cap")
    with (out / name).open("xb") as stream:
        stream.write(raw)


def runtime():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hide CUDA explicitly")
    require(len(os.sched_getaffinity(0)) == 1, "pin this diagnostic to exactly one CPU")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        require(os.environ.get(name) == "1", name + " must equal1")
    require(not children(), "diagnostic must have no child processes")


def children():
    found = set()
    for task in Path("/proc/self/task").iterdir():
        try:
            found.update(int(x) for x in (task / "children").read_text().split())
        except FileNotFoundError:
            continue
    return sorted(found)


def deny_spawn(event, args):
    if event in ("subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.posix_spawnp"):
        raise RuntimeError("registered CPU diagnostic forbids children: " + event)


def alarm(deadline):
    remaining = deadline - time.time()
    require(remaining > 0, "registered cumulative/report deadline expired; waiting counts")
    old = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("recorded risk CPU deadline")))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    return old


def restore_alarm(old):
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old)


def fixed_cases(p):
    require(p["commit"] == COMMIT and p["collect"] is False and p["kind"] == "DEV"
        and p["permissions"] == PERMISSIONS and p["performance_claim"] is False
        and p["horizon"] == 800 and len(p["cases"]) == 96, "original fixed DEV contract differs")
    rows = []
    for task, init in PAIRS:
        for arm in ARMS:
            got = [r for r in p["cases"] if r["arm"] == arm and r["task"] == task and r["init"] == init
                and r["role"] == "risk_DEV" and r["rep"] == 0]
            require(len(got) == 1, "missing/duplicated fixed known-risk case")
            rows.append(got[0])
    return rows


def input_inventory(job, cases):
    paths = [job / n for n in GLOBAL_FILES]
    for arm in ARMS:
        paths += [job / f"{arm}-READY.json", job / "arms" / arm / "TARGET-INITIAL-RNG.pt",
            job / "arms" / arm / "OWNED-CLEANUP.json"]
    inventories = {}
    for case in cases:
        folder = job / "episodes" / case["key"]
        require(folder.is_dir(), "recorded episode directory missing")
        files = sorted(x for x in folder.rglob("*") if x.is_file())
        require(files, "empty recorded episode directory")
        inventories[case["key"]] = [str(x.relative_to(folder)) for x in files]
        paths += files
    present, absent = {}, []
    for path in sorted(set(paths)):
        if not path.exists():
            absent.append(str(path))
            continue
        require(path.is_file() and path.resolve().is_relative_to(job), "source receipt escaped original job")
        stat = path.stat()
        present[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(path))
    return dict(files=present, absent=absent, episode_file_inventory=inventories)


def verify_inventory(reg):
    job = Path(reg["input_job"])
    require(input_inventory(job, reg["cases"]) == reg["inputs"], "bound four-case input evidence changed")
    for name, digest in reg["source_sha256"].items():
        require(sha(name) == digest, "diagnostic/helper source changed after registration")


def prepare(job, out):
    birth = time.time()
    runtime()
    job, out = Path(job).resolve(strict=True), Path(out).resolve()
    require(job == SOURCE_JOB and out.parent == RT and out != job and not out.exists(),
        "fixed original source and new independent runtime output required")
    deadline = min(birth + 600, datetime.fromisoformat(HARD).timestamp())
    require(time.time() < deadline, "08:50 or cumulative600s cap exhausted")
    out.mkdir(exist_ok=False)
    write(out, "PREPARE-STARTED.json", dict(status="CPU_REGISTRATION_ONLY", birth_epoch=birth,
        deadline_epoch=deadline, CPU_cores_max=1, GPU=False, permissions=PERMISSIONS))
    old_alarm = alarm(deadline)
    try:
        p = campaign.load(job)
        cases = fixed_cases(p)
        source_files = (OWN / "audit.py", Path(base.__file__), Path(campaign.__file__),
            ROOT / "投机采样/wam-p0-worktree/wan_va/wan_va_server.py",
            ROOT / "投机采样/wam-p0-worktree/wan_va/configs/va_libero_cfg.py")
        reg = dict(schema="tf-effect-recorded-risk-component-audit/v1", source_commit=COMMIT,
            goal="only inspect task1/init29 and task3/init26 recorded T/TF_ON components; no quality certificate",
            input_job=str(job), cases=cases, inputs=input_inventory(job, cases),
            source_sha256={str(x): sha(x) for x in source_files},
            created_at=datetime.now(timezone.utc).isoformat(), birth_epoch=birth, deadline_epoch=deadline,
            cumulative_prepare_wait_run_cap_sec=600, hard_report_by=HARD, waiting_counts=True,
            CPU_cores_max=1, GPU=False, storage_cap_bytes=CAP, permissions=PERMISSIONS,
            no_children=True, no_CUDA=True, no_forward=True, no_new_episode=True, no_retry=True,
            no_quality_certificate=True, no_TEST=True, no_120=True,
            cleanup="single CPU process exits; no signals or writes to source/active jobs",
            terminal_status="REGISTERED_NOT_RUN", evidence_path=str(out))
        write(out, "PREREGISTRATION.json", reg)
        write(out, "PLAN.json", dict(reg, preregistration_sha256=sha(out / "PREREGISTRATION.json")))
        return reg
    except BaseException as exc:
        write(out, "FAILED.json", dict(status="FAILED_PREPARING_RECORDED_COMPONENT_DIAGNOSTIC", error=repr(exc),
            traceback=traceback.format_exc(), GPU=False, permissions=PERMISSIONS))
        raise
    finally:
        restore_alarm(old_alarm)


def pin_exited(pin):
    try:
        fields = Path(f"/proc/{pin['pid']}/stat").read_text().rpartition(") ")[2].split()
        return int(fields[19]) != pin["start_ticks"] or fields[0] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        return True


def global_cleanup(job, p):
    names = ("ROOT-LAUNCH.json", "CLOCK-LAUNCH.json", "RUN-STARTED.json", "CLOCK-GUARD.json",
        "CLOCK-RESULT.json", "SUPERVISOR-RESULT.json", "FINAL-OWNED-CLEANUP.json", "CLOCK-STOP.json")
    launch, clock_launch, started, guard, clock, sup, clean, stop = [read(job / x) for x in names]
    root, clock_pin = launch["pin"], clock_launch["pin"]
    require(launch["start_new_session"] is True and launch["command"] == root["argv"]
        and root == guard["supervisor_pin"] and clock_pin == guard["guard_pin"]
        and root["pid"] == started["pid"] and root["uid"] == clock_pin["uid"] == os.getuid()
        and str(job) in root["argv"] and str(EXP / "tf_effect_campaign_v1/run.py") in root["argv"]
        and str(job) in clock_pin["argv"] and str(EXP / "tf_effect_campaign_v1/clock.py") in clock_pin["argv"],
        "original root/clock lifetime identity differs")
    require(guard["clock"] == clock["clock"] == stop["clock"]
        and clock["clock"]["plan_sha256"] == sha(job / "plan.json")
        and clock["clock"]["absolute_work_stop"] == p["absolute_work_stop"]
        and clock["clock"]["absolute_cleanup_deadline"] == p["absolute_cleanup_deadline"],
        "original clock/source-plan binding differs")
    require(clock["external_stop_reason"] == stop["reason"] == "registered complete TF campaign deadline"
        and datetime.fromisoformat(stop["at"]) >= datetime.fromisoformat(p["absolute_work_stop"])
        and sup["status"] == "FAILED_OR_PARTIAL" and (job / "FAILED.json").is_file()
        and sup["permissions"] == PERMISSIONS and sup["performance_claim"] is False,
        "original budget PARTIAL/FAILED proof differs; it must remain unchanged")
    require(clean["all_roots_reaped"] is True and sup["cleanup"] == clean
        and clock["all_observed_owned_live_processes_exited"] is True and clock["error"] is None
        and clock["no_foreign_signals"] is True and clock["supervisor_exit_observation"]["live"] is False
        and all(x["observation"]["live"] is False for x in clock["process_observations"]),
        "original global owned cleanup proof incomplete")
    pins = sup["observed_pins"]
    require(pins and all(x["uid"] == os.getuid() and pin_exited(x) for x in pins)
        and pin_exited(root) and pin_exited(clock_pin), "an original pinned own lifetime is still live")
    return dict(component_status="RECORDED_GLOBAL_EXIT_PROOF_VERIFIED", original_status=sup["status"],
        original_failure=sup["failure"], completed_ledger=sup["full_episodes"],
        root_lifetime=root, clock_lifetime=clock_pin, root_and_clock_lifetimes_exited=True,
        observed_owned_lifetimes_checked=len(pins), observed_owned_lifetimes_exited=True,
        global_receipt_does_not_replace_missing_arm_receipt=True,
        observation_scope=clock["observation_scope"], old_FAILED_sha256=sha(job / "FAILED.json"),
        proof_sha256={n: sha(job / n) for n in names})


def arm_components(job, p, arm):
    import torch
    ready = read(job / f"{arm}-READY.json")
    require(ready["status"] == "READY" and ready["arm"] == arm and ready["collect"] is False
        and ready["collector_installed"] is False and ready["setup_RNG_restored"] is True
        and ready["target_RNG_captured_before_draft_and_CAL"] is True
        and ready["ready_rng"] == ready["target_initial_rng"] and ready["permissions"] == PERMISSIONS
        and ready["bank_explicitly_enabled"] is True and ready["synthetic_queries"] == 0
        and ready["performance_claim"] is False, "actual READY component differs")
    require((arm == "T" and ready["predictor_checkpoint"] is None and ready["O1_O2_O3"] is False)
        or (arm == "TF_ON" and ready["predictor_checkpoint"] == p["selected"][arm]["checkpoint"]
            and ready["O1_O2_O3"] is True and ready["selector"] == p["selector"]),
        "recorded target/draft API differs")
    path = job / "arms" / arm / "TARGET-INITIAL-RNG.pt"
    rng = torch.load(path, map_location="cpu", weights_only=True)
    require(set(rng) == {"CPU", "CUDA"}, "actual target initial RNG fields differ")
    fingerprints = {k: base.rng_fingerprint(rng[k]) for k in ("CPU", "CUDA")}
    require(fingerprints["CPU"] == ready["target_initial_rng"]["cpu"]
        and fingerprints["CUDA"] == ready["target_initial_rng"]["cuda"], "actual RNG bytes/shape/dtype not bound to READY")
    cleanup_path = job / "arms" / arm / "OWNED-CLEANUP.json"
    if cleanup_path.is_file():
        require(read(cleanup_path)["all_roots_reaped"] is True, "existing local arm cleanup is not complete")
    return dict(ready=ready, rng=rng, public=dict(arm=arm, actual_CPU_CUDA_RNG_bound_to_READY=True,
        RNG_fingerprints=fingerprints, READY_sha256=sha(job / f"{arm}-READY.json"), RNG_file_sha256=sha(path),
        local_arm_cleanup_receipt_present=cleanup_path.is_file(),
        local_arm_cleanup_receipt_sha256=sha(cleanup_path) if cleanup_path.is_file() else None,
        local_arm_cleanup_audit_gap=None if cleanup_path.is_file() else
            "MISSING original OWNED-CLEANUP.json; global exit proof is separate and cannot restore the old strict gate"))


def raw_outcome(job, case):
    path = job / "episodes" / case["key"] / "client/episodes.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    require(len(records) == 1, "one original complete recorded episode required")
    r = records[0]
    require(r["status"] == "completed" and r["task_index"] == case["task"] and r["episode_index"] == case["init"]
        and r["libero_horizon"] == 800 and r["max_policy_chunks"] is None
        and r["truncated_after_max_policy_chunks"] is False and type(r["success"]) is bool
        and (r["success"] or r["horizon_reached"] is True) and 0 < r["env_steps"] <= 800,
        "full recorded terminal differs")
    return dict(case=case, success=r["success"], actual_env_steps=r["env_steps"],
        failure800_penalty=r["env_steps"] if r["success"] else 800,
        terminal_sha256=sha(path), original_episode_attempt_id=r["episode_attempt_id"],
        episode_wall_sec=r["elapsed_sec"], timing_scope="instrumented shared-resource diagnostic; not formal performance")


def episode_components(job, p, case, arm):
    import torch
    result = base.episode(job, case, arm["ready"], p)
    folder = job / "episodes" / case["key"]
    counter, mapping = read(folder / "COUNTER.json"), read(folder / "client/COMMAND-MAPPING.json")
    contexts = [r["trace_context"] for r in counter["requests"]] + [r["context"] for r in counter["acceptance"]]
    contexts += [r["context"] for r in mapping["chunks"]]
    require(all(c.get("campaign_id") == job.name for c in contexts), "recorded context source job differs")
    for path in (folder / "server_output").rglob("actions_*.pt"):
        value = torch.load(path, map_location="cpu", weights_only=True)
        require(value.dtype == torch.bfloat16 and tuple(value.shape) == (1, 30, 4, 4, 1),
            "actual saved production normalized action geometry/dtype differs")
    return dict(component_status="RECORDED_EPISODE_COMPONENTS_VERIFIED", **result,
        original_helper=str(Path(base.__file__)), original_helper_unchanged=True,
        normalized_action_dtype="torch.bfloat16", no_teacher_forward=True,
        no_arm_cleanup_requirement_removed_from_old_gate=True, not_a_quality_certificate=True)


def pair_components(job, task, init, cases, arms):
    selected = {r["arm"]: r for r in cases if (r["task"], r["init"]) == (task, init)}
    summary = {}
    for arm in ARMS:
        case = selected[arm]
        meta = read(job / "episodes" / case["key"] / "client/repro_metadata.json")
        summary[arm] = dict(case, initial_pairing={k: meta[k] for k in
            ("seed", "model_xml_sha256", "init_state_sha256", "initial_state_hashes", "initial_obs_hashes", "initial_rng")})
    rng_exact = {k: base.exact_value(arms["T"]["rng"][k], arms["TF_ON"]["rng"][k]) for k in ("CPU", "CUDA")}
    paired = base.pairing(job, summary["T"], summary["TF_ON"], all(rng_exact.values()))
    return dict(paired, actual_RNG_tensor_exact=rng_exact,
        recorded_comparison_complete=True, quality_gate_PASS=False,
        pixel_difference_cannot_be_deleted_or_explained_by_global_cleanup=True)


def analyze(job, p, cases):
    arms, episodes, pairs, gaps = {}, [], [], []
    global_proof = global_cleanup(job, p)
    for arm in ARMS:
        try:
            arms[arm] = arm_components(job, p, arm)
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            gaps.append(dict(scope="arm_components", arm=arm, error=repr(exc)))
    for case in cases:
        raw = raw_outcome(job, case)
        try:
            raw["recorded_component_audit"] = episode_components(job, p, case, arms[case["arm"]])
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            raw["recorded_component_audit"] = dict(component_status="RECORDED_COMPONENT_AUDIT_GAP", error=repr(exc))
            gaps.append(dict(scope="episode_components", key=case["key"], error=repr(exc)))
        episodes.append(raw)
    for task, init in PAIRS:
        try:
            pairs.append(pair_components(job, task, init, cases, arms))
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            gaps.append(dict(scope="pair_components", task=task, init=init, error=repr(exc)))
    complete = len(episodes) == 4 and len(pairs) == 2 and not gaps
    local_gaps = [x["public"]["local_arm_cleanup_audit_gap"] for x in arms.values()
        if x["public"]["local_arm_cleanup_audit_gap"] is not None]
    return dict(schema="tf-effect-recorded-risk-component-result/v1", source_commit=COMMIT,
        status="COMPLETE_RECORDED_COMPONENT_DIAGNOSTIC" if complete else "PARTIAL_RECORDED_COMPONENT_DIAGNOSTIC",
        recorded_component_diagnostic_complete=complete, not_a_quality_certificate=True,
        source_job=str(job), original_namespace_untouched=True, original_campaign_status="FAILED_OR_PARTIAL",
        fixed_recorded_cases=cases, recorded_outcomes=episodes, recorded_pair_comparisons=pairs,
        arm_components={k: v["public"] for k, v in arms.items()}, global_owned_exit_component=global_proof,
        component_audit_gaps=gaps, original_local_arm_cleanup_gaps=local_gaps,
        old_strict_gate="BLOCKED; original campaign PARTIAL and missing local TF_ON cleanup receipt are retained",
        quality_gate_PASS=False, permissions=PERMISSIONS, independent_TEST_started=False,
        noninferiority_established=False, formal_performance_permission=False,
        sum_T_round_sec=UNKNOWN, cost_per_legal_commit=UNKNOWN, performance_claim=False,
        no_root_cause_inference=True, no_new_forward=True, no_new_episode=True,
        limitations=["Two fixed previously registered known-risk pairs only; no DEV/TEST aggregation or selection.",
            "Global observed exit proof does not recreate or satisfy an absent original per-arm receipt.",
            "Only saved physical/controller/RNG/camera states can be compared; unsaved simulator state UNKNOWN.",
            "Local recorded acceptance/residual and action audits do not establish environment safety or explain failure."])


def prose(result):
    lines = ["# 已记录风险对的组件诊断", "", f"终态：{result['status']}。仅检查既有四局，未运行模型或新 episode。",
        "", "|task/init|arm|成功|实际环境步|800失败罚分|命令/保存输出组件审计|", "|---|---|---|---|---|---|"]
    for row in result["recorded_outcomes"]:
        case = row["case"]
        lines.append(f"|{case['task']}/{case['init']}|{case['arm']}|{row['success']}|{row['actual_env_steps']}|{row['failure800_penalty']}|{row['recorded_component_audit']['component_status']}|")
    lines += ["", "|task/init|sim/controller数组精确|CPU/CUDA RNG精确|完整reset精确|两相机像素精确|",
        "|---|---|---|---|---|"]
    for row in result["recorded_pair_comparisons"]:
        lines.append(f"|{row['task']}/{row['init']}|{row['actual_sim_controller_arrays_exact']}|{row['actual_RNG_tensor_exact']}|{row['full_reset_state_equal']}|{ {k:v['exact'] for k,v in row['camera_pixels'].items()} }|")
    lines += ["", "原 namespace 的 FAILED/PARTIAL 和 TF_ON 局部 arm 清理收据缺口全部保留。全局退出证明单独核验，不能补造局部收据。",
        "四项原权限全部 false；strict quality gate 仍阻塞。本报告不生成质量证书，不推断失败根因。",
        "每相机像素差异、实际 RNG 指纹、四局命令与保存输出细节见 RESULT.json。",
        "完整局耗时属于共享资源且带记录的描述，ΣT_round/Σg 未采集，均为 UNKNOWN。"]
    return "\n".join(lines) + "\n"


def run(out):
    runtime()
    out = Path(out).resolve(strict=True)
    reg = read(out / "PLAN.json")
    require(out.parent == RT and reg["source_commit"] == COMMIT and reg["permissions"] == PERMISSIONS
        and sha(out / "PREREGISTRATION.json") == reg["preregistration_sha256"]
        and not any((out / n).exists() for n in ("RUN-STARTED.json", "FAILED.json", "CPU-FINAL.json")),
        "fresh separately registered diagnostic only; no overwrite/retry")
    require(reg["deadline_epoch"] <= reg["birth_epoch"] + 600
        and reg["deadline_epoch"] <= datetime.fromisoformat(HARD).timestamp(), "CPU cap/deadline differs")
    old_alarm, start = None, time.monotonic()
    try:
        old_alarm = alarm(reg["deadline_epoch"])
        write(out, "RUN-STARTED.json", dict(status="RUNNING_CPU_RECORDED_COMPONENT_AUDIT", pid=os.getpid(),
            at=datetime.now(timezone.utc).isoformat(), GPU=False, permissions=PERMISSIONS))
        sys.addaudithook(deny_spawn)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized(), "no CUDA initialization permitted")
        verify_inventory(reg)
        job = Path(reg["input_job"])
        require(job == SOURCE_JOB, "only fixed original recorded namespace allowed")
        p = campaign.load(job)
        require(fixed_cases(p) == reg["cases"], "fixed four-case plan changed")
        result = analyze(job, p, reg["cases"])
        verify_inventory(reg)
        require(not torch.cuda.is_initialized() and not children(), "unexpected CUDA or child")
        require(time.time() < reg["deadline_epoch"], "CPU cumulative/report deadline")
        result.update(CPU_run_wall_sec=time.monotonic() - start,
            CPU_prepare_wait_run_wall_sec=time.time() - reg["birth_epoch"], GPU=False,
            CUDA_initialized=False, subprocesses_started=0, input_plan_sha256=sha(job / "plan.json"),
            diagnostic_source_sha256=sha(__file__))
        write(out, "RESULT.json", result)
        text = prose(result).encode()
        require(sum(x.stat().st_size for x in out.iterdir() if x.is_file()) + len(text) < CAP, "diagnostic storage cap")
        with (out / "RECORDED-COMPONENT-DIAGNOSTIC.md").open("xb") as stream:
            stream.write(text)
        write(out, "CPU-FINAL.json", dict(status=result["status"], GPU=False, CUDA_initialized=False,
            children_at_exit=children(), CPU_cores_max=1, cumulative_wall_sec=time.time() - reg["birth_epoch"],
            cleanup="single CPU process exits; no source processes signalled", permissions=PERMISSIONS,
            not_a_quality_certificate=True))
        print(json.dumps(dict(status=result["status"], output=str(out)), ensure_ascii=False))
        return 0 if result["recorded_component_diagnostic_complete"] else 2
    except BaseException as exc:
        failure = dict(status="FAILED_RECORDED_COMPONENT_DIAGNOSTIC", error=repr(exc),
            traceback=traceback.format_exc(), GPU=False, CUDA_initialized=False,
            cumulative_wall_sec=time.time() - reg["birth_epoch"], children_at_exit=children(),
            original_evidence_not_modified=True, permissions=PERMISSIONS, not_a_quality_certificate=True)
        write(out, "FAILED.json", failure)
        if not (out / "CPU-FINAL.json").exists():
            write(out, "CPU-FINAL.json", failure)
        raise
    finally:
        if old_alarm is not None:
            restore_alarm(old_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job", type=Path, help="prepare only; fixed original DEV namespace")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        require(args.job is not None, "prepare requires --job")
        prepare(args.job, args.out)
        print(json.dumps(dict(status="REGISTERED_NOT_RUN", output=str(args.out)), ensure_ascii=False))
        return 0
    require(args.job is None, "run input comes only from frozen diagnostic PLAN")
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())

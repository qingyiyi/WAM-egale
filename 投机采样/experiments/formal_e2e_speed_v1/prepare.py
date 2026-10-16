"""Freeze a balanced local A_off/T episode plan without touching CUDA."""
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from speed_contract import (OWN, PRESET, REGISTRY, SOURCE_PATHS, TARGET_MANIFEST,
                      check_job, read, schedule, sha, write)


def build(job: Path, pairs: list[str], label: str, port: int, master_port: int) -> dict:
    job = check_job(job)
    if job.exists():
        raise ValueError("job must be fresh; no resume or overwrite")
    if not (1024 <= port <= 65535 and 1024 <= master_port <= 65535 and port != master_port):
        raise ValueError("distinct unprivileged ports required")
    if label not in ("single_task", "multi_task"):
        raise ValueError("label must be single_task or multi_task")
    tasks = {x["task"] for x in schedule(pairs)}
    if (label == "single_task" and len(tasks) != 1) or (label == "multi_task" and len(tasks) < 2):
        raise ValueError("label/task coverage mismatch")
    preset, registry, inventory = read(PRESET), read(REGISTRY), read(TARGET_MANIFEST)
    if (preset["action_draft_mode"] != "verifier_k2_b2_bonus"
            or preset["selector"] != "prefix_then_cost" or not preset["shared_KV"]
            or preset["host_optimization"] != "O1+O2+O3"):
        raise ValueError("current candidate is not frozen A_off/O3/prefix preset")
    target = registry["suites"]["libero_10"]
    if target["status"] != "available" or target["official_checkpoint"] != inventory["target_checkpoint_id"]:
        raise ValueError("LIBERO Long checkpoint registry/manifest mismatch")
    weights = [x for x in inventory["weight_files"] if "/._____temp/" not in x["path"]]
    if len(weights) != 7:
        raise ValueError("expected seven loaded target safetensors")
    for row in weights:
        stat = Path(row["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise ValueError("target weight metadata differs from hashed inventory")
    if Path(preset["draft_checkpoint"]).resolve() != Path(preset["draft_checkpoint"]):
        raise ValueError("absolute draft checkpoint required")
    rows = schedule(pairs)
    plan = dict(schema="local-formal-e2e-speed/v1", created_at=datetime.now(
        ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"), job=str(job),
        label=label, pairs=pairs, schedule=rows, arms=["T", "A_OFF"],
        server_host="local", gpu_model="A100-SXM4-40GB", cuda_device="0",
        seed=20260922, horizon=800, trace_mode="off", selector="prefix_then_cost",
        threshold=preset["threshold"], mode="verifier_k2_b2_bonus",
        draft_checkpoint=preset["draft_checkpoint"], draft_sha256=sha(Path(preset["draft_checkpoint"])),
        target_checkpoint_path=target["local_path"],
        target_checkpoint_id=target["official_checkpoint"], registry=str(REGISTRY),
        target_manifest_sha256=sha(TARGET_MANIFEST), target_weights=weights,
        preset_sha256=sha(PRESET), registry_sha256=sha(REGISTRY),
        source_sha256={str(x): sha(x) for x in SOURCE_PATHS},
        port=port, master_port=master_port, one_model_and_one_EGL_at_a_time=True,
        fresh_model_per_episode=True, order="T/A_OFF then A_OFF/T per consecutive pairs",
        sample_interval_sec=5, idle_admission_samples=5,
        per_episode_wall_timeout_sec=1500, max_run_wall_sec=max(3600, len(rows) * 1100),
        cold_start_and_natural_episode_separately_reported=True,
        startup_and_artifact_drain_excluded_from_original_client_elapsed=True,
        formal_speed_local_only=True, no_GPU_use_during_prepare=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--label", choices=("single_task", "multi_task"), required=True)
    ap.add_argument("--port", type=int, default=19481)
    ap.add_argument("--master-port", type=int, default=29481)
    args = ap.parse_args()
    plan = build(args.job, args.pairs, args.label, args.port, args.master_port)
    args.job.mkdir(parents=True)
    write(args.job / "plan.json", plan)
    print(str(args.job / "plan.json"))


if __name__ == "__main__":
    main()

"""CPU metadata-only preparation for the explicitly approved fourteen retries."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tf_suite_pipeline_v1.protocol import ROOT, PERMISSIONS, read, write, require


def rebase(value, old, new):
    if isinstance(value, str) and (value == str(old) or value.startswith(str(old) + "/")):
        return str(new) + value[len(str(old)):]
    if isinstance(value, list):
        return [rebase(x, old, new) for x in value]
    if isinstance(value, dict):
        return {k: rebase(v, old, new) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()
    allowed = (Path.home() / "cxz").resolve(strict=True)
    prior = args.prior.resolve(strict=True)
    job = args.job.resolve()
    require(ROOT.is_relative_to(allowed) and prior.is_relative_to(allowed)
            and job.is_relative_to(allowed), "all work stays under ~/cxz")
    require(not job.exists(), "new namespace only; do not overwrite a prior attempt")
    r = read(prior / "registration.json")
    require(r["permissions"] == PERMISSIONS and datetime.now(timezone.utc).timestamp()
            < datetime.fromisoformat(r["absolute_work_stop"]).timestamp(), "original M17 budget/deadline")
    require(read(prior / "TERMINAL.json")["status"] == "PARTIAL_REMOTE_OBJECT_NO_RETRY",
            "preserve actual previous terminal")
    require(read(prior / "CURRENT_DATA/GUARD-FINAL.json")["status"] == "CLEAN",
            "previous current phase already cleaned")
    index = read(prior / "PACK_TARGET/INDEX.json")
    require(index["FIT_rows"] == 80605 and index["CAL_rows"] == 9065
            and index["all_episodes_terminal"] is True, "immutable qualified Target200 source")
    p = read(prior / "CURRENT_DATA/plan.json")
    require(len(p["rows"]) == 30 and p["migration_readonly_completed_row_ids"] == list(range(16)),
            "same registered thirty rows and sixteen qualified imported rows")
    old_attempts, cases = 0, []
    for i, row in enumerate(p["rows"]):
        c = read(Path(row["out"]) / "COUNTER.json")
        t = read(Path(row["out"]) / "EPISODE-TERMINAL.json")
        require(row["row_id"] == i and t["complete_episode"] is True,
                "every old robot episode is retained as a complete outcome")
        if i < 16:
            require(c["collector_invalid"] is False and not c["collector_errors"],
                    "do not rerun a qualified imported capture")
        else:
            require(c["collector_invalid"] is True and c["collector_errors"],
                    "only the fourteen rejected captures may be retried")
            old_attempts += c["target_forward_attempts"]
            cases.append(dict(row_id=i, task=row["task"], init=row["init"], case_id=row["case_id"]))
    require(len(cases) == 14, "only explicitly approved fourteen")
    job.mkdir()
    current = job / "CURRENT_DATA"
    current.mkdir()
    old_prefix = prior / "CURRENT_DATA/episodes"
    new_prefix = current / "episodes"
    for row in p["rows"]:
        previous = Path(row["out"])
        row["out"] = str(new_prefix / f"case-{row['case_id']:04d}" / row["arm"])
        if row["row_id"] >= 16:
            continue
        dest = Path(row["out"])
        dest.mkdir(parents=True)
        for name in ("EPISODE-TERMINAL.json", "COUNTER.json", "AUDIT.json",
                     "FINISH_READY.json", "COLLECTION-AUDIT.json"):
            write(dest / name, rebase(read(previous / name), old_prefix, new_prefix))
        write(dest / "RECOVERY-PROVENANCE.json", dict(original_out=str(previous),
              new_episode=False, only_host_path_strings_rebound=True, tensor_shards_unchanged=True))
    for item in p["migration_context_out_prefix_mapping"]:
        item["actual"] = str(new_prefix)
    version = "object_capture_recovery_v2-20261019-authorized14"
    p.update(job=str(current), root_path=str(ROOT), source_code_commit=version,
             migration_readonly_row_results=str(prior / "CURRENT_DATA"))
    # Old valid shards remain under the parent; new worker writes a unique child.
    require(Path(p["collector_output_dir"]).is_relative_to(Path(p["suite_storage_root"])),
            "preserve one shared compact budget root")
    write(current / "plan.json", p)
    write(current / "DEPLOYMENT-COMPLETE.json", dict(
        status="REGISTERED_AUTHORIZED_14_CAPTURE_RECOVERY", permissions=PERMISSIONS,
        readonly_old_rows=list(range(16)), retry_rows=list(range(16, 30)),
        no_old_record_overwritten=True, source_code_commit=version))
    for name in ("PACK_TARGET", "TARGET_TRAINING_DATA"):
        (job / name).symlink_to(prior / name, target_is_directory=True)
    write(job / "FINAL-TEMPLATE.json", read(prior / "FINAL-TEMPLATE.json"))
    r.update(job=str(job), source_code_commit=version, prior_failed_job=str(prior),
             shared_capture_root=p["suite_storage_root"],
             remaining_forward_budget=r["new_job_forward_cap"] + 56000 - old_attempts,
             approved_extra_complete_episodes=14, approved_extra_forward_attempts=56000,
             previous_invalid14_actual_attempts=old_attempts,
             no_budget_origin_or_deadline_reset=True,
             recovery_authorization="用户2026-10-19明确允许补采14局并接续原任务",
             recovery_case_manifest=cases)
    write(job / "registration.json", r)
    write(job / "PREPARED.json", dict(status="PREPARED_METADATA_ONLY_NO_GPU_STARTED",
          additional_robot_episodes_started=0, retry_cases=cases, old_invalid_attempts=old_attempts,
          remaining_forward_budget=r["remaining_forward_budget"], permissions=PERMISSIONS))
    print(dict(prepared=str(job), fixed_retry_count=14, previous_attempts=old_attempts,
               absolute_work_stop=r["absolute_work_stop"], no_GPU_started=True), flush=True)


if __name__ == "__main__":
    main()

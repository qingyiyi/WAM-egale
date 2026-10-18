"""Small-source provenance and frozen-evidence audit; no Torch/CUDA imports."""
import argparse
import ast
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

REVIEW_COMMIT = "ca5f37e0482c2e38e3738e3d8e71e3b185990a23"
MODEL = "投机采样/wam-p0-worktree/wan_va/modules/model.py"
ALIGNED = "投机采样/experiments/operator_torch_alignment_v2/"
EXPERIMENT = ".pipeline/experiments/operator-torch-alignment-20261018/"
SOURCES = [
    MODEL, "投机采样/wam-p0-worktree/wan_va/wan_va_server.py",
    ALIGNED + "integration.py", ALIGNED + "setup.py",
    ALIGNED + "csrc/standalone_flash_sdpa.cu", ALIGNED + "server_entry.py",
    ALIGNED + "worker_full.py", ALIGNED + "worker_performance.py",
    ALIGNED + "worker_full30_performance.py", ALIGNED + "saved_input_validate.py",
    "投机采样/experiments/shared_prefix_torch_v2/csrc/shared_torch.cu",
    "投机采样/experiments/shared_prefix_torch_v2/csrc/shared_flash_fwd_kernel.h",
    "投机采样/experiments/shared_prefix_torch_v2/csrc/shared_load.h",
    "投机采样/experiments/wam_candidate_tile_v1/csrc/group.cu",
    "投机采样/experiments/wam_candidate_tile_v1/csrc/group_kernel.h",
    "投机采样/experiments/wam_candidate_tile_v1/csrc/group_load.h",
    "投机采样/experiments/wam_candidate_tile_guarded_v2/guarded_worker.py",
    "投机采样/experiments/wam_static_prompt_baseline_v2/static_projection.py",
    "投机采样/experiments/wam_static_prompt_baseline_v2/baseline_contract.py",
    "投机采样/experiments/single_worker_chc_gpu_v4/adapter.py",
    "投机采样/wam-p0-worktree/LICENSE.txt",
]
EVIDENCE = [
    "review/CURRENT-PROGRESS-20261018.md", "review/PROJECT-REVIEW-PROMPT-20261018.md",
    ".pipeline/cycles/wam-research-20261016/STATUS-20261018.md",
    EXPERIMENT + "RESULT.json", EXPERIMENT + "RESULT.md",
    EXPERIMENT + "ATTENTION.csv", EXPERIMENT + "BLOCK.csv",
    EXPERIMENT + "FULL30.csv", EXPERIMENT + "POLICY.csv",
    EXPERIMENT + "performance/RESULT.json", EXPERIMENT + "performance/RESULT.md",
    EXPERIMENT + "performance/PERFORMANCE-ONLY.csv",
    EXPERIMENT + "performance/PERFORMANCE-ONLY-ROUNDS.csv",
    EXPERIMENT + "performance/FULL30.csv", EXPERIMENT + "performance/FULL30-ROUNDS.csv",
    EXPERIMENT + "performance/CPU-INPUT-IDENTITY-AUDIT.json",
]


def symbols(source):
    def walk(nodes, parent=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                name = parent + node.name
                yield name, dict(line=node.lineno, end_line=node.end_lineno)
                if isinstance(node, ast.ClassDef):
                    yield from walk(node.body, name + ".")
    return dict(walk(ast.parse(source).body))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {4, 5})
    args.output.mkdir(parents=True, exist_ok=True)
    def git(*argv):
        return subprocess.check_output(["git", *argv], cwd=args.repository)
    records = []
    for relative in SOURCES + EVIDENCE:
        try:
            raw = git("show", f"{REVIEW_COMMIT}:{relative}")
        except subprocess.CalledProcessError:
            records.append(dict(path=relative, status="UNKNOWN_MISSING_AT_FIXED_COMMIT"))
            continue
        if len(raw) > 2 * 2**20:
            raise RuntimeError("Audit only small sources/reports, never large tensor/model hashes")
        live = args.local_root / relative
        row = dict(path=relative, status="READ_FIXED_GIT_OBJECT", commit=REVIEW_COMMIT,
                   bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                   local_present=live.is_file(), local_byte_equal=(live.read_bytes() == raw if live.is_file() else None),
                   interpretation="source/result identity, not independently rerun old GPU evidence")
        if relative.endswith(".py"):
            row["symbols"] = symbols(raw.decode())
        if relative.endswith(".csv"):
            table = list(csv.DictReader(io.StringIO(raw.decode())))
            row["table_rows"] = len(table)
            row["columns"] = list(table[0]) if table else []
        if relative.endswith(".json"):
            data = json.loads(raw)
            if isinstance(data, dict):
                row["top_level_fields"] = list(data)
                row["status"] = "READ_FIXED_GIT_OBJECT"
                row["recorded_status"] = data.get("status")
        records.append(row)
    result = dict(schema="wam-frozen-small-source-audit/v1", review_commit=REVIEW_COMMIT,
                  audited_at=datetime.now().astimezone().isoformat(),
                  publication_checkout_HEAD=git("rev-parse", "HEAD").decode().strip(),
                  publication_checkout_dirty=git("status", "--short").decode(),
                  source_records=records,
                  no_main_used_for_scientific_baseline=True, no_model_or_tensor_hash=True,
                  no_torch_import=True, no_cuda_context=True)
    (args.output / "SOURCE-MANIFEST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(sources=len(SOURCES), evidence=len(EVIDENCE),
                         missing=[row["path"] for row in records if row["status"].startswith("UNKNOWN")],
                         local_differences=[row["path"] for row in records if row.get("local_byte_equal") is False]), ensure_ascii=False))


if __name__ == "__main__":
    main()

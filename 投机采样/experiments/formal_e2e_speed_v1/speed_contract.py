"""CPU-safe, immutable contract for local-server natural-episode timing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
EXP = OWN.parent
WORKTREE = ROOT / "投机采样/wam-p0-worktree"
RUNTIME = ROOT / "投机采样/results/runtime"
# The project volume can be full while the local scratch filesystem has room
# for complete episode artifacts. The plan freezes the chosen absolute path.
SCRATCH_RUNTIME = Path("/var/tmp/cxz-formal-e2e-speed")
VA = Path("/rjs/cxz/miniconda3/envs/va/bin/python")
CLIENT = Path("/rjs/cxz/miniconda3/envs/libero-va/bin/python")
PREFIX = "p0-formal-e2e-speed-v1-"
PRESET = ROOT / "投机采样/configs/current_speculative_candidate.json"
REGISTRY = ROOT / "投机采样/configs/libero_checkpoint_registry.json"
TARGET_MANIFEST = ROOT / ".pipeline/research/speculative-sampling/evidence/fusion-withdraw-draft-onpolicy-20261014/MORNING-CURRENT-TEACHER-WEIGHT-SHA256.json"
SOURCE_PATHS = (
    WORKTREE / "wan_va/wan_va_server.py",
    ROOT / "投机采样/scripts/p0_libero_multichunk_client.py",
    EXP / "speculative_e2e_v3/client_entry.py",
    EXP / "speculative_e2e_v3/server_entry.py",
    *(EXP / "prefix_acceptance_v1" / name for name in (
        "selection.py", "online.py", "measured_online.py", "buffers.py",
        "scalar.py", "template.py", "prefix_kernel.py", "paths.py")),
    *(OWN / name for name in ("speed_contract.py", "prepare.py", "client_entry.py",
                                  "server_entry.py", "run.py", "analyze.py")),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text())


def write(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_job(job: Path) -> Path:
    job = Path(job).resolve()
    if job.parent not in (RUNTIME, SCRATCH_RUNTIME) or not job.name.startswith(PREFIX):
        raise ValueError("fresh local formal E2E results job required")
    return job


def task_init(pair: str) -> tuple[int, int]:
    task, init = pair.split(":")
    task, init = int(task), int(init)
    if not 0 <= task < 10 or not 0 <= init < 50:
        raise ValueError("LIBERO Long task 0..9 and init 0..49 required")
    return task, init


def schedule(pairs: list[str]) -> list[dict]:
    parsed = [task_init(pair) for pair in pairs]
    if len(parsed) < 2 or len(parsed) % 2 or len(set(parsed)) != len(parsed):
        raise ValueError("even number >=2 of unique task:init pairs required for AB/BA balance")
    rows = []
    for index, (task, init) in enumerate(parsed):
        for arm in (("T", "A_OFF") if index % 2 == 0 else ("A_OFF", "T")):
            rows.append(dict(pair=index, arm=arm, task=task, init=init,
                             key=f"p{index:02d}-{arm}-t{task}-i{init}"))
    return rows


def validate(job: Path, *, check_sources=True) -> dict:
    job = check_job(job)
    p = read(job / "plan.json")
    if (p.get("schema") != "local-formal-e2e-speed/v1"
            or p.get("job") != str(job)
            or p.get("server_host") != "local"
            or p.get("gpu_model") != "A100-SXM4-40GB"
            or p.get("arms") != ["T", "A_OFF"]
            or p.get("schedule") != schedule(p["pairs"])
            or p.get("selector") != "prefix_then_cost"
            or p.get("mode") != "verifier_k2_b2_bonus"
            or p.get("trace_mode") != "off"
            or p.get("seed") != 20260922
            or p.get("horizon") != 800):
        raise ValueError("formal E2E plan contract changed")
    if check_sources:
        for path, expected in p["source_sha256"].items():
            if sha(Path(path)) != expected:
                raise ValueError("source changed after plan freeze: " + path)
        if sha(PRESET) != p["preset_sha256"] or sha(REGISTRY) != p["registry_sha256"]:
            raise ValueError("candidate/registry changed after plan freeze")
        if sha(Path(p["draft_checkpoint"])) != p["draft_sha256"]:
            raise ValueError("draft bytes changed after plan freeze")
        for row in p["target_weights"]:
            stat = Path(row["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
                raise ValueError("target weight metadata changed: " + row["path"])
    return p

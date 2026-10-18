"""Explicit resource gates. Importing this module does not initialize CUDA."""
from datetime import datetime
import json
import os
from pathlib import Path

from .reference import REVIEW_COMMIT


def cpu_only(cpus=(4, 5)):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only task started inside an initialized CUDA process")
    allowed = os.sched_getaffinity(0)
    chosen = set(cpus)
    if not chosen or len(chosen) > 2 or not chosen.issubset(allowed):
        raise ValueError("CPU task needs one or two available registered cores")
    os.sched_setaffinity(0, chosen)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(len(chosen))
    torch.set_num_interop_threads(1)

    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU_ONLY: CUDA context creation is forbidden")
    torch.cuda._lazy_init = forbidden
    return dict(affinity=sorted(chosen), torch_num_threads=torch.get_num_threads(),
                cuda_context=False)


def require_gpu_authorization(path, phase):
    """Fail closed before native imports, availability probes or CUDA calls.

    This gate consumes a future explicit task authorization; it does not create
    one. The CPU registration supplied with this task intentionally fails it.
    """
    if not path:
        raise PermissionError("NOT_RUN_NO_GPU_AUTHORIZATION")
    data = json.loads(Path(path).read_text())
    if (data.get("allow_new_gpu_jobs") is not True
            or data.get("review_commit") != REVIEW_COMMIT
            or phase not in data.get("allowed_phases", [])):
        raise PermissionError("NOT_RUN_NO_VALID_PHASE_AUTHORIZATION")
    now = datetime.now().astimezone()
    deadline = datetime.fromisoformat(data["cleanup_deadline"])
    if deadline.tzinfo is None or deadline <= now:
        raise PermissionError("NOT_RUN_EXPIRED_GPU_AUTHORIZATION")
    if not (0 < data.get("gpu_lifecycle_seconds_cap", 0) <= 7200):
        raise PermissionError("NOT_RUN_INVALID_GPU_BUDGET")
    if not data.get("user_authorization_evidence"):
        raise PermissionError("NOT_RUN_MISSING_USER_AUTHORIZATION_EVIDENCE")
    if phase == "performance" and data.get("exclusive_window") is not True:
        raise PermissionError("NOT_RUN_NO_EXCLUSIVE_PERFORMANCE_WINDOW")
    return data

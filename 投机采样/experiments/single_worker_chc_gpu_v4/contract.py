"""CPU-safe task contract. Nothing in this module launches a model or GPU job."""
from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path('/rjs/cxz/论文探索')
WORKTREE = ROOT / '投机采样/wam-p0-worktree'
MODEL_SOURCE = WORKTREE / 'wan_va/modules/model.py'
MODEL_SHA256 = 'f579ee02b490dbffcf78add6452742dbdbef43286600c0eeae3dede3f053cb5b'
PROPOSAL = ROOT / '.pipeline/research/speculative-sampling/evidence/single-worker-chc-plan-20261012/PLAN.json'
FEATURE_ENV = 'P0_SINGLE_WORKER_CHC_ENABLED'
ROWS = (0,) * 7 + (1,) * 7
LAYER_COUNT = 30
FORWARD_CAP = 900
CACHE_BYTE_CAP = 12 * 2**30


def enabled_from_env():
    # Old KV-attribution/GPU approval flags do NOT enable this new task.
    return os.environ.get(FEATURE_ENV, '0') == '1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_proposal():
    p = json.loads(PROPOSAL.read_text())
    if (p['GPU_model_worker_cap'] != 1 or p['hard_target_forward_cap'] != FORWARD_CAP
            or sum(p['full_target_forward_budget'].values()) != 895
            or p['new_full_episode_cap'] or p['EGL_initialization_cap']
            or p['complete_policy_chunk_cap'] or p['training']):
        raise ValueError('single-worker frozen proposal changed')
    return p


class Budget:
    """Reserve BEFORE the attempted forward; exceptions do not refund it."""
    def __init__(self, cap=FORWARD_CAP, *, seconds=45 * 60, clock=time.monotonic):
        if type(cap) is not int or not 0 < cap <= FORWARD_CAP:
            raise ValueError('forward cap')
        if not 0 < seconds <= 45 * 60:
            raise ValueError('deadline')
        self.cap, self.used, self.clock = cap, 0, clock
        self.deadline = clock() + seconds

    def reserve(self, count=1):
        if type(count) is not int or count <= 0:
            raise ValueError('reservation')
        if self.clock() >= self.deadline:
            raise TimeoutError('task deadline reached')
        if self.used + count > self.cap:
            raise RuntimeError('target forward budget exhausted; no retries')
        self.used += count


def job_path(job_root, relative):
    """Explicit relocation, not string substitution of escaped Chinese paths."""
    root = Path(job_root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('path must belong to this job')
    out = (root / rel).resolve()
    if not out.is_relative_to(root):
        raise ValueError('path escaped job through symlink')
    return out


def cpu_job_metadata(job_root):
    """Pure metadata only; cannot enqueue/start GPU or inherit old approval."""
    return dict(job=str(Path(job_root).resolve()), lane='speculative-sampling NOT tile',
                CPU_only=True, GPU_run_approved=False, new_GPU_started=False,
                training=False, model_workers=1, resident_peer=False,
                forward_cap=FORWARD_CAP, GPU_minutes_hard_cap=45,
                full_episodes=0, EGL=0, full_policy_chunks=0)

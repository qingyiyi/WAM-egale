"""Unmodified complete LIBERO loop, deterministic initial state and RAM commands."""
import importlib.util
import os
from pathlib import Path
import sys

from protocol import EXP, ROOT, load


def main():
    job = Path(os.environ['RESET_ONLINE_JOB']).resolve()
    p = load(job)
    key = os.environ['RESET_ONLINE_CASE']
    row, = [r for r in p['cases'] if r['key'] == key]
    if any(os.getenv(k) != v for k, v in dict(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
            MUJOCO_EGL_DEVICE_ID='0', CUDA_VISIBLE_DEVICES='0').items()):
        raise ValueError('actual EGL and visible GPU configuration required')
    sys.path.insert(0, str(ROOT / '投机采样/scripts'))
    import p0_libero_multichunk_client as client
    args = client.parse_args()
    out = job / 'episodes' / key / 'client'
    if (args.task_indices != [row['task']] or args.episode_start != row['init']
            or args.episodes_per_task != 1 or args.out_dir.resolve() != out
            or args.trace_mode != 'off' or not args.latency_benchmark or args.resume
            or args.save_video or args.condition != key or args.benchmarks != ['libero_10']):
        raise ValueError('complete-task assignment mismatch')
    path = EXP / 'speculative_e2e_v3/client_entry.py'
    spec = importlib.util.spec_from_file_location('reset_quality_lean_client', path)
    lean = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lean)
    lean.install(client, out)
    return client.main()


if __name__ == '__main__':
    raise SystemExit(main())

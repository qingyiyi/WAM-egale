"""Original full-horizon LIBERO client, initial pixels and command mapping.

No short episode, video save, smoke, FIT/CAL reshuffle, or policy replacement.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

OWN = Path(__file__).resolve().parent
EXP, ROOT = OWN.parent, OWN.parents[2]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from tf_effect_campaign_v1.protocol import load


def main():
    job = Path(os.environ['TF_EFFECT_JOB']).resolve(strict=True)
    p = load(job)
    key, arm = os.environ['TF_EFFECT_CASE'], os.environ['TF_EFFECT_ARM']
    rows = [row for row in p['cases'] if row['key'] == key and row['arm'] == arm]
    if len(rows) != 1:
        raise ValueError('one registered complete episode task/init/arm assignment required')
    row = rows[0]
    expected = dict(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0', CUDA_VISIBLE_DEVICES='0')
    if any(os.getenv(name) != value for name, value in expected.items()) or any(os.getenv(name) for name in
            ('WAM_P0_STATE_TEACHER_DIR', 'P0_NATIVE_DIAGNOSTIC_DIR')):
        raise ValueError('one real EGL client with heavy native diagnostics disabled required')
    if len(os.sched_getaffinity(0)) > 2:
        raise ValueError('client must inherit its registered two CPU cores')
    sys.path.insert(0, str(ROOT / '投机采样/scripts'))
    import p0_libero_multichunk_client as client
    if Path(client.__file__).resolve() != ROOT / '投机采样/scripts/p0_libero_multichunk_client.py':
        raise ImportError('actual complete LIBERO client origin differs')
    args = client.parse_args()
    out = job / 'episodes' / key / 'client'
    if (args.task_indices != [row['task']] or args.episode_start != row['init'] or args.episodes_per_task != 1
            or args.out_dir.resolve() != out or args.trace_mode != 'off' or not args.latency_benchmark
            or args.resume or args.save_video or args.condition != key or args.benchmarks != ['libero_10']
            or getattr(args, 'max_policy_chunks', None) is not None
            or args.campaign_id != job.name or args.worker_id != key
            or client.LIBERO_HORIZON != 800):
        raise ValueError('fixed complete horizon800 task flags differ from registered plan')
    from tf_usable_quality_v1.capture_client import install
    import tf_usable_quality_v1.capture_client as capture
    if (Path(capture.__file__).resolve() != EXP / 'tf_usable_quality_v1/capture_client.py'
            or str(Path(capture.__file__).resolve()) not in p['source_sha256']):
        raise ImportError('initial-pixel/physical-state capture source lacks registered origin')
    install(client, out)
    from task4_direction_diagnostic_v1.client_observer import install as install_mapping
    import task4_direction_diagnostic_v1.client_observer as mapping
    if str(Path(mapping.__file__).resolve()) not in p['source_sha256']:
        raise ImportError('submitted/environment command mapping source lacks registered origin')
    install_mapping(client, out)
    return client.main()


if __name__ == '__main__':
    raise SystemExit(main())

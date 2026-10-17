"""Unshortened full LIBERO task, RAM commands and initial/final state artifacts."""
import os
from pathlib import Path
import sys

EXP=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(EXP))
from task4_direction_diagnostic_v1.protocol import ROOT,load


def main():
    job=Path(os.environ['DIRECTION_DIAG_JOB']).resolve()
    p=load(job)
    key=os.environ['DIRECTION_DIAG_CASE']
    row,=[r for r in p['cases'] if r['key']==key]
    expected=dict(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',CUDA_VISIBLE_DEVICES='0')
    if any(os.getenv(k)!=v for k,v in expected.items()) or os.getenv('WAM_P0_STATE_TEACHER_DIR') or os.getenv('P0_NATIVE_DIAGNOSTIC_DIR'):
        raise ValueError('one real EGL client with heavy diagnostics disabled required')
    sys.path.insert(0,str(ROOT/'投机采样/scripts'))
    import p0_libero_multichunk_client as client
    if Path(client.__file__).resolve()!=ROOT/'投机采样/scripts/p0_libero_multichunk_client.py':
        raise ImportError('actual full LIBERO client source differs from frozen follow-up')
    args=client.parse_args()
    out=job/'episodes'/key/'client'
    if (args.task_indices!=[row['task']] or args.episode_start!=row['init'] or args.episodes_per_task!=1
            or args.out_dir.resolve()!=out or args.trace_mode!='off' or not args.latency_benchmark
            or args.resume or args.save_video or args.condition!=key or args.benchmarks!=['libero_10']
            or getattr(args,'max_policy_chunks',None) is not None):
        raise ValueError('fixed complete DEV client flags differ')
    from tf_usable_quality_v1.capture_client import install
    import tf_usable_quality_v1.capture_client as capture
    if Path(capture.__file__).resolve()!=Path(__file__).resolve().parent.parent/'tf_usable_quality_v1/capture_client.py':
        raise ImportError('fresh initial-pixel capture origin differs')
    install(client,out)
    from task4_direction_diagnostic_v1.client_observer import install as install_diagnostic
    install_diagnostic(client,out)
    return client.main()


if __name__=='__main__':
    raise SystemExit(main())

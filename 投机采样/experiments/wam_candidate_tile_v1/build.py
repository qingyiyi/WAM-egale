"""Finite two-core CPU build of the isolated research extension."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
OWN=Path(__file__).resolve().parent

def main():
    os.sched_setaffinity(0,{12,13})
    job=Path(sys.argv[1]).resolve();job.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MAX_JOBS='2',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCH_CUDA_ARCH_LIST='8.0',
        CUDA_HOME='/usr/local/cuda-12.4',PYTHONDONTWRITEBYTECODE='1')
    started=time.monotonic();child=None;error=None
    command=[sys.executable,'-B',str(OWN/'setup.py'),'build_ext','--inplace']
    (job/'PLAN.json').write_text(json.dumps(dict(command=command,CPU=2,wall_cap_seconds=1500,
        build_cap_bytes=512*2**20,GPU=False,original_vendor_modified=False),indent=2)+'\n')
    try:
        with (job/'build.log').open('x') as log:
            child=subprocess.Popen(command,cwd=OWN,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if time.monotonic()-started>1490:raise TimeoutError('CPU compile time cap')
                if sum(p.stat().st_size for p in OWN.rglob('*') if p.is_file())>512*2**20:raise RuntimeError('build size cap')
                time.sleep(.5)
        if child.returncode:raise RuntimeError('compiler returned '+str(child.returncode))
        assert len(list(OWN.glob('_wam_candidate_tile*.so')))==1
    except BaseException as exc:error=repr(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        result=dict(status='COMPLETE_CPU_BUILD' if error is None else 'FAILED_CPU_BUILD',error=error,
            CPU_affinity=sorted(os.sched_getaffinity(0)),elapsed_seconds=time.monotonic()-started,
            owned_root_reaped=child is None or child.poll() is not None,GPU_used=False,
            output_bytes=sum(p.stat().st_size for p in OWN.rglob('*') if p.is_file()))
        (job/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if error:raise RuntimeError(error)

if __name__=='__main__':main()

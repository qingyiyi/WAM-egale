"""One finite numerical worker; shared GPU permitted, no performance result."""
from pathlib import Path
import fcntl,hashlib,json,os,signal,sys,time,traceback
OWN=Path(__file__).resolve().parent;ROOT=OWN.parents[2];RUNTIME=ROOT/'投机采样/results/runtime'
sys.path.insert(0,str(OWN.parent));sys.path.insert(0,str(OWN.parent/'wam_shape_audit_v1'))
from run_matrix_r2 import bounded,NVML
from target_trace_draft_fresh_dev_v1.legacy import launcher

def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    os.sched_setaffinity(0,{12,13});out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    assert out.parent==RUNTIME
    files=[OWN/'run.py',OWN/'correctness.py',OWN/'GENERATION.json',*OWN.glob('csrc/*'),*OWN.glob('_wam_candidate_tile*.so')]
    assert len(list(OWN.glob('_wam_candidate_tile*.so')))==1
    build=RUNTIME/'p0-wam-candidate-tile-build-v1-20261016-r2/RESULT.json'
    assert json.loads(build.read_text())['status']=='COMPLETE_CPU_BUILD'
    inputs=RUNTIME/'p0-shared-prefix-observe-v2-20261012-182534/observation'
    input_stats={}
    for layer in (0,8,29):
        for suffix in ('tensors.pt','metadata.json'):
            f=inputs/f'layer{layer:02d}-{suffix}';st=f.stat()
            input_stats[str(f)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
            if suffix=='metadata.json':input_stats[str(f)]['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
    write(out/'PLAN.json',dict(GPU_seconds_cap=600,cleanup_seconds=60,CPU=2,output_cap_bytes=128*2**20,
        operator_call_cap=64,performance_claim=False,shared_GPU_allowed=True,required_free_GiB=2,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        input_dir=str(inputs),input_stats=input_stats,build_result=str(build),
        build_result_sha256=hashlib.sha256(build.read_bytes()).hexdigest()))
    Base,cleanup=launcher()
    class Driver:
        def __init__(self):self.roots=[];self.pins={}
    d=Driver();nv=None;worker=None;error=None;closed=None;resources=None;locks=[];started=time.monotonic()
    def stop(*_):raise SystemExit('signal requested owned cleanup')
    signal.signal(signal.SIGTERM,stop)
    try:
        for name in ('.speculative_minimal_gpu.lock','.shared-prefix-GPU.lock'):
            f=(RUNTIME/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        nv=bounded(NVML);resources=bounded(nv.sample);assert resources['memory_free_bytes']>=2*2**30
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        worker=Base.spawn(d,[sys.executable,'-B',str(OWN/'correctness.py'),str(out)],out/'worker.log',env)
        while worker.poll() is None:
            assert time.monotonic()-started<540,'finite numeric replay time cap'
            assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<128*2**20
            assert not (out/'STOP_REQUESTED').exists()
            time.sleep(.5)
        assert worker.returncode==0,'worker failed'
        r=json.loads((out/'WORKER-RESULT.json').read_text())
        assert r['status']=='COMPLETE_FINITE_KERNEL_NUMERIC_RESEARCH' and len(r['rows'])==9 and r['calls']==12
    except BaseException as exc:error=repr(exc)
    finally:
        try:
            closed=cleanup(d.roots,d.pins,deadline=started+600,wait_seconds=.5)
            if nv:
                resources=bounded(nv.sample)
                assert all(p['pid'] not in d.pins for key in ('compute','graphics') for p in resources[key])
                bounded(nv.close)
        except BaseException as exc:error=(error or '')+' cleanup:'+repr(exc)
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
        result=dict(status='COMPLETE_KERNEL_NUMERIC_RESEARCH' if error is None else 'FAILED_OR_BLOCKED',
            error=error,cleanup=closed,resources=resources,GPU_wall_budget_charge_sec=time.monotonic()-started,
            integration_permission=False,performance_claim=False)
        write(out/'RESULT.json',result);print(json.dumps(result))
    if error:raise RuntimeError(error)

if __name__=='__main__':main()

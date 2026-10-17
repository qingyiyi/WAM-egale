"""CPU initial-state identity check; no environment, model, or TEST outcome."""
import os,sys,json,hashlib,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RT=ROOT/'投机采样/results/runtime'
OUT=RT/'p0-tf-effect-initial-state-cpu-v1-20261016-r2'
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    OUT.mkdir();start=time.monotonic()
    first=RT/'p0-tf-effect-initial-state-cpu-v1-20261016-r1'
    remaining=300-(time.time()-(first/'PREREGISTRATION.json').stat().st_mtime)
    assert remaining>0 and (first/'FAILED.json').exists()
    write(OUT/'PREREGISTRATION.json',dict(goal='Check proposed new FIT30/31 CAL32 and DEV36/37 against prior0to29 exact initial-state identities within each task',
        commit='b0dfac3a02f65de33a1217246b83003615166640',CPU_affinity=sorted(os.sched_getaffinity(0)),threads=2,
        wall_cap_sec=300,storage_cap_bytes=32*2**20,GPU=0,environment_steps=0,model_forwards=0,
        independent_TEST_outcomes_read=False,inspected_indices=list(range(33))+[36,37],terminal_status='REGISTERED_CPU_ONLY',predecessor=str(first),budget_origin='first registration mtime cumulative300s',remaining_wall_sec=remaining,correction='trusted original LIBERO load environment'))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('CPU identity cap')));signal.alarm(max(1,int(remaining)))
    try:
        import torch
        torch.set_num_threads(2)
        sys.path.insert(0,str(ROOT/'third_party/LIBERO-official-20260922'))
        from libero.libero import benchmark
        suite=benchmark.get_benchmark('libero_10')();rows=[];all_unique=True
        for task in range(10):
            values=suite.get_task_init_states(task);assert len(values)>=38
            hashes={i:hashlib.sha256(values[i].contiguous().numpy().tobytes()).hexdigest() for i in list(range(33))+[36,37]}
            selected=[30,31,32,36,37]
            collisions=[dict(new=i,old=j) for i in selected for j in range(i) if j in hashes and hashes[i]==hashes[j]]
            all_unique &= not collisions
            rows.append(dict(task=task,task_name=suite.get_task(task).name,container_count=len(values),
                new_initial_state_hashes={str(i):hashes[i] for i in selected},collisions=collisions))
        assert not torch.cuda.is_initialized()
        result=dict(status='PASS_NEW_SPLIT_EXACT_INITIAL_STATE_IDENTITY' if all_unique else 'FAILED_EXACT_INITIAL_STATE_OVERLAP',
            rows=rows,all_new_initial_states_exact_unique_within_task=all_unique,
            note='Exact initial-state-vector identity only; does not prove statistical independence or upstream training absence',
            GPU=0,environment_steps=0,model_forwards=0,TEST_outcomes_read=False,elapsed_sec=time.monotonic()-start)
        write(OUT/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        import traceback
        write(OUT/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU=0));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()

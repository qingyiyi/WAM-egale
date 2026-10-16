"""Only real frozen long-KV kernel inputs; no timings or production integration."""
from pathlib import Path
import json,os,sys,time,traceback
OWN=Path(__file__).resolve().parent;ROOT=OWN.parents[2]
OBS=ROOT/'投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/observation'

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    os.sched_setaffinity(0,{12,13});out=Path(sys.argv[1]);started=time.monotonic()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import _wam_candidate_tile as candidate
    sys.path.insert(0,str(OWN.parent/'shared_prefix_torch_v2'))
    import _shared_torch_v1 as native
    result=dict(status='RUNNING',rows=[],calls=0,production_integrated=False,
        scope='three real long2160 attention layers only',performance_claim=False)
    def check():
        assert time.monotonic()-started<500 and not (out/'STOP_REQUESTED').exists()
        assert result['calls']<64
    def call(fn):
        check();result['calls']+=1;y=fn();torch.cuda.synchronize();return y
    def compare(a,b):
        a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous();d=a.float()-b.float()
        return dict(bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),
            finite=bool(torch.isfinite(a).all()),different=int(torch.count_nonzero(d)),
            elements=a.numel(),max_abs=float(d.abs().max()),
            relative_RMS=float(d.square().mean().sqrt()/(b.float().square().mean().sqrt()+1e-30)))
    try:
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        for layer in (0,8,29):
            x=torch.load(OBS/f'layer{layer:02d}-tensors.pt',map_location='cpu',weights_only=True)
            meta=json.loads((OBS/f'layer{layer:02d}-metadata.json').read_text())
            q,ck,cv,pk,pv=[x[k].cuda() for k in ('q','current_k','current_v','prefix_k','prefix_v')]
            assert q.shape==(14,16,24,128) and pk.shape==(2,2144,24,128) and pv.shape==pk.shape
            assert all(bool(torch.isfinite(t).all()) for t in (q,ck,cv,pk,pv))
            host_slots=torch.tensor(meta['valid_slots'],dtype=torch.long)
            assert host_slots.numel()==2144 and bool((host_slots>=0).all()) and bool((host_slots<2160).all())
            assert bool((host_slots[1:]>host_slots[:-1]).all())
            slots=host_slots.cuda();rows=torch.tensor([0]*7+[1]*7,device='cuda')
            poolk=torch.full((2,2160,24,128),float('nan'),dtype=q.dtype,device='cuda');poolv=torch.full_like(poolk,float('nan'))
            poolk[:,slots]=pk;poolv[:,slots]=pv
            oracle=call(lambda:native.shared(q,poolk,poolv,ck,cv,slots,rows))
            assert compare(oracle,x['original_out'])['bitwise'],'original native no longer reproduces original fixture'
            assert bool(torch.isfinite(oracle).all()),'original native produced a nonfinite output'
            for group in (1,2,4):
                y=call(lambda:candidate.grouped(q,poolk,poolv,ck,cv,slots,group))
                row=dict(layer=layer,n_v=7,group=group,Q_per_candidate=16,BM=64,BN=64,
                    CTAs=2*((7+group-1)//group)*24,output=compare(y,oracle),
                    virtual_private_suffix=True,other_candidate_private_keys_masked=True,
                    algebraic_semantics_not_assumed_BF16_equal=True)
                result['rows'].append(row);print(json.dumps(row),flush=True)
            for actual,key in ((q,'q'),(ck,'current_k'),(cv,'current_v'),(poolk[:,slots],'prefix_k'),(poolv[:,slots],'prefix_v')):
                assert compare(actual,x[key])['bitwise'],'actual valid input mutated by custom kernel'
            assert torch.equal(slots.cpu(),host_slots) and torch.equal(rows.cpu(),torch.tensor([0]*7+[1]*7))
            del x,q,ck,cv,pk,pv,poolk,poolv,oracle,y
            torch.cuda.empty_cache()
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        groups={str(g):all(r['output']['bitwise'] and r['output']['finite'] for r in result['rows'] if r['group']==g) for g in (1,2,4)}
        result.update(status='COMPLETE_FINITE_KERNEL_NUMERIC_RESEARCH',all_real_inputs_and_RNG_unchanged=True,
            group_exact_on_three_long_fixtures=groups,
            eligible_for_further_numeric_fixtures=[g for g in (2,4) if groups[str(g)]],
            integration_permission=False,short_middle_block_verifier_chunk_episode_unvalidated=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        result.update(status='FAILED_KERNEL_REPLAY',error=repr(exc),traceback=traceback.format_exc());raise
    finally:write(out/'WORKER-RESULT.json',result)

if __name__=='__main__':main()

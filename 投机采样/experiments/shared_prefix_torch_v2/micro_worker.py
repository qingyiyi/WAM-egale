"""Same real Q/K/V: original torch vs literal original math vs shared addresses."""
import argparse
import json
import os
from pathlib import Path
import time
from task_contract import OWN, OBS, digest, gpu_entry_gate, write, stamp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', required=True)
    job = Path(ap.parse_args().job).resolve()
    p = gpu_entry_gate(job, claim=True) # Before importing torch or the CUDA extension.
    import torch
    import _shared_torch_v1 as native
    from direct_counter import Counter
    import sys
    sys.path.append(str(OWN.parent / 'shared_prefix_observe_v2'))
    from observation import numerical, flags
    deadline = json.loads((job / 'GPU-ATTEMPT.json').read_text())['monotonic_deadline'] - 15
    def check():
        if time.monotonic() >= deadline or (job / 'STOP_REQUESTED').exists():
            raise TimeoutError('micro cleanup-reserved deadline/stop')
    identity = dict(job=str(job), plan_sha256=digest(job / 'plan.json'), kind=p['execution_revision'])
    counter = Counter(job / 'direct-counter.jsonl', identity, cap=24, check=check)
    records = []
    initial_flags = flags()
    try:
        for layer in p['layers']:
            check()
            payload = torch.load(OBS / 'observation' / f'layer{layer:02d}-tensors.pt', map_location='cpu', weights_only=True)
            meta = json.loads((OBS / 'observation' / f'layer{layer:02d}-metadata.json').read_text())
            q, ck, cv, pk, pv = [payload[k].cuda() for k in ('q','current_k','current_v','prefix_k','prefix_v')]
            rows = torch.tensor([0]*7+[1]*7,device='cuda',dtype=torch.long)
            slots = torch.tensor(meta['valid_slots'],device='cuda',dtype=torch.long)
            compact = torch.arange(pk.shape[1],device='cuda',dtype=torch.long)
            full_pk = torch.full((2,2160,24,128),float('nan'),dtype=q.dtype,device=q.device)
            full_pv = torch.full_like(full_pk,float('nan'))
            full_pk[:,slots] = pk; full_pv[:,slots] = pv
            # Fourteen-row materialization belongs only to the original reference.
            k = torch.cat((pk[rows],ck),dim=1); v = torch.cat((pv[rows],cv),dim=1)
            torch_sdpa = lambda Q,K,V: torch.nn.functional.scaled_dot_product_attention(
                Q.transpose(1,2),K.transpose(1,2),V.transpose(1,2)).transpose(1,2)
            call = lambda name, fn: counter.call(f'layer{layer}:{name}', fn)
            versions = [x._version for x in (q,ck,cv,pk,pv,rows,slots,full_pk,full_pv)]
            ref = call('original_torch_default',lambda: torch_sdpa(q,k,v))
            repeated = call('original_torch_repeat',lambda: torch_sdpa(q,k,v))
            dense = call('original_math_dense',lambda: native.dense(q,k,v))
            shared = call('shared_compact_CFG2',lambda: native.shared(q,pk,pv,ck,cv,compact,rows))
            masked = call('shared_masked_original_pool',lambda: native.shared(q,full_pk,full_pv,ck,cv,slots,rows))
            repeated_shared = call('shared_repeat',lambda: native.shared(q,full_pk,full_pv,ck,cv,slots,rows))
            changed_cv = cv.clone(); changed_cv[0].add_(.5)
            changed_v = torch.cat((pv[rows],changed_cv),dim=1)
            changed_ref = call('original_torch_current_row0_changed',lambda: torch_sdpa(q,k,changed_v))
            changed_shared = call('shared_current_row0_changed',lambda: native.shared(q,full_pk,full_pv,ck,changed_cv,slots,rows))
            torch.cuda.synchronize()
            comparisons = {name:numerical(value,ref) for name,value in (
                ('original_torch_repeat',repeated),('original_math_dense',dense),
                ('shared_compact_CFG2',shared),('shared_masked_original_pool',masked),
                ('shared_repeat',repeated_shared))}
            frozen_compare = numerical(ref,payload['original_out'])
            isolation = dict(perturbed_matches_original_torch=numerical(changed_shared,changed_ref),
                             other13_bitwise=torch.equal(changed_shared[1:].view(torch.uint8),masked[1:].view(torch.uint8)),
                             changed_row_nontrivial=not torch.equal(changed_shared[0],masked[0]))
            row = dict(layer=layer, comparisons=comparisons, original_matches_frozen=frozen_compare,
                       isolation=isolation, flags_unchanged=flags()==initial_flags,
                       inputs_versions_unchanged=versions==[x._version for x in (q,ck,cv,pk,pv,rows,slots,full_pk,full_pv)],
                       invalid_slots_poisoned_with_NaN=True, shared_history_rows=2, candidate_current_rows=14)
            write(job / f'layer{layer:02d}.json',row,exclusive=True); records.append(row)
            print(json.dumps(dict(event='LAYER_COMPARED',layer=layer,bitwise={k:v['bitwise'] for k,v in comparisons.items()})),flush=True)
        passed = all(all(v['bitwise'] and v['finite'] for v in r['comparisons'].values())
                     and r['original_matches_frozen']['bitwise'] and r['flags_unchanged'] and r['inputs_versions_unchanged']
                     and r['isolation']['perturbed_matches_original_torch']['bitwise']
                     and r['isolation']['other13_bitwise'] and r['isolation']['changed_row_nontrivial'] for r in records)
        write(job / 'MICRO-RESULT.json',dict(status='PASS' if passed else 'FAIL_NUMERICAL',timestamp=stamp(),
             layers=records,counter=counter.summary(),target_forwards=0,
             compiled_cuda=[native.compiled_cuda_major,native.compiled_cuda_minor],
             wheel_cuda=torch.version.cuda,full_model_acceptance=False),exclusive=True)
    finally:
        counter.close()


if __name__ == '__main__':
    main()

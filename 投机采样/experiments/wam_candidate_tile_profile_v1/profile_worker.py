"""Three actual attention launches; profiler counters only, no formal timings."""
import hashlib
import json
import os
from pathlib import Path
import sys

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = OWN.parents[2]
OBS = ROOT / '投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/observation'


def main():
    os.sched_setaffinity(0, {12, 13})
    out = Path(sys.argv[1])
    plan = json.loads((out / 'PLAN.json').read_text())
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(EXP / 'shared_prefix_torch_v2'))
    import _shared_torch_v1 as native
    sys.path.insert(0, str(EXP / 'wam_candidate_tile_v1'))
    import _wam_candidate_tile as candidate
    for module in (native, candidate):
        path = Path(module.__file__).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == plan['sources'][str(path)]
    x = torch.load(OBS / 'layer00-tensors.pt', map_location='cpu', weights_only=True)
    meta = json.loads((OBS / 'layer00-metadata.json').read_text())
    q, ck, cv, pk, pv = [x[k].cuda() for k in ('q', 'current_k', 'current_v', 'prefix_k', 'prefix_v')]
    assert q.shape == (14, 16, 24, 128) and pk.shape == pv.shape == (2, 2144, 24, 128)
    assert all(bool(torch.isfinite(t).all()) for t in (q, ck, cv, pk, pv))
    host_slots = torch.tensor(meta['valid_slots'], dtype=torch.int64)
    assert host_slots.numel() == 2144 and bool((host_slots[1:] > host_slots[:-1]).all())
    assert bool((host_slots >= 0).all()) and bool((host_slots < 2160).all())
    slots = host_slots.cuda()
    rows = torch.tensor([0] * 7 + [1] * 7, device='cuda')
    poolk = torch.full((2, 2160, 24, 128), float('nan'), dtype=q.dtype, device='cuda')
    poolv = torch.full_like(poolk, float('nan'))
    poolk[:, slots], poolv[:, slots] = pk, pv
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    tracked = (q, ck, cv, poolk, poolv, slots, rows)
    versions = [(t, t._version) for t in tracked]

    def exact(a, b):
        a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    engines = [('native_BM128', lambda: native.shared(q, poolk, poolv, ck, cv, slots, rows)),
               ('G1_BM64', lambda: candidate.grouped(q, poolk, poolv, ck, cv, slots, 1)),
               ('G2_BM64', lambda: candidate.grouped(q, poolk, poolv, ck, cv, slots, 2))]
    result = []
    for name, call in engines:
        assert not (out / 'STOP_REQUESTED').exists()
        for _ in range(2):
            warm = call()
        torch.cuda.synchronize()
        assert exact(warm, x['original_out']) and bool(torch.isfinite(warm).all())
        torch.cuda.nvtx.range_push('CANDIDATE_PROFILE')
        value = call()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        assert exact(value, x['original_out']) and bool(torch.isfinite(value).all())
        result.append(dict(arm=name, frozen_bitwise=True, finite=True))
    for value, key in ((q, 'q'), (ck, 'current_k'), (cv, 'current_v'),
                       (poolk[:, slots], 'prefix_k'), (poolv[:, slots], 'prefix_v')):
        assert exact(value, x[key])
    assert exact(slots, host_slots) and exact(rows, torch.tensor([0] * 7 + [1] * 7))
    assert all(t._version == version for t, version in versions)
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
    with (out / 'INPUTS.json').open('x') as f:
        json.dump(dict(status='COMPLETE_REAL_ATTENTION_PROFILE_INPUTS', rows=result,
            valid_KV_Q_current_KV_slots_CFG_RNG_versions_unchanged=True,
            python_attention_calls=9, profiled_launches=3,
            profiler_internal_replays_not_in_python_call_count=True,
            source=str(OBS / 'layer00-tensors.pt'), profile_only=True), f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()

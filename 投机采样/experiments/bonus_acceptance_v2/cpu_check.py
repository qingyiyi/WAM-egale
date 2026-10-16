"""Real tiny production BF16 K2+bonus helper equivalence, with no CUDA."""
import argparse
import copy
import json
import os
import tempfile
from types import SimpleNamespace
import threading
import time
from collections import Counter
from pathlib import Path

from paths import ROOT, OWN, BASE, load
from experiment import write, stamp, assignments, fixed_slots
from selection import install, NAME, choose, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', choices=['shared'], default='shared')
    args = ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only boundary')
    f = load('test_cpu', BASE / 'test_cpu.py')
    t = f.torch
    from production_fixture import loaded_model
    from measured_online import control, INSTALLS
    from online import OnlineShared
    from template import stats
    import buffers
    checks, pairs = [], []
    started = time.monotonic()
    for path in OWN.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    checks.append('private_source_syntax')
    for name in ['paths.py', 'template.py', 'scalar.py', 'buffers.py', 'online.py',
                 'measured_online.py', 'state_check.py', 'runtime.py', 'cleanup.py',
                 'prefix_kernel.py']:
        assert (OWN / name).read_bytes() == (OWN.parent / 'bonus_e2e_v1' / name).read_bytes()
    checks.append('accepted_O1_O2_O3_core_and_original_client_byte_identical')
    assert sum(x['cap'] for wave in assignments() for x in wave)+640 == 16640
    assert len(fixed_slots())==4
    from types import SimpleNamespace
    from itertools import product
    api=load('accept_original_api',ROOT/'投机采样/scripts/p0_batched_verify.py')
    tested=0
    for costs in product((0.,.5,1.,1.5), repeat=6):
        scores=SimpleNamespace(first_costs=costs[:2],second_costs=(costs[2:4],costs[4:6]))
        g=api.select_k2_b2_route(scores,selector='global'); d=choose(api,scores,1.)
        gd=depth(g.first_cost,g.second_cost,1.); dd=depth(d.first_cost,d.second_cost,1.)
        assert dd>=gd
        if dd==gd: assert (d.parent,d.child)==(g.parent,g.child)
        tested+=1
    checks.append('4096_exhaustive_same_tree_depth_non_decrease_global_tie_and_boundary_cases')
    calls=[]
    dummy=SimpleNamespace(_load_p0_shadow_api=lambda:api, infer=lambda request:calls.append(request) or request,
                          _p0_action_draft_threshold=1.)
    proxy=install(dummy,'global')
    request={'p0_trace_context':{'rollout_chunk':7,'env_steps_before':108},'obs':'untouched'}
    assert dummy.infer(request) is request and proxy.context==request['p0_trace_context']
    assert not hasattr(dummy,'_p0_teacher_context')
    dummy.infer({'reset':True}); assert proxy.context=={} and len(calls)==2
    checks.append('request_context_observer_with_teacher_absent_policy_reset_and_transport_unchanged')
    m = loaded_model(dtype=t.bfloat16)
    with tempfile.TemporaryDirectory(prefix='bonus-BF16-CPU-') as tmp:
        s = f.reset_server(m, tmp)
        s.dtype = t.bfloat16
        for b in m.blocks:
            c = b.attn1.attn_caches['pos']
            c['k'], c['v'] = c['k'].to(t.bfloat16), c['v'].to(t.bfloat16)
        s.prompt_embeds = s.prompt_embeds.to(t.bfloat16)
        s.negative_prompt_embeds = s.negative_prompt_embeds.to(t.bfloat16)
        s._p0_state_teacher = None
        s._cache_mode = 'original'
        if hasattr(s, '_p0_teacher_context'): del s._p0_teacher_context
        s._p0_action_draft_selector = 'global'
        s._p0_action_draft_mode = 'verifier_k2_b2_bonus'
        s._p0_action_draft_b2_checkpoint = str(ROOT / '投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/training/A_off.pt')
        s.action_scheduler = f.public_server.FlowMatchScheduler(shift=3., sigma_min=0., extra_one_step=True)
        proxy = install(s, 'global')
        buffers.prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
        s._load_p0_action_draft_b2_predictor()
        initial = f.cache_state(m)
        actions = t.randn(1, 30, 4, 4, 1, dtype=t.bfloat16)
        original = s._infer_action_verifier_k2_b2
        methods = [b.attn1.forward for b in m.blocks]

        def run(optimized, frame, threshold, selector):
            proxy.effective=selector
            proxy.records.clear()
            proxy.pending=None
            s._p0_action_draft_selector=selector
            for b, saved in zip(m.blocks, initial):
                f.SNAPSHOT.restore_cache(b.attn1.attn_caches['pos'], saved)
            s.action_scheduler.set_timesteps(50)
            times = t.nn.functional.pad(s.action_scheduler.timesteps, (0, 1))
            scheduler = copy.deepcopy(vars(s.action_scheduler))
            rng = t.get_rng_state().clone()
            s.frame_st_id, s._p0_action_draft_threshold = frame, threshold
            counts = Counter()
            s._phase_profile_counter = lambda name, value=1: counts.update({name: value})
            c = (control(s, 'O3', allow_cuda=False, cpu_reference_enabled=True) if optimized
                 else OnlineShared(s, enabled=True, allow_cuda=False, cpu_reference_enabled=True).install()
                 if args.reference == 'shared' else None)
            result = {}

            def infer():
                try:
                    with t.no_grad():
                        result['out'] = s._infer_action_verifier_k2_b2(
                            actions.clone(), times, frame, 50, 'CPU', None, bonus_target_step=True)
                except BaseException as exc:
                    result['error'] = exc

            try:
                thread = threading.Thread(target=infer)
                thread.start()
                thread.join(timeout=120)
                assert not thread.is_alive()
                if 'error' in result:
                    raise result['error']
                if c is not None:
                    assert c.last_clean and c.control is None
                assert buffers.ACTIVE.get() is None and t.equal(rng, t.get_rng_state())
                for k, v in scheduler.items():
                    now = vars(s.action_scheduler)[k]
                    assert t.equal(v, now) if isinstance(v, t.Tensor) else v == now
                assert counts['committed_scheduler_update_count'] == 50
                return result['out'], f.cache_state(m), dict(counts), list(proxy.records)
            finally:
                if c is not None:
                    c.close()
                assert s._infer_action_verifier_k2_b2 == original
                assert [b.attn1.forward for b in m.blocks] == methods

        for selector in ('global',NAME):
            for frame in (0,4):
                for threshold in (0.,1e6,.007839680526450805):
                    a,ca,pa,ra=run(False,frame,threshold,selector)
                    b,cb,pb,rb=run(True,frame,threshold,selector)
                    assert t.equal(a,b) and f.same_state(ca,cb) and pa==pb and ra==rb, dict(selector=selector, frame=frame, threshold=threshold, output=t.equal(a,b), KV=f.same_state(ca,cb), counters=pa==pb, records=ra==rb, record_counts=[len(ra),len(rb)])
                    assert len(ra)==pa['tree_count']
                    assert a.shape==(1,30,4,4,1)
                    if frame==0: assert t.count_nonzero(a[:,:,:1])==0
                    if threshold==1e6: assert pa['bonus_target_scheduler_update_count']==16
                    if threshold==0.: assert pa.get('bonus_target_scheduler_update_count',0)==0
                    pairs.append(dict(selector=selector,frame=frame,threshold=threshold,full30_BF16_bitwise=True,
                        all30_KV_bitwise=True,scheduler_RNG_equal=True,records_exact=True,phase_counters=pa,
                        output_hash=f.SNAPSHOT.fingerprint(a)))
        checks += ['six_original_helper_vs_O3_bonus_full50_BF16_pairs_with_same_attention_backend',
                   'first_and_later_frame_projection_terminal_tail_and_bonus_counts',
                   'all30_KV_scheduler_RNG_routes_equal_and_lifecycle_restored']
        assert len(INSTALLS) == 12 and stats()['templates'] == 1
        assert all(x['AST_builds_observed'] == 0 for x in INSTALLS[1:])
        checks.append('warm_bonus_reuses_compiled_template_and_workspace_cleanup')
    assert not t.cuda.is_initialized() and not f.CUDA_ATTEMPTS
    checks.append('no_CUDA_initialization_or_attempt')
    write(args.out, dict(status='PASS_CPU_BONUS_ACCEPTANCE_FULL50_BF16_PARITY',
        at=stamp(), checks=checks, checks_run=len(checks), pairs=pairs,
        reference_attention_backend=args.reference,
        raw_unshared_original_Torch_bonus_GPU_gate_still_required=True,
        CUDA_initialized=False, CUDA_attempts=[], elapsed_seconds=time.monotonic()-started),
        exclusive=True, durable=True)
    print(json.dumps(dict(status='PASS', checks=len(checks), pairs=len(pairs))))


if __name__ == '__main__':
    main()

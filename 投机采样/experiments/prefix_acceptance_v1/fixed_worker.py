"""Eight complete numerical oracles, no speed measurement or new EGL."""
import copy
import sys
import traceback
from pathlib import Path

from experiment import worker_entry, write, stamp
from paths import WORKTREE, EXP, S0, load


def run(s, source, job, out, p):
    import torch
    from runtime import Runtime
    from measured_online import control
    from selection import install
    import buffers
    import state_check
    snap = load('accept_snapshot', EXP / 'fixed_load_cost/snapshot.py')
    validate = load('accept_validator', S0 / 'server_entry.py')
    validate.validate_server(s)
    s.infer(dict(reset=True, prompt=None))
    s._load_p0_action_draft_b2_predictor().model.to(s.device).eval()
    proxy = install(s, 'depth_then_cost')
    buffers.prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
    rt = Runtime(s, out, p['fixed_target_cap'], shared=False, allow_cuda=True)
    save_original, saved = source.save_async, {}

    def save(value, path):
        name = Path(path).name
        key = ('video' if name.startswith('latents_') else
               'normalized_action' if name.startswith('actions_') else None)
        if key is None:
            raise ValueError('unexpected original artifact')
        saved[key] = value
        return save_original(value, path)

    source.save_async = save
    methods = [b.attn1.forward for b in s.transformer.blocks]
    details = []
    try:
        for ref in p['snapshots']:
            payload = torch.load(ref['path'], map_location='cpu', weights_only=False)
            if (snap.fingerprint(payload['state']) != ref['state_hash'] or
                    snap.fingerprint(payload['request']) != ref['request_hash']):
                raise ValueError('frozen real-state input changed')
            if any(c['slots'].numel() != 2016 or c['pool_shape'][1] != 2160
                   for c in payload['state']['caches']):
                raise ValueError('original long KV geometry changed')
            expected, end = {}, {}
            initial = None
            for item in p['fixed_slots']:
                selector = item['selector']
                rt.shared.close()
                proxy.effective = s._p0_action_draft_selector = selector
                proxy.records.clear()
                proxy.pending = None
                snap.restore(s, payload['state'])
                if initial is None:
                    if snap.fingerprint(snap.capture(s)) != ref['state_hash']:
                        raise ValueError('snapshot restore failed')
                    initial = state_check.references(s.transformer, s.cache_name)
                if (not state_check.equal(s.transformer, initial, s.cache_name) or
                        snap.fingerprint(state_check.capture_light(s)) !=
                        snap.fingerprint(state_check.light(payload['state']))):
                    raise ValueError('input all30 KV/metadata/scheduler/RNG mismatch')
                if item['optimized']:
                    rt.shared = control(s, 'O3', allow_cuda=True)
                folder = out / 'calls' / f"init{ref['init']}" / item['arm']
                folder.mkdir(parents=True)
                s.exp_save_root = str(folder)
                saved.clear()
                before = dict(rt.phase)
                returned = s.infer(copy.deepcopy(payload['request']))
                torch.cuda.synchronize()
                shared = rt.shared.summary()
                rt.shared.close()
                if [b.attn1.forward for b in s.transformer.blocks] != methods:
                    raise ValueError('attention lease not restored')
                if set(saved) != {'video', 'normalized_action'}:
                    raise ValueError('original full outputs missing')
                data = dict(saved=snap.cpu(saved), returned=snap.cpu(returned),
                            final_state_light=state_check.capture_light(s))
                phase = {k: v-before.get(k, 0) for k, v in rt.phase.items() if v-before.get(k, 0)}
                if phase.get('committed_scheduler_update_count') != 50:
                    raise ValueError('full50 action updates required')
                if len(proxy.records) != phase['tree_count'] or proxy.pending is not None:
                    raise ValueError('per-tree observation lost')
                identity = dict(data_hash=snap.fingerprint(data), phase=phase,
                    records=copy.deepcopy(proxy.records), forwards=rt.requests[-1]['forward_intents'])
                if not item['optimized']:
                    expected[selector] = identity
                    end[selector] = state_check.references(s.transformer, s.cache_name)
                    if selector == 'depth_then_cost':
                        old = torch.load(Path(p['prior_quality_job']) / 'fixed/worker0' /
                            f"init{ref['init']}-RAW_DEPTH-output.pt", map_location='cpu', weights_only=False)
                        if snap.fingerprint(data) != snap.fingerprint(old):
                            raise ValueError('observer changed archived current DEPTH math/RNG')
                elif (identity != expected[selector] or
                      not state_check.equal(s.transformer, end[selector], s.cache_name)):
                    raise ValueError('raw Torch vs shared O3 bonus all30/output/routes/RNG mismatch')
                if item['optimized'] and (not shared['last_request_clean'] or shared['candidates'] <= 0):
                    raise ValueError('shared verifier unused or leaked')
                torch.save(data, out / f"init{ref['init']}-{item['arm']}-output.pt")
                detail = dict(init=ref['init'], **item, **identity,
                    all30_GPU_KV_metadata_equal=True, shared=shared,
                    observation_only_no_speed_claim=True, at=stamp())
                details.append(detail)
                write(out / f"init{ref['init']}-{item['arm']}.json", detail, exclusive=True, durable=True)
            del payload, expected, end, initial
        save_original.__globals__['executor'].shutdown(wait=True)
        torch.cuda.synchronize()
        rt.close()
        write(out / 'WORKER-RESULT.json', dict(status='PASS_FIXED_ACCEPTANCE_ORACLES',
            chunks=len(details), details=details, target_intents=rt.attempted,
            current_DEPTH_archived_output_exact=True,
            raw_vs_shared_O3_all30_KV_output_routes_RNG_exact_both_selectors=True,
            speed_comparison_deferred=True), exclusive=True, durable=True)
    finally:
        source.save_async = save_original


def main():
    job, out, p = worker_entry('fixed')
    sys.path.insert(0, str(WORKTREE))
    from wan_va import wan_va_server as source

    def execute(s, rank, host, port):
        try:
            if rank != 0:
                raise ValueError('one model rank required')
            run(s, source, job, out, p)
        except BaseException as exc:
            write(out / 'WORKER-FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc()),
                  exclusive=True, durable=True)
            raise

    source.run_async_server_mode = execute
    source.init_logger()
    source.main()


if __name__ == '__main__':
    main()

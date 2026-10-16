"""Four TF candidate teacher OFF/ON oracles, no timing or EGL."""
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
    from candidate_predictor import install as install_candidate
    if hasattr(s, 'action_per_frame'):
        raise ValueError('new native cold-install proof must start before original reset')
    predictor = install_candidate(s,p,allow_device_transfer=True)
    cold_status = predictor.layout_status()
    if cold_status['status'] != 'AWAITING_ORIGINAL_RESET_NO_GEOMETRY_CLAIM':
        raise ValueError('cold geometry must be pending without fabricated layout')
    reset_original = s._reset.__func__
    s.infer(dict(reset=True, prompt=None))
    if s._reset.__func__ is not reset_original:
        raise ValueError('original reset was replaced')
    reset_status = predictor.validate_layout()
    write(out/'COLD-INSTALL-PROOF.json',dict(status='PASS_NATIVE_COLD_INSTALL_ORIGINAL_RESET',
        constructor_had_no_action_per_frame=True,install_before_original_reset=True,
        original_public_infer_reset_unchanged=True,cold_status=cold_status,
        original_reset_status=reset_status,candidate_checkpoint_sha256=predictor.checkpoint_sha256,
        additional_target_forwards=0,new_source_requires_new_four_GPU_oracles=True),exclusive=True,durable=True)
    proxy = install(s, 'prefix_then_cost')
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
                selector = 'prefix_then_cost'
                checkpoint_key = item['checkpoint']
                if checkpoint_key!='NEW':raise ValueError('candidate-only no-mutation oracle')
                s._p0_state_teacher = None
                collector = None
                if item['teacher']:
                    from collector import install as install_labels
                    collector = install_labels(s,out/'oracle_teacher'/f"init{ref['init']}-{item['arm']}",dict(task=0,init=ref['init'],split='oracle',draft_sha256=p['new_checkpoint_sha256'],plan_sha256=__import__('experiment').digest(job/'plan.json'),target_checkpoint_id=p['frozen_reference']['target_checkpoint_id']))
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
                request = copy.deepcopy(payload['request'])
                request['p0_trace_context'] = dict(campaign_id=job.name,parallel_worker_id='fixed',episode_attempt_id=f"oracle-init{ref['init']}",task_index=0,episode_index=ref['init'],rollout_chunk=14)
                returned = s.infer(request)
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
                if checkpoint_key not in expected:
                    expected[checkpoint_key] = identity
                    end[checkpoint_key] = state_check.references(s.transformer, s.cache_name)
                elif (identity != expected[checkpoint_key] or
                      not state_check.equal(s.transformer, end[checkpoint_key], s.cache_name)):
                    raise ValueError('teacher/cache variant changed output/KV/routes/RNG or forwards')
                if collector is not None:
                    if collector.errors or collector.total != phase['tree_count']:
                        raise ValueError('same-query GPU teacher data incomplete: '+repr(collector.errors))
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
        write(out / 'WORKER-RESULT.json', dict(status='PASS_FIXED_TEACHER_TRAIN_ORACLES',
            chunks=len(details), details=details, target_intents=rt.attempted,
            same_candidate_teacher_OFF_ON_exact=True,
            teacher_variants_all30_output_routes_RNG_exact=True,
            new_native_cold_install_proof=True,
            previous_four_GPU_oracles_reused=False,
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

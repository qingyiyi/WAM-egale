"""Original server; bonus mode only, with frozen O1/O2/O3 action control."""
import os
import sys
import threading
import time
import traceback

from experiment import worker_entry, WORKTREE, write, stamp, MODES, SELECTORS
from paths import S0, load


def main():
    job, out, p = worker_entry()
    sys.path.insert(0, str(WORKTREE))
    from wan_va import wan_va_server as source
    from runtime import Runtime
    from measured_online import control, INSTALLS
    from template import stats
    import buffers
    import scalar
    validate = load('bonus_frozen_server_validator', S0 / 'server_entry.py')
    original_run = source.run_async_server_mode

    def serve(s, local_rank, host, port):
        validate.validate_server(s)
        from candidate_predictor import install as install_candidate
        predictor = install_candidate(s,p,allow_device_transfer=True)
        arm = os.environ['P0_SPEC_E2E_ARM']
        item = next(x for wave in p['assignments'] for x in wave if x['id'] == os.environ['DRAFT_TRAIN_ID'])
        init = item['init']
        if local_rank != 0 or s._p0_action_draft_mode != MODES[arm]:
            raise ValueError('unchanged server rank and exact bonus/off mode required')
        setup = time.monotonic()
        from selection import install
        proxy = install(s, SELECTORS[arm]) if arm != 'T' else None
        if arm != 'T':
            buffers.prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
        cap = item['cap']
        rt = Runtime(s, out, cap, shared=False, allow_cuda=True)
        labels = None
        if p['stage'] == 'collection':
            from collector import install as install_labels
            labels = install_labels(s, out / 'teacher', dict(item, target_checkpoint_id=p['frozen_reference']['target_checkpoint_id'], draft_sha256=p['new_checkpoint_sha256'], plan_sha256=__import__('experiment').digest(job/'plan.json')))
        if arm != 'T':
            rt.shared.close()
            rt.shared = control(s, 'O3', allow_cuda=True)
        write(out / 'READY.json', dict(status='PASS', pid=os.getpid(), arm=arm, init=init,
            mode=MODES[arm], effective_selector=SELECTORS[arm], bonus_enabled=(arm != 'T'), speed_comparison_deferred=True, port=port, master_port=os.environ['MASTER_PORT'],
            target_device=str(s.device), vae_device=str(s.vae_device), EGL=os.environ['MUJOCO_GL'],
            profile_OFF=True, teacher_OFF=(labels is None), trace_OFF=True,
            candidate_checkpoint_sha256=predictor.checkpoint_sha256,
            candidate_layout_status=predictor.layout_status(),
            candidate_geometry_guard='read original reset geometry before every predict; no private geometry assignment',
            private_setup_seconds=time.monotonic()-setup), exclusive=True)

        def finish():
            while not (out / 'FINISH_REQUESTED').exists():
                time.sleep(.5)
            try:
                source.save_async.__globals__['executor'].shutdown(wait=True)
                source.torch.cuda.synchronize()
                rt.close()
                if labels is not None:
                    if labels.errors or labels.total<=0:
                        raise ValueError('teacher collection incomplete: '+repr(labels.errors))
                    write(out/'LABELS-SUMMARY.json',dict(saved_trees=labels.total, errors=labels.errors, extra_teacher_forwards=0),exclusive=True,durable=True)
                write(out / 'ACCEPTANCE-TREES.json', dict(schema='bonus-acceptance-trees/v1', selector=SELECTORS[arm], threshold=s._p0_action_draft_threshold, records=proxy.records if proxy else [], extra_target_forwards=0, CPU_costs_only=True), exclusive=True, durable=True)
                write(out / 'OPTIMIZATION-STATS.json', dict(arm=arm, template_stats=stats(),
                    scalar_stats=scalar.stats(), installs=INSTALLS, workspaces=buffers.EVENTS),
                    exclusive=True)
                write(out / 'FINISH_READY.json', dict(status='PASS', timestamp=stamp(),
                    async_artifacts_drained=True, counter_closed=True), exclusive=True, durable=True)
            except BaseException as exc:
                write(out / 'FINISH_FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc()),
                      exclusive=True, durable=True)

        threading.Thread(target=finish, daemon=True).start()
        return original_run(s, local_rank, host, port)

    source.run_async_server_mode = serve
    source.init_logger()
    source.main()


if __name__ == '__main__':
    main()

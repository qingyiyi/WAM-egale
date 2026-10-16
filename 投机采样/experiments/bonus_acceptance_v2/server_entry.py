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
        arm, init = os.environ['P0_SPEC_E2E_ARM'], int(os.environ['BONUS_ACCEPT_INIT'])
        if local_rank != 0 or s._p0_action_draft_mode != MODES[arm]:
            raise ValueError('unchanged server rank and exact bonus/off mode required')
        setup = time.monotonic()
        from selection import install
        proxy = install(s, SELECTORS[arm])
        if arm in MODES:
            buffers.prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
        cap = next(x['cap'] for wave in p['assignments'] for x in wave
                   if (x['arm'], x['init']) == (arm, init))
        rt = Runtime(s, out, cap, shared=False, allow_cuda=True)
        if arm in MODES:
            rt.shared.close()
            rt.shared = control(s, 'O3', allow_cuda=True)
        write(out / 'READY.json', dict(status='PASS', pid=os.getpid(), arm=arm, init=init,
            mode=MODES[arm], effective_selector=SELECTORS[arm], bonus_enabled=True, speed_comparison_deferred=True, port=port, master_port=os.environ['MASTER_PORT'],
            target_device=str(s.device), vae_device=str(s.vae_device), EGL=os.environ['MUJOCO_GL'],
            profile_OFF=True, teacher_OFF=True, trace_OFF=True,
            private_setup_seconds=time.monotonic()-setup), exclusive=True)

        def finish():
            while not (out / 'FINISH_REQUESTED').exists():
                time.sleep(.5)
            try:
                source.save_async.__globals__['executor'].shutdown(wait=True)
                source.torch.cuda.synchronize()
                rt.close()
                write(out / 'ACCEPTANCE-TREES.json', dict(schema='bonus-acceptance-trees/v1', selector=SELECTORS[arm], threshold=s._p0_action_draft_threshold, records=proxy.records, extra_target_forwards=0, CPU_costs_only=True), exclusive=True, durable=True)
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

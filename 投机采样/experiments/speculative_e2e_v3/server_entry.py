"""Owned full-episode model server. Validate private gate before WAM/CUDA import."""
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from task_contract import worker_entry, WORKTREE, write, stamp

def validate_server(s):
    if (s._cache_mode != 'original' or len(s.transformer.blocks) != 30
            or s.streaming_vae_half is not None or s._p0_trace is not None
            or s._p0_state_teacher is not None or s._nsys_profile_mode != 'off'
            or s._phase_profile_enabled
            or s.job_config.action_num_inference_steps != 50
            or s.job_config.num_inference_steps != 20
            or s.job_config.action_guidance_scale != 1 or s.job_config.guidance_scale != 5):
        raise ValueError('original frozen model/config boundary mismatch')

def main():
    job, out, p = worker_entry()
    sys.path.insert(0,str(WORKTREE))
    from wan_va import wan_va_server as source
    from runtime import Runtime
    original_run = source.run_async_server_mode
    def serve(s, local_rank, host, port):
        if local_rank != 0:
            raise ValueError('one model rank per independent worker')
        validate_server(s)
        arm = os.environ['P0_SPEC_E2E_ARM']
        if arm not in p['arms']:
            raise ValueError('owned arm mismatch')
        expected = 'off' if arm == 'T' else 'verifier_k2_b2'
        if s._p0_action_draft_mode != expected:
            raise ValueError('original action mode mismatch')
        rt = Runtime(s,out,p['per_episode_forward_cap'],shared=arm=='S',allow_cuda=True)
        write(out/'READY.json',dict(status='PASS',pid=os.getpid(),arm=arm,
              mode=expected,port=port,master_port=os.environ['MASTER_PORT'],
              target_device=str(s.device),vae_device=str(s.vae_device),
              EGL=os.environ['MUJOCO_GL'],profile_OFF=True,teacher_OFF=True,
              trace_OFF=True,default_public_source_unchanged=True),exclusive=True)

        def finish():
            while not (out/'FINISH_REQUESTED').exists():
                time.sleep(.5)
            try:
                # After both clients in this wave have returned. Async artifact
                # writes and CUDA completion are drained outside episode timing.
                source.save_async.__globals__['executor'].shutdown(wait=True)
                source.torch.cuda.synchronize()
                rt.close()
                write(out/'FINISH_READY.json',dict(status='PASS',timestamp=stamp(),
                                                  async_artifacts_drained=True),exclusive=True,durable=True)
            except BaseException as exc:
                write(out/'FINISH_FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()),
                      exclusive=True,durable=True)
        threading.Thread(target=finish,daemon=True).start()
        return original_run(s,local_rank,host,port)
    source.run_async_server_mode = serve
    source.init_logger()
    source.main()

if __name__ == '__main__':
    main()

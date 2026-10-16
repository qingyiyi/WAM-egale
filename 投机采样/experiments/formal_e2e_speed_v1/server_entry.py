"""Current A_off/T server with RAM-only accounting; one local EGL client."""
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import types
from collections import Counter

from speed_contract import EXP, OWN, WORKTREE, sha, validate, write


def _paths():
    base = EXP / "single_worker_chc_gpu_v4"
    prefix = EXP / "prefix_acceptance_v1"
    for directory in (EXP, WORKTREE, prefix, base):
        name = str(directory)
        if name in sys.path:
            sys.path.remove(name)
        sys.path.insert(0, name)
    import importlib.util
    found = importlib.util.find_spec("adapter")
    if found is None or Path(found.origin).resolve() != base / "adapter.py":
        raise ImportError("frozen prefix adapter path changed")


def validate_server(s):
    if (s._cache_mode != "original" or len(s.transformer.blocks) != 30
            or s.streaming_vae_half is not None or s._p0_trace is not None
            or s._p0_state_teacher is not None or s._nsys_profile_mode != "off"
            or s._phase_profile_enabled or s.job_config.action_num_inference_steps != 50
            or s.job_config.num_inference_steps != 20
            or s.job_config.action_guidance_scale != 1
            or s.job_config.guidance_scale != 5):
        raise ValueError("frozen target/cache/scheduler contract changed")


class RamAccounting:
    def __init__(self, s, cap):
        self.s, self.cap = s, cap
        self.lock = threading.RLock()
        self.attempted = self.returned = self.active = self.failed = 0
        self.phase, self.requests = Counter(), []
        self.original_infer = s.infer
        self.original_phase = s._phase_profile_counter
        self.hooks = [
            s.transformer.register_forward_pre_hook(self.pre, with_kwargs=True),
            s.transformer.register_forward_hook(self.post, with_kwargs=True, always_call=True),
        ]
        s._phase_profile_counter = types.MethodType(self.count_phase, s)
        s.infer = types.MethodType(self.infer, s)

    def pre(self, _model, args, kwargs):
        with self.lock:
            if self.attempted >= self.cap:
                raise RuntimeError("formal speed target-forward cap exceeded")
            self.attempted += 1
            self.active += 1

    def post(self, _model, args, kwargs, output):
        with self.lock:
            if self.active:
                self.active -= 1
                if output is None:
                    self.failed += 1
                else:
                    self.returned += 1

    def count_phase(self, _server, name, value=1):
        self.phase[name] += int(value)
        return self.original_phase(name, value)

    def infer(self, _server, request):
        kind = "reset" if request.get("reset") else "cache" if request.get("compute_kv_cache") else "policy"
        before, start = self.attempted, time.monotonic()
        ok = False
        try:
            result = self.original_infer(request)
            ok = True
            return result
        finally:
            self.requests.append(dict(kind=kind, ok=ok,
                wall_sec=time.monotonic() - start,
                target_forwards=self.attempted - before,
                trace_context=request.get("p0_trace_context")))

    def close(self):
        if self.failed or self.active or self.attempted != self.returned:
            raise RuntimeError("target forwards did not close cleanly")
        for hook in self.hooks:
            hook.remove()
        self.s.infer = self.original_infer
        self.s._phase_profile_counter = self.original_phase
        return dict(status="CLOSED", cap=self.cap, target_intents=self.attempted,
                    returned=self.returned, failed=self.failed, active=self.active,
                    phase_counters=dict(self.phase), requests=self.requests,
                    per_request_file_io=False, cuda_events=False)


def main():
    job = Path(os.environ["FORMAL_SPEED_JOB"]).resolve()
    plan = validate(job)
    key = os.environ["FORMAL_SPEED_KEY"]
    rows = [row for row in plan["schedule"] if row["key"] == key]
    if len(rows) != 1 or os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("one frozen local worker assignment and CUDA device 0 required")
    row = rows[0]
    out = job / "episodes" / key
    _paths()
    from wan_va import wan_va_server as source
    from measured_online import control
    from selection import install as install_selector
    import buffers

    original_init = source.VA_Server.__init__
    init_timing = {}

    def timed_init(s, *args, **kwargs):
        start = time.monotonic()
        try:
            return original_init(s, *args, **kwargs)
        finally:
            init_timing["target_server_init_sec"] = time.monotonic() - start

    source.VA_Server.__init__ = timed_init
    original_run = source.run_async_server_mode

    def serve(s, local_rank, host, port):
        if local_rank != 0:
            raise ValueError("single-rank model server required")
        validate_server(s)
        mode = "off" if row["arm"] == "T" else plan["mode"]
        if (s._p0_action_draft_mode != mode
                or s._p0_action_draft_threshold != plan["threshold"]
                or s._p0_action_draft_b2_checkpoint != plan["draft_checkpoint"]
                or sha(Path(plan["draft_checkpoint"])) != plan["draft_sha256"]):
            raise ValueError("frozen candidate mode/draft/threshold changed")
        setup_start = time.monotonic()
        s.action_per_frame = int(s.job_config.action_per_frame)
        selector = None
        online = None
        if row["arm"] == "A_OFF":
            s._load_p0_action_draft_b2_predictor().model.to(s.device).eval()
            selector = install_selector(s, "prefix_then_cost")
            buffers.prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
            online = control(s, "O3", allow_cuda=True)
        accounting = RamAccounting(s, cap=4500)
        ready = dict(status="READY", key=key, arm=row["arm"], task=row["task"],
                     init=row["init"], mode=mode,
                     selector="global" if selector is None else "prefix_then_cost",
                     threshold=plan["threshold"], draft_sha256=plan["draft_sha256"],
                     target_checkpoint_id=plan["target_checkpoint_id"],
                     target_server_init_sec=init_timing["target_server_init_sec"],
                     private_setup_sec=time.monotonic() - setup_start,
                     ready_monotonic=time.monotonic(), port=port,
                     no_target_forward_before_READY=True,
                     profile_off=True, teacher_off=True, trace_off=True,
                     O1_O2_O3=(row["arm"] == "A_OFF"),
                     per_request_file_io=False)
        write(out / "READY.json", ready)

        def finish():
            while not (out / "FINISH_REQUESTED").exists():
                time.sleep(.5)
            try:
                source.save_async.__globals__["executor"].shutdown(wait=True)
                source.torch.cuda.synchronize()
                if selector is not None and selector.pending is not None:
                    raise ValueError("unfinished prefix selection")
                snapshot = accounting.close()
                if selector is not None and len(selector.records) != snapshot["phase_counters"].get("tree_count", 0):
                    raise ValueError("prefix tree record coverage differs from count")
                if online is not None:
                    online.close()
                write(out / "COUNTER-CLOSED.json", snapshot)
                write(out / "ACCEPTANCE-TREES.json", dict(
                    selector="global" if selector is None else "prefix_then_cost",
                    records=[] if selector is None else selector.records))
                write(out / "FINISH_READY.json", dict(status="PASS", drained_after_episode=True))
            except BaseException as exc:
                write(out / "FINISH_FAILED.json", dict(error=repr(exc), traceback=traceback.format_exc()))

        threading.Thread(target=finish, daemon=True).start()
        return original_run(s, local_rank, host, port)

    source.run_async_server_mode = serve
    source.init_logger()
    source.main()


if __name__ == "__main__":
    main()

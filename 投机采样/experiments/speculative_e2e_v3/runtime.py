"""Common lightweight RAM accounting. No CUDA events or per-forward disk I/O.

A finite reservation is persisted BEFORE launching a worker. If terminal
durable closure is missing the supervisor charges the whole reservation.
"""
import os
import threading
import time
import types
from collections import Counter

from task_contract import write, stamp


class Runtime:
    def __init__(self, server, out, cap, *, shared=False, allow_cuda=False):
        from online_shared import OnlineShared
        self.s, self.out, self.cap = server, out, cap
        self.lock = threading.RLock()
        self.attempted = self.returned = self.failed = self.active = 0
        self.closed = False
        self.calls = Counter()
        self.phase = Counter()
        self.requests = []
        self.request_active = False
        self.shared = OnlineShared(server, enabled=shared, allow_cuda=allow_cuda).install()
        self.original_infer = server.infer
        self.original_counter = server._phase_profile_counter
        self.hooks = [server.transformer.register_forward_pre_hook(self.pre, with_kwargs=True),
                      server.transformer.register_forward_hook(self.post, with_kwargs=True,
                                                              always_call=True)]
        server._phase_profile_counter = types.MethodType(self.counter, server)
        server.infer = types.MethodType(self.infer, server)
        self.checkpoint()

    def pre(self, model, args, kwargs):
        with self.lock:
            if self.closed or self.attempted >= self.cap:
                raise RuntimeError('finite target forward cap/closed worker')
            self.attempted += 1
            self.active += 1
            inp = args[0] if args else kwargs['input_dict']
            key = ('action' if kwargs.get('action_mode',False) else 'video',
                   int(kwargs.get('update_cache',0)), int(inp['noisy_latents'].shape[0]),
                   kwargs.get('candidate_cache_rows') is not None)
            self.calls[str(key)] += 1

    def post(self, model, args, kwargs, output):
        with self.lock:
            # always_call also fires after a rejecting pre-hook.
            if self.active:
                self.active -= 1
                if output is None: self.failed += 1
                else: self.returned += 1

    def counter(self, _server, name, value=1):
        self.phase[name] += int(value)
        return self.original_counter(name, value)

    def infer(self, _server, request):
        with self.lock:
            if self.request_active or self.closed:
                raise RuntimeError('one request per model worker required')
            self.request_active = True
        start, count = time.monotonic(), self.attempted
        kind = 'reset' if request.get('reset') else 'cache' if request.get('compute_kv_cache') else 'policy'
        ok = False
        try:
            result = self.original_infer(request)
            ok = True
            return result
        finally:
            with self.lock:
                self.request_active = False
                self.requests.append(dict(kind=kind, ok=ok,
                    wall_seconds=time.monotonic()-start,
                    forward_intents=self.attempted-count,
                    frame=getattr(self.s,'frame_st_id',None)))
                # Atomic visibility, no per-request fsync; reservations cover
                # an unclosed/crashed worker conservatively.
                self.checkpoint()

    def summary(self):
        return dict(status='CLOSED' if self.closed else 'RUNNING', pid=os.getpid(),
                    timestamp=stamp(), cap=self.cap, target_intents=self.attempted,
                    host_returned=self.returned, failed_forwards=self.failed,
                    active_forwards=self.active, request_active=self.request_active,
                    calls=dict(self.calls), phase_counters=dict(self.phase),
                    requests=self.requests, shared=self.shared.summary(),
                    per_forward_filesystem_IO=False, CUDA_events=False,
                    on_failure_charge_full_reservation=True)

    def checkpoint(self):
        write(self.out/'counter-state.json', self.summary())

    def close(self):
        with self.lock:
            if self.active or self.request_active or self.attempted != self.returned or self.failed:
                raise RuntimeError('cannot close incomplete forward/request accounting')
            if self.shared.control is not None or not self.shared.last_clean:
                raise RuntimeError('shared request lease leaked')
            self.shared.close()
            self.closed = True
            write(self.out/'COUNTER-CLOSED.json', self.summary(), exclusive=True, durable=True)
            for h in self.hooks: h.remove()
            self.s.infer = self.original_infer
            self.s._phase_profile_counter = self.original_counter

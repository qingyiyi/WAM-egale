"""Private online lifecycle for the unchanged, accepted S0 attention adapter.

Only the original K2 helper and CFG candidate packer on this server instance
are wrapped. Ordinary target calls, scheduler, scoring and outputs are original.
An ordinary call revokes the lease; the next candidate creates a fresh lease.
The adapter is constructed in the inference thread and removed every chunk.
"""
import threading
import types

import torch
from prefix_adapter import SharedPrefix, LeaseError


class OnlineShared:
    def __init__(self, server, *, enabled=False, allow_cuda=False,
                 cpu_reference_enabled=False, expected_layers=30):
        self.s = server
        self.enabled = bool(enabled)
        self.options = dict(enabled=True, allow_cuda=allow_cuda,
                            cpu_reference_enabled=cpu_reference_enabled,
                            expected_layers=expected_layers)
        self.installed = False
        self.patches = []
        self.control = self.ctx = self.token = None
        self.request_thread = None
        self.requests = self.leases = self.candidates = self.revocations = 0
        self.shared_layer_calls = 0
        self.last_clean = True

    def _patch(self, name, value):
        self.patches.append((name, name in vars(self.s), vars(self.s).get(name)))
        setattr(self.s, name, value)

    def install(self):
        if not self.enabled:
            return self
        if self.installed or self.control is not None:
            raise LeaseError('online adapter already installed')
        original_action = self.s._infer_action_verifier_k2_b2
        original_pack = self.s._repeat_action_candidates_for_cfg

        def action(_s, *args, **kwargs):
            if self.control is not None:
                raise LeaseError('nested online action request')
            self.requests += 1
            self.last_clean = False
            self.request_thread = threading.get_ident()
            self.control = SharedPrefix(_s.transformer, **self.options)
            try:
                self.control.install()
                return original_action(*args, **kwargs)
            finally:
                try:
                    self._end_lease()
                finally:
                    self.control.close()
                    self.last_clean = (not self.control.installed and not self.control.entries
                                       and self.control.active is None)
                    self.control = None
                    self.request_thread = None

        def pack(_s, inp):
            if self.control is None or threading.get_ident() != self.request_thread:
                raise LeaseError('candidate pack outside owned action thread')
            packed, original_rows = original_pack(inp)
            if (not _s.use_cfg or inp['noisy_latents'].shape[0] != 7
                    or packed['noisy_latents'].shape[0] != 14):
                raise LeaseError('online shared requires original seven candidates and CFG2')
            if self.token is not None and self.token.revoked:
                self.revocations += 1
                self._end_lease()
            if self.token is None:
                self.ctx = self.control.lease(request='online-chunk-' + str(self.requests),
                                              arm='S', cache_name=_s.cache_name,
                                              device=packed['noisy_latents'].device)
                self.token = self.ctx.__enter__()
                self.leases += 1
            # Validate the ACTUAL original packer's map, not just batch geometry.
            # This small GPU equality synchronisation is included in E2E timing.
            if (original_rows.dtype != self.token.rows.dtype
                    or original_rows.device != self.token.rows.device
                    or not torch.equal(original_rows, self.token.rows)):
                self.control.invalidate('original row map mismatch')
                raise LeaseError('original CFG row values changed')
            self.candidates += 1
            return packed, self.token.rows

        self._patch('_infer_action_verifier_k2_b2', types.MethodType(action, self.s))
        self._patch('_repeat_action_candidates_for_cfg', types.MethodType(pack, self.s))
        self.installed = True
        return self

    def _end_lease(self):
        if self.ctx is not None:
            self.shared_layer_calls += sum(v.get('shared_attention', 0)
                                          for v in self.control.ops.values())
            ctx, self.ctx = self.ctx, None
            self.token = None
            ctx.__exit__(None, None, None)

    def close(self):
        if self.control is not None:
            raise LeaseError('cannot close active online request')
        for name, had, old in reversed(self.patches):
            if had:
                setattr(self.s, name, old)
            else:
                delattr(self.s, name)
        self.patches.clear()
        self.installed = False

    def summary(self):
        return dict(enabled=self.enabled, requests=self.requests, leases=self.leases,
                    candidates=self.candidates, revocations=self.revocations,
                    shared_layer_calls_diagnostic=None,
                    request_active=self.control is not None, last_request_clean=self.last_clean,
                    prompt_shared=False, rowproof_reuse=False)

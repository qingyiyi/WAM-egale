"""O1 and O1+O2 share the same template attention and online lifecycle."""
import time

from online import frozen_online
from template import TemplateSharedPrefix, clone_function, stats

INSTALLS = []


class MeasuredO1(TemplateSharedPrefix):
    arm = 'O1'

    def install(self):
        before = stats()
        start = time.perf_counter()
        ok = False
        try:
            result = super().install()
            ok = True
            return result
        finally:
            after = stats()
            INSTALLS.append(dict(arm=self.arm, wall_sec=time.perf_counter()-start, ok=ok,
                AST_builds_observed=after['AST_builds']-before['AST_builds'],
                bindings=after['bindings']-before['bindings'],
                template_stats_before=before, template_stats_after=after))


class MeasuredO2(MeasuredO1):
    arm = 'O2'


class OnlineO1(frozen_online.OnlineShared):
    install = clone_function(frozen_online.OnlineShared.install, SharedPrefix=MeasuredO1)


class OnlineO2(frozen_online.OnlineShared):
    _base_install = clone_function(frozen_online.OnlineShared.install, SharedPrefix=MeasuredO2)

    def install(self):
        if not self.enabled:
            return self
        if self.installed or self.patches:
            raise RuntimeError('O2 already installed')
        import scalar
        compiled = scalar.bind(self.s._infer_action_verifier_k2_b2, self.s)
        self._patch('_infer_action_verifier_k2_b2', compiled)
        try:
            return self._base_install()
        except BaseException:
            self.close()
            raise


class MeasuredO3(MeasuredO1):
    arm = 'O3'


class OnlineO3(frozen_online.OnlineShared):
    _base_install = clone_function(frozen_online.OnlineShared.install, SharedPrefix=MeasuredO3)

    def install(self):
        if not self.enabled:return self
        if self.installed or self.patches:raise RuntimeError('O3 already installed')
        import types
        import torch
        import scalar
        import buffers
        original=self.s._infer_action_verifier_k2_b2
        base=types.MethodType(scalar.action_function(original),self.s)
        new=types.MethodType(buffers.action_function(original),self.s)
        original_scatter=self.s._p0_action_scatter_compact_batch

        @torch.no_grad()
        def action(server,*args,**kwargs):
            trace=kwargs.get('p0_trace_chunk',args[5] if len(args)>5 else None)
            if trace is not None or getattr(server,'_p0_state_teacher',None) is not None:
                buffers.EVENTS.append(dict(enabled=False,reason='trace_or_teacher',closed=True))
                return base(*args,**kwargs)
            if buffers.ACTIVE.get() is not None:raise RuntimeError('nested O3 workspace')
            ws=buffers.Workspace();token=buffers.ACTIVE.set(ws)
            ok=False
            try:
                result=new(*args,**kwargs);ok=True
                return result
            finally:
                ws.close();buffers.ACTIVE.reset(token)
                buffers.EVENTS.append(dict(enabled=True,ok=ok,closed=ws.closed,**ws.counts))

        def scatter(server,*args,**kwargs):
            ws=buffers.ACTIVE.get()
            if ws is None:return original_scatter(*args,**kwargs)
            return ws.scatter(server,*args,**kwargs)

        self._patch('_infer_action_verifier_k2_b2',types.MethodType(action,self.s))
        self._patch('_p0_action_scatter_compact_batch',types.MethodType(scatter,self.s))
        try:return self._base_install()
        except BaseException:self.close();raise


def control(server, arm, *, allow_cuda=True, cpu_reference_enabled=False):
    cls = {'O1': OnlineO1, 'O2': OnlineO2, 'O3':OnlineO3}[arm]
    return cls(server, enabled=True, allow_cuda=allow_cuda,
               cpu_reference_enabled=cpu_reference_enabled).install()

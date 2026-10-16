"""Read-only actual call observation; no trace flag or scheduler replacement.

record_update locals are observed at its existing counter call. This keeps the
original O3 workspace active and does not fabricate commits from tree counts.
Every commit must bind actual scheduler input and projected output records.
"""
import inspect
import types


class Difference(RuntimeError):
    pass


class Observer:
    def __init__(self, server, snap, state_check, check, *, reference=None):
        import torch
        self.s, self.snap, self.sc, self.check = server, snap, state_check, check
        self.torch = torch
        self.reference = reference
        self.events, self.commits, self.transitions = [], [], []
        self.groups = {}
        self.query = None
        self.query_count = self.candidate_queries = self.attention_calls = 0
        self.stage = 'policy'
        self.stage_query_count = 0
        self.refs = None
        self.diff = None
        self.parameters = [(v, v._version) for v in server.transformer.parameters()]
        predictor = server._load_p0_action_draft_b2_predictor()
        self.parameters += [(v, v._version) for v in predictor.model.parameters()]
        self.original_counter = server._phase_profile_counter
        self.original_step = server.action_scheduler.step
        if 'step' in vars(server.action_scheduler):
            raise RuntimeError('expected original class scheduler step, no preexisting instance patch')
        self.original_project = server._p0_action_project_state
        server._phase_profile_counter = types.MethodType(self.counter, server)
        server.action_scheduler.step = self.step
        server._p0_action_project_state = types.MethodType(self.project, server)
        self.hooks = [server.transformer.register_forward_pre_hook(self.pre, with_kwargs=True),
            server.transformer.register_forward_hook(self.post, with_kwargs=True, always_call=True)]

    def event(self, kind, value):
        self.check()
        item = dict(kind=kind, value=value)
        index = len(self.events)
        if self.reference is not None:
            if index >= len(self.reference) or self.reference[index] != item:
                self.diff = dict(event=index, kind=kind, actual=item,
                    expected=self.reference[index] if index < len(self.reference) else None)
                raise Difference('first actual numerical/state/path difference at event ' + str(index))
        self.events.append(item)

    def digest(self, value):
        return self.snap.fingerprint(value)

    def light(self):
        # The diagnostic's temporary bound step is not scheduler numerical
        # state. All original scheduler fields are retained, never mutated.
        caches = []
        for c in self.sc.pools(self.s.transformer, self.s.cache_name):
            caches.append(dict(pool_shape=list(c['k'].shape), pool_dtype=str(c['k'].dtype),
                slots=self.snap.cpu(c['mask'].nonzero(as_tuple=False).flatten()),
                **{k: self.snap.cpu(c[k]) for k in ('id', 'mask', 'is_pred')}))
        return dict(fields={k: self.snap.cpu(getattr(self.s, k)) for k in self.snap.FIELDS},
            caches=caches, streaming=self.snap.cpu(self.s.streaming_vae.feat_cache),
            scheduler=self.snap.cpu(vars(self.s.scheduler)),
            action_scheduler=self.snap.cpu({k: v for k, v in vars(self.s.action_scheduler).items() if k != 'step'}),
            rng=self.snap.cpu(self.snap.rng(True)))

    def layout(self, value):
        t = self.torch
        if isinstance(value, t.Tensor):
            return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                        device=str(value.device), storage_offset=value.storage_offset())
        if isinstance(value, dict): return {k: self.layout(v) for k, v in sorted(value.items())}
        if isinstance(value, (tuple, list)): return [self.layout(v) for v in value]
        return value

    def cache_epoch(self, reason):
        self.refs = None
        caches = [self.snap.capture_cache(b.attn1.attn_caches[self.s.cache_name]) for b in self.s.transformer.blocks]
        self.event('all_valid_KV_epoch', dict(stage=self.stage, query=self.query_count,
            reason=reason, complete30_hash=self.digest(caches),
            counts=[int(c['slots'].numel()) for c in caches]))
        del caches
        self.refs = self.sc.references(self.s.transformer, self.s.cache_name)

    def pre(self, model, args, kwargs):
        self.check()
        if self.query is not None: raise RuntimeError('nested target query')
        if self.refs is None:
            self.cache_epoch('initial_restored_history')
        elif not self.sc.equal(model, self.refs, self.s.cache_name):
            if not (self.stage == 'cache' and self.stage_query_count == 0):
                raise RuntimeError('KV changed outside the declared original cache clear')
            self.cache_epoch('original_compute_cache_clear_pred_cache')
        if not all(v._version == version for v, version in self.parameters):
            raise RuntimeError('target/draft parameter version changed')
        inp = args[0] if args else kwargs['input_dict']
        candidate = kwargs.get('candidate_cache_rows') is not None
        if candidate:
            if tuple(inp['noisy_latents'].shape[:1]) != (14,) or kwargs['update_cache'] != 0:
                raise RuntimeError('only actual original B14 read-only candidate queries')
            self.candidate_queries += 1
        self.query = dict(index=self.query_count, stage=self.stage, update=int(kwargs.get('update_cache', 0)),
            candidate=candidate, attention_start=self.attention_calls,
            input_hash=self.digest((args, kwargs)), input_layout=self.layout((args, kwargs)),
            RNG=self.digest(self.snap.rng(True)),
            light_state=self.digest(self.light()))
        self.event('target_input', self.query)

    def post(self, model, args, kwargs, output):
        if self.query is None: return
        # Preserve a primary kernel/input mismatch exception raised inside
        # forward; the supervisor still refuses the incomplete query/result.
        if output is None: return
        q = self.query
        if self.digest((args, kwargs)) != q['input_hash']:
            raise RuntimeError('target input bytes changed')
        if q['update'] == 0:
            if not self.sc.equal(model, self.refs, self.s.cache_name):
                raise RuntimeError('read-only target changed any valid KV or metadata')
        else:
            self.cache_epoch('original_update_cache_' + str(q['update']))
        if not all(v._version == version for v, version in self.parameters):
            raise RuntimeError('parameter changed during target query')
        count = self.attention_calls - q['attention_start']
        if count != (30 if q['candidate'] else 0):
            raise RuntimeError('every actual B14 must traverse thirty observed shared callbacks')
        self.event('target_output', dict(index=q['index'], stage=self.stage,
            output_hash=self.digest(output), output_layout=self.layout(output),
            all_valid_KV_readonly=q['update'] == 0, shared_layers=count,
            RNG=self.digest(self.snap.rng(True)), light_state=self.digest(self.light())))
        self.query_count += 1
        self.stage_query_count += 1
        self.query = None

    def attention_pre(self, q, k, v, slots, rows):
        if self.query is None or not self.query['candidate']:
            raise RuntimeError('candidate attention outside an observed actual B14 query')
        value = dict(query=self.query_count, layer=self.attention_calls - self.query['attention_start'],
            input_hash=self.digest((q, k, v, slots, rows)), layout=self.layout((q, k, v, slots, rows)))
        self.event('candidate_attention_input', value)
        return value['input_hash']

    def attention_post(self, q, k, v, slots, rows, output, before):
        if before != self.digest((q, k, v, slots, rows)):
            raise RuntimeError('candidate kernel changed actual Q/K/V or private maps')
        self.event('candidate_attention_output', dict(query=self.query_count,
            layer=self.attention_calls - self.query['attention_start'], output_hash=self.digest(output)))
        self.attention_calls += 1

    def step(self, *args, **kwargs):
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name != 'scheduler_step':
            raise RuntimeError('unexpected authoritative action scheduler caller')
        if len(args) != 3 or kwargs != dict(return_dict=False):
            raise RuntimeError('original authoritative scheduler call interface changed')
        direction, timestep, state = args
        row = dict(index=len(self.transitions), state_hash=self.digest(self.s._p0_action_compact(state[0])),
            direction_hash=self.digest(self.s._p0_action_compact(direction[0])), timestep=float(timestep),
            full_input_hash=self.digest((args, kwargs)))
        output = self.original_step(*args, **kwargs)
        row['raw_full_next_hash'] = self.digest(output)
        self.transitions.append(row)
        self.event('actual_action_scheduler_step', dict(row))
        return output

    def project(self, _server, full_state, action_cond, frame_st_id):
        caller = inspect.currentframe().f_back
        output = self.original_project(full_state, action_cond, frame_st_id)
        if caller.f_code.co_name == 'scheduler_step':
            if not self.transitions or full_state is not caller.f_locals['raw_full_next']:
                raise RuntimeError('actual raw/projected transition binding missing')
            row = self.transitions[-1]
            if self.digest(full_state) != row['raw_full_next_hash'] or 'projected_full_next_hash' in row:
                raise RuntimeError('scheduler raw result mutated or duplicate projection')
            row['projected_full_next_hash'] = self.digest(output)
            row['projected_compact_next_hash'] = self.digest(self.s._p0_action_compact(output[0]))
            self.event('actual_action_scheduler_projection', dict(row))
        return output

    def counter(self, _server, name, value=1):
        if name == 'committed_scheduler_update_count':
            caller = inspect.currentframe().f_back
            local = caller.f_locals
            required = {'step', 'state', 'direction', 'source', 'transition',
                        'bonus_target_step_executed', 'action_timesteps', 'self'}
            if caller.f_code.co_name != 'record_update' or not required.issubset(local) or local['self'] is not self.s or value != 1:
                raise RuntimeError('actual record_update local contract changed; no inferred commits allowed')
            if local['transition'] is not None:
                raise RuntimeError('full trace enabled unexpectedly; original O3 must remain active')
            index = int(local['step'])
            if index != len(self.commits): raise RuntimeError('actual committed scheduler step sequence broken')
            row = dict(step=index, source=local['source'], bonus=bool(local['bonus_target_step_executed']),
                state_hash=self.digest(local['state']), direction_hash=self.digest(local['direction']),
                timestep=float(local['action_timesteps'][index]),
                sigma=float(self.s.action_scheduler.sigmas[index]),
                next_sigma=float(self.s.action_scheduler.sigmas[index + 1])
                    if index + 1 < len(self.s.action_scheduler.sigmas) else 0.)
            matching = [r for r in self.transitions if r['state_hash'] == row['state_hash']
                and r['direction_hash'] == row['direction_hash'] and r['timestep'] == row['timestep']
                and 'projected_compact_next_hash' in r]
            if not matching or len({(r['raw_full_next_hash'], r['projected_full_next_hash'], r['projected_compact_next_hash']) for r in matching}) != 1:
                raise RuntimeError('committed update lacks equal actual scheduler/projection evidence')
            row.update(actual_transition_indices=[r['index'] for r in matching],
                raw_full_next_hash=matching[0]['raw_full_next_hash'],
                projected_full_next_hash=matching[0]['projected_full_next_hash'],
                projected_compact_next_hash=matching[0]['projected_compact_next_hash'])
            if self.commits and self.commits[-1]['projected_compact_next_hash'] != row['state_hash']:
                raise RuntimeError('consecutive actual committed states disagree')
            self.event('actual_committed_scheduler_update', row)
            self.commits.append(row)
        return self.original_counter(name, value)

    def change_stage(self):
        if self.query is not None or len(self.commits) != 50:
            raise RuntimeError('full fifty actual policy commits required before cache handoff')
        self.stage = 'cache'
        self.stage_query_count = 0

    def close(self, *, complete):
        for handle in self.hooks: handle.remove()
        self.s._phase_profile_counter = self.original_counter
        del self.s.action_scheduler.step
        self.s._p0_action_project_state = self.original_project
        if complete:
            if self.query is not None or len(self.commits) != 50 or self.stage_query_count != 2:
                raise RuntimeError('incomplete actual full policy plus two-forward cache path')
            if self.reference is not None and len(self.events) != len(self.reference):
                raise Difference('actual complete event count differs')
        self.refs = None

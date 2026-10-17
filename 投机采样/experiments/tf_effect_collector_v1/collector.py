"""Read-only O3 teacher side channel; no additional target or score call.

The caller brackets each *complete policy request* with begin_chunk/end_chunk.
This module never enables _p0_state_teacher or _p0_trace, replaces an action
body, changes a score, or selects a route. Importing it initializes no CUDA.
Capture failures invalidate the data; wrappers still return original results.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import types


SCHEMA = 'tf-effect-o3-same-query-columns/v1'
CONTEXT_KEYS = ('campaign_id', 'parallel_worker_id', 'episode_attempt_id',
                'task_index', 'episode_index', 'rollout_chunk')


class Collector:
    """Per-server observer. A successful shard has exactly three rows per tree.

    metadata binds the generator checkpoint, target identity and data role.
    storage_cap_bytes defaults to 512 MiB and covers this output directory.
    end_chunk returns {path, trees, rows, reached_rows, status}; it raises when
    any capture failed, so the caller must never admit that chunk as FIT/CAL.
    The original target/execute calls themselves are never retried or replaced.
    """

    def __init__(self, s, proxy, output_dir, metadata):
        import torch
        self.torch = torch
        self.s, self.proxy = s, proxy
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = copy.deepcopy(metadata)
        self.storage_cap = int(self.metadata.get('storage_cap_bytes', 512 * 2**20))
        if not 0 < self.storage_cap <= 512 * 2**20:
            raise ValueError('collector storage cap must be in (0, 512 MiB]')
        self.installed = False
        self.active = self.pending = None
        self.errors = []
        self.chunk_index = self.total_trees = self.total_rows = 0
        self.closed = False

    def _check_mode(self):
        if self.s._p0_state_teacher is not None or self.s._p0_trace is not None:
            raise ValueError('original O3 requires native trace and teacher to remain None')
        if self.s._p0_action_draft_mode != 'verifier_k2_b2_bonus':
            raise ValueError('fixed K2/B2 bonus production API required')

    def _validate_source(self):
        # Inspect the authoritative class method, never the online wrapper.
        raw = inspect.unwrap(type(self.s)._infer_action_verifier_k2_b2)
        tree = ast.parse(textwrap.dedent(inspect.getsource(raw)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)]
        for name in ('_target_action_candidate_directions', 'execute_k2_b2_route'):
            if sum(n.func.attr == name for n in calls) != 1:
                raise ValueError('one authoritative action call site required: ' + name)
        assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                    and isinstance(n.ctx, ast.Store)}
        if not {'step_index', 'tree', 'scores', 'route', 'frame_st_id'}.issubset(assigned | set(raw.__code__.co_varnames)):
            raise ValueError('authoritative action local names changed')
        self.source_proof = dict(function=raw.__name__, file=raw.__code__.co_filename,
            source_sha256=hashlib.sha256(inspect.getsource(raw).encode()).hexdigest(),
            unique_candidate_and_execute_sites=True)

    def _workspace(self):
        ws = self.buffers.ACTIVE.get()
        if ws is None or ws.closed:
            raise ValueError('teacher side channel observed scalar fallback or closed O3 workspace')
        return ws

    def _caller(self, frame, *, tree=None, scores=None, route=None):
        local = frame.f_locals
        required = {'self', 'step_index', 'tree', 'action_timesteps', 'frame_st_id'}
        if (frame.f_code.co_name != '_infer_action_verifier_k2_b2'
                or not required.issubset(local) or local['self'] is not self.s):
            raise ValueError('actual generated action caller/local contract changed')
        for name, value in (('tree', tree), ('scores', scores), ('route', route)):
            if value is not None and local.get(name) is not value:
                raise ValueError('actual caller identity differs: ' + name)
        step = int(local['step_index'])
        if step < 0 or step >= len(local['action_timesteps']):
            raise ValueError('actual scheduler step outside action timestep table')
        return step, int(local['frame_st_id'])

    def _error(self, where, exc):
        item = dict(where=where, error=repr(exc), chunk_index=self.chunk_index,
                    context=None if self.active is None else copy.deepcopy(self.active['context']))
        self.errors.append(item)
        if self.active is not None:
            self.active['errors'].append(item)

    def _cpu(self, x, *, float32=False):
        if not isinstance(x, self.torch.Tensor):
            raise TypeError('real tensor required for teacher field')
        y = x.detach().to(device='cpu')
        if float32:
            y = y.float()
        return y.contiguous().clone()

    def _same_bytes(self, a, b):
        return (a.shape == b.shape and a.dtype == b.dtype
                and self.torch.equal(a.contiguous().view(self.torch.uint8),
                                     b.contiguous().view(self.torch.uint8)))

    def install(self):
        if self.installed or self.closed:
            raise RuntimeError('collector is already installed or closed')
        self._check_mode()
        self._validate_source()
        wanted = Path(__file__).resolve().parent.parent / 'prefix_acceptance_v1/buffers.py'
        found = list({id(m): m for m in tuple(sys.modules.values())
                      if getattr(m, '__file__', None) and Path(m.__file__).resolve() == wanted}.values())
        if len(found) != 1:
            raise ImportError('one already-loaded frozen O3 buffers module required')
        self.buffers = found[0]
        self.original_candidate = self.s._target_action_candidate_directions
        self.original_execute = self.proxy.execute_k2_b2_route
        self.had_candidate = '_target_action_candidate_directions' in vars(self.s)
        self.had_execute = 'execute_k2_b2_route' in vars(self.proxy)
        collector = self

        def candidates(_server, states, times, frame_st_id):
            # Call exactly once, preserving its returned object. Logging errors
            # may not turn a rejected draft into an accepted one or vice versa.
            result = collector.original_candidate(states, times, frame_st_id)
            try:
                if collector.active is None:
                    raise ValueError('target candidate call outside declared policy chunk')
                collector.active['candidate_calls'] += 1
                if collector.pending is not None:
                    raise ValueError('previous candidate query not consumed by execute')
                collector._check_mode()
                ws = collector._workspace()
                caller = inspect.currentframe().f_back
                try:
                    step, actual_frame = collector._caller(caller)
                    actual_tree = caller.f_locals['tree']
                finally:
                    del caller
                if int(frame_st_id) != actual_frame or states.shape[0] != 7 or len(times) != 7:
                    raise ValueError('actual seven-state candidate/frame contract differs')
                if dict(collector.proxy.context) != collector.active['context']:
                    raise ValueError('request identity changed within teacher chunk')
                queries = collector.torch.stack(tuple(
                    collector.s._p0_action_compact(states[i]) for i in range(3)), dim=0)
                collector.pending = dict(tree=actual_tree, step=step, frame=actual_frame,
                    workspace=ws, generation=ws.generation,
                    query=collector._cpu(queries),
                    times=tuple(float(v) for v in times),
                    target=collector._cpu(collector.torch.stack(tuple(
                        collector.s._p0_action_compact(result[i]) for i in range(3)), dim=0)))
            except Exception as exc:
                collector._error('candidate_capture', exc)
            return result

        def execute(_proxy, tree, scores, route, **kwargs):
            result = collector.original_execute(tree, scores, route, **kwargs)
            try:
                if collector.active is None:
                    raise ValueError('route execute outside declared policy chunk')
                collector.active['execute_calls'] += 1
                collector._check_mode()
                pending = collector.pending
                if pending is None:
                    raise ValueError('execute lacks actual target candidate capture')
                caller = inspect.currentframe().f_back
                try:
                    step, frame = collector._caller(caller, tree=tree, scores=scores, route=route)
                finally:
                    del caller
                ws = collector._workspace()
                if (pending['tree'] is not tree or pending['step'] != step or pending['frame'] != frame
                        or pending['workspace'] is not ws or pending['generation'] != ws.generation):
                    raise ValueError('candidate/execute binding or O3 generation changed')
                collector._capture(tree, scores, route, result, pending, kwargs)
            except Exception as exc:
                collector._error('execute_capture', exc)
            finally:
                collector.pending = None
            return result

        self.s._target_action_candidate_directions = types.MethodType(candidates, self.s)
        self.proxy.execute_k2_b2_route = types.MethodType(execute, self.proxy)
        self.installed = True
        return self

    def begin_chunk(self, context):
        if not self.installed or self.closed or self.active is not None or self.pending is not None:
            raise RuntimeError('installed collector and no active chunk required')
        self._check_mode()
        context = copy.deepcopy(dict(context))
        if any(k not in context for k in CONTEXT_KEYS):
            raise ValueError('complete actual task/episode/chunk request identity required')
        # Ensure the identity can be persisted without tensor aliases or NaNs.
        json.dumps(context, ensure_ascii=False, allow_nan=False)
        self.active = dict(context=context, records=[], candidate_calls=0, execute_calls=0, errors=[])

    def _capture(self, tree, scores, route, result, pending, kwargs):
        t = self.torch
        if dict(self.proxy.context) != self.active['context']:
            raise ValueError('execute request identity differs from begin_chunk')
        threshold = float(kwargs['threshold'])
        if threshold != float(self.s._p0_action_draft_threshold):
            raise ValueError('original threshold changed')
        depth, parent, child = int(result.accepted_depth), int(route.parent), int(route.child)
        if depth not in (0, 1, 2) or parent not in (0, 1) or child not in (0, 1):
            raise ValueError('invalid actually executed prefix/route')
        native_state = self._cpu(t.stack((tree.root_state, *tree.first_states), dim=0))
        native_previous = self._cpu(t.stack((tree.root_previous_direction, *tree.first_directions), dim=0))
        native_target = self._cpu(scores.target_directions[:3])
        native_proposals = self._cpu(t.stack((tree.first_directions, *tree.second_directions), dim=0))
        if not self._same_bytes(native_state, pending['query']):
            raise ValueError('predictor state bytes differ from actual target query')
        if not self._same_bytes(native_target, pending['target']):
            raise ValueError('score teacher bytes differ from actual target return')
        if native_state.shape != native_previous.shape or native_state.shape != native_target.shape:
            raise ValueError('compact state/previous/target shapes differ')
        if native_state.ndim != 2 or native_state.shape != (3, 112) or native_proposals.shape != (3, 2, 112):
            raise ValueError('fixed 7-channel 16-token TF feature/proposal shape required')
        if not all(bool(t.isfinite(x).all()) for x in (native_state, native_previous, native_target, native_proposals)):
            raise ValueError('nonfinite actual teacher/proposal fields')
        sigma = tuple(float(v) for v in tree.sigma_path)
        times = tuple(float(v) for v in tree.target_timesteps)
        if len(sigma) != 3 or len(times) != 3 or not all(math.isfinite(v) for v in sigma + times):
            raise ValueError('finite original sigma/time triplets required')
        if pending['times'] != (times[0], times[1], times[1], times[2], times[2], times[2], times[2]):
            raise ValueError('actual packed timesteps do not match the tree')
        first = tuple(float(v) for v in scores.first_costs)
        second = tuple(tuple(float(v) for v in q) for q in scores.second_costs)
        if len(first) != 2 or len(second) != 2 or any(len(q) != 2 for q in second):
            raise ValueError('exact six original FP32 edge costs required')
        if not all(math.isfinite(v) and v >= 0 for v in (*first, *second[0], *second[1])):
            raise ValueError('finite nonnegative original edge costs required')
        expected_depth = 0 if first[parent] > threshold else 1 if second[parent][child] > threshold else 2
        if depth != expected_depth:
            raise ValueError('actual execution depth disagrees with unchanged gate')
        self.active['records'].append(dict(state=native_state.float(), previous=native_previous.float(),
            target=native_target.float(), proposals=native_proposals.float(),
            source_dtype=dict(state=str(native_state.dtype), previous=str(native_previous.dtype),
                              target=str(native_target.dtype), proposals=str(native_proposals.dtype)),
            sigma=sigma, times=times, first=first, second=second, threshold=threshold,
            depth=depth, parent=parent, child=child, step=pending['step'], frame=pending['frame'],
            generation=pending['generation']))

    def _failure_marker(self, error):
        path = self.output_dir / ('chunk-%05d-FAILED.json' % self.chunk_index)
        payload = dict(schema=SCHEMA, status='FAILED_CAPTURE_NOT_FIT_CAL', error=repr(error),
            context=None if self.active is None else self.active['context'], errors=self.errors,
            extra_teacher_forwards=0, performance_claim=False)
        try:
            with path.open('x') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write('\n')
        except Exception as write_exc:
            print('TF_EFFECT_COLLECTOR_FAILURE_MARKER_ERROR', repr(write_exc), flush=True)

    def end_chunk(self):
        if self.active is None:
            raise RuntimeError('no declared teacher chunk')
        active = self.active
        tmp = None
        try:
            if self.pending is not None or active['errors'] or self.errors:
                raise ValueError('collector capture error or unconsumed candidate; data are invalid')
            records = active['records']
            n = len(records)
            if not n or (active['candidate_calls'], active['execute_calls']) != (n, n):
                raise ValueError('candidate/execute/saved tree coverage is not exactly one-to-one')
            t = self.torch
            context = active['context']
            state = t.cat([r['state'] for r in records], dim=0)
            offsets = t.tensor([0, 1, 1], dtype=t.int64).repeat(n)
            tree_steps = t.tensor([r['step'] for r in records], dtype=t.int64).repeat_interleave(3)
            nodes = t.tensor([0, 1, 2], dtype=t.int64).repeat(n)
            parent = t.tensor([r['parent'] for r in records], dtype=t.int64).repeat_interleave(3)
            depth = t.tensor([r['depth'] for r in records], dtype=t.int64).repeat_interleave(3)
            selected = (nodes == 0) | (nodes - 1 == parent)
            reached = (nodes == 0) | (selected & (depth >= 1))
            committed = ((nodes == 0) & (depth >= 1)) | ((nodes > 0) & selected & (depth == 2))
            scalars = t.tensor([[r['sigma'][o], r['sigma'][o + 1], r['times'][o] / 1000.]
                                for r in records for o in (0, 1, 1)], dtype=t.float32)
            columns = dict(state=state, previous=t.cat([r['previous'] for r in records]),
                target=t.cat([r['target'] for r in records]),
                proposals=t.cat([r['proposals'] for r in records]), scalars=scalars,
                delta=t.tensor([abs(r['sigma'][o + 1] - r['sigma'][o])
                                for r in records for o in (0, 1, 1)], dtype=t.float32),
                task=t.full((3*n,), int(context['task_index']), dtype=t.int64),
                init=t.full((3*n,), int(context['episode_index']), dtype=t.int64),
                episode=t.full((3*n,), int(context['task_index'])*100 + int(context['episode_index']), dtype=t.int64),
                tree_index=t.arange(n, dtype=t.int64).repeat_interleave(3),
                tree_step_index=tree_steps, step_index=tree_steps + offsets, node=nodes,
                accepted_depth=depth, selected_branch=selected, state_reached=reached,
                draft_edge_committed=committed)
            source_dtypes = records[0]['source_dtype']
            if any(r['source_dtype'] != source_dtypes for r in records):
                raise ValueError('source dtype changed within a policy chunk')
            payload = dict(schema=SCHEMA, status='COMPLETE_CHUNK_PENDING_EXTERNAL_COUNTER_CHECK',
                context=context, metadata=self.metadata, source_proof=self.source_proof,
                source_dtypes=source_dtypes, columns=columns,
                trees=dict(sigma_path=t.tensor([r['sigma'] for r in records], dtype=t.float64),
                    target_timesteps=t.tensor([r['times'] for r in records], dtype=t.float64),
                    first_costs=t.tensor([r['first'] for r in records], dtype=t.float32),
                    second_costs=t.tensor([r['second'] for r in records], dtype=t.float32),
                    threshold=t.tensor([r['threshold'] for r in records], dtype=t.float64),
                    selected_parent=t.tensor([r['parent'] for r in records], dtype=t.int64),
                    selected_child=t.tensor([r['child'] for r in records], dtype=t.int64),
                    accepted_depth=t.tensor([r['depth'] for r in records], dtype=t.int64),
                    start_step=t.tensor([r['step'] for r in records], dtype=t.int64),
                    frame_st_id=t.tensor([r['frame'] for r in records], dtype=t.int64),
                    O3_generation=t.tensor([r['generation'] for r in records], dtype=t.int64)),
                candidate_calls=active['candidate_calls'], execute_calls=active['execute_calls'],
                tree_count=n, row_count=3*n, reached_row_count=int(reached.sum()),
                query_and_teacher_bytes_exact=True, original_O3_active_at_every_capture=True,
                extra_teacher_forwards=0, original_results_returned_unchanged=True,
                counterfactual_rows_retained=True, default_training_rows='state_reached_only',
                external_counter_coverage_checked=False, performance_claim=False)
            path = self.output_dir / ('chunk-%05d.pt' % self.chunk_index)
            if path.exists():
                raise FileExistsError(path)
            tmp = path.with_suffix('.pt.tmp')
            if tmp.exists():
                raise FileExistsError(tmp)
            current_bytes = sum(p.stat().st_size for p in self.output_dir.rglob('*') if p.is_file())
            t.save(payload, tmp)
            if current_bytes + tmp.stat().st_size > self.storage_cap:
                raise ValueError('teacher storage cap reached; no shard admission')
            os.replace(tmp, path)
            self.total_trees += n
            self.total_rows += 3*n
            return dict(path=str(path), trees=n, rows=3*n, reached_rows=int(reached.sum()),
                        status=payload['status'], extra_teacher_forwards=0)
        except Exception as exc:
            self._failure_marker(exc)
            raise
        finally:
            # A failed temporary shard remains evidence and counts storage.
            self.active = self.pending = None
            self.chunk_index += 1

    def close(self):
        if self.closed:
            return
        if self.active is not None or self.pending is not None:
            self._failure_marker(RuntimeError('collector closed before complete chunk'))
        if self.installed:
            if self.had_candidate:
                self.s._target_action_candidate_directions = self.original_candidate
            else:
                delattr(self.s, '_target_action_candidate_directions')
            if self.had_execute:
                self.proxy.execute_k2_b2_route = self.original_execute
            else:
                delattr(self.proxy, 'execute_k2_b2_route')
        self.active = self.pending = None
        self.closed = True

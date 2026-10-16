"""Default-off bounded capture of changing verifier inputs, not training data."""
from dataclasses import dataclass
import torch
from contract import Budget, ROWS
from adapter import _tensor_key


class CapturedBeforeTerminal(RuntimeError):
    pass


def cpu_copy(value, *, allow_cuda=False):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' and not allow_cuda:
            raise RuntimeError('CUDA capture not authorized for CPU task')
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v, allow_cuda=allow_cuda) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v, allow_cuda=allow_cuda) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v, allow_cuda=allow_cuda) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError('unknown capture value')


@dataclass
class Query:
    input_dict: dict
    kwargs: dict
    output: torch.Tensor

    def replay(self, model, *, rows, device='cpu', allow_cuda=False):
        if torch.device(device).type != 'cpu' and not allow_cuda:
            raise RuntimeError('CUDA replay not authorized for CPU task')
        inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.input_dict.items()}
        kw = dict(self.kwargs)
        # Capture verified all original row values. Canonical replacement has
        # identical values/batch, prevents per-tree identity changes defeating
        # HC or silently allowing a differently mapped row to hit old history.
        kw['candidate_cache_rows'] = rows
        return model(inp, **kw)


def _history_key(model, cache_name, *, allow_cuda=False):
    keys = []
    for block in model.blocks:
        cache = block.attn1.attn_caches[cache_name]
        if set(cache) != {'k', 'v', 'mask', 'id', 'is_pred'}:
            raise ValueError('unknown original history schema')
        if not allow_cuda and any(t.device.type != 'cpu' for t in cache.values()):
            raise RuntimeError('CUDA history capture not authorized for CPU task')
        keys.append((id(cache), tuple(_tensor_key(cache[k]) for k in ['k','v','mask','id','is_pred'])))
    return tuple(keys)


def _snapshot_history(model, cache_name, *, allow_cuda=False):
    # Same valid-slot schema as the sealed fixed_load_cost snapshot helper.
    # Retain this in RAM for replay; do NOT read undefined torch.empty slots
    # or serialize a new multi-GiB diagnostic fixture automatically.
    snapshots = []
    for block in model.blocks:
        cache = block.attn1.attn_caches[cache_name]
        shape = list(cache['k'].shape)
        if (shape != list(cache['v'].shape) or len(shape) != 4
                or any(list(cache[k].shape) != [shape[1]] for k in ['id','mask','is_pred'])):
            raise ValueError('original history shape changed')
        slots = cache['mask'].nonzero(as_tuple=False).flatten()
        snapshots.append(dict(pool_shape=shape, pool_dtype=str(cache['k'].dtype),
            slots=cpu_copy(slots, allow_cuda=allow_cuda),
            k_valid=cpu_copy(cache['k'][:,slots], allow_cuda=allow_cuda),
            v_valid=cpu_copy(cache['v'][:,slots], allow_cuda=allow_cuda),
            **{k:cpu_copy(cache[k], allow_cuda=allow_cuda) for k in ['id','mask','is_pred']}))
    return snapshots


def capture_queries(server, request, *, enabled=False, allow_cuda=False,
                    expected_trees=29, expected_nonterminal_calls=52, budget=None):
    if not enabled:
        return dict(status='DISABLED', queries=[], complete_policy_chunks=0)
    if expected_trees != 29 or expected_nonterminal_calls != 52:
        raise ValueError('frozen capture bounds require 29 trees/52 calls')
    budget = budget if budget is not None else Budget(cap=52)
    model = server.transformer
    original = model.forward
    had_instance = 'forward' in model.__dict__
    old_instance = model.__dict__.get('forward')
    queries, calls = [], []
    history_key, history_before_candidates, history_cache_name = None, None, None
    stopped = False

    def forward(input_dict, update_cache=0, cache_name='pos', action_mode=False,
                train_mode=False, candidate_cache_rows=None):
        nonlocal stopped, history_key, history_before_candidates, history_cache_name
        if action_mode and update_cache == 1:
            if len(queries) != expected_trees or len(calls) != expected_nonterminal_calls:
                raise ValueError('terminal arrived before exact capture contract')
            stopped = True
            raise CapturedBeforeTerminal('stop BEFORE action terminal enters Transformer')
        if train_mode:
            raise ValueError('training forbidden')
        if len(calls) >= expected_nonterminal_calls:
            raise ValueError('capture nonterminal call cap exceeded')
        if not allow_cuda and any(isinstance(v, torch.Tensor) and v.device.type != 'cpu' for v in input_dict.values()):
            raise RuntimeError('CUDA capture not authorized for CPU task')
        candidate = candidate_cache_rows is not None
        if not candidate and 0 < len(queries) < expected_trees:
            raise ValueError('non-candidate call inside fixed-history verifier segment')
        kw = dict(update_cache=update_cache, cache_name=cache_name, action_mode=action_mode,
                  train_mode=train_mode, candidate_cache_rows=candidate_cache_rows)
        captured = None
        if candidate:
            if (not action_mode or update_cache != 0 or input_dict['noisy_latents'].shape[0] != 14
                    or len(queries) >= expected_trees):
                raise ValueError('read-only batch14 candidate contract')
            rows = cpu_copy(candidate_cache_rows, allow_cuda=allow_cuda)
            if rows.dtype != torch.long or rows.ndim != 1 or tuple(rows.tolist()) != ROWS:
                raise ValueError('original candidate row mapping changed')
            key = _history_key(model, cache_name, allow_cuda=allow_cuda)
            if history_key is None:
                history_key, history_cache_name = key, cache_name
                history_before_candidates = _snapshot_history(model, cache_name, allow_cuda=allow_cuda)
            elif cache_name != history_cache_name or key != history_key:
                raise ValueError('history changed between captured verifier queries')
            captured = (cpu_copy(input_dict, allow_cuda=allow_cuda), cpu_copy(kw, allow_cuda=allow_cuda))
        budget.reserve()
        calls.append(dict(action=action_mode, update=update_cache, candidate=candidate))
        out = original(input_dict, **kw)
        if captured is not None:
            if _history_key(model, cache_name, allow_cuda=allow_cuda) != history_key:
                raise ValueError('candidate forward wrote supposedly read-only history')
            queries.append(Query(*captured, cpu_copy(out, allow_cuda=allow_cuda)))
        return out

    model.forward = forward
    try:
        try:
            server.infer(request)
        except CapturedBeforeTerminal:
            pass
        else:
            raise ValueError('capture unexpectedly completed a policy chunk')
    finally:
        if had_instance:
            model.forward = old_instance
        else:
            delattr(model, 'forward')
    if not stopped or len(queries) != 29 or len(calls) != 52:
        raise ValueError('capture incomplete')
    if ([c['action'] for c in calls[:21]] != [False] * 21
            or sum(c['candidate'] for c in calls) != 29
            or sum(c['action'] and not c['candidate'] for c in calls) != 2):
        raise ValueError('video21/ordinary2/candidate29 call ledger changed')
    first = queries[0].input_dict
    if all(all(torch.equal(q.input_dict[k], first[k]) for k in ['noisy_latents', 'timesteps']) for q in queries[1:]):
        raise ValueError('capture repeated the same query, not real changing inputs')
    return dict(status='PASS', queries=queries, calls=calls, target_calls=52,
                stopped_before_action_terminal=True, complete_policy_chunks=0,
                teacher_training_dataset=False, history_before_candidates=history_before_candidates,
                history_cache_name=history_cache_name, history_constant_across_all29=True,
                history_snapshot_in_RAM_only=True)

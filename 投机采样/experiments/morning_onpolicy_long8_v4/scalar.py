"""O2: unchanged scalar math, one six-error readback and CPU scheduler time.

Only unbound functions/code are retained. Tables are rebuilt inside each action
request. Original API globals and public methods are never mutated.
"""
import ast
import copy
import inspect
import math
import textwrap
import threading
import time
import types

import torch

_lock = threading.RLock()
_actions = {}
_scores = {}
_counts = dict(action_templates=0, score_templates=0, template_compile_sec=0.,
               table_reads=0, error_readbacks=0, error_scalars=0)


def stats():
    return dict(_counts)


def _tables(timesteps, sigmas):
    # No cast: tolist returns the same exact Python scalar as each old item().
    result = (tuple(float(v) for v in timesteps.detach().cpu().tolist()),
              tuple(float(v) for v in sigmas.detach().cpu().tolist()))
    _counts['table_reads'] += 2
    return result


def _materialize(first, second):
    # Each input scalar has already been computed separately by the original
    # expression. Stack/copy preserve those float32 bits without re-reduction.
    values = torch.stack((*first, *second[0], *second[1])).detach().cpu().tolist()
    values = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('direction error is non-finite')
    _counts['error_readbacks'] += 1
    _counts['error_scalars'] += 6
    return values[:2], (values[2:4], values[4:6])


def _source(fn):
    raw = inspect.unwrap(fn)
    tree = ast.parse(textwrap.dedent(inspect.getsource(raw)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('one original function required')
    return raw, tree


def _dump(tree):
    return ast.dump(tree, include_attributes=False)


def _compile(tree, raw, **overrides):
    namespace = {**raw.__globals__, **overrides}
    ast.fix_missing_locations(tree)
    exec(compile(tree, raw.__code__.co_filename + ':private-O2', 'exec'), namespace)
    return namespace[raw.__name__]


class API:
    def __init__(self, original, score):
        self.original = original
        self.score_k2_b2_tree = score

    def __getattr__(self, name):
        return getattr(self.original, name)


def api(original):
    key = original.score_k2_b2_tree
    with _lock:
        entry = _scores.get(key)
        if entry is None:
            start = time.perf_counter()
            raw_norm, norm = _source(original._weighted_direction_l2)
            gold_norm = copy.deepcopy(norm)
            body = norm.body[0].body
            # Preserve all shape/channel/scale checks and the exact value AST.
            if [type(n) for n in body[-4:]] != [ast.Assign, ast.Assign, ast.If, ast.Return]:
                raise ValueError('original norm return structure changed')
            removed = copy.deepcopy(body[-3:])
            body[-3:] = ast.parse('return value').body
            norm.body[0].returns = None
            restored = copy.deepcopy(norm)
            restored.body[0].body[-1:] = removed
            restored.body[0].returns = gold_norm.body[0].returns
            if _dump(restored) != _dump(gold_norm):
                raise ValueError('norm executable math changed')
            tensor_norm = _compile(norm, raw_norm)
            raw_score, score = _source(key)
            gold_score = copy.deepcopy(score)
            score.body[0].body.insert(-1, ast.parse(
                'first, second = __o2_materialize(first, second)').body[0])
            restored = copy.deepcopy(score)
            del restored.body[0].body[-2]
            if _dump(restored) != _dump(gold_score):
                raise ValueError('score structure changed')
            fn = _compile(score, raw_score, _weighted_direction_l2=tensor_norm,
                          __o2_materialize=_materialize)
            entry = API(original, fn)
            _scores[key] = entry
            _counts['score_templates'] += 1
            _counts['template_compile_sec'] += time.perf_counter() - start
        return entry


def action_function(original):
    """Bind the original helper with three narrow, reversible AST edits."""
    original = getattr(original, '__func__', original)
    with _lock:
        if original in _actions:
            return _actions[original]
        start = time.perf_counter()
        raw, tree = _source(original)
        gold = copy.deepcopy(tree)
        originals = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name == 'api':
                originals[name] = copy.deepcopy(node.value)
                node.value = ast.Call(func=ast.Name(id='__o2_api', ctx=ast.Load()),
                                      args=[node.value], keywords=[])
            elif name in ('sigma_path', 'target_times'):
                originals[name] = copy.deepcopy(node.value)
                text = ('[__o2_sigmas[step_index], __o2_sigmas[step_index+1], '
                        '__o2_sigmas[step_index+2] if step_index+2<len(__o2_sigmas) else 0.]'
                        if name == 'sigma_path' else
                        '[__o2_times[step_index], __o2_times[step_index+1], __o2_times[step_index+2]]')
                node.value = ast.parse(text, mode='eval').body
        if set(originals) != {'api', 'sigma_path', 'target_times'}:
            raise ValueError('exact original table sites required')
        cpu_time_nodes = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef) and node.name == 'scheduler_step'):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == 'tensor':
                        for kw in inner.keywords:
                            if kw.arg == 'device':
                                cpu_time_nodes.append(copy.deepcopy(kw.value))
                                kw.value = ast.Constant(value='cpu')
        if len(cpu_time_nodes) != 1:
            raise ValueError('one exact scheduler scalar construction required')
        table_node = ast.parse(
            '__o2_times, __o2_sigmas = __o2_tables(action_timesteps, self.action_scheduler.sigmas)').body[0]
        # Place table reads after original setup, immediately before the loop.
        body = tree.body[0].body
        loop_index = next(i for i, node in enumerate(body) if isinstance(node, ast.While))
        body.insert(loop_index, table_node)
        restored = copy.deepcopy(tree)
        del restored.body[0].body[loop_index]
        for node in ast.walk(restored):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in originals:
                    node.value = copy.deepcopy(originals[name])
            if isinstance(node, ast.FunctionDef) and node.name == 'scheduler_step':
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == 'tensor':
                        for kw in inner.keywords:
                            if kw.arg == 'device':
                                kw.value = copy.deepcopy(cpu_time_nodes[0])
        if _dump(restored) != _dump(gold):
            raise ValueError('original action executable structure failed reverse proof')
        result = _compile(tree, raw, __o2_api=api, __o2_tables=_tables)
        _actions[original] = result
        _counts['action_templates'] += 1
        _counts['template_compile_sec'] += time.perf_counter() - start
        return result


def bind(original, server):
    return types.MethodType(action_function(original), server)

"""Instance-only guarded history callback; original O3 lifecycle is retained.

No public module, model, scheduler or original adapter globals are modified.
The guarded wrapper changes only the original wrapper's lazy final kernel call.
"""
import ast
import copy
import inspect
import textwrap


def guarded_attention(kernel):
    from template import frozen
    raw = frozen.SharedPrefix._history.__globals__['shared_attention']
    tree = ast.parse(textwrap.dedent(inspect.getsource(raw)))
    original = copy.deepcopy(tree)
    body = tree.body[0].body
    expected = ast.parse('from prefix_kernel import run\nreturn run(q, pk, pv, k, v, slots, rows)').body
    if ast.dump(ast.Module(body=body[-2:], type_ignores=[]), include_attributes=False) != ast.dump(ast.Module(body=expected, type_ignores=[]), include_attributes=False):
        raise RuntimeError('original shared wrapper final import/call changed')
    removed = copy.deepcopy(body[-2:])
    body[-2:] = ast.parse('return __s04_kernel(q, pk, pv, k, v, slots, rows)').body
    restored = copy.deepcopy(tree)
    restored.body[0].body[-1:] = removed
    if ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise RuntimeError('shared wrapper reverse executable proof failed')
    ast.fix_missing_locations(tree)
    namespace = {**raw.__globals__, '__s04_kernel': kernel}
    exec(compile(tree, raw.__code__.co_filename + ':private-S04', 'exec'), namespace)
    return namespace[raw.__name__]


def control(server, arm, observer, candidate):
    import os
    from experiment import ENABLED, APPROVED
    from measured_online import OnlineO3, MeasuredO3, frozen_online
    from template import clone_function, frozen

    native_wrapper = frozen.SharedPrefix._history.__globals__['shared_attention']

    def kernel(q, pk, pv, k, v, slots, rows):
        if os.environ.get(ENABLED) != '1' or os.environ.get(APPROVED) != '1':
            raise RuntimeError('original default-off CUDA authorization required')
        prefix = slots.numel()
        if prefix <= 0:
            raise RuntimeError('actual validated nonempty history required')
        group = 2 if (prefix + 32 + 63) // 64 == (prefix + 16 + 63) // 64 else 1
        observer.groups[group] = observer.groups.get(group, 0) + 1
        return candidate.grouped(q, pk, pv, k, v, slots, group)

    custom_wrapper = guarded_attention(kernel)

    def attention(q, pk, pv, k, v, slots, rows, **kwargs):
        before = observer.attention_pre(q, k, v, slots, rows)
        fn = native_wrapper if arm == 'native' else custom_wrapper
        output = fn(q, pk, pv, k, v, slots, rows, **kwargs)
        observer.attention_post(q, k, v, slots, rows, output, before)
        return output

    class ObservedPrefix(MeasuredO3):
        _history = clone_function(frozen.SharedPrefix._history, shared_attention=attention)

    class ObservedO3(OnlineO3):
        _base_install = clone_function(frozen_online.OnlineShared.install, SharedPrefix=ObservedPrefix)

    return ObservedO3(server, enabled=True, allow_cuda=True, cpu_reference_enabled=False).install()

"""Prefer the longest threshold-admissible prefix; retain global cost ties."""
import math
import types

NAME = 'depth_then_cost'


def depth(first, second, threshold):
    return 0 if first > threshold else 1 if second > threshold else 2


def choose(api, scores, threshold):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('finite nonnegative frozen threshold required')
    pairs = []
    for parent in range(2):
        first = float(scores.first_costs[parent])
        for child in range(2):
            second = float(scores.second_costs[parent][child])
            if not math.isfinite(first + second) or min(first, second) < 0:
                raise ValueError('finite nonnegative verifier costs required')
            pairs.append((parent, child, first, second))
    parent, child, first, second = min(pairs, key=lambda q:
        (-depth(q[2], q[3], threshold), q[2] + q[3], q[0], q[1]))
    return api.K2B2Route(parent=parent, child=child, first_cost=first,
        second_cost=second, total_cost=first + second, selector=NAME)


class ObservedAPI:
    """Per-instance adapter; original target/model/API module stays untouched."""
    def __init__(self, original, server, effective='global'):
        if effective not in ('global', NAME):
            raise ValueError('two declared selectors only')
        self.original, self.server, self.effective = original, server, effective
        self.records = []
        self.pending = None
        self.context = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def select_k2_b2_route(self, scores, *, selector):
        threshold = float(self.server._p0_action_draft_threshold)
        original = self.original.select_k2_b2_route(scores, selector='global')
        candidate = choose(self.original, scores, threshold)
        if selector != self.effective:
            raise ValueError('effective selector provenance mismatch')
        chosen = original if selector == 'global' else candidate
        old_depth = depth(original.first_cost, original.second_cost, threshold)
        new_depth = depth(candidate.first_cost, candidate.second_cost, threshold)
        if new_depth < old_depth:
            raise ValueError('same-tree admissible depth decreased')
        self.pending = dict(first_costs=list(scores.first_costs),
            second_costs=[list(v) for v in scores.second_costs], threshold=threshold,
            global_parent=original.parent, global_child=original.child,
            depth_parent=candidate.parent, depth_child=candidate.child,
            global_depth=old_depth, maximum_admissible_depth=new_depth,
            actual_parent=chosen.parent, actual_child=chosen.child,
            effective_selector=selector,
            changed_route=(original.parent, original.child) != (candidate.parent, candidate.child))
        return chosen

    def execute_k2_b2_route(self, tree, scores, route, **kwargs):
        result = self.original.execute_k2_b2_route(tree, scores, route, **kwargs)
        if self.pending is None or kwargs['threshold'] != self.pending['threshold']:
            raise ValueError('one score/select/execute record required')
        expected = (self.pending['global_depth'] if self.effective == 'global'
                    else self.pending['maximum_admissible_depth'])
        if result.accepted_depth != expected:
            raise ValueError('executed prefix disagrees with frozen gate')
        self.records.append({**self.pending,
            'sigma': float(tree.sigma_path[0]), 'next_sigma': float(tree.sigma_path[1]),
            'timestep': float(tree.target_timesteps[0]),
            'frame_st_id': int(self.server.frame_st_id),
            'context': dict(self.context),
            'actual_depth': result.accepted_depth})
        self.pending = None
        return result


def install(server, effective='global'):
    original = server._load_p0_shadow_api()
    proxy = ObservedAPI(original, server, effective)
    server._load_p0_shadow_api = lambda: proxy
    server._p0_action_draft_selector = effective
    original_infer = server.infer
    def context_infer(_server, request):
        proxy.context = dict(request.get('p0_trace_context') or {})
        return original_infer(request)
    server.infer = types.MethodType(context_infer, server)
    return proxy

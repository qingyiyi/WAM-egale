"""Longest valid prefix, then cost of accepted edges only."""
import math
import types
from paths import OWN, load

BASELINE_NAME = 'depth_then_cost'
NAME = 'prefix_then_cost'
baseline = load('prefix_frozen_baseline_selection', OWN.parent / 'bonus_acceptance_v2/selection.py')


def depth(first, second, threshold):
    return 0 if first > threshold else 1 if second > threshold else 2


def choose(api, scores, threshold):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('finite nonnegative frozen threshold required')
    if len(scores.first_costs) != 2 or len(scores.second_costs) != 2:
        raise ValueError('exact two-parent tree required')
    pairs = []
    for parent in range(2):
        if len(scores.second_costs[parent]) != 2:
            raise ValueError('exact two-child parent required')
        first = float(scores.first_costs[parent])
        for child in range(2):
            second = float(scores.second_costs[parent][child])
            if not all(math.isfinite(x) and x >= 0 for x in (first, second)):
                raise ValueError('finite nonnegative verifier costs required')
            n = depth(first, second, threshold)
            cost = first + second if n == 2 else first if n == 1 else 0.
            if not math.isfinite(first + second):
                raise ValueError('finite route diagnostic cost required')
            pairs.append((parent, child, first, second, n, cost))
    if max(q[4] for q in pairs) == 0:
        old = baseline.choose(api, scores, threshold)
        parent, child, first, second = old.parent, old.child, old.first_cost, old.second_cost
    else:
        parent, child, first, second, _, _ = min(pairs,
            key=lambda q: (-q[4], q[5], q[0], q[1]))
    return api.K2B2Route(parent=parent, child=child, first_cost=first,
        second_cost=second, total_cost=first + second, selector=NAME)


def prefix_cost(route, threshold):
    n = depth(route.first_cost, route.second_cost, threshold)
    return route.total_cost if n == 2 else route.first_cost if n == 1 else 0.


class ObservedAPI:
    """Per-instance adapter; original target/model/API module stays untouched."""
    def __init__(self, original, server, effective=BASELINE_NAME):
        if effective not in (BASELINE_NAME, NAME):
            raise ValueError('two declared selectors only')
        self.original, self.server, self.effective = original, server, effective
        self.records = []
        self.pending = None
        self.context = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def select_k2_b2_route(self, scores, *, selector):
        threshold = float(self.server._p0_action_draft_threshold)
        original = baseline.choose(self.original, scores, threshold)
        candidate = choose(self.original, scores, threshold)
        if selector != self.effective:
            raise ValueError('effective selector provenance mismatch')
        chosen = original if selector == BASELINE_NAME else candidate
        old_depth = depth(original.first_cost, original.second_cost, threshold)
        new_depth = depth(candidate.first_cost, candidate.second_cost, threshold)
        if new_depth != old_depth or prefix_cost(candidate, threshold) > prefix_cost(original, threshold):
            raise ValueError('same-tree accepted depth/cost contract violated')
        effective_parent_changed = old_depth > 0 and original.parent != candidate.parent
        self.pending = dict(first_costs=list(scores.first_costs),
            second_costs=[list(v) for v in scores.second_costs], threshold=threshold,
            baseline_parent=original.parent, baseline_child=original.child,
            prefix_parent=candidate.parent, prefix_child=candidate.child,
            baseline_depth=old_depth, maximum_admissible_depth=new_depth,
            baseline_prefix_cost=prefix_cost(original, threshold),
            candidate_prefix_cost=prefix_cost(candidate, threshold),
            actual_parent=chosen.parent, actual_child=chosen.child,
            effective_selector=selector,
            effective_parent_changed=effective_parent_changed,
            bookkeeping_child_only_change=(old_depth == 1 and not effective_parent_changed
                and original.child != candidate.child),
            changed_route=(original.parent, original.child) != (candidate.parent, candidate.child))
        return chosen

    def execute_k2_b2_route(self, tree, scores, route, **kwargs):
        result = self.original.execute_k2_b2_route(tree, scores, route, **kwargs)
        if self.pending is None or kwargs['threshold'] != self.pending['threshold']:
            raise ValueError('one score/select/execute record required')
        expected = self.pending['maximum_admissible_depth']
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


def install(server, effective=BASELINE_NAME):
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

import math
NAME="prefix_then_cost"
import depth_selector as baseline
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
            if not all((math.isfinite(x) and x >= 0 for x in (first, second))):
                raise ValueError('finite nonnegative verifier costs required')
            n = depth(first, second, threshold)
            cost = first + second if n == 2 else first if n == 1 else 0.0
            if not math.isfinite(first + second):
                raise ValueError('finite route diagnostic cost required')
            pairs.append((parent, child, first, second, n, cost))
    if max((q[4] for q in pairs)) == 0:
        old = baseline.choose(api, scores, threshold)
        parent, child, first, second = (old.parent, old.child, old.first_cost, old.second_cost)
    else:
        parent, child, first, second, _, _ = min(pairs, key=lambda q: (-q[4], q[5], q[0], q[1]))
    return api.K2B2Route(parent=parent, child=child, first_cost=first, second_cost=second, total_cost=first + second, selector=NAME)

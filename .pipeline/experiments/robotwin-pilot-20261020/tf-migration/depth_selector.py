import math
NAME="depth_then_cost"
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
    parent, child, first, second = min(pairs, key=lambda q: (-depth(q[2], q[3], threshold), q[2] + q[3], q[0], q[1]))
    return api.K2B2Route(parent=parent, child=child, first_cost=first, second_cost=second, total_cost=first + second, selector=NAME)

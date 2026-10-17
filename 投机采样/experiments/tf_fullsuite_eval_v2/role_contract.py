"""Independent S/T audits; original ObservedAPI assertions are not changed."""
from __future__ import annotations

from collections import Counter


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_request(*, kind, role, phase, routes, events, shared_requests,
                  target_counts, calls, target_action_updates, timing):
    require(all(v['attempted'] == v['returned'] and v['failed'] == 0
                for v in calls.values()), 'attempted transformer forwards must all return')
    count = lambda key: calls.get(key, {}).get('attempted', 0)
    if kind == 'reset':
        require(not calls and not routes and not events and shared_requests == 0,
            'bank reset must execute no target/candidate forward')
        return dict(status='PASS_COMPLETE_ROLE_REQUEST', kind=kind, g=0)
    if kind == 'cache':
        require(set(calls) == {('video', 2, 2, False), ('action', 2, 2, False)}
            and all(v['attempted'] == 1 for v in calls.values()),
            'one video plus one action original cache update required')
        require(not routes and not events and shared_requests == 0
            and phase.get('committed_scheduler_update_count', 0) == 0,
            'cache requests cannot count as policy or commits')
        return dict(status='PASS_COMPLETE_ROLE_REQUEST', kind=kind, g=0)
    require(kind == 'policy' and role in ('S', 'T'), 'complete policy role required')
    action = {k: v for k, v in calls.items() if k[0] == 'action'}
    video = {k: v for k, v in calls.items() if k[0] == 'video'}
    require(set(action).issubset({('action', 0, 2, False),
        ('action', 0, 14, True), ('action', 1, 2, False)}), 'action geometry or cache role changed')
    require(set(video) == {('video', 0, 2, False), ('video', 1, 2, False)}
        and count(('video', 0, 2, False)) == 20 and count(('video', 1, 2, False)) == 1,
        'full20 video updates plus terminal forward required')
    require(count(('action', 1, 2, False)) == 1, 'one terminal action cache call required')
    if role == 'T':
        require(not routes and not events and shared_requests == 0,
            'native T must not execute speculative helper/O3')
        require(not any(phase.get(k, 0) for k in ('tree_count',
            'committed_scheduler_update_count', 'committed_draft_scheduler_update_count',
            'bonus_target_scheduler_update_count')), 'T cannot fabricate S phase counters')
        require(target_action_updates == 50 and count(('action', 0, 2, False)) == 50
            and count(('action', 0, 14, True)) == 0, 'T requires50 scheduler updates /51 B2')
        for key, expected in dict(action_scheduler_updates=50, action_target_intents=51,
                action_target_returned=51, action_candidate_calls=0,
                action_terminal_calls=1, action_regular_calls=50, action_CFG2_calls=51).items():
            require(target_counts.get(key, 0) == expected, 'canonical T observer mismatch: ' + key)
        natural_edges, tree_commits, g, depths = 0, 0, 50, Counter()
    else:
        require(target_action_updates == 0, 'S counterfactual transitions cannot count as T commits')
        require(shared_requests == 1 and len(events) == 1
            and all(e.get('enabled') and e.get('ok') and e.get('closed') for e in events),
            'S requires one unchanged, enabled and closed O3 request')
        require(len(routes) == phase.get('tree_count', 0)
            == count(('action', 0, 14, True)) == phase.get('verifier_forward_count', 0),
            'tree/route/B14/phase coverage mismatch')
        depths = Counter(int(route['actual_depth']) for route in routes)
        require(set(depths).issubset({0, 1, 2}), 'frozen K2 depths required')
        for depth in (0, 1, 2):
            require(depths[depth] == phase.get('accepted_depth_' + str(depth), 0),
                'natural accepted depth does not match production counter')
        natural_edges = depths[1] + 2 * depths[2]
        tree_commits = depths[0] + 2 * depths[1] + 3 * depths[2]
        singles = phase.get('target_forward_count', 0)
        g = phase.get('committed_scheduler_update_count', 0)
        require(g == 50 == singles + tree_commits, 'actual commits must sum to50')
        require(count(('action', 0, 2, False)) == singles
            and phase.get('committed_draft_scheduler_update_count', 0) == natural_edges
            and phase.get('bonus_target_scheduler_update_count', 0) == depths[2]
            and phase.get('ordinary_target_scheduler_update_count', 0)
                == singles + depths[0] + depths[1], 'anchor/tail/bonus/fallback mismatch')
    require(timing['action_complete'] and timing['sum_g'] == g
        and abs(timing['sum_round_host_seconds'] - timing['action_host_seconds']) <= 1e-6,
        'rounds must partition the full native action interval and all50 commits')
    return dict(status='PASS_COMPLETE_ROLE_REQUEST', kind=kind, role=role, g=g,
        natural_accepted_draft_edges=natural_edges, actual_tree_commits=tree_commits,
        accepted_depth_counts={str(k): depths[k] for k in (0, 1, 2)},
        sum_T_round_sec=timing['sum_round_cuda_seconds'],
        sum_T_round_host_sec=timing['sum_round_host_seconds'],
        cost_per_valid_commit=timing['sum_round_cuda_seconds']/g,
        clock='CUDA events on original current stream; exploratory shared-resource timing',
        time_boundary='native action_total including anchor,tail,fallback,bonus,terminal',
        performance_claim=False)

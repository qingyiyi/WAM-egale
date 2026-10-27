"""Historical PF2 at whole policy-chunk granularity, reset per episode."""


def policy_role(arm, ordinal):
    if arm not in ('T', 'S', 'P') or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError((arm, ordinal))
    if arm == 'P':
        return 'T' if ordinal % 3 == 2 else 'S'
    return arm

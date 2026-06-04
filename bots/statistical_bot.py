from math import comb
from collections import Counter

BLUFF_THRESHOLD = 0.35


def _binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def statistical_bot(state):
    legal      = state['legal_actions']
    legal_bids = state['legal_bids']
    my_rolls   = state['your_rolls']
    prev       = state['previous_bid']
    total      = state['total_dice']
    unknown    = total - len(my_rolls)

    if prev is None:
        counts  = Counter(my_rolls)
        jokers  = counts.get(1, 0)
        best_f  = max(range(2, 7), key=lambda f: counts.get(f, 0) + jokers)
        own_cnt = counts.get(best_f, 0) + jokers
        qty     = max(1, round((own_cnt + unknown * (2 / 6)) * 0.75))
        target  = (qty, best_f)
        if target in legal_bids:
            return legal[legal_bids.index(target)]
        return next(a for a, b in zip(legal, legal_bids) if b not in ('liar', 'spot_on'))

    qty, face = prev
    p_hit     = 2 / 6 if face != 1 else 1 / 6
    own_cnt   = sum(1 for r in my_rolls if r == face or (face != 1 and r == 1))
    p_valid   = _binom_sf(qty - own_cnt, unknown, p_hit)

    if p_valid < BLUFF_THRESHOLD and 'liar' in legal_bids:
        return legal[legal_bids.index('liar')]

    for nf in range(face + 1, 7):
        if (qty, nf) in legal_bids:
            return legal[legal_bids.index((qty, nf))]
    for nq in range(qty + 1, total + 1):
        for nf in range(2, 7):
            if (nq, nf) in legal_bids:
                return legal[legal_bids.index((nq, nf))]

    return legal[legal_bids.index('liar')]

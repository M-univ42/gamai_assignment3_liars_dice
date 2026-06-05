import random


def random_bot(state):
    legal      = state['legal_actions']
    legal_bids = state['legal_bids']

    if state['previous_bid'] and 'liar' in legal_bids and random.random() < 0.20:
        return legal[legal_bids.index('liar')]

    bid_actions = [a for a, b in zip(legal, legal_bids) if b not in ('liar', 'spot_on')]
    if not bid_actions:
        return legal[legal_bids.index('liar')]
    return random.choice(bid_actions)

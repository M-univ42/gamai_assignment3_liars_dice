import random
from math import comb
from collections import defaultdict


TRAIN_ITERS = 300

cumulRegrets  = defaultdict(lambda: defaultdict(float))
cumulStrategy = defaultdict(lambda: defaultdict(float))


def _allBids(prevBid, totalDice):
    bids = []
    for n in range(1, totalDice + 1):
        for v in range(2, 7):
            if prevBid is None:
                bids.append((n,v))
            else:
                pn, pv = prevBid
                if n > pn or (n == pn and v > pv):
                    bids.append((n,v))
    return bids


def _countDice(allDice, val):
    return sum(1 for d in allDice if d == val or (d == 1 and val != 1))


def _evalAction(bid, allDice, totalDice):
    if bid == 'liar':
        return None
    if bid == 'spot_on':
        return None
    amt, face = bid
    p = 2/6 if face != 1 else 1/6
    pSurvive = sum(comb(totalDice, i) * p**i * (1-p)**(totalDice-i) for i in range(amt, totalDice+1))
    return pSurvive * 2 - 1


def _evalChallenge(bid, action, allDice):
    if bid is None:
        return 0.0
    amt, face = bid
    actual = _countDice(allDice, face)
    if action == 'liar':
        return +1.0 if actual < amt else -1.0
    elif action == 'spot_on':
        return +1.5 if actual == amt else -1.0
    return 0.0


def _regretMatch(regretDict, actions):
    pos = {a: max(0.0, regretDict[a]) for a in actions}
    total = sum(pos.values())
    if total > 0:
        return {a: pos[a]/total for a in actions}
    return {a: 1.0/len(actions) for a in actions}


def _infoKey(myRolls, prevBid):
    return (tuple(sorted(myRolls)), prevBid)


def cfr_bot(game):
    myId   = game['your_id']
    myDice = game['your_rolls']
    active = game['active_players']
    diceCounts = game['dice_counts']
    totalDice  = game['total_dice']
    prevBid    = game.get('previous_bid')
    legalActs  = game['legal_actions']
    legalBids  = game['legal_bids']
    isFirst    = (prevBid is None)

    internalActions = []
    if not isFirst:
        internalActions.append('liar')
        internalActions.append('spot_on')
    internalActions += _allBids(prevBid, totalDice)

    infoKey = _infoKey(myDice, prevBid)

    for _ in range(TRAIN_ITERS):
        oppDice = []
        for pid in active:
            if pid != myId:
                oppDice += [random.randint(1,6) for _ in range(diceCounts[pid])]
        allDice = list(myDice) + oppDice

        strat = _regretMatch(cumulRegrets[infoKey], internalActions)

        vals = {}
        for a in internalActions:
            if a in ('liar', 'spot_on'):
                vals[a] = _evalChallenge(prevBid, a, allDice)
            else:
                vals[a] = _evalAction(a, allDice, totalDice)

        expected = sum(strat[a] * vals[a] for a in internalActions)

        for a in internalActions:
            cumulRegrets[infoKey][a]  += vals[a] - expected
            cumulStrategy[infoKey][a] += strat[a]

    total = sum(max(0, cumulStrategy[infoKey][a]) for a in internalActions)
    if total == 0:
        chosen = random.choice(internalActions)
    else:
        r = random.random() * total
        chosen = internalActions[-1]
        running = 0
        for a in internalActions:
            running += max(0, cumulStrategy[infoKey][a])
            if running >= r:
                chosen = a
                break

    if chosen == 'liar':
        if 'liar' in legalBids:
            return legalActs[legalBids.index('liar')]
    elif chosen == 'spot_on':
        if 'spot_on' in legalBids:
            return legalActs[legalBids.index('spot_on')]
    elif chosen in legalBids:
        return legalActs[legalBids.index(chosen)]

    for a, b in zip(legalActs, legalBids):
        if b not in ('liar', 'spot_on'):
            return a
    return legalActs[0]

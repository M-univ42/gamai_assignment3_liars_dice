
import math
from math import comb
import random
from copy import deepcopy


ITERATIONS = 1000
MAX_DEPTH  = 20
numWorlds  = 20
C = 1.4


def get_all_bids(prev_bet, totalDice):
    bids = []
    for n in range(1, totalDice + 1):
        for v in range(2, 7):
            if prev_bet is None:
                bids.append((n, v))
            else:
                pn, pv = prev_bet
                if n > pn or (n == pn and v > pv):
                    bids.append((n,v))
    return bids


def countDice(dice_lists, val):
    total = 0
    for dice in dice_lists:
        for d in dice:
            if d == val or (d == 1 and val != 1):
                total += 1
    return total


def getLegalActions(prev_bet, totalDice, firstPlayer):
    actions = []

    if not firstPlayer:
        actions.append("bluff")
        actions.append("spot on")

    actions += get_all_bids(prev_bet, totalDice)
    return actions


def sample_world(game):
    myId = game["your_id"]
    active = game["active_players"]
    dice_counts = game["dice_counts"]

    # build players list, randomise opponent dice
    players = []
    for pid in active:
        if pid == myId:
            rolls = list(game["your_rolls"])
        else:
            rolls = [random.randint(1, 6) for _ in range(dice_counts[pid])]
        players.append({"id": pid, "num_die": dice_counts[pid], "rolls": rolls})

    return {
        "players": players,
        "your_id": myId,
        "your_rolls": list(game["your_rolls"]),
        "previous_bet": game.get("previous_bid"),
    }


def rollout_policy(prev_bet, totalDice, allDice, isFirst):

    if prev_bet is None or isFirst:
        return (1, random.randint(2,6))

    amount, value = prev_bet
    actual = countDice(allDice, value)

    p = actual / totalDice if totalDice > 0 else 0.0

    bid_pressure = amount / totalDice if totalDice > 0 else 0.0

    if bid_pressure > 0.8 and random.random() < 0.8:
        return "bluff"

    if p < 0.25 and random.random() < 0.7:
        return "bluff"

    if 0.4 < p < 0.6 and random.random() < 0.15:
        return "spot on"

    bids = get_all_bids(prev_bet, totalDice)
    if not bids:
        return "bluff"
    if p > 0.6:
        return random.choice(bids[:max(1, len(bids)//2)])
    else:
        return random.choice(bids[:3])


def simulate(world, action, ourId):
    allDice = [p["rolls"] for p in world["players"]]
    n_dice = sum(p["num_die"] for p in world["players"])
    prev_bet = world.get("previous_bet")
    myDice = next(p["num_die"] for p in world["players"] if p["id"] == ourId)

    if action == "bluff":
        if prev_bet is None:
            return 0.0
        amt, val = prev_bet
        actual = countDice(allDice, val)
        return +1.0 if actual < amt else -1.0 / max(1, myDice)

    elif action == "spot on":
        if prev_bet is None:
            return 0.0
        amt, val = prev_bet
        actual = countDice(allDice, val)
        return +1.5 if actual == amt else -1.0

    else:
        amt, val = action
        p_face = 2/6 if val != 1 else 1/6
        p_survive = sum(comb(n_dice, i) * p_face**i * (1-p_face)**(n_dice-i) for i in range(amt, n_dice+1))
        return p_survive * 2 - 1


def run_mcts(world, ourId, iters=ITERATIONS):
    totalDice = sum(p["num_die"] for p in world["players"])
    prevBet = world.get("previous_bet")
    isFirst = (prevBet is None)

    actions = getLegalActions(prevBet, totalDice, isFirst)
    counts = {a: 0 for a in actions}
    totals = {a: 0.0 for a in actions}

    for i in range(1, iters + 1):
        bestA = None
        bestVal = -float("inf")
        for a in actions:
            if counts[a] == 0:
                bestA = a
                break
            u = totals[a] / counts[a] + C * math.sqrt(math.log(i) / counts[a])
            if u > bestVal:
                bestVal = u
                bestA = a

        r = simulate(world, bestA, ourId)
        counts[bestA] += 1
        totals[bestA] += r

    return {a: totals[a] / counts[a] for a in actions if counts[a] > 0}


def mcts_bot(game,debug=False):
    ourId = game["your_id"]
    legal_actions = game["legal_actions"]
    legal_bids = game["legal_bids"]
    allScores = {}

    for _ in range(numWorlds):
        w = sample_world(game)
        s = run_mcts(w, ourId)

        for action, score in s.items():
            if action not in allScores:
                allScores[action] = []
            allScores[action].append(score)

    if not allScores:
        # fallback: smallest valid bid
        for a, b in zip(legal_actions, legal_bids):
            if b not in ('liar', 'spot_on'):
                return a
        return legal_actions[0]

    # pick best action by average score
    best = max(allScores, key=lambda a: sum(allScores[a]) / len(allScores[a]))
    if debug:
        print(f"[MCTS] scores: { {str(a): round(sum(v)/len(v), 2) for a, v in allScores.items()} }")
        print(f"[MCTS] chose: {best}")

    # map internal action back to integer action code
    if best == "bluff" and 'liar' in legal_bids:
        return legal_actions[legal_bids.index('liar')]
    elif best == "spot on" and 'spot_on' in legal_bids:
        return legal_actions[legal_bids.index('spot_on')]
    elif best in legal_bids:
        return legal_actions[legal_bids.index(best)]

    # fallback: smallest valid bid
    for a, b in zip(legal_actions, legal_bids):
        if b not in ('liar', 'spot_on'):
            return a
    return legal_actions[0]
import random
import math



### BOTS ###

def crapbot(game, silent = True):
    value = 0
    if game['current_id'] == game['your_id']:
        if not silent:
            print ('im player {0}'.format(game['your_id']))
            print ('my dice: ',game['your_rolls'])

        if game['your_id'] == 3:
            return 'spot on'

        if not game['previous_bet']:
            number = 5

        if game['previous_bet']:
            if game['previous_bet'][0] > 7:
                return 'bluff'
            number = game['previous_bet'][0] + 1
            value = 5

        return (number, value)

    return



def random_bot(game):
    if game['current_id'] != game['your_id']:
        return None

    previous_bet = game['previous_bet']

    if previous_bet and random.random() < 0.2:
        return 'bluff'

    if previous_bet and random.random() < 0.1:
        return 'spot on'

    if not previous_bet:
        number = 1
        value = random.randint(2, 6)
        return (number, value)

    prev_number, prev_value = previous_bet

    if prev_value < 6:
        return (prev_number, prev_value + 1)
    else:
        return (prev_number + 1, 2)
    


def statistical_bot(game):
    # Only act on your turn
    if game["current_id"] != game["your_id"]:
        return None

    own_dice = game["your_rolls"]
    previous_bet = game["previous_bet"]

    # Total dice still in the game
    total_dice = sum(player["num_die"] for player in game["players"])

    def probability_at_least(bid):
        bid_amount, bid_value = bid

        own_matches = sum(
            1 for die in own_dice
            if die == bid_value or die == 1
        )

        # P(match) = P(face) + P(joker)
        p_match = 1 / 3

        unknown_dice = total_dice - len(own_dice)

        # Amount still needed from opponents
        needed = bid_amount - own_matches

        if needed <= 0:
            return 1.0

        if needed > unknown_dice:
            return 0.0

        probability = 0.0

        for k in range(needed, unknown_dice + 1):
            probability += (
                math.comb(unknown_dice, k)
                * (p_match ** k)
                * ((1 - p_match) ** (unknown_dice - k))
            )

        return probability

    def probability_exactly(bid):
        bid_amount, bid_value = bid

        own_matches = sum(
            1 for die in own_dice
            if die == bid_value or die == 1
        )

        p_match = 1 / 3

        unknown_dice = total_dice - len(own_dice)

        needed = bid_amount - own_matches

        if needed < 0 or needed > unknown_dice:
            return 0.0

        return (
            math.comb(unknown_dice, needed)
            * (p_match ** needed)
            * ((1 - p_match) ** (unknown_dice - needed))
        )

    def next_bid():
        if previous_bet is None:

            # Ignore jokers for opening bid
            non_jokers = [die for die in own_dice if die != 1]

            if non_jokers:
                best_value = max(
                    set(non_jokers),
                    key=non_jokers.count
                )
            else:
                best_value = random.randint(2, 6)

            return (1, best_value)

        number, value = previous_bet

        # Increase face value first
        if value < 6:
            return (number, value + 1)

        # Then increase amount
        return (number + 1, 2)

    if previous_bet is None:
        return next_bid()


    prob_true = probability_at_least(previous_bet)
    prob_exact = probability_exactly(previous_bet)

    if prob_exact > 0.35:
        return "spot on"

    if prob_true < 0.30:
        return "bluff"

    return next_bid()
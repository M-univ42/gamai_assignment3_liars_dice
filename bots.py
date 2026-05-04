import random


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
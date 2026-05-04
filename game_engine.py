#!/bin/python
# Evan Widloski - 2015-10-09

from random import randint
from itertools import cycle

### GAME ###

class Player(object):
    def __init__(self,function,id):
        self.function = function
        self.bets = []
        self.num_die = 5
        self.rolls = []
        self.id = id


def active_players(players):
    return [player for player in players if player.num_die > 0]


def count_rolls(players):
    rolls_sum = [0,0,0,0,0,0]
    for player in players:
        for roll in player.rolls:
            rolls_sum[roll - 1] += 1

    return rolls_sum


def generate_round(players,current_player,player,previous_player):
    round = {'players':[],'previous_id':None,'previous_bet':None,'current_id':None,'your_id':None,'your_roll':[]}
    round['your_id'] = player.id
    round['your_rolls'] = player.rolls
    for player in players:
        round['players'].append({'id':player.id,'bet':None,'num_die':player.num_die})
    round['current_id'] = current_player.id
    if previous_player:
        round['previous_id'] = previous_player.id
        round['previous_bet'] = previous_player.bets[-1]
    else:
        round['previous_id'] = None
        round['previous_bet'] = None
    return round



def play_game(bot_funcs, silent = True):

    players = [Player(bot_func,id) for id,bot_func in enumerate(bot_funcs)]

    game_cycle = iter(cycle(players))

    while len(active_players(players)) > 1:
        if not silent:
            print ("\n### NEW ROUND - {0}\n".format([player.num_die for player in players]))
        previous_player = None

        for player in players:
            player.rolls = [randint(1,6) for _ in range(player.num_die)]

        for current_player in game_cycle:

            for player in players:
                round = generate_round(players,current_player,player,previous_player)

                if current_player == player:
                    if current_player.num_die >= 1:
                        response = player.function(round)
                        if response not in ['spot on','bluff'] and type(response) != tuple:
                            raise ValueError('Player {0} gave an invalid bet:{1}'.format(current_player.id, response))
                    else:
                        response = None
                else:
                    player.function(round)

            current_player.bets.append(response)

            if response == 'bluff':
                if not previous_player:
                    raise ValueError('The first player tried to call bluff!')
                if not silent:
                    print("### Player {0} called a bluff".format(current_player.id))
                bet = previous_player.bets[-1]
                if count_rolls(players)[bet[1] - 1] >= bet[0]:
                    if not silent:
                        print("The bluff was False!")
                    current_player.num_die -= 1
                else:
                    if not silent:
                        print("The bluff was True!")
                    previous_player.num_die -= 1
                break

            elif response == 'spot on':
                if not previous_player:
                    raise ValueError('The first player tried to call spot on!')
                if not silent:
                    print("### Player {0} called a spot on".format(current_player.id))
                bet = previous_player.bets[-1]
                if count_rolls(players)[bet[1] - 1] == bet[0]:
                    if not silent:
                        print("The spot on was True!")
                    for player in players:
                        if player != current_player:
                            player.num_die -= 1
                else:
                    if not silent:
                        print("The spot on was False!")
                    current_player.num_die -= 1
                break

            elif type(response) == tuple:
                if not silent:
                    print("### Player {0} bet {1}".format(current_player.id, response))

                if previous_player:
                    prev_number, prev_value = previous_player.bets[-1]
                    new_number, new_value = response
                    if not ( new_number > prev_number or (new_number == prev_number and new_value > prev_value)):
                        raise ValueError(f"Player {current_player.id} made invalid bet {response} after {(prev_number, prev_value)}")
                    
            if current_player.num_die > 0:
                previous_player = current_player
        if not silent:
            print("\nDice total:",count_rolls(players))
    if not silent:
        print("\n### WINNER - Player {0}".format(active_players(players)[0].id))
        
    return active_players(players)[0].id

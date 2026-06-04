#!/usr/bin/env python3
import random
import pyspiel
import liars_dice_mp
from liars_dice_mp import action_to_bid

from bots import random_bot, statistical_bot

NUM_PLAYERS = liars_dice_mp._DEFAULT_PLAYERS
NUM_DICE    = liars_dice_mp._DEFAULT_DICE
GAME        = pyspiel.load_game("liars_dice_mp")


def play_game(bots, verbose=True):
    assert len(bots) == NUM_PLAYERS

    state     = GAME.new_initial_state()
    own_rolls = {p: [] for p in range(NUM_PLAYERS)}
    round_num = 0

    if verbose:
        print(f'=== Liar\'s Dice | {NUM_PLAYERS} players, {NUM_DICE} dice ===')

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(random.choices(outcomes, weights=probs)[0])
            if not state.is_terminal() and not state.is_chance_node():
                round_num += 1
                for p in range(NUM_PLAYERS):
                    own_rolls[p] = list(state._rolls[p])
                if verbose:
                    print(f'\n--- Round {round_num} | Dice: { {p: state._dice_counts[p] for p in range(NUM_PLAYERS)} }')
                    print(f'    Rolls: { {p: own_rolls[p] for p in state._active} }')
            continue

        pid    = state.current_player()
        legal  = state.legal_actions()
        bids   = [action_to_bid(a, NUM_PLAYERS, NUM_DICE) for a in legal]
        action = bots[pid]({
            'your_id':        pid,
            'your_rolls':     own_rolls[pid],
            'previous_bid':   state._prev_bid,
            'bid_history':    list(state._bid_history),
            'legal_actions':  legal,
            'legal_bids':     bids,
            'active_players': list(state._active),
            'dice_counts':    list(state._dice_counts),
            'total_dice':     sum(state._dice_counts[p] for p in state._active),
        })

        if verbose:
            bid = action_to_bid(action, NUM_PLAYERS, NUM_DICE)
            if bid == 'liar':
                print(f'  Player {pid} calls LIAR on {state._prev_bid}')
            elif bid == 'spot_on':
                print(f'  Player {pid} calls SPOT ON on {state._prev_bid}')
            else:
                print(f'  Player {pid} bids {bid[0]}x{bid[1]}')

        state.apply_action(action)

    returns = state.returns()
    winner  = returns.index(max(returns))
    if verbose:
        print(f'\n=== Winner: Player {winner} ({bots[winner].__name__}) ===')
    return winner


if __name__ == '__main__':
    bots = [statistical_bot, random_bot, statistical_bot, random_bot, statistical_bot, random_bot]

    print('--- Single game ---')
    play_game(bots, verbose=True)

    print('\n--- Win rate: 200 games, 6 statistical bots ---')
    wins = {i: 0 for i in range(NUM_PLAYERS)}
    for _ in range(200):
        wins[play_game(bots, verbose=False)] += 1
    for p, w in wins.items():
        print(f'  Player {p}: {w}/200')

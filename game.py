#!/usr/bin/env python3
import random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyspiel

import liars_dice_mp
from liars_dice_mp import action_to_bid
from bots import random_bot, statistical_bot, mcts_bot
from bots.human_bot import create_human_agent

NUM_PLAYERS = liars_dice_mp._DEFAULT_PLAYERS
NUM_DICE    = liars_dice_mp._DEFAULT_DICE
GAME        = pyspiel.load_game("liars_dice_mp")

# ── Palette used throughout plots ────────────────────────────────────────────
_PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']


# ─────────────────────────────────────────────────────────────────────────────
# Core game runner
# ─────────────────────────────────────────────────────────────────────────────

def play_game(bots, verbose=True, observer=None):
    """Run one complete game and return (winner_idx, stats_dict).

    observer: optional callable(state_dict) called after every bidding action,
              useful for live display updates (e.g. the human agent's figure).
    """
    assert len(bots) == NUM_PLAYERS

    state     = GAME.new_initial_state()
    own_rolls = {p: [] for p in range(NUM_PLAYERS)}
    round_num = 0
    bot_names = [b.__name__ for b in bots]

    # Per-game tracking
    elimination_order = []   # [(player_id, bot_name, round_eliminated)]
    liar_calls        = []   # [{'caller': pid, 'bot': name, 'correct': bool}]
    spot_on_calls     = []   # [{'caller': pid, 'bot': name, 'correct': bool}]
    action_counts     = Counter()  # 'bid' | 'liar' | 'spot_on'
    bids_per_round    = []   # number of bid actions placed each round

    if verbose:
        print(f"=== Liar's Dice | {NUM_PLAYERS} players, {NUM_DICE} dice ===")

    while not state.is_terminal():
        # ── Chance node: roll dice one at a time ─────────────────────────────
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(random.choices(outcomes, weights=probs)[0])
            # Transition to bidding phase detected
            if not state.is_terminal() and not state.is_chance_node():
                round_num += 1
                for p in range(NUM_PLAYERS):
                    own_rolls[p] = list(state._rolls[p])
                bids_per_round.append(0)
                if verbose:
                    print(f'\n--- Round {round_num} | Dice: '
                          f'{ {p: state._dice_counts[p] for p in range(NUM_PLAYERS)} }')
                    print(f'    Rolls: { {p: own_rolls[p] for p in state._active} }')
            continue

        # ── Bidding node ──────────────────────────────────────────────────────
        pid   = state.current_player()
        legal = state.legal_actions()
        bids  = [action_to_bid(a, NUM_PLAYERS, NUM_DICE) for a in legal]

        dice_before   = list(state._dice_counts)
        active_before = list(state._active)

        state_dict = {
            'your_id':        pid,
            'your_rolls':     own_rolls[pid],
            'previous_bid':   state._prev_bid,
            'bid_history':    list(state._bid_history),
            'legal_actions':  legal,
            'legal_bids':     bids,
            'active_players': list(state._active),
            'dice_counts':    list(state._dice_counts),
            'total_dice':     sum(state._dice_counts[p] for p in state._active),
            'bot_names':      bot_names,
        }
        action = bots[pid](state_dict)

        decoded    = action_to_bid(action, NUM_PLAYERS, NUM_DICE)
        is_liar    = decoded == 'liar'
        is_spot_on = decoded == 'spot_on'

        if is_liar or is_spot_on:
            action_counts[decoded] += 1
        else:
            action_counts['bid'] += 1
            if bids_per_round:
                bids_per_round[-1] += 1

        if verbose:
            if is_liar:
                print(f'  Player {pid} calls LIAR on {state._prev_bid}')
            elif is_spot_on:
                print(f'  Player {pid} calls SPOT ON on {state._prev_bid}')
            else:
                print(f'  Player {pid} bids {decoded[0]}x{decoded[1]}')

        bid_challenged = state._prev_bid if (is_liar or is_spot_on) else None
        state.apply_action(action)

        # ── Record challenge outcome ──────────────────────────────────────────
        dice_after   = list(state._dice_counts)
        active_after = list(state._active)

        # Notify observer with full action context for live display
        if observer:
            observer({
                'acting_player':  pid,
                'action_taken':   decoded,
                'bid_challenged': bid_challenged,
                'dice_before':    dice_before,
                'eliminated_now': [p for p in active_before if p not in active_after],
                'previous_bid':   state._prev_bid,
                'bid_history':    list(state._bid_history),
                'active_players': list(state._active),
                'dice_counts':    list(state._dice_counts),
                'total_dice':     sum(state._dice_counts[p] for p in state._active),
                'bot_names':      bot_names,
            })

        if is_liar:
            caller_lost = dice_after[pid] < dice_before[pid]
            liar_calls.append({'caller': pid, 'bot': bot_names[pid],
                               'correct': not caller_lost})
        elif is_spot_on:
            caller_gained = dice_after[pid] > dice_before[pid]
            spot_on_calls.append({'caller': pid, 'bot': bot_names[pid],
                                  'correct': caller_gained})

        # ── Record eliminations ───────────────────────────────────────────────
        for p in active_before:
            if p not in active_after:
                elimination_order.append((p, bot_names[p], round_num))
                if verbose:
                    print(f'  *** Player {p} ({bot_names[p]}) eliminated in round {round_num} ***')

    returns = state.returns()
    winner  = returns.index(max(returns))

    if verbose:
        print(f'\n=== Winner: Player {winner} ({bot_names[winner]}) ===')

    return winner, {
        'winner':            winner,
        'winner_bot':        bot_names[winner],
        'bot_names':         bot_names,
        'rounds':            round_num,
        'winner_dice':       state._dice_counts[winner],
        'elimination_order': elimination_order,
        'action_counts':     dict(action_counts),
        'liar_calls':        liar_calls,
        'spot_on_calls':     spot_on_calls,
        'bids_per_round':    bids_per_round,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tournament runner
# ─────────────────────────────────────────────────────────────────────────────

def run_tournament(bots, n_games=200):
    print(f'\n--- Running {n_games}-game tournament ---')
    all_stats = []
    for i in range(n_games):
        _, stats = play_game(bots, verbose=(i == 0))
        all_stats.append(stats)
        print(f'  game {i+1}/{n_games} done — winner: {stats["winner_bot"]}')
    return all_stats


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_tournament_stats(all_stats, bots, out_path='tournament_stats.png'):
    """Generate and save a 9-panel tournament breakdown figure."""
    bot_names   = [b.__name__ for b in bots]
    n_players   = len(bots)
    n_games     = len(all_stats)
    unique_bots = sorted(set(bot_names))

    # ── Aggregations ──────────────────────────────────────────────────────────
    win_by_player = Counter(s['winner'] for s in all_stats)
    wins_by_type  = Counter(s['winner_bot'] for s in all_stats)
    game_lengths  = [s['rounds'] for s in all_stats]

    survival = {p: [] for p in range(n_players)}
    for s in all_stats:
        elim = {pid: rnd for pid, _, rnd in s['elimination_order']}
        for p in range(n_players):
            survival[p].append(elim.get(p, s['rounds']))

    total_actions = Counter()
    for s in all_stats:
        total_actions.update(s['action_counts'])

    liar_by_bot = defaultdict(lambda: [0, 0])   # [correct, total]
    spot_by_bot = defaultdict(lambda: [0, 0])
    for s in all_stats:
        for c in s['liar_calls']:
            liar_by_bot[c['bot']][0] += c['correct']
            liar_by_bot[c['bot']][1] += 1
        for c in s['spot_on_calls']:
            spot_by_bot[c['bot']][0] += c['correct']
            spot_by_bot[c['bot']][1] += 1

    winner_dice_by_type = defaultdict(list)
    for s in all_stats:
        winner_dice_by_type[s['winner_bot']].append(s['winner_dice'])

    round_bids = defaultdict(list)
    for s in all_stats:
        for i, cnt in enumerate(s['bids_per_round']):
            round_bids[i + 1].append(cnt)

    p_color  = {p: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(range(n_players))}
    bt_color = {b: _PALETTE[i % len(_PALETTE)] for i, b in enumerate(unique_bots)}

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    type_summary = ', '.join(
        f'{bot_names.count(b)}× {b}' for b in unique_bots
    )
    fig.suptitle(
        f'Tournament Results  |  {n_games} games  |  {n_players} players  |  {type_summary}',
        fontsize=14, fontweight='bold',
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)

    tick_labels = [f'P{p}\n{bot_names[p][:7]}' for p in range(n_players)]

    ax1 = fig.add_subplot(gs[0, 0])
    win_pcts = [win_by_player.get(p, 0) / n_games * 100 for p in range(n_players)]
    bars1 = ax1.bar(tick_labels, win_pcts,
                    color=[p_color[p] for p in range(n_players)])
    ax1.axhline(100 / n_players, color='gray', linestyle='--', linewidth=1,
                label=f'Equal share ({100/n_players:.1f}%)')
    ax1.set_title('Win Rate by Player', fontweight='bold')
    ax1.set_ylabel('Win %')
    ax1.set_ylim(0, max(win_pcts, default=1) * 1.25 + 3)
    ax1.legend(fontsize=7)
    for bar, w in zip(bars1, [win_by_player.get(p, 0) for p in range(n_players)]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.4, str(w),
                 ha='center', va='bottom', fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    type_counts = [bot_names.count(b) for b in unique_bots]
    type_wins   = [wins_by_type.get(b, 0) for b in unique_bots]
    type_pcts   = [w / n_games * 100 for w in type_wins]
    ax2.bar(unique_bots, type_pcts, color=[bt_color[b] for b in unique_bots])
    for i, (b, exp) in enumerate(zip(unique_bots,
                                      [c / n_players * 100 for c in type_counts])):
        ax2.bar(i, exp, width=0.55, color='none',
                edgecolor='gray', linestyle='--', linewidth=1.5,
                label='Random-chance baseline' if i == 0 else None)
    ax2.set_title('Win Rate by Bot Type\n(dashed outline = random-chance baseline)',
                  fontweight='bold')
    ax2.set_ylabel('Win %')
    ax2.legend(fontsize=7)
    for i, w in enumerate(type_wins):
        ax2.text(i, type_pcts[i] + 0.4, str(w),
                 ha='center', va='bottom', fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(game_lengths, bins=25, color='steelblue', edgecolor='white', alpha=0.85)
    mean_r, med_r = np.mean(game_lengths), np.median(game_lengths)
    ax3.axvline(mean_r,  color='red',    linestyle='--', linewidth=1.5,
                label=f'Mean={mean_r:.1f}')
    ax3.axvline(med_r,   color='orange', linestyle=':',  linewidth=1.5,
                label=f'Median={int(med_r)}')
    ax3.set_title('Game Length Distribution', fontweight='bold')
    ax3.set_xlabel('Rounds')
    ax3.set_ylabel('Games')
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    avg_surv = [np.mean(survival[p]) for p in range(n_players)]
    std_surv = [np.std(survival[p])  for p in range(n_players)]
    ax4.bar(tick_labels, avg_surv, yerr=std_surv,
            color=[p_color[p] for p in range(n_players)],
            capsize=4, error_kw={'linewidth': 1.2})
    ax4.set_title('Avg Survival Rounds (± σ)', fontweight='bold')
    ax4.set_ylabel('Rounds')

    ax5 = fig.add_subplot(gs[1, 1])
    action_keys = ['bid', 'liar', 'spot_on']
    action_vals = [total_actions.get(k, 0) for k in action_keys]
    pie_labels  = [f'Bid\n{action_vals[0]:,}',
                   f'Liar\n{action_vals[1]:,}',
                   f'Spot-On\n{action_vals[2]:,}']
    ax5.pie(action_vals, labels=pie_labels, autopct='%1.1f%%',
            colors=[_PALETTE[0], _PALETTE[1], _PALETTE[2]],
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax5.set_title('Action Distribution (all games)', fontweight='bold')

    ax6 = fig.add_subplot(gs[1, 2])
    x_tick_labels = []
    x_tick_pos    = []
    for i, btype in enumerate(unique_bots):
        lc, lt = liar_by_bot[btype]
        sc, st = spot_by_bot[btype]
        l_acc = lc / lt * 100 if lt else 0
        s_acc = sc / st * 100 if st else 0
        xl, xs = i * 2.6, i * 2.6 + 1.1
        for x, acc, n in [(xl, l_acc, lt), (xs, s_acc, st)]:
            ax6.bar(x, acc,         color='#55A868', width=0.9)
            ax6.bar(x, 100 - acc,   bottom=acc, color='#C44E52', width=0.9)
            if acc > 12:
                ax6.text(x, acc / 2, f'{acc:.0f}%',
                         ha='center', va='center',
                         color='white', fontsize=8, fontweight='bold')
            ax6.text(x, 102, f'n={n}', ha='center', va='bottom', fontsize=6.5)
        ax6.text((xl + xs) / 2, -15, btype[:11], ha='center', fontsize=7)
        x_tick_pos  += [xl, xs]
        x_tick_labels += ['Liar', 'Spot']
    ax6.set_xticks(x_tick_pos)
    ax6.set_xticklabels(x_tick_labels, fontsize=7)
    ax6.set_ylim(-20, 122)
    ax6.set_ylabel('%')
    ax6.set_title('Challenge Accuracy by Bot Type\n(green = correct, red = wrong)',
                  fontweight='bold')

    ax7 = fig.add_subplot(gs[2, 0])
    bins = [x - 0.5 for x in range(1, NUM_DICE + 2)]
    for btype in unique_bots:
        data = winner_dice_by_type.get(btype, [])
        if data:
            ax7.hist(data, bins=bins, alpha=0.7,
                     label=btype, color=bt_color[btype], edgecolor='white')
    ax7.set_title("Winner's Dice at Game End", fontweight='bold')
    ax7.set_xlabel('Dice remaining')
    ax7.set_ylabel('Games')
    ax7.set_xticks(range(1, NUM_DICE + 1))
    ax7.legend(fontsize=8)

    ax8 = fig.add_subplot(gs[2, 1])
    max_r    = max(game_lengths) if game_lengths else 1
    n_buckets = 12
    bsz       = max(1, max_r // n_buckets)
    elim_mat  = np.zeros((n_players, n_buckets))
    for s in all_stats:
        elim = {pid: rnd for pid, _, rnd in s['elimination_order']}
        for pid in range(n_players):
            if pid != s['winner']:
                r = elim.get(pid, s['rounds'])
                b = min(int((r - 1) / bsz), n_buckets - 1)
                elim_mat[pid, b] += 1
    im = ax8.imshow(elim_mat / n_games, aspect='auto', cmap='YlOrRd')
    ax8.set_yticks(range(n_players))
    ax8.set_yticklabels([f'P{p} {bot_names[p][:5]}' for p in range(n_players)],
                        fontsize=8)
    ax8.set_xticks(range(n_buckets))
    ax8.set_xticklabels([str(i * bsz + 1) for i in range(n_buckets)],
                        fontsize=7, rotation=45)
    ax8.set_title('Elimination Round Heatmap\n(fraction of games)',
                  fontweight='bold')
    ax8.set_xlabel('Round bucket start')
    plt.colorbar(im, ax=ax8, shrink=0.85, label='Fraction of games')

    ax9 = fig.add_subplot(gs[2, 2])
    threshold    = max(1, int(n_games * 0.05))
    common_rounds = sorted(r for r, lst in round_bids.items()
                           if len(lst) >= threshold)
    if common_rounds:
        means = [np.mean(round_bids[r]) for r in common_rounds]
        stds  = [np.std(round_bids[r])  for r in common_rounds]
        ax9.plot(common_rounds, means, marker='o', markersize=3,
                 color=_PALETTE[0], linewidth=1.5)
        ax9.fill_between(common_rounds,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.2, color=_PALETTE[0])
    ax9.set_title('Avg Bids Placed per Round\n(shading = ±σ)', fontweight='bold')
    ax9.set_xlabel('Round')
    ax9.set_ylabel('Bids')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    all_liar = [c for s in all_stats for c in s['liar_calls']]
    all_spot  = [c for s in all_stats for c in s['spot_on_calls']]
    print('\n' + '=' * 52)
    print(f'  Tournament Summary  ({n_games} games)')
    print('=' * 52)
    print(f'  Avg rounds   : {np.mean(game_lengths):.1f}  '
          f'(σ={np.std(game_lengths):.1f}, '
          f'min={min(game_lengths)}, max={max(game_lengths)})')
    print()
    for p in range(n_players):
        w = win_by_player.get(p, 0)
        print(f'  P{p} {bot_names[p]:>15}: {w:>3} wins '
              f'({w/n_games*100:>5.1f}%)  '
              f'avg survival {np.mean(survival[p]):.1f} rds')
    print()
    if all_liar:
        acc = sum(c['correct'] for c in all_liar) / len(all_liar) * 100
        print(f'  Liar calls   : {len(all_liar):>5} total  {acc:.1f}% correct')
    if all_spot:
        acc = sum(c['correct'] for c in all_spot) / len(all_spot) * 100
        print(f'  Spot-on calls: {len(all_spot):>5} total  {acc:.1f}% correct')
    print(f'\n  Plot saved → {out_path}')


def play_interactive(human_player_id=0, n_opponents=5):
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass

    ai_pool = [statistical_bot] * 2 + [random_bot] * 3 + [mcts_bot] * 5
    bots = []
    ai_idx = 0
    for p in range(NUM_PLAYERS):
        if p == human_player_id:
            bots.append(None)          # placeholder; replaced below
        else:
            bots.append(ai_pool[ai_idx % len(ai_pool)])
            ai_idx += 1

    bot_names = [b.__name__ if b else 'human' for b in bots]
    human, observer = create_human_agent(player_id=human_player_id, bot_names=bot_names)
    bots[human_player_id] = human

    print(f'\n  You are Player {human_player_id}.')
    print('  Opponents: ' +
          ', '.join(f'P{p} ({bot_names[p]})' for p in range(NUM_PLAYERS)
                    if p != human_player_id))
    print('  (matplotlib analysis window will open)\n')

    winner, stats = play_game(bots, verbose=False, observer=observer)

    import matplotlib.pyplot as plt
    plt.ioff()
    if winner == human_player_id:
        print('\n  *** You win! ***')
    else:
        print(f'\n  Player {winner} ({bot_names[winner]}) wins.')
    plt.show(block=True)    # keep figure open until closed manually
    return winner, stats


if __name__ == '__main__':
    from bots import random_bot, statistical_bot, mcts_bot, cfr_bot, llm_bot
    bots = [llm_bot, cfr_bot, mcts_bot, random_bot, statistical_bot,random_bot]
    all_stats = run_tournament(bots, n_games=10)
    plot_tournament_stats(all_stats, bots)
    import os, subprocess
    subprocess.Popen(['explorer', os.path.abspath('tournament_stats.png')])

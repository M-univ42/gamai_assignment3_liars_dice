from math import comb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

_console = Console()

_GLYPHS = ['', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣']
_PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']


def _binom_sf(k, n, p):
    """P(X ≥ k) for X ~ Binomial(n, p)."""
    if k <= 0: return 1.0
    if k > n:  return 0.0
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def _prob_valid(qty, face, own_rolls, unknown):
    """P(bid qty×face is valid) from this player's perspective."""
    p_hit   = 1 / 6 if face == 1 else 2 / 6
    own_cnt = sum(1 for r in own_rolls if r == face or (face != 1 and r == 1))
    return _binom_sf(max(0, qty - own_cnt), unknown, p_hit)


def _setup_figure():
    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle('Live Bid Analysis', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)
    axes = {
        'face_probs':    fig.add_subplot(gs[0, :]),
        'dice_counts':   fig.add_subplot(gs[1, 0]),
        'round_history': fig.add_subplot(gs[1, 1]),
    }
    plt.show(block=False)
    plt.pause(0.05)
    return fig, axes


def _update_figure(fig, axes, state):
    own     = state['your_rolls']
    prev    = state['previous_bid']
    total   = state['total_dice']
    unk     = total - len(own)
    counts  = state['dice_counts']
    active  = state['active_players']
    hist    = state['bid_history']
    pid     = state['your_id']
    names   = state.get('bot_names', [])
    n_p     = len(counts)

    qty = prev[0] if prev else 1

    ax = axes['face_probs']
    ax.clear()
    faces = list(range(1, 7))
    probs = [_prob_valid(qty, f, own, unk) for f in faces]
    colors = []
    for f, p in zip(faces, probs):
        if prev and f == prev[1]:
            colors.append('#4C72B0')
        elif p >= 0.50:
            colors.append('#55A868')
        elif p >= 0.35:
            colors.append('#DD8452')
        else:
            colors.append('#C44E52')

    face_labels = [f'Face {f}' for f in faces]
    bars = ax.bar(face_labels, probs, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(0.50, color='#55A868', linestyle='--', linewidth=1.3, alpha=0.8, label='50%')
    ax.axhline(0.35, color='#C44E52', linestyle=':',  linewidth=1.3, alpha=0.8,
               label='35% bluff threshold')
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('P(bid valid)')
    ax.legend(fontsize=8, loc='upper right')
    bid_label = (f'Current bid: {qty} x {prev[1]}' if prev else 'No bid yet')
    ax.set_title(f'P(>= {qty} dice show face)   |   {bid_label}',
                 fontweight='bold', fontsize=10)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f'{p:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2 = axes['dice_counts']
    ax2.clear()
    plabels = [f'P{p}*' if p == pid else f'P{p}' for p in range(n_p)]
    bcolors = ['#4C72B0' if p == pid
               else '#55A868' if p in active
               else '#BBBBBB'
               for p in range(n_p)]
    ax2.bar(plabels, counts, color=bcolors, edgecolor='white')
    ax2.set_ylim(0, max(counts, default=1) + 1.5)
    ax2.set_title('Dice Remaining  (blue=you, green=active, grey=out)',
                  fontweight='bold', fontsize=9)
    ax2.set_ylabel('Dice')
    for i, d in enumerate(counts):
        if d > 0:
            short = names[i][:5] if names and i < len(names) else ''
            ax2.text(i, d + 0.08, f'{d}\n{short}',
                     ha='center', va='bottom', fontsize=7)

    ax3 = axes['round_history']
    ax3.clear()
    if hist:
        confs = [_prob_valid(q, f, own, unk) for q, f in hist]
        xlabs = [f'{q}x{f}' for q, f in hist]
        xs    = list(range(len(confs)))
        ax3.plot(xs, confs, marker='o', markersize=6, color='#4C72B0', linewidth=2)
        ax3.fill_between(xs, confs, alpha=0.15, color='#4C72B0')
        ax3.axhline(0.35, color='#C44E52', linestyle='--',
                    linewidth=1.2, alpha=0.8, label='35% threshold')
        ax3.set_xticks(xs)
        ax3.set_xticklabels(xlabs, rotation=40, fontsize=7)
        ax3.set_ylim(0, 1.08)
        ax3.set_ylabel('P(valid)')
        ax3.legend(fontsize=7)
    else:
        ax3.text(0.5, 0.5, 'No bids yet this round',
                 ha='center', va='center', transform=ax3.transAxes,
                 fontsize=11, color='gray')
    ax3.set_title('Round Bid History — Confidence Trend', fontweight='bold', fontsize=9)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.02)


def _print_opponent_action(state):
    """Print one line describing what an opponent just did."""
    acting  = state['acting_player']
    decoded = state['action_taken']
    names   = state.get('bot_names', [])
    name    = names[acting][:14] if names and acting < len(names) else f'P{acting}'
    dc_before = state.get('dice_before', [])
    dc_after  = state['dice_counts']

    if decoded == 'liar':
        bid = state.get('bid_challenged')
        bid_str = f'{bid[0]}x{bid[1]}' if bid else '?'
        losers = [p for p in range(len(dc_after))
                  if dc_before and dc_after[p] < dc_before[p]]
        if losers:
            p = losers[0]
            outcome = f'[red]P{p} loses a die[/] ({dc_before[p]} → {dc_after[p]})'
        else:
            outcome = 'no change'
        _console.print(f'  [yellow]P{acting}[/] {name} calls [bold red]LIAR[/] on {bid_str} → {outcome}')

    elif decoded == 'spot_on':
        bid = state.get('bid_challenged')
        bid_str = f'{bid[0]}x{bid[1]}' if bid else '?'
        gainers = [p for p in range(len(dc_after))
                   if dc_before and dc_after[p] > dc_before[p]]
        losers  = [p for p in range(len(dc_after))
                   if dc_before and dc_after[p] < dc_before[p]]
        if gainers:
            p = gainers[0]
            outcome = f'[green]correct! P{p} gains a die[/] ({dc_before[p]} → {dc_after[p]})'
        elif losers:
            p = losers[0]
            outcome = f'[red]wrong! P{p} loses a die[/] ({dc_before[p]} → {dc_after[p]})'
        else:
            outcome = 'no change'
        _console.print(f'  [yellow]P{acting}[/] {name} calls [bold green]SPOT-ON[/] on {bid_str} → {outcome}')

    else:
        qty, face = decoded
        _console.print(f'  [yellow]P{acting}[/] {name} bids [bold]{qty}x{face} {_GLYPHS[face]}[/]')

    for elim in state.get('eliminated_now', []):
        ename = names[elim][:14] if names and elim < len(names) else f'P{elim}'
        _console.print(f'  [bold red]  *** P{elim} ({ename}) eliminated! ***[/]')


def _display_state(state):
    pid    = state['your_id']
    own    = state['your_rolls']
    prev   = state['previous_bid']
    hist   = state['bid_history']
    active = state['active_players']
    counts = state['dice_counts']
    total  = state['total_dice']
    unk    = total - len(own)
    names  = state.get('bot_names', [])
    n_p    = len(counts)

    _console.rule(f'[bold cyan]  Your turn — Player {pid}  [/]')

    # own dice
    row = ('   '.join(f'[bold yellow]{_GLYPHS[r]}[/] [dim]{r}[/]' for r in own)
           if own else '[dim]No dice[/]')
    _console.print(Panel(row, title=f'[bold]Your Dice ({len(own)} dice)[/]',
                         border_style='yellow', expand=False))

    # player table
    tbl = Table(box=box.SIMPLE, show_header=True, header_style='bold magenta', expand=False)
    tbl.add_column('Player')
    tbl.add_column('Bot', width=16)
    tbl.add_column('Dice')
    tbl.add_column('Status')
    for p in range(n_p):
        d     = counts[p]
        me    = p == pid
        alive = p in active
        style  = 'bold cyan' if me else 'green' if alive else 'dim'
        status = '[bold cyan]YOU[/]' if me else ('[green]active[/]' if alive else '[dim]out[/]')
        bar    = '█' * d + '░' * (5 - d)
        bname  = names[p][:15] if names and p < len(names) else '—'
        tbl.add_row(f'P{p}', bname, f'{bar} {d}', status, style=style)
    _console.print(tbl)

    # current bid with probability
    if prev:
        qty, face = prev
        p_val = _prob_valid(qty, face, own, unk)
        pc = 'green' if p_val >= 0.5 else 'yellow' if p_val >= 0.35 else 'red'
        content = (f'[bold]{qty} x {face} {_GLYPHS[face]}[/]'
                   f'    P(valid): [{pc}]{p_val:.0%}[/]')
    else:
        content = '[dim]None — you open the bidding[/]'
    _console.print(Panel(content, title='[bold]Current Bid[/]',
                         border_style='blue', expand=False))

    if hist:
        _console.print('[dim]History: ' + ' → '.join(f'{q}x{f}' for q, f in hist) + '[/]')
    _console.print(f'[dim]Total dice on table: {total}[/]\n')


def _prompt_action(state):
    legal = state['legal_actions']
    bids  = state['legal_bids']

    liar_act = next((a for a, b in zip(legal, bids) if b == 'liar'),    None)
    spot_act = next((a for a, b in zip(legal, bids) if b == 'spot_on'), None)
    bid_pairs = [(a, b) for a, b in zip(legal, bids) if b not in ('liar', 'spot_on')]
    bid_lookup = {b: a for a, b in bid_pairs}

    quick = bid_pairs[:12]
    _console.print('[bold]Actions:[/]')
    for i, (_, b) in enumerate(quick):
        qty, face = b
        end = '\n' if (i + 1) % 4 == 0 or i == len(quick) - 1 else '   '
        _console.print(f'  [[cyan]{i+1:>2}[/]] {qty}x{face} {_GLYPHS[face]}', end=end)
    if len(bid_pairs) > 12:
        _console.print(f'  [dim]... {len(bid_pairs)-12} more — type QTY FACE e.g. "8 3"[/]')

    extras = []
    if liar_act is not None:
        extras.append('  [[ [red bold]L[/] ]] [bold red]CALL LIAR[/]')
    if spot_act is not None:
        extras.append('  [[ [green bold]S[/] ]] [bold green]CALL SPOT-ON[/]')
    if extras:
        _console.print('   '.join(extras))
    _console.print()

    while True:
        raw = _console.input('[bold]> [/]').strip().lower()
        if raw == 'l' and liar_act is not None:
            return liar_act
        if raw == 's' and spot_act is not None:
            return spot_act
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(quick):
                return quick[idx][0]
        parts = raw.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            key = (int(parts[0]), int(parts[1]))
            if key in bid_lookup:
                return bid_lookup[key]
        _console.print('[red]Invalid. Enter a number, L, S, or "QTY FACE" (e.g. 7 4).[/]')

def create_human_agent(player_id=0, bot_names=None):
    fig, axes = _setup_figure()

    _human = {'your_id': player_id, 'your_rolls': []}

    def _human_view(state):
        return dict(state,
                    your_id=_human['your_id'],
                    your_rolls=_human['your_rolls'],
                    bot_names=bot_names or [])

    def observer(state):
        _update_figure(fig, axes, _human_view(state))
        if state.get('acting_player') != player_id:
            _print_opponent_action(dict(state, bot_names=bot_names or []))

    def human_bot(state):
        # Update cached rolls for this round before rendering
        _human['your_id']    = state['your_id']
        _human['your_rolls'] = state['your_rolls']
        view = _human_view(state)
        _update_figure(fig, axes, view)
        _console.print()   # blank line separator after opponent action log
        _display_state(view)
        return _prompt_action(view)

    human_bot.__name__ = 'human'
    return human_bot, observer

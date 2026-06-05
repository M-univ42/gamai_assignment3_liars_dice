from math import comb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

_console = Console()

_GLYPHS   = ['', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣']
_PALETTE  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
_BID_CMAP = LinearSegmentedColormap.from_list(
    'bid_safe', ['#C44E52', '#DD8452', '#F5C518', '#55A868', '#2d6a4f'], N=256
)


def _binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0: return 1.0
    if k > n:  return 0.0
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def _prob_valid(qty, face, own_rolls, unknown):
    """P(bid qty×face is valid) from this player's perspective."""
    p_hit   = 1 / 6 if face == 1 else 2 / 6
    own_cnt = sum(1 for r in own_rolls if r == face or (face != 1 and r == 1))
    return _binom_sf(max(0, qty - own_cnt), unknown, p_hit)


def _prob_exact(qty, face, own_rolls, unknown):
    """P(exactly qty dice show face) from this player's perspective."""
    p_hit   = 1 / 6 if face == 1 else 2 / 6
    own_cnt = sum(1 for r in own_rolls if r == face or (face != 1 and r == 1))
    need    = qty - own_cnt
    if need < 0 or need > unknown:
        return 0.0
    return comb(unknown, need) * p_hit**need * (1 - p_hit)**(unknown - need)



def _setup_figure():
    plt.ion()
    fig = plt.figure(figsize=(17, 12))
    fig.suptitle('Live Bid Analysis', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.60, wspace=0.38)
    axes = {
        'face_probs':    fig.add_subplot(gs[0, :2]),   # top-left  2/3
        'outcome':       fig.add_subplot(gs[0, 2]),    # top-right 1/3
        'bid_heatmap':   fig.add_subplot(gs[1, :]),    # mid full-width
        'dice_counts':   fig.add_subplot(gs[2, 0]),    # bot-left  1/3
        'round_history': fig.add_subplot(gs[2, 1:]),   # bot-right 2/3
    }
    plt.show(block=False)
    plt.pause(0.05)
    return fig, axes


def _update_figure(fig, axes, state):
    """Redraw all panels from the human player's perspective."""
    own    = state['your_rolls']
    prev   = state['previous_bid']
    total  = state['total_dice']
    unk    = total - len(own)
    counts = state['dice_counts']
    active = state['active_players']
    hist   = state['bid_history']
    pid    = state['your_id']
    names  = state.get('bot_names', [])
    n_p    = len(counts)
    qty    = prev[0] if prev else 1

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

    bars = ax.bar([f'Face {f}' for f in faces], probs,
                  color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(0.50, color='#55A868', linestyle='--', linewidth=1.3, alpha=0.8, label='50%')
    ax.axhline(0.35, color='#C44E52', linestyle=':',  linewidth=1.3, alpha=0.8,
               label='35% bluff threshold')
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('P(bid valid)')
    ax.legend(fontsize=8, loc='upper right')
    bid_label = f'Current bid: {qty} x {prev[1]}' if prev else 'No bid yet'
    ax.set_title(f'P(>= {qty} dice show face)   |   {bid_label}',
                 fontweight='bold', fontsize=10)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f'{p:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2 = axes['outcome']
    ax2.clear()
    if prev:
        cur_qty, cur_face = prev
        p_valid = _prob_valid(cur_qty, cur_face, own, unk)
        p_exact = _prob_exact(cur_qty, cur_face, own, unk)
        p_over  = max(0.0, p_valid - p_exact)
        p_under = max(0.0, 1.0 - p_valid)

        segments = [
            (p_under, '#C44E52', 'Under\n(Liar ✓)'),
            (p_exact, '#F5C518', 'Exact\n(Spot-On ✓)'),
            (p_over,  '#55A868', 'Over\n(bid low)'),
        ]
        left = 0.0
        for val, col, lbl in segments:
            ax2.barh(0, val, left=left, color=col, height=0.45, edgecolor='white', linewidth=1)
            if val > 0.07:
                tc = 'black' if col == '#F5C518' else 'white'
                ax2.text(left + val / 2, 0, f'{val:.0%}',
                         ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
            left += val

        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.5, 0.8)
        ax2.set_yticks([])
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=7)
        ax2.set_title(f'Outcome: {cur_qty}x{cur_face}\n'
                      f'P(valid)={p_valid:.0%}   P(exact)={p_exact:.0%}',
                      fontweight='bold', fontsize=9)
        legend_els = [Patch(facecolor=c, label=l.split('\n')[0])
                      for _, c, l in segments]
        ax2.legend(handles=legend_els, loc='upper center',
                   bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7)
    else:
        ax2.text(0.5, 0.5, 'No bid yet', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=10, color='gray')
        ax2.set_title('Outcome Distribution', fontweight='bold', fontsize=9)

    ax3 = axes['bid_heatmap']
    ax3.clear()

    if prev:
        q_start = prev[0]
        q_end   = min(prev[0] + 7, total)
    else:
        q_start = 1
        q_end   = min(8, total)
    qtys = list(range(q_start, q_end + 1))

    # Build P(valid) matrix; NaN where not legal
    legal_set = {
        (q, f)
        for q in range(1, total + 1)
        for f in range(1, 7)
        if prev is None or q > prev[0] or (q == prev[0] and f > prev[1])
    }
    matrix = np.full((len(qtys), 6), np.nan)
    for i, q in enumerate(qtys):
        for j, f in enumerate(faces):
            if (q, f) in legal_set:
                matrix[i, j] = _prob_valid(q, f, own, unk)

    _BID_CMAP.set_bad('#DDDDDD')
    im = ax3.imshow(matrix, cmap=_BID_CMAP, vmin=0, vmax=1, aspect='auto')

    # Annotate each cell
    for i, q in enumerate(qtys):
        for j, f in enumerate(faces):
            val = matrix[i, j]
            if not np.isnan(val):
                tc = 'white' if val < 0.35 or val > 0.75 else 'black'
                ax3.text(j, i, f'{val:.0%}',
                         ha='center', va='center', fontsize=8.5,
                         fontweight='bold', color=tc)

    if prev and prev[0] in qtys:
        qi = qtys.index(prev[0])
        fi = prev[1] - 1
        ax3.add_patch(Rectangle(
            (fi - 0.5, qi - 0.5), 1, 1,
            fill=False, edgecolor='#4C72B0', linewidth=3, zorder=5
        ))

    ax3.set_xticks(range(6))
    ax3.set_xticklabels([f'Face {f}' for f in faces], fontsize=9)
    ax3.set_yticks(range(len(qtys)))
    ax3.set_yticklabels([str(q) for q in qtys], fontsize=9)
    ax3.set_xlabel('Face', fontsize=9)
    ax3.set_ylabel('Quantity', fontsize=9)
    ax3.set_title(
        'P(valid) for Your Legal Bids   '
        '(blue border = current bid  |  grey = not legal  |  '
        'green > 50% > orange > 35% > red)',
        fontweight='bold', fontsize=9
    )
    old_cbar = getattr(fig, "_bid_heatmap_colorbar", None)
    if old_cbar is not None:
        try:
            old_cbar.remove()
        except Exception:
            pass

    fig._bid_heatmap_colorbar = fig.colorbar(
        im, ax=ax3,fraction=0.015,pad=0.01,label="P(valid)",)

    ax4 = axes['dice_counts']
    ax4.clear()
    plabels = [f'P{p}*' if p == pid else f'P{p}' for p in range(n_p)]
    bcolors = ['#4C72B0' if p == pid
               else '#55A868' if p in active
               else '#BBBBBB'
               for p in range(n_p)]
    ax4.bar(plabels, counts, color=bcolors, edgecolor='white')
    ax4.set_ylim(0, max(counts, default=1) + 1.5)
    ax4.set_title('Dice Remaining\n(blue=you, green=active, grey=out)',
                  fontweight='bold', fontsize=9)
    ax4.set_ylabel('Dice')
    for i, d in enumerate(counts):
        if d > 0:
            short = names[i][:5] if names and i < len(names) else ''
            ax4.text(i, d + 0.08, f'{d}\n{short}',
                     ha='center', va='bottom', fontsize=7)

    ax5 = axes['round_history']
    ax5.clear()
    if hist:
        confs = [_prob_valid(q, f, own, unk) for q, f in hist]
        xlabs = [f'{q}x{f}' for q, f in hist]
        xs    = list(range(len(confs)))
        ax5.plot(xs, confs, marker='o', markersize=6, color='#4C72B0', linewidth=2)
        ax5.fill_between(xs, confs, alpha=0.15, color='#4C72B0')
        ax5.axhline(0.35, color='#C44E52', linestyle='--',
                    linewidth=1.2, alpha=0.8, label='35% threshold')
        ax5.set_xticks(xs)
        ax5.set_xticklabels(xlabs, rotation=40, fontsize=7)
        ax5.set_ylim(0, 1.08)
        ax5.set_ylabel('P(valid)')
        ax5.legend(fontsize=7)
    else:
        ax5.text(0.5, 0.5, 'No bids yet this round',
                 ha='center', va='center', transform=ax5.transAxes,
                 fontsize=11, color='gray')
    ax5.set_title('Round Bid History — Confidence Trend', fontweight='bold', fontsize=9)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.02)


def _print_opponent_action(state):
    acting    = state['acting_player']
    decoded   = state['action_taken']
    names     = state.get('bot_names', [])
    name      = names[acting][:14] if names and acting < len(names) else f'P{acting}'
    dc_before = state.get('dice_before', [])
    dc_after  = state['dice_counts']

    if decoded == 'liar':
        bid = state.get('bid_challenged')
        bid_str = f'{bid[0]}x{bid[1]}' if bid else '?'
        losers = [p for p in range(len(dc_after))
                  if dc_before and dc_after[p] < dc_before[p]]
        if losers:
            p = losers[0]
            outcome = f'[red]P{p} loses a die[/] ({dc_before[p]} -> {dc_after[p]})'
        else:
            outcome = 'no change'
        _console.print(f'  [yellow]P{acting}[/] {name} calls [bold red]LIAR[/] on {bid_str} -> {outcome}')

    elif decoded == 'spot_on':
        bid = state.get('bid_challenged')
        bid_str = f'{bid[0]}x{bid[1]}' if bid else '?'
        gainers = [p for p in range(len(dc_after))
                   if dc_before and dc_after[p] > dc_before[p]]
        losers  = [p for p in range(len(dc_after))
                   if dc_before and dc_after[p] < dc_before[p]]
        if gainers:
            p = gainers[0]
            outcome = f'[green]correct! P{p} gains a die[/] ({dc_before[p]} -> {dc_after[p]})'
        elif losers:
            p = losers[0]
            outcome = f'[red]wrong! P{p} loses a die[/] ({dc_before[p]} -> {dc_after[p]})'
        else:
            outcome = 'no change'
        _console.print(f'  [yellow]P{acting}[/] {name} calls [bold green]SPOT-ON[/] on {bid_str} -> {outcome}')

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

    row = ('   '.join(f'[bold yellow]{_GLYPHS[r]}[/] [dim]{r}[/]' for r in own)
           if own else '[dim]No dice[/]')
    _console.print(Panel(row, title=f'[bold]Your Dice ({len(own)} dice)[/]',
                         border_style='yellow', expand=False))

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

    if prev:
        qty, face = prev
        p_val   = _prob_valid(qty, face, own, unk)
        p_exact = _prob_exact(qty, face, own, unk)
        pc = 'green' if p_val >= 0.5 else 'yellow' if p_val >= 0.35 else 'red'
        content = (f'[bold]{qty} x {face} {_GLYPHS[face]}[/]'
                   f'    P(valid): [{pc}]{p_val:.0%}[/]'
                   f'    P(exact): [cyan]{p_exact:.0%}[/]')
    else:
        content = '[dim]None — you open the bidding[/]'
    _console.print(Panel(content, title='[bold]Current Bid[/]',
                         border_style='blue', expand=False))

    if hist:
        _console.print('[dim]History: ' + ' -> '.join(f'{q}x{f}' for q, f in hist) + '[/]')
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
        _console.print('  [[ [yellow bold]?[/] ]] HELP   [[ [red bold]Q[/] ]] QUIT')
    _console.print()

    while True:
        raw = _console.input('[bold]> [/]').strip().lower()
        if raw in ("q", "quit", "exit"):
            _console.print("[yellow]Exiting game...[/]")
            raise KeyboardInterrupt
        if raw in ("?", "help", "h"):
            guide = Table(
                title="🎲 Liar's Dice Quick Guide",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold yellow",
                border_style="yellow",
                expand=False,
            )

            guide.add_column("Input", style="bold cyan", width=12)
            guide.add_column("Action", style="bold white", width=22)
            guide.add_column("Meaning", style="white", width=58)

            guide.add_row(
                "1, 2, 3...",
                "Choose listed bid",
                "Pick one of the visible legal bids from the action list.",
            )
            guide.add_row(
                "8 3",
                "Direct bid",
                "Bid directly: at least 8 dice showing face 3 on the whole table.",
            )
            guide.add_row(
                "L",
                "Call LIAR",
                "Challenge the previous bid. Use it when P(valid) looks low.",
            )
            guide.add_row(
                "S",
                "Call SPOT-ON",
                "Claim the previous bid is exactly correct. Use it when P(exact) looks high.",
            )
            guide.add_row(
                "? / h / help",
                "Show this guide",
                "Open the quick guide again during your turn.",
            )

            _console.print()
            _console.print(Panel(
                guide,
                title="[bold yellow]Captain's Table Manual[/]",
                subtitle="[dim]Read the table, trust the odds, then bluff wisely.[/]",
                border_style="yellow",
                padding=(1, 2),
                expand=False,
            ))

            _console.print(Panel(
                "[bold cyan]P(valid)[/]  Probability that the current bid is true.\n"
                "[bold cyan]P(exact)[/]  Probability that the current bid is exactly true.\n\n"
                "[green]Tip:[/] If P(valid) is high, calling LIAR is risky.\n"
                "[yellow]Tip:[/] If P(exact) is high, SPOT-ON can be powerful.\n"
                "[red]Tip:[/] If both are low, the previous player may be bluffing.",
                title="[bold]Reading the odds[/]",
                border_style="cyan",
                padding=(1, 2),
                expand=False,
            ))
        continue
        if raw in ("q", "quit", "exit"):
            _console.print("[yellow]Exiting game...[/]")
            raise KeyboardInterrupt
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
        _console.print('[red]Invalid. Enter a number, L, S, or "QTY FACE" (e.g. 7 4), or ? for help.[/]')


def create_human_agent(player_id=0, bot_names=None):
    fig, axes = _setup_figure()
    _human = {'your_id': player_id, 'your_rolls': []}

    def _human_view(state):
        return dict(state,
                    your_id=_human['your_id'],
                    your_rolls=_human['your_rolls'],
                    bot_names=bot_names or [])

    def observer(state):
        try:
            _update_figure(fig, axes, _human_view(state))
        except Exception as exc:
            _console.print(
            f"[yellow]Live analysis window update failed, continuing game: {exc}[/]"
        )
        if state.get('acting_player') != player_id:
            _print_opponent_action(dict(state, bot_names=bot_names or []))

    def human_bot(state):
        _human['your_id']    = state['your_id']
        _human['your_rolls'] = state['your_rolls']
        view = _human_view(state)
        try:
            _update_figure(fig, axes, view)
        except Exception as exc:
            _console.print(
            f"[yellow]Live analysis window update failed, continuing game: {exc}[/]"
            )
        _console.print()
        _display_state(view)
        return _prompt_action(view)

    human_bot.__name__ = 'human'
    return human_bot, observer

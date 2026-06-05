"""
Per-player behavioural model for Liar's Dice.

At the end of every round (when a challenge fires and actual dice are
revealed), PlayerModel retroactively labels every bid in that round as
honest or a bluff by comparing the claimed quantity against the true count
of matching dice.  It also records liar/spot-on call accuracy.

The resulting per-player statistics are exposed in two forms:

  bluff_rate(pid)      -> float in (0, 1)
      Laplace-smoothed probability that a given player's bid is a bluff.
      Directly useful for the Stage-2 statistical bot (weight bid
      credibility before deciding whether to call liar).

  feature_vector(reference_player) -> np.ndarray, shape (num_players * 5,)
      Fixed-size float32 block ready to be appended to the RL observation
      tensor.  Ordering is relative to reference_player so the network
      sees the same layout regardless of absolute seat position.

Integration with game.py
------------------------
1.  game.py creates PlayerModel(NUM_PLAYERS) once per game (or reuses one
    across games for a persistent tournament model).
2.  It appends `player_model` to every state_dict so bots can read it.
3.  After each challenge it calls player_model.update(...) with the bid log,
    a snapshot of rolls taken *before* the state clears them, and the
    challenge outcome.
"""

import numpy as np

_CONFIDENCE_CAP = 30   # bids observed before confidence saturates at 1.0


def _empty():
    return {
        'bids':         0,
        'bluffs':       0,   # bids where actual dice count < claimed qty
        'liar_calls':   0,
        'liar_correct': 0,
        'spot_calls':   0,
        'spot_correct': 0,
    }


class PlayerModel:
    """Accumulates bluffing and challenge statistics for each player."""

    FEATURES_PER_PLAYER = 5   # RL agents use this to size their input layer

    def __init__(self, num_players):
        self.num_players = num_players
        self._s = [_empty() for _ in range(num_players)]

    # ─────────────────────────────────────────────────────────────────
    # Update (called once per round, after challenge resolves)
    # ─────────────────────────────────────────────────────────────────

    def update(self, bid_log, rolls, active_players, challenge=None):
        """
        Retroactively label every bid from the just-resolved round.

        Parameters
        ----------
        bid_log : list of (player_id, qty, face)
            Every bid placed this round in order, with the bidder's id.
        rolls : list of list[int]
            Actual dice for all players, indexed by absolute player id.
            Must be captured *before* game.py clears them for the next round.
        active_players : list[int]
            Players active this round (whose dice count toward actual).
        challenge : dict | None
            {'caller': int, 'type': 'liar'|'spot_on', 'correct': bool}
            Pass None for rounds that end without a challenge (shouldn't
            happen in a normal game but keeps the API robust).
        """
        for pid, qty, face in bid_log:
            # Mirror the resolve formula in liars_dice_mp._resolve:
            # a die counts if it shows `face` or shows 1 (joker).
            # When face==1 the two conditions are identical, so no double-count.
            actual = sum(
                1 for p in active_players
                for r in rolls[p]
                if r == face or r == 1
            )
            s = self._s[pid]
            s['bids'] += 1
            if actual < qty:
                s['bluffs'] += 1

        if challenge:
            s = self._s[challenge['caller']]
            if challenge['type'] == 'liar':
                s['liar_calls'] += 1
                s['liar_correct'] += int(challenge['correct'])
            else:
                s['spot_calls'] += 1
                s['spot_correct'] += int(challenge['correct'])

    # ─────────────────────────────────────────────────────────────────
    # Queries
    # ─────────────────────────────────────────────────────────────────

    def bluff_rate(self, player_id):
        """
        Laplace-smoothed P(bid is a bluff) for use by the Stage-2 bot.
        Returns 0.5 for an unseen player (prior of 50/50).
        """
        s = self._s[player_id]
        return (s['bluffs'] + 1) / (s['bids'] + 2)

    def feature_vector(self, reference_player=None):
        """
        Flat float32 array of shape (num_players * FEATURES_PER_PLAYER,).

        Features per player slot (all values in [0, 1]):
          0  bluff_rate     P(bid is bluff), Laplace smoothed
          1  call_rate      challenge actions / total actions
          2  liar_accuracy  correct liar calls / liar calls (0.5 prior)
          3  spot_accuracy  correct spot calls / spot calls (0.5 prior)
          4  confidence     min(bids_seen, 30) / 30

        Slot ordering:
          If reference_player is given, slot 0 is the next player clockwise,
          and self occupies the last slot.  This makes the vector invariant
          to absolute seat number, which helps the RL agent generalise.
          If reference_player is None, slots follow absolute player index.
        """
        n = self.num_players
        if reference_player is None:
            order = list(range(n))
        else:
            order = [(reference_player + 1 + i) % n for i in range(n)]

        out = np.zeros(n * self.FEATURES_PER_PLAYER, dtype=np.float32)
        for slot, pid in enumerate(order):
            s     = self._s[pid]
            total = s['bids'] + s['liar_calls'] + s['spot_calls']
            base  = slot * self.FEATURES_PER_PLAYER

            out[base + 0] = (s['bluffs'] + 1) / (s['bids'] + 2)
            out[base + 1] = (s['liar_calls'] + s['spot_calls']) / max(total, 1)
            out[base + 2] = (s['liar_correct'] + 1) / (s['liar_calls'] + 2)
            out[base + 3] = (s['spot_correct'] + 1) / (s['spot_calls'] + 2)
            out[base + 4] = min(s['bids'], _CONFIDENCE_CAP) / _CONFIDENCE_CAP

        return out

    def reset(self):
        """Wipe all statistics.  Call between independent games when needed."""
        self._s = [_empty() for _ in range(self.num_players)]

    def __repr__(self):
        lines = [f'PlayerModel({self.num_players} players)']
        for pid, s in enumerate(self._s):
            br = (s['bluffs'] + 1) / (s['bids'] + 2)
            lines.append(
                f'  P{pid}: bids={s["bids"]} bluff_rate={br:.2f} '
                f'liar={s["liar_calls"]}(acc={s["liar_correct"]/max(s["liar_calls"],1):.0%})'
            )
        return '\n'.join(lines)

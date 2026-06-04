"""
Multi-player Liar's Dice (PERUDO) — OpenSpiel Python game.

Implements the pyspiel.Game / pyspiel.State interface so every OpenSpiel
algorithm (CFR, NFSP, DQN, MCTS, exploitability, …) can be run on it.

Differences from OpenSpiel's built-in liars_dice:
  - Supports 2–6 players (built-in caps at 2)
  - Full multi-round game: players lose dice, last player standing wins
  - Jokers: 1s count toward any face bid
  - Spot-on call: caller gains a die if exact, loses one if wrong

Action encoding (fixed for entire game):
  bid(qty, face)  ->  (qty-1) * NUM_FACES + (face-1)   [0 .. MAX_TOTAL_DICE*NUM_FACES - 1]
  LIAR            ->  MAX_TOTAL_DICE * NUM_FACES
  SPOT_ON         ->  MAX_TOTAL_DICE * NUM_FACES + 1

Chance action encoding:
  face F (1-6)  ->  F - 1   [0 .. 5]

Note on performance (from OpenSpiel docs):
  Python games are slower than C++ for algorithms that repeatedly step
  through states (MCTS, online RL). They are fine for CFR-style algorithms
  that build the tree once, and acceptable for batch RL training.
"""

import numpy as np
import pyspiel

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

_DEFAULT_PLAYERS = 6
_DEFAULT_DICE = 5
_NUM_FACES = 6

# History window encoded in the info-state tensor
_BID_HISTORY_LEN = 6

_LIAR_ACTION_OFFSET = 0     # relative to bid action count
_SPOT_ON_ACTION_OFFSET = 1


def _max_total_dice(num_players, num_dice):
    return num_players * num_dice


def _num_bid_actions(num_players, num_dice):
    return _max_total_dice(num_players, num_dice) * _NUM_FACES


def _num_distinct_actions(num_players, num_dice):
    return _num_bid_actions(num_players, num_dice) + 2  # + LIAR + SPOT_ON


def _liar_action(num_players, num_dice):
    return _num_bid_actions(num_players, num_dice)


def _spot_on_action(num_players, num_dice):
    return _num_bid_actions(num_players, num_dice) + 1


def bid_to_action(qty, face, num_players, num_dice):
    assert 1 <= qty <= _max_total_dice(num_players, num_dice)
    assert 1 <= face <= _NUM_FACES
    return (qty - 1) * _NUM_FACES + (face - 1)


def action_to_bid(action, num_players, num_dice):
    """Return (qty, face) tuple, 'liar', or 'spot_on'."""
    liar = _liar_action(num_players, num_dice)
    spot = _spot_on_action(num_players, num_dice)
    if action == liar:
        return 'liar'
    if action == spot:
        return 'spot_on'
    qty = action // _NUM_FACES + 1
    face = action % _NUM_FACES + 1
    return (qty, face)


# ---------------------------------------------------------------------------
# GameType & GameInfo (fixed for default params)
# ---------------------------------------------------------------------------

_GAME_TYPE = pyspiel.GameType(
    short_name="liars_dice_mp",
    long_name="Multi-Player Liar's Dice",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_DEFAULT_PLAYERS,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_num_distinct_actions(_DEFAULT_PLAYERS, _DEFAULT_DICE),
    max_chance_outcomes=_NUM_FACES,
    num_players=_DEFAULT_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=10_000,
)


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class LiarsDiceGame(pyspiel.Game):
    """Multi-player Liar's Dice, conforming to OpenSpiel's Game interface."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
        self.num_dice = _DEFAULT_DICE
        self.np = _DEFAULT_PLAYERS  # shorthand for num_players

    def new_initial_state(self):
        return LiarsDiceState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return LiarsDiceObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
            self.np,
            self.num_dice,
        )


# ---------------------------------------------------------------------------
# State class
# ---------------------------------------------------------------------------

class LiarsDiceState(pyspiel.State):
    """Full multi-round Liar's Dice state."""

    def __init__(self, game):
        super().__init__(game)
        self._g = game
        np_ = game.np
        nd = game.num_dice

        # Persistent across rounds
        self._dice_counts = [nd] * np_        # dice remaining per player
        self._active = list(range(np_))       # indices of players still in

        # Per-round state
        self._rolls = [[] for _ in range(np_)]
        self._bid_history = []               # list of (qty, face)
        self._prev_bid = None                # (qty, face) or None
        self._prev_bidder = None             # player index (absolute)

        # Rolling phase: tracks which die of which player we're rolling
        self._roll_player = 0               # index in self._active
        self._roll_die = 0                  # die index within that player

        # Bidding phase: index of current bidder within self._active
        self._first_bidder = 0             # index in self._active who opens bidding
        self._bid_turn = 0
        self._in_bidding = False

        self._game_over = False
        self._returns = [0.0] * np_

    # ------------------------------------------------------------------
    # OpenSpiel required API
    # ------------------------------------------------------------------

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if not self._in_bidding:
            return pyspiel.PlayerId.CHANCE
        return self._active[self._bid_turn]

    def _legal_actions(self, player):
        assert self._in_bidding and not self._game_over
        np_ = self._g.np
        nd = self._g.num_dice
        total = sum(self._dice_counts[p] for p in self._active)
        actions = []

        prev = self._prev_bid
        for qty in range(1, total + 1):
            for face in range(1, _NUM_FACES + 1):
                if prev is None or qty > prev[0] or (qty == prev[0] and face > prev[1]):
                    actions.append(bid_to_action(qty, face, np_, nd))

        if prev is not None:
            actions.append(_liar_action(np_, nd))
            actions.append(_spot_on_action(np_, nd))

        return sorted(actions)

    def chance_outcomes(self):
        assert not self._in_bidding and not self._game_over
        p = 1.0 / _NUM_FACES
        return [(f, p) for f in range(_NUM_FACES)]   # 0-5 represent faces 1-6

    def _apply_action(self, action):
        np_ = self._g.np
        nd = self._g.num_dice

        if not self._in_bidding:
            # Chance node: record die roll (action 0-5 = face 1-6)
            face = action + 1
            p = self._active[self._roll_player]
            self._rolls[p].append(face)
            self._roll_die += 1

            if self._roll_die >= self._dice_counts[p]:
                # Move to next active player
                self._roll_player += 1
                self._roll_die = 0
                if self._roll_player >= len(self._active):
                    # All active players have rolled — start bidding
                    self._in_bidding = True
                    self._bid_turn = self._first_bidder
            return

        # Bidding action
        liar = _liar_action(np_, nd)
        spot = _spot_on_action(np_, nd)
        caller = self._active[self._bid_turn]

        if action == liar or action == spot:
            self._resolve(action, caller)
            return

        bid = action_to_bid(action, np_, nd)
        self._prev_bid = bid
        self._prev_bidder = caller
        self._bid_history.append(bid)
        # Advance to next active bidder
        self._bid_turn = (self._bid_turn + 1) % len(self._active)

    def _resolve(self, action, caller):
        """Handle liar / spot-on call and set up next round or end game."""
        np_ = self._g.np
        nd = self._g.num_dice
        qty, face = self._prev_bid
        bidder = self._prev_bidder

        # Count total dice showing face or joker (1)
        actual = sum(
            1 for p in self._active
            for r in self._rolls[p]
            if r == face or r == 1
        )

        if action == _liar_action(np_, nd):
            if actual >= qty:                     # bid was valid → caller loses
                loser = caller
            else:                                 # bid was wrong → bidder loses
                loser = bidder
        else:                                     # SPOT_ON
            if actual == qty:                     # exact → caller gains a die
                self._dice_counts[caller] = min(
                    self._dice_counts[caller] + 1, nd
                )
                loser = None
            else:                                 # wrong → caller loses
                loser = caller

        if loser is not None:
            self._dice_counts[loser] -= 1

        # Eliminate players at 0 dice
        next_start_player = loser if (loser is not None and self._dice_counts[loser] > 0) else None
        self._active = [p for p in self._active if self._dice_counts[p] > 0]

        if len(self._active) == 1:
            winner = self._active[0]
            self._returns[winner] = 1.0
            for p in range(self._g.np):
                if p != winner:
                    self._returns[p] = -1.0
            self._game_over = True
            return

        # Start next round
        self._start_new_round(loser, next_start_player)

    def _start_new_round(self, loser, next_start_player):
        self._rolls = [[] for _ in range(self._g.np)]
        self._bid_history = []
        self._prev_bid = None
        self._prev_bidder = None
        self._in_bidding = False
        self._roll_player = 0    # always roll through active list from index 0
        self._roll_die = 0

        # Loser of the round opens bidding in the next round
        if next_start_player is not None and next_start_player in self._active:
            self._first_bidder = self._active.index(next_start_player)
        elif loser is not None and loser in self._active:
            self._first_bidder = self._active.index(loser)
        else:
            self._first_bidder = 0

    def is_terminal(self):
        return self._game_over

    def returns(self):
        return list(self._returns)

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def _action_to_string(self, player, action):
        np_ = self._g.np
        nd = self._g.num_dice
        bid = action_to_bid(action, np_, nd)
        if bid == 'liar':
            return 'Liar'
        if bid == 'spot_on':
            return 'SpotOn'
        if player == pyspiel.PlayerId.CHANCE:
            return f'Roll:{action + 1}'
        return f'{bid[0]}-{bid[1]}'

    def __str__(self):
        counts = ' '.join(f'P{p}:{self._dice_counts[p]}' for p in range(self._g.np))
        bid = f'bid={self._prev_bid}' if self._prev_bid else 'no bid'
        return f'[{counts}] {bid}'

    # ------------------------------------------------------------------
    # Information state (required for imperfect-info algorithms)
    # ------------------------------------------------------------------

    def information_state_string(self, player):
        own = self._rolls[player]
        counts = self._dice_counts
        hist = self._bid_history[-_BID_HISTORY_LEN:]
        return (f'p{player} dice={own} counts={counts} '
                f'prev={self._prev_bid} hist={hist}')

    def observation_string(self, player):
        return self.information_state_string(player)


# ---------------------------------------------------------------------------
# Observer (required for information_state_tensor / observation_tensor)
# ---------------------------------------------------------------------------

class LiarsDiceObserver:
    """Builds flat tensors for RL algorithms.

    Tensor layout (all values normalised to [0, 1]):
      [own_dice]       NUM_DICE × 6  one-hot per die slot (zeros if die lost)
      [dice_counts]    NUM_PLAYERS   normalised dice count per player
      [prev_bid_qty]   1             qty / max_total_dice  (0 if no bid)
      [prev_bid_face]  7             one-hot face 0-6 (index 0 = no bid)
      [bid_history]    BID_HIST×7    last K bids, each (qty_norm + face_one_hot)
      [player_id]      NUM_PLAYERS   one-hot
    """

    def __init__(self, iig_obs_type, num_players, num_dice):
        self._np = num_players
        self._nd = num_dice
        self._max = _max_total_dice(num_players, num_dice)

        face_enc = _NUM_FACES + 1          # 0 = "no bid", 1-6 = face
        bid_enc = 1 + face_enc             # qty_norm + face_one_hot

        size = (
            num_dice * _NUM_FACES           # own dice one-hot
            + num_players                   # dice counts
            + 1                             # prev bid qty
            + face_enc                      # prev bid face
            + _BID_HISTORY_LEN * bid_enc    # bid history
            + num_players                   # player id one-hot
        )
        self.tensor = np.zeros(size, np.float32)
        self.dict = {}
        idx = 0

        def _slice(name, length):
            self.dict[name] = self.tensor[idx: idx + length]
            return idx + length

        idx = _slice('own_dice', num_dice * _NUM_FACES)
        idx = _slice('dice_counts', num_players)
        idx = _slice('prev_bid_qty', 1)
        idx = _slice('prev_bid_face', face_enc)
        idx = _slice('bid_history', _BID_HISTORY_LEN * bid_enc)
        idx = _slice('player_id', num_players)
        assert idx == size

    def set_from(self, state, player):
        self.tensor.fill(0)
        nd = self._nd

        # Own dice (one-hot per slot)
        own = state._rolls[player]
        own_t = self.dict['own_dice'].reshape(nd, _NUM_FACES)
        for i, face in enumerate(own):
            own_t[i, face - 1] = 1.0

        # Dice counts (normalised)
        for p in range(self._np):
            self.dict['dice_counts'][p] = state._dice_counts[p] / nd

        # Previous bid
        prev = state._prev_bid
        if prev is not None:
            self.dict['prev_bid_qty'][0] = prev[0] / self._max
            self.dict['prev_bid_face'][prev[1]] = 1.0   # index 1-6
        # else stays zero (index 0 of face encoding = no bid)

        # Bid history (last K bids, most recent last)
        face_enc = _NUM_FACES + 1
        bid_enc = 1 + face_enc
        hist = state._bid_history[-_BID_HISTORY_LEN:]
        hist_t = self.dict['bid_history'].reshape(_BID_HISTORY_LEN, bid_enc)
        for i, (qty, face) in enumerate(hist):
            hist_t[i, 0] = qty / self._max
            hist_t[i, 1 + face] = 1.0

        # Player id
        self.dict['player_id'][player] = 1.0

    def string_from(self, state, player):
        return state.information_state_string(player)


# ---------------------------------------------------------------------------
# Register with OpenSpiel
# ---------------------------------------------------------------------------

pyspiel.register_game(_GAME_TYPE, LiarsDiceGame)

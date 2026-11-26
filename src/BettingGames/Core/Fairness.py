from typing import Callable

import numpy as np

from src.BettingGames.Core.Bet import Bet
from src.BettingGames.Core.Strategy import Strategy
from src.BettingGames.Core.Tournament import play

__author__ = 'frank.ma'


def assert_fair_play(bet: Bet, make_strategy: Callable[[], Strategy], wins_a: np.ndarray, wins_b: np.ndarray) -> None:
    """
    the betting games' fair-play rule: a player's stake may depend only on the public terms and what that player has
    seen so far. Two tables are compared player by player: wherever a player's own outcomes agree up to a round, that
    player must have staked the same in every round up to and including it, whatever the other players' outcomes or
    the outcomes still to come
    :param make_strategy: builds a fresh strategy, seeded alike each time if it is random
    :raises AssertionError: naming the first player and round that differ
    """
    wins_a, wins_b = np.asarray(wins_a, dtype=bool), np.asarray(wins_b, dtype=bool)
    if wins_a.shape != wins_b.shape:
        raise AssertionError('the two tables should have the same shape.')
    stakes_a = play(bet, make_strategy(), wins_a, keep_paths=True).stakes
    stakes_b = play(bet, make_strategy(), wins_b, keep_paths=True).stakes
    n_rounds, n_players = wins_a.shape
    for i in range(n_players):
        differ = np.flatnonzero(wins_a[:, i] != wins_b[:, i])
        agree_through = differ[0] if differ.size else n_rounds - 1  # the first round whose outcome differs is unseen
        for t in range(agree_through + 1):
            if stakes_a[t, i] != stakes_b[t, i]:
                raise AssertionError('player %i staked %r and %r in round %i although everything they had seen was '
                                     'the same.' % (i, stakes_a[t, i], stakes_b[t, i], t))

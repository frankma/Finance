from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.BettingGames.Core.Bet import Bet
from src.BettingGames.Core.Strategy import Strategy, View

__author__ = 'frank.ma'


@dataclass
class Result(object):
    final: np.ndarray  # (n_players,) wealth after the last round
    paths: Optional[np.ndarray] = None  # (n_rounds + 1, n_players) wealth before each round and at the end
    stakes: Optional[np.ndarray] = None  # (n_rounds, n_players) the fraction each player staked


def play(bet: Bet, strategy: Strategy, wins: np.ndarray, start: float = 1.0, pool_size: Optional[int] = None,
         keep_paths: bool = False) -> Result:
    """
    play every player through every round of a pre-drawn table of outcomes

    the strategy sees only the outcomes of rounds already played, so the outcomes being drawn in advance tells it
    nothing; drawing them first lets different strategies face exactly the same luck

    :param wins: (n_rounds, n_players), True where a player's bet wins that round
    :param pool_size: when set, consecutive groups of this many players pool their wealth after every round and
        share it equally: each still bets on their own outcome, the group shares the result
    """
    wins = np.asarray(wins, dtype=bool)
    n_rounds, n_players = wins.shape
    if pool_size is not None and (pool_size < 1 or n_players % pool_size):
        raise ValueError('%i players do not split into pools of %r.' % (n_players, pool_size))
    past = wins.view()
    past.flags.writeable = False
    wealth = np.full(n_players, float(start))
    paths = np.empty((n_rounds + 1, n_players)) if keep_paths else None
    stakes = np.empty((n_rounds, n_players)) if keep_paths else None
    for t in range(n_rounds):
        if keep_paths:
            paths[t] = wealth
        seen = wealth.copy()
        seen.flags.writeable = False
        f = np.asarray(strategy.fraction(View(bet, t, n_rounds, seen, past[:t])), dtype=float)
        if f.shape != (n_players,) or np.any(f < 0.0) or np.any(f > 1.0):
            raise ValueError('%s should stake a fraction in [0, 1] for each of %i players.' % (strategy.name, n_players))
        if keep_paths:
            stakes[t] = f
        wealth = wealth * bet.multiplier(f, wins[t])
        if pool_size is not None:
            wealth = np.repeat(wealth.reshape(-1, pool_size).mean(axis=1), pool_size)
    if keep_paths:
        paths[n_rounds] = wealth
    return Result(wealth, paths, stakes)

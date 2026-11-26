from dataclasses import dataclass

import numpy as np

from src.BettingGames.Core.Bet import Bet

__author__ = 'frank.ma'


@dataclass(frozen=True, eq=False)
class View(object):
    """
    what the players at a table know before a round: the public terms of the bet and the round count, each player's
    own wealth and the outcomes already drawn. Never the outcomes still to come

    the arrays are read-only, one column per player; a fair strategy sets each player's stake from that player's
    column (and the public terms) only
    """
    bet: Bet
    round: int  # rounds already played, 0 before the first
    n_rounds: int
    wealth: np.ndarray  # (n_players,)
    past_wins: np.ndarray  # (round, n_players), True where the bet won


class Strategy(object):
    """
    sets the fraction of wealth each player stakes this round
    """
    name = 'strategy'

    def fraction(self, view: View) -> np.ndarray:
        raise NotImplementedError


class FixedFraction(Strategy):
    def __init__(self, fraction: float, name: str = ''):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError('the fraction %r should be in [0, 1].' % fraction)
        self.value = fraction
        self.name = name or 'stake %g%%' % (100.0 * fraction)

    def fraction(self, view: View) -> np.ndarray:
        return np.full(view.wealth.shape, self.value)


class AllIn(FixedFraction):
    # stake everything, every round
    def __init__(self):
        super(AllIn, self).__init__(1.0, 'all in')


class Kelly(Strategy):
    """
    stake the growth-optimal fraction of the bet's public terms, times `multiple` (0.5 is half Kelly)
    """
    def __init__(self, multiple: float = 1.0, cap: float = 1.0):
        self.multiple = multiple
        self.cap = cap
        self.name = 'Kelly' if multiple == 1.0 else '%g Kelly' % multiple

    def fraction(self, view: View) -> np.ndarray:
        f = min(self.multiple * view.bet.kelly_fraction(cap=self.cap), self.cap)
        return np.full(view.wealth.shape, f)

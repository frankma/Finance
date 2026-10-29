from typing import Dict, List

from src.CardGames.Core.Card import Card, Rank
from src.CardGames.Core.CardCounter import CardCounter

__author__ = 'frank.ma'

# hi-lo: small cards leaving the shoe help the player, tens and aces leaving it hurt
HI_LO = {Rank.TWO: 1, Rank.THREE: 1, Rank.FOUR: 1, Rank.FIVE: 1, Rank.SIX: 1,
         Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 0,
         Rank.TEN: -1, Rank.JACK: -1, Rank.QUEEN: -1, Rank.KING: -1, Rank.ACE: -1}  # type: Dict[Rank, float]


def hi_lo_counter(composition: List[Card]) -> CardCounter:
    return CardCounter(composition, HI_LO)


def kelly_fraction(edge: float, variance: float) -> float:
    """
    growth-optimal fraction of the bankroll to stake on a bet with the given expected return and variance per unit
    staked, the small-edge approximation edge / variance; never negative, since a bet without edge is not taken
    """
    if variance <= 0.0:
        raise ValueError('variance %r should be strictly positive.' % variance)
    return max(edge, 0.0) / variance


class KellyBetSizer(object):
    """
    bet a fraction of Kelly on the edge implied by the true count, within the table limits

    edge(tc) = base_edge + edge_per_true_count * tc; the defaults are the usual rule of thumb for a six-deck game,
    about -0.5% off the top and +0.5% per true count, with variance about 1.3 per unit bet
    """

    def __init__(self, min_bet: float, max_bet: float, base_edge: float = -0.005, edge_per_true_count: float = 0.005,
                 variance: float = 1.3, kelly_multiplier: float = 1.0):
        if not 0.0 < min_bet <= max_bet:
            raise ValueError('expect 0 < min_bet <= max_bet, received %r and %r.' % (min_bet, max_bet))
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.base_edge = base_edge
        self.edge_per_true_count = edge_per_true_count
        self.variance = variance
        self.kelly_multiplier = kelly_multiplier  # e.g. 0.5 for half Kelly

    def edge(self, true_count: float) -> float:
        return self.base_edge + self.edge_per_true_count * true_count

    def bet(self, bankroll: float, true_count: float) -> float:
        stake = self.kelly_multiplier * kelly_fraction(self.edge(true_count), self.variance) * bankroll
        return min(max(stake, self.min_bet), self.max_bet)


class FlatBetSizer(object):
    def __init__(self, bet: float):
        self.min_bet = bet
        self.max_bet = bet

    def bet(self, bankroll: float, true_count: float) -> float:
        return self.min_bet

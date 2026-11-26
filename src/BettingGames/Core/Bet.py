import math
from dataclasses import dataclass
from typing import List

import numpy as np

__author__ = 'frank.ma'


@dataclass(frozen=True)
class Bet(object):
    """
    a repeated bet with two outcomes: each round the staked part of a player's wealth gains `up` with probability
    `p_up` and loses `down` otherwise; what is not staked is kept as it is

    the terms are public: every player knows them before the first round, so a strategy may use them
    """
    up: float = 0.5  # gain on the stake when the bet wins, 0.5 is +50%
    down: float = 0.4  # loss on the stake when it loses, 0.4 is -40%
    p_up: float = 0.5

    def __post_init__(self):
        if not self.up > 0.0:
            raise ValueError('the gain %r should be strictly positive.' % self.up)
        if not 0.0 < self.down <= 1.0:
            raise ValueError('the loss %r should be in (0, 1].' % self.down)
        if not 0.0 < self.p_up < 1.0:
            raise ValueError('the win probability %r should be in (0, 1).' % self.p_up)

    @property
    def p_down(self) -> float:
        return 1.0 - self.p_up

    def multiplier(self, fraction, win):
        """
        what wealth is multiplied by in one round when `fraction` of it is staked
        """
        fraction = np.asarray(fraction, dtype=float)
        return np.where(win, 1.0 + fraction * self.up, 1.0 - fraction * self.down)

    def mean_return(self, fraction: float = 1.0) -> float:
        # the expected one-round return: the average across many players in one round
        return fraction * (self.p_up * self.up - self.p_down * self.down)

    def growth_rate(self, fraction: float = 1.0) -> float:
        """
        the expected log return per round: what one player's wealth grows at over many rounds
        """
        loss = 1.0 - fraction * self.down
        if loss <= 0.0:
            return -math.inf
        return self.p_up * math.log1p(fraction * self.up) + self.p_down * math.log(loss)

    def kelly_fraction(self, cap: float = 1.0) -> float:
        """
        the stake that maximises the growth rate, p / down - q / up for two outcomes; zero when the bet has no edge,
        and at most `cap` of wealth (1.0: no borrowing to bet more than you have)
        """
        return min(max(self.p_up / self.down - self.p_down / self.up, 0.0), cap)

    def draw(self, rng: np.random.Generator, size) -> np.ndarray:
        # True where the bet wins
        return rng.random(size) < self.p_up

    # the exact distribution of wealth after n rounds at a fixed fraction: it depends only on the number of wins

    def wins_pmf(self, n_rounds: int) -> np.ndarray:
        k = np.arange(n_rounds + 1)
        log_choose = np.array([math.lgamma(n_rounds + 1) - math.lgamma(i + 1) - math.lgamma(n_rounds - i + 1)
                               for i in k])
        return np.exp(log_choose + k * math.log(self.p_up) + (n_rounds - k) * math.log(self.p_down))

    def wealth_after(self, fraction: float, n_rounds: int) -> np.ndarray:
        # wealth from 1 after n rounds with k = 0..n wins
        k = np.arange(n_rounds + 1)
        return (1.0 + fraction * self.up) ** k * (1.0 - fraction * self.down) ** (n_rounds - k)

    def mean_wealth(self, fraction: float, n_rounds: int) -> float:
        return (1.0 + self.mean_return(fraction)) ** n_rounds

    def prob_above(self, fraction: float, n_rounds: int, level: float = 1.0) -> float:
        # the chance of ending with more than `level` times the starting wealth
        return float(self.wins_pmf(n_rounds)[self.wealth_after(fraction, n_rounds) > level].sum())

    def quantiles(self, fraction: float, n_rounds: int, qs: List[float]) -> List[float]:
        # wealth quantiles after n rounds, from the binomial count of wins (the lower quantile at a jump)
        wealth = self.wealth_after(fraction, n_rounds)
        order = np.argsort(wealth, kind='stable')
        cdf = np.cumsum(self.wins_pmf(n_rounds)[order])
        return [float(wealth[order][min(np.searchsorted(cdf, q - 1e-12), n_rounds)]) for q in qs]

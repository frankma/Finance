import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.CardGames.BlackJack.Counting import hi_lo_counter
from src.CardGames.BlackJack.Round import Round
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.Core.Game import play_out
from src.CardGames.Core.Player import Strategy
from src.CardGames.Core.Shoe import Shoe
from src.CardGames.Core.Simulator import MonteCarloResult

__author__ = 'frank.ma'


class Table(object):
    """
    a shoe dealt round after round, reshuffled at the cut card, with a hi-lo counter watching every card shown
    """

    def __init__(self, rules: Rules, rng: Optional[random.Random] = None, cards: Optional[list] = None):
        """
        :param rules: table rules
        :param rng: random source, pass a seeded one for reproducible play
        :param cards: explicit shoe composition, rules.n_decks standard decks if not given
        """
        self.rules = rules
        self.shoe = Shoe(rules.n_decks, penetration=rules.penetration, rng=rng, cards=cards)
        self.counter = hi_lo_counter(self.shoe.composition)
        self.shoe.watch(self.counter)

    @property
    def true_count(self) -> float:
        return self.counter.true_count()

    def play_round(self, strategy: Strategy, bet: float = 1.0) -> float:
        if self.shoe.needs_shuffle:
            self.shoe.reshuffle()
        game = Round(self.shoe, self.rules, bet, self.counter)
        return play_out(game, [strategy])[0]


def simulate_ev(rules: Rules, strategy: Strategy, n_rounds: int, rng: Optional[random.Random] = None,
                cards: Optional[list] = None) -> MonteCarloResult:
    """
    expected return per unit of initial flat bet; rounds dealt from one shoe are mildly correlated, so the standard
    error is a slight understatement
    """
    table = Table(rules, rng, cards)
    return MonteCarloResult.from_samples([table.play_round(strategy) for _ in range(n_rounds)])


def ev_by_true_count(rules: Rules, strategy: Strategy, n_rounds: int, rng: Optional[random.Random] = None,
                     max_count: int = 6) -> Dict[int, MonteCarloResult]:
    """
    expected return per unit flat bet bucketed by the hi-lo true count before the deal, truncated toward zero and
    capped at +/- max_count
    """
    table = Table(rules, rng)
    buckets = {}  # type: Dict[int, list]
    for _ in range(n_rounds):
        if table.shoe.needs_shuffle:
            table.shoe.reshuffle()
        bucket = int(max(-max_count, min(max_count, table.true_count)))
        buckets.setdefault(bucket, []).append(table.play_round(strategy))
    return {bucket: MonteCarloResult.from_samples(results) for bucket, results in sorted(buckets.items())}


@dataclass(frozen=True)
class BankrollResult(object):
    paths: np.ndarray  # bankroll after each round, shape (n_paths, n_rounds + 1); stays flat once ruined
    ruined: np.ndarray  # whether each path fell below the minimum bet

    @property
    def ruin_probability(self) -> float:
        return float(self.ruined.mean())

    @property
    def final(self) -> np.ndarray:
        return self.paths[:, -1]


def simulate_bankroll(rules: Rules, strategy: Strategy, bet_sizer, bankroll: float, n_rounds: int, n_paths: int,
                      rng: Optional[random.Random] = None) -> BankrollResult:
    """
    bankroll paths under a bet sizing scheme keyed off the true count, each path on its own shoe
    :param bet_sizer: object with min_bet and bet(bankroll, true_count), e.g. KellyBetSizer or FlatBetSizer
    """
    rng = rng if rng is not None else random.Random()
    paths = np.empty((n_paths, n_rounds + 1))
    ruined = np.zeros(n_paths, dtype=bool)
    for i in range(n_paths):
        table = Table(rules, random.Random(rng.random()))
        wealth = bankroll
        paths[i, 0] = wealth
        for j in range(1, n_rounds + 1):
            if not ruined[i]:
                if table.shoe.needs_shuffle:
                    table.shoe.reshuffle()
                # splits and doubles may stake more than is left, so a ruined path can end slightly below zero
                stake = min(bet_sizer.bet(wealth, table.true_count), wealth)
                if stake < bet_sizer.min_bet:
                    ruined[i] = True
                else:
                    wealth += table.play_round(strategy, stake)
            paths[i, j] = wealth
    return BankrollResult(paths, ruined)

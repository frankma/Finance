import random
from typing import Dict, List, Optional

from src.CardGames.Core.Game import play_out
from src.CardGames.Core.Player import Strategy
from src.CardGames.Core.Simulator import MonteCarloResult
from src.CardGames.Holdem.Round import Round
from src.CardGames.Holdem.Rules import Rules

__author__ = 'frank.ma'


class Table(object):
    """
    hands played one after another with the button moving left, stacks carried over and busted seats sitting out
    """

    def __init__(self, stacks: List[int], rules: Rules = Rules(), rng: Optional[random.Random] = None):
        self.stacks = list(stacks)
        self.rules = rules
        self.rng = rng if rng is not None else random.Random()
        self.button = 0
        self.n_hands = 0

    @property
    def n_live(self) -> int:
        return sum(stack > 0 for stack in self.stacks)

    def play_hand(self, strategies: List[Strategy]) -> List[float]:
        if self.n_live < 2:
            raise ValueError('need at least two seats with chips to deal.')
        game = Round(self.stacks, self.rules, self.button, self.rng)
        payoffs = play_out(game, strategies)
        self.stacks = game.stacks
        self.n_hands += 1
        n = len(self.stacks)
        self.button = next((self.button + k) % n for k in range(1, n + 1) if self.stacks[(self.button + k) % n] > 0)
        return payoffs


def simulate_win_rate(strategies: List[Strategy], n_hands: int, stack: int, rules: Rules = Rules(),
                      rng: Optional[random.Random] = None) -> Dict[int, MonteCarloResult]:
    """
    big blinds won per hand by each seat, with every stack topped back up to the start before each hand so the
    hands are independent and the comparison is not skewed by a seat going broke
    """
    rng = rng if rng is not None else random.Random()
    results = [[] for _ in strategies]  # type: List[List[float]]
    button = 0
    for _ in range(n_hands):
        game = Round([stack] * len(strategies), rules, button, rng)
        for seat, payoff in enumerate(play_out(game, strategies)):
            results[seat].append(payoff / rules.big_blind)
        button = (button + 1) % len(strategies)
    return {seat: MonteCarloResult.from_samples(values) for seat, values in enumerate(results)}

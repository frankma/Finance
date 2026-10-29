import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional

__author__ = 'frank.ma'


class Strategy(ABC):
    @abstractmethod
    def decide(self, observation: Any, legal_moves: List[Any]) -> Any:
        pass


class RandomStrategy(Strategy):
    # baseline that picks uniformly among the legal moves
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()

    def decide(self, observation: Any, legal_moves: List[Any]) -> Any:
        return self.rng.choice(legal_moves)


class Player(object):
    def __init__(self, name: str, strategy: Strategy, bankroll: float = 0.0):
        self.name = name
        self.strategy = strategy
        self.bankroll = bankroll

    def settle(self, payoff: float) -> None:
        self.bankroll += payoff

    def __str__(self):
        return '%s (bankroll %.2f)' % (self.name, self.bankroll)

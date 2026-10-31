import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

__author__ = 'frank.ma'


class Strategy(ABC):
    """
    chooses a move from a seat's observation and legal moves alone, the only inputs play_out gives it
    """

    @abstractmethod
    def decide(self, observation: Any, legal_moves: List[Any]) -> Any:
        pass


@dataclass(frozen=True)
class Advice(object):
    """
    a recommended move and why, in terms of facts the player can check for themselves
    """
    move: Any
    reason: str


class Advisor(Strategy):
    """
    a strategy that explains itself, so it can coach a human seat as well as play a computer one; it plays the move
    it would advise
    """

    @abstractmethod
    def advise(self, observation: Any, legal_moves: List[Any]) -> Advice:
        pass

    def decide(self, observation: Any, legal_moves: List[Any]) -> Any:
        return self.advise(observation, legal_moves).move


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

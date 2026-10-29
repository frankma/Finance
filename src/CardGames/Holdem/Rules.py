from dataclasses import dataclass
from typing import List

from src.CardGames.Core.Card import Card
from src.CardGames.Holdem.HandEvaluator import HandRank, evaluate, evaluate_omaha

__author__ = 'frank.ma'


@dataclass(frozen=True)
class Rules(object):
    """
    no-limit hold'em and its variants: n_hole hole cards each, and with omaha a hand uses exactly two of them plus
    exactly three board cards; without it any five of hole and board play, so n_hole = 2 is Texas hold'em and
    n_hole = 4 with omaha is (no-limit) Omaha
    """
    small_blind: int = 1
    big_blind: int = 2
    n_hole: int = 2
    omaha: bool = False

    def __post_init__(self):
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError('expect 0 < small_blind <= big_blind, received %i and %i.'
                             % (self.small_blind, self.big_blind))
        if not 2 <= self.n_hole <= 5:
            raise ValueError('n_hole %i should be between 2 and 5.' % self.n_hole)

    def max_seats(self) -> int:
        # every seat's hole cards plus five board cards and three burns come out of one deck
        return (52 - 8) // self.n_hole

    def hand_rank(self, hole: List[Card], board: List[Card]) -> HandRank:
        return evaluate_omaha(hole, board) if self.omaha else evaluate(list(hole) + list(board))


HOLDEM = Rules()
OMAHA = Rules(n_hole=4, omaha=True)

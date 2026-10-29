import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional

from src.CardGames.Core.Card import Card, standard_deck
from src.CardGames.Core.CardCounter import CardCounter
from src.CardGames.Core.Simulator import MonteCarloResult
from src.CardGames.Holdem.HandEvaluator import HandCategory, evaluate
from src.CardGames.Holdem.Rules import Rules

__author__ = 'frank.ma'


def _share(mine, others) -> float:
    # this hand's share of the pot against the best of the others, ties split
    if not others:
        return 1.0
    best = max(others)
    if mine > best:
        return 1.0
    if mine < best:
        return 0.0
    return 1.0 / (1 + sum(other == best for other in others))


def _unseen(known: List[Card]) -> List[Card]:
    counter = CardCounter(standard_deck())
    counter.see_many(known)
    return counter.unseen()


@dataclass(frozen=True)
class EquityResult(object):
    equity: MonteCarloResult  # share of the pot won per trial, ties split
    categories: Dict[HandCategory, float]  # how often the hand finishes in each category


def simulate_equity(hole: List[Card], board: List[Card], n_opponents: int, n_sims: int,
                    rules: Rules = Rules(), rng: Optional[random.Random] = None,
                    dead: Optional[List[Card]] = None) -> EquityResult:
    """
    Monte Carlo equity against opponents holding random hands: deal each opponent rules.n_hole of the unseen cards
    and the rest of the board, score everyone under the rules, and average the pot share
    :param dead: cards known to be out of play besides hole and board, e.g. ones seen exposed
    """
    rng = rng if rng is not None else random.Random()
    unseen = _unseen(list(hole) + list(board) + list(dead or []))
    n_hole, to_come = rules.n_hole, 5 - len(board)
    need = n_opponents * n_hole + to_come
    if need > len(unseen):
        raise ValueError('%i opponents with %i hole cards do not fit in the unseen cards.' % (n_opponents, n_hole))
    shares, categories = [], Counter()
    for _ in range(n_sims):
        sample = rng.sample(unseen, need)
        full_board = list(board) + sample[n_opponents * n_hole:]
        mine = rules.hand_rank(hole, full_board)
        others = [rules.hand_rank(sample[k * n_hole:(k + 1) * n_hole], full_board) for k in range(n_opponents)]
        shares.append(_share(mine, others))
        categories[mine.category] += 1
    return EquityResult(MonteCarloResult.from_samples(shares),
                        {category: categories[category] / n_sims for category in HandCategory})


def exact_equity(holes: List[List[Card]], board: List[Card], rules: Rules = Rules()) -> List[float]:
    """
    each known hand's pot share averaged over every possible run-out of the board; practical from the flop on
    (990 run-outs heads-up) or for a turn board
    """
    unseen = _unseen([card for hole in holes for card in hole] + list(board))
    totals = [0.0] * len(holes)
    n = 0
    for rest in combinations(unseen, 5 - len(board)):
        full_board = list(board) + list(rest)
        ranks = [rules.hand_rank(hole, full_board) for hole in holes]
        for k, rank in enumerate(ranks):
            totals[k] += _share(rank, ranks[:k] + ranks[k + 1:])
        n += 1
    return [total / n for total in totals]


def _board_category(cards: List[Card]) -> HandCategory:
    # what the board makes on its own; four cards can only pair up
    if len(cards) >= 5:
        return evaluate(cards).category
    counts = sorted(Counter(card.rank for card in cards).values(), reverse=True)
    if counts[0] == 4:
        return HandCategory.FOUR_OF_A_KIND
    if counts[0] == 3:
        return HandCategory.THREE_OF_A_KIND
    if counts[:2] == [2, 2]:
        return HandCategory.TWO_PAIR
    return HandCategory.PAIR if counts[0] == 2 else HandCategory.HIGH_CARD


@dataclass(frozen=True)
class Outs(object):
    category: HandCategory  # what the hand makes now
    by_category: Dict[HandCategory, List[Card]]  # the cards that lift it into each better category
    n_unseen: int

    @property
    def n_outs(self) -> int:
        return sum(len(cards) for cards in self.by_category.values())

    @property
    def p_next_card(self) -> float:
        return self.n_outs / self.n_unseen

    def p_by_river(self, cards_to_come: int) -> float:
        # chance at least one out arrives in the remaining cards
        miss = 1.0
        for k in range(cards_to_come):
            miss *= (self.n_unseen - self.n_outs - k) / (self.n_unseen - k)
        return 1.0 - miss


def outs(hole: List[Card], board: List[Card], rules: Rules = Rules()) -> Outs:
    """
    unseen cards that lift the hand into a better category on the next card; in hold'em a card that only improves
    the board helps everyone, so it is not an out, while under the omaha rule the board never plays on its own
    """
    if len(board) not in (3, 4):
        raise ValueError('outs are counted on the flop or the turn, received a board of %i cards.' % len(board))
    unseen = _unseen(list(hole) + list(board))
    now = rules.hand_rank(hole, board).category
    by_category = {}  # type: Dict[HandCategory, List[Card]]
    for card in unseen:
        category = rules.hand_rank(hole, list(board) + [card]).category
        if category > now and (rules.omaha or category > _board_category(list(board) + [card])):
            by_category.setdefault(category, []).append(card)
    return Outs(now, by_category, len(unseen))


def pot_odds(pot: int, to_call: int) -> float:
    """
    the equity a call needs to break even: the call over the pot it would make
    :param pot: chips in the pot before calling, the opponents' bets included
    """
    return to_call / (pot + to_call) if to_call > 0 else 0.0


def call_ev(equity: float, pot: int, to_call: int) -> float:
    """
    expected chips won by calling and checking down, relative to folding
    """
    return equity * (pot + to_call) - to_call

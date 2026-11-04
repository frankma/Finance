from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

from src.CardGames.Core.Card import Card, standard_deck
from src.CardGames.Holdem.HandEvaluator import HandCategory
from src.CardGames.Holdem.Round import Action, Observation, Street

__author__ = 'frank.ma'

# how much a preflop action narrows a hand: the percentile it must reach, and the chance of it with a weaker hand
FIRST_RAISE = (0.60, 0.10)
RERAISE = (0.88, 0.05)
CALL_A_RAISE = (0.30, 0.30)
# how strong a hand is on a board, for how likely it is to bet or call there
MADE, TOP_PAIR, LOW_PAIR, FLUSH_DRAW, NOTHING = 1.0, 0.8, 0.45, 0.45, 0.1
CALL_LIFT = 0.3  # calling takes less than betting
WEIGHT_FLOOR = 0.2  # the least any hand counts, however badly it fits, since players do bluff


def _chen(high: int, low: int, suited: bool) -> float:
    # the Chen formula: a standard hand-rule score for two hole cards, from the high card, pairing, suit and gap
    points = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}.get(high, high / 2.0)
    if high == low:
        return max(5.0, points * 2.0)
    gap = high - low - 1
    points += 2.0 if suited else 0.0
    points -= {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}.get(gap, 5.0)
    if gap <= 1 and high < 12:
        points += 1.0
    return points


def _percentiles() -> Dict[Tuple[int, int, bool], float]:
    # the share of all starting hands a hand class beats, half of its ties included
    scores = {}  # type: Dict[Tuple[int, int, bool], float]
    counted = []
    for a, b in combinations(standard_deck(), 2):
        high, low = max(a.rank.value, b.rank.value), min(a.rank.value, b.rank.value)
        key = (high, low, a.suit == b.suit)
        scores[key] = _chen(*key)
        counted.append(scores[key])
    n = float(len(counted))
    return {key: (sum(c < s for c in counted) + 0.5 * sum(c == s for c in counted)) / n for key, s in scores.items()}


PERCENTILE = _percentiles()


def preflop_percentile(hole: List[Card]) -> float:
    a, b = hole
    return PERCENTILE[(max(a.rank.value, b.rank.value), min(a.rank.value, b.rank.value), a.suit == b.suit)]


def board_strength(hole: List[Card], board: List[Card], rules) -> float:
    """
    how strong a hand reads on a board, roughly: made hands from the hole cards, then top pair, lower pairs and flush
    draws, then nothing; a pair or better that the board makes alone does not count
    """
    ranks = [card.rank.value for card in board]
    paired_board = len(set(ranks)) < len(ranks)
    category = rules.hand_rank(hole, board).category
    if category >= HandCategory.THREE_OF_A_KIND or (category == HandCategory.TWO_PAIR and not paired_board):
        return MADE
    hole_ranks = [card.rank.value for card in hole]
    if hole_ranks[0] == hole_ranks[1] or any(r in ranks for r in hole_ranks):
        pair = hole_ranks[0] if hole_ranks[0] == hole_ranks[1] else max(r for r in hole_ranks if r in ranks)
        return TOP_PAIR if pair >= max(ranks) else LOW_PAIR
    # four to a flush with a card still to come, using at least one hole card
    suits = [card.suit for card in board] + [card.suit for card in hole]
    if len(board) < 5 and any(suits.count(card.suit) == 4 for card in hole):
        return FLUSH_DRAW
    return NOTHING


def _actions(o: Observation) -> Dict[int, List[Tuple[Street, Action, float, int]]]:
    # each seat's calls and raises: street, action, its size as a share of the pot, and the raises before it this street
    actions = {}  # type: Dict[int, List[Tuple[Street, Action, float, int]]]
    pot, street_now, current, raises = o.rules.small_blind + o.rules.big_blind, Street.PREFLOP, o.big_blind, 0
    bets = {}  # type: Dict[int, int]
    for seat, street, move in o.history:
        if street != street_now:
            street_now, current, raises, bets = street, 0, 0, {}
        put = move.to - bets.get(seat, 0)
        if move.action == Action.RAISE:
            actions.setdefault(seat, []).append((street, Action.RAISE, (move.to - current) / float(max(pot, 1)), raises))
            current, raises = move.to, raises + 1
        elif move.action == Action.CALL and (street != Street.PREFLOP or current > o.big_blind):
            actions.setdefault(seat, []).append((street, Action.CALL, put / float(max(pot, 1)), raises))
        if move.action in (Action.RAISE, Action.CALL):
            pot += put
            bets[seat] = move.to
    return actions


def betting_weights(o: Observation) -> Optional[List[Callable[[List[Card]], float]]]:
    """
    for each opponent still in the hand, how likely a two-card hand is to have played as that opponent has, from
    public actions only: who raised or called, how much, and on which board. None when no opponent has shown anything
    or the game is not two-card hold'em. Only a model: real players bluff, and the weights never rule a hand out
    """
    if o.rules.n_hole != 2 or not o.in_hand:
        return None
    actions = _actions(o)
    opponents = [seat for seat, still_in in enumerate(o.in_hand) if still_in and seat != o.seat]
    if not any(seat in actions for seat in opponents):
        return None

    def weight_of(moves):
        memo = {}  # type: Dict[Tuple[Card, Card], float]

        def weight(hole: List[Card]) -> float:
            key = (hole[0], hole[1])
            if key not in memo:
                memo[key] = _weight(hole, moves, o)
            return memo[key]
        return weight
    return [weight_of(actions.get(seat, [])) for seat in opponents]


def _weight(hole: List[Card], moves, o: Observation) -> float:
    weight = 1.0
    for street, action, size, raises in moves:
        if street == Street.PREFLOP:
            need, otherwise = (CALL_A_RAISE if action == Action.CALL else FIRST_RAISE if raises == 0 else RERAISE)
            weight *= 1.0 if preflop_percentile(hole) >= need else otherwise
        else:
            strength = board_strength(hole, list(o.board[:2 + street.value]), o.rules)
            weight *= min(1.0, strength + CALL_LIFT) if action == Action.CALL else strength ** (1.0 + min(size, 2.0))
    return WEIGHT_FLOOR + (1.0 - WEIGHT_FLOOR) * weight

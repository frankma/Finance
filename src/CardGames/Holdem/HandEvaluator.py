from enum import IntEnum
from itertools import combinations
from typing import List, NamedTuple, Tuple

from src.CardGames.Core.Card import Card

__author__ = 'frank.ma'


class HandCategory(IntEnum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


class HandRank(NamedTuple):
    # compares as a tuple: category first, then the tiebreaking ranks in order (ace = 14)
    category: HandCategory
    tiebreakers: Tuple[int, ...]


_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
          11: 'jack', 12: 'queen', 13: 'king', 14: 'ace'}


def _plural(rank: int) -> str:
    return 'sixes' if rank == 6 else _WORDS[rank] + 's'


def describe(rank: HandRank) -> str:
    t = rank.tiebreakers
    if rank.category == HandCategory.STRAIGHT_FLUSH:
        return 'Royal flush' if t[0] == 14 else 'Straight flush, %s high' % _WORDS[t[0]]
    if rank.category == HandCategory.FOUR_OF_A_KIND:
        return 'Four %s' % _plural(t[0])
    if rank.category == HandCategory.FULL_HOUSE:
        return 'Full house, %s full of %s' % (_plural(t[0]), _plural(t[1]))
    if rank.category == HandCategory.FLUSH:
        return 'Flush, %s high' % _WORDS[t[0]]
    if rank.category == HandCategory.STRAIGHT:
        return 'Straight, %s high' % _WORDS[t[0]]
    if rank.category == HandCategory.THREE_OF_A_KIND:
        return 'Three %s' % _plural(t[0])
    if rank.category == HandCategory.TWO_PAIR:
        return 'Two pair, %s and %s' % (_plural(t[0]), _plural(t[1]))
    if rank.category == HandCategory.PAIR:
        return 'Pair of %s' % _plural(t[0])
    return '%s high' % _WORDS[t[0]].capitalize()


def _straight_high(mask: int) -> int:
    # mask has bit r set for each rank r present; the ace also plays low in the wheel
    if mask & (1 << 14):
        mask |= 1 << 1
    for high in range(14, 4, -1):
        if (mask >> (high - 4)) & 0b11111 == 0b11111:
            return high
    return 0


def evaluate(cards: List[Card]) -> HandRank:
    """
    best five-card hand among any five or more cards, read straight from the rank and suit counts rather than by
    scoring every five-card subset
    """
    if len(cards) < 5:
        raise ValueError('need at least 5 cards to evaluate a hand, received %i.' % len(cards))
    rank_counts = [0] * 15
    mask = 0
    suit_masks = {}
    suit_counts = {}
    for card in cards:
        r = card.rank.value
        rank_counts[r] += 1
        mask |= 1 << r
        suit_masks[card.suit] = suit_masks.get(card.suit, 0) | (1 << r)
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

    # with ten cards (five hole cards) two suits can both make a flush
    flush, straight_flush = None, 0
    for suit, n in suit_counts.items():
        if n < 5:
            continue
        straight_flush = max(straight_flush, _straight_high(suit_masks[suit]))
        top = HandRank(HandCategory.FLUSH, tuple(r for r in range(14, 1, -1) if suit_masks[suit] & (1 << r))[:5])
        flush = max(flush, top) if flush else top
    if straight_flush:
        return HandRank(HandCategory.STRAIGHT_FLUSH, (straight_flush,))

    quads, trips, pairs = [], [], []
    for r in range(14, 1, -1):
        n = rank_counts[r]
        if n >= 4:
            quads.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)

    def kickers(excluded, k):
        return tuple(r for r in range(14, 1, -1) if rank_counts[r] and r not in excluded)[:k]

    if quads:
        return HandRank(HandCategory.FOUR_OF_A_KIND, (quads[0],) + kickers((quads[0],), 1))
    if trips and (len(trips) > 1 or pairs):
        second = max(trips[1] if len(trips) > 1 else 0, pairs[0] if pairs else 0)
        return HandRank(HandCategory.FULL_HOUSE, (trips[0], second))
    if flush:
        return flush
    high = _straight_high(mask)
    if high:
        return HandRank(HandCategory.STRAIGHT, (high,))
    if trips:
        return HandRank(HandCategory.THREE_OF_A_KIND, (trips[0],) + kickers((trips[0],), 2))
    if len(pairs) >= 2:
        return HandRank(HandCategory.TWO_PAIR, (pairs[0], pairs[1]) + kickers((pairs[0], pairs[1]), 1))
    if pairs:
        return HandRank(HandCategory.PAIR, (pairs[0],) + kickers((pairs[0],), 3))
    return HandRank(HandCategory.HIGH_CARD, kickers((), 5))


def evaluate_omaha(hole: List[Card], board: List[Card]) -> HandRank:
    """
    best hand using exactly two hole cards and exactly three board cards
    """
    if len(hole) < 2 or len(board) < 3:
        raise ValueError('the Omaha rule needs at least 2 hole and 3 board cards, received %i and %i.'
                         % (len(hole), len(board)))
    return max(evaluate(list(two) + list(three)) for two in combinations(hole, 2) for three in combinations(board, 3))


def best_five(cards: List[Card]) -> List[Card]:
    """
    the five cards that make the hand, e.g. for display
    """
    target = evaluate(cards)
    return list(next(five for five in combinations(cards, 5) if evaluate(list(five)) == target))

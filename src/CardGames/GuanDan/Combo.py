from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.CardGames.Core.Card import Card, Rank, Suit

__author__ = 'frank.ma'

BLACK_JOKER = Rank.BLACK_JOKER.value  # 16
RED_JOKER = Rank.RED_JOKER.value  # 17
LEVEL_VALUE = 15  # the level rank jumps above the ace, just under the jokers
SUIT_ORDER = [Suit.SPADE, Suit.HEART, Suit.DIAMOND, Suit.CLUB]


def is_wild(card: Card, level: Rank) -> bool:
    # the two hearts of the level rank stand for any card except a joker
    return card.suit == Suit.HEART and card.rank == level


def group_value(rank: int, level: Rank) -> int:
    """
    order for singles, pairs, triples, full houses and same-rank bombs: 2 low, then the level rank above the ace,
    then the jokers; sequences use the natural ranks instead
    """
    if rank >= BLACK_JOKER:
        return rank
    return LEVEL_VALUE if rank == level.value else rank


def sort_hand(hand: List[Card], level: Rank) -> List[Card]:
    return sorted(hand, key=lambda c: (is_wild(c, level), group_value(c.rank.value, level),
                                       SUIT_ORDER.index(c.suit) if c.suit else 0))


class ComboType(Enum):
    SINGLE = 'single'
    PAIR = 'pair'
    TRIPLE = 'triple'
    FULL_HOUSE = 'full house'
    STRAIGHT = 'straight'
    TUBE = 'tube'  # three consecutive pairs
    PLATE = 'plate'  # two consecutive triples
    BOMB = 'bomb'  # four to ten of a kind
    STRAIGHT_FLUSH = 'straight flush'
    JOKERS = 'four jokers'


BOMBS = (ComboType.BOMB, ComboType.STRAIGHT_FLUSH, ComboType.JOKERS)


@dataclass(frozen=True)
class Combo(object):
    type: ComboType
    cards: Tuple[Card, ...]
    value: int  # group value for groups, the natural top rank for sequences
    rank: int  # the rank it is named after
    pair_rank: Optional[int] = None  # the pair under a full house; the level rank when wilds make it up

    @property
    def n(self) -> int:
        return len(self.cards)

    @property
    def is_bomb(self) -> bool:
        return self.type in BOMBS

    @property
    def power(self) -> int:
        # bomb ladder: 4 < 5 < straight flush < 6 < 7 < ... < 10 of a kind < four jokers
        if self.type == ComboType.JOKERS:
            return 10000
        if self.type == ComboType.STRAIGHT_FLUSH:
            return 600 + self.value
        return (self.n * 100 if self.n <= 5 else (self.n + 1) * 100) + self.value

    def beats(self, other: Optional['Combo']) -> bool:
        """
        whether this may be played on other: anything leads, a bomb beats any ordinary combination and only a bigger
        bomb, and an ordinary combination beats the same type and size with a higher value
        """
        if other is None:
            return True
        if self.is_bomb != other.is_bomb:
            return self.is_bomb
        if self.is_bomb:
            return self.power > other.power
        return self.type == other.type and self.n == other.n and self.value > other.value

    def describe(self) -> str:
        r = self.rank
        if self.type == ComboType.SINGLE:
            return rank_name(r) if r >= BLACK_JOKER else 'single ' + rank_name(r)
        if self.type == ComboType.PAIR:
            return 'pair of ' + rank_plural(r)
        if self.type == ComboType.TRIPLE:
            return 'three ' + rank_plural(r)
        if self.type == ComboType.FULL_HOUSE:
            return 'full house of %s with %s' % (rank_plural(r), rank_plural(self.pair_rank))
        if self.type == ComboType.STRAIGHT:
            return 'straight to ' + rank_name(r)
        if self.type == ComboType.TUBE:
            return 'tube up to ' + rank_plural(r)
        if self.type == ComboType.PLATE:
            return 'plate up to ' + rank_plural(r)
        if self.type == ComboType.BOMB:
            return '%i-card bomb of %s' % (self.n, rank_plural(r))
        if self.type == ComboType.STRAIGHT_FLUSH:
            return 'straight flush to ' + rank_name(r)
        return 'four jokers'

    def __str__(self):
        return '%s [%s]' % (self.describe(), ' '.join(str(card) for card in self.cards))


_NAMES = {11: 'J', 12: 'Q', 13: 'K', 14: 'A', BLACK_JOKER: 'black joker', RED_JOKER: 'red joker'}


def rank_name(rank: int) -> str:
    return _NAMES.get(rank, str(rank))


def rank_plural(rank: int) -> str:
    return rank_name(rank) + 's'


def make(combo_type: ComboType, cards, value: int, rank: int, pair_rank: Optional[int] = None) -> Combo:
    return Combo(combo_type, tuple(sorted(cards, key=Card.sort_key)), value, rank, pair_rank)


def _natural(q: int) -> int:
    # window position 1 is the ace played low
    return 14 if q == 1 else q


def _sequence_tops(counts: Dict[int, int], n_natural: int, n_wild: int, length: int, per: int) -> List[int]:
    """
    tops of the windows of length consecutive ranks (ace low or high) that these natural cards plus n_wild wilds
    fill exactly, per cards to a rank
    """
    if any(rank >= BLACK_JOKER for rank in counts):
        return []
    tops = []
    for start in range(1, 16 - length):
        need, used, ok = 0, 0, True
        for q in range(start, start + length):
            c = counts.get(_natural(q), 0)
            if c > per:
                ok = False
                break
            need += per - c
            used += c
        if ok and used == n_natural and need == n_wild:
            tops.append(start + length - 1)
    return tops


def interpret(cards: List[Card], level: Rank) -> List[Combo]:
    """
    every legal reading of a set of cards, bombs first, then strongest first; wilds take whatever role fits, so the
    same cards may read as several full houses or sequences
    """
    n = len(cards)
    out = []
    if not n:
        return out
    natural = [card for card in cards if not is_wild(card, level)]
    n_wild = n - len(natural)
    counts = Counter(card.rank.value for card in natural)
    if n == 4 and counts[RED_JOKER] == 2 and counts[BLACK_JOKER] == 2:
        out.append(make(ComboType.JOKERS, cards, 100, RED_JOKER))
    if len(counts) <= 1:
        r = next(iter(counts)) if counts else level.value
        joker, v = r >= BLACK_JOKER, group_value(r, level)
        if n == 1:
            out.append(make(ComboType.SINGLE, cards, v, r))
        elif n == 2 and (not joker or n_wild == 0):
            out.append(make(ComboType.PAIR, cards, v, r))
        elif n == 3 and not joker:
            out.append(make(ComboType.TRIPLE, cards, v, r))
        elif 4 <= n <= 10 and not joker:
            out.append(make(ComboType.BOMB, cards, v, r))
    if n == 5:
        # wilds can make the same cards read as more than one full house or straight; each is a legal reading
        for t in range(2, 15):
            ct = counts.get(t, 0)
            if ct > 3:
                continue
            for p in list(range(2, 15)) + [BLACK_JOKER, RED_JOKER]:
                cp = counts.get(p, 0)
                if p == t or cp > 2 or ct + cp != len(natural) or (p >= BLACK_JOKER and cp != 2):
                    continue
                out.append(make(ComboType.FULL_HOUSE, cards, group_value(t, level), t, p if cp else level.value))
                break
        tops = _sequence_tops(counts, len(natural), n_wild, 5, 1)
        flush = all(card.suit == natural[0].suit for card in natural)
        for top in tops:
            if flush:
                out.append(make(ComboType.STRAIGHT_FLUSH, cards, top, top))
            out.append(make(ComboType.STRAIGHT, cards, top, top))
    if n == 6:
        out += [make(ComboType.TUBE, cards, top, top) for top in _sequence_tops(counts, len(natural), n_wild, 3, 2)]
        out += [make(ComboType.PLATE, cards, top, top) for top in _sequence_tops(counts, len(natural), n_wild, 2, 3)]
    return sorted(out, key=lambda c: (not c.is_bomb, -c.power if c.is_bomb else -c.value))


def generate_moves(hand: List[Card], level: Rank) -> List[Combo]:
    """
    every combination the hand can play, using wilds only where natural cards run short; other readings of the same
    cards (another suit of a rank, a wild in another role) are legal too, see Round.is_legal
    """
    wild = [card for card in hand if is_wild(card, level)]
    n_wild = len(wild)
    by_rank = {}  # type: Dict[int, List[Card]]
    for card in hand:
        if not is_wild(card, level):
            by_rank.setdefault(card.rank.value, []).append(card)
    moves, seen = [], set()

    def add(combo_type, cards, value, rank, pair_rank=None):
        combo = make(combo_type, cards, value, rank, pair_rank)
        key = (combo_type, combo.cards)
        if key in seen:
            return None
        seen.add(key)
        moves.append(combo)
        return combo

    trips, pairs = [], []  # (rank, cards, wilds used)
    for r in list(range(2, 15)) + [BLACK_JOKER, RED_JOKER]:
        cs = by_rank.get(r)
        if not cs:
            continue
        c, joker, v = len(cs), r >= BLACK_JOKER, group_value(r, level)
        add(ComboType.SINGLE, cs[:1], v, r)
        for uw in range(min(n_wild, 1) + 1):
            if joker and uw:
                continue
            if c >= 2 - uw:
                cards = cs[:2 - uw] + wild[:uw]
                add(ComboType.PAIR, cards, v, r)
                pairs.append((r, cards, uw))
        if not joker:
            for uw in range(min(n_wild, 2) + 1):
                if c >= 3 - uw:
                    cards = cs[:3 - uw] + wild[:uw]
                    add(ComboType.TRIPLE, cards, v, r)
                    trips.append((r, cards, uw))
            for k in range(4, min(10, c + n_wild) + 1):
                uw = max(0, k - c)
                add(ComboType.BOMB, cs[:k - uw] + wild[:uw], v, r)
    if n_wild:
        add(ComboType.SINGLE, wild[:1], LEVEL_VALUE, level.value)
    if n_wild == 2:
        add(ComboType.PAIR, wild, LEVEL_VALUE, level.value)
    if len(by_rank.get(BLACK_JOKER, [])) == 2 and len(by_rank.get(RED_JOKER, [])) == 2:
        add(ComboType.JOKERS, by_rank[BLACK_JOKER] + by_rank[RED_JOKER], 100, RED_JOKER)
    for t_rank, t_cards, t_wild in trips:
        for p_rank, p_cards, p_wild in pairs:
            if t_rank == p_rank or t_wild + p_wild > n_wild or (p_wild and t_wild):
                continue
            pair_cards = p_cards[:1] + wild[t_wild:t_wild + 1] if p_wild else p_cards
            add(ComboType.FULL_HOUSE, t_cards + pair_cards, group_value(t_rank, level), t_rank, p_rank)

    def sequences(combo_type, length, per, pick):
        for start in range(1, 16 - length):
            need, cards = 0, []
            for q in range(start, start + length):
                cs = pick(_natural(q))
                take = min(per, len(cs))
                cards += cs[:take]
                need += per - take
            if need <= n_wild:
                top = start + length - 1
                add(combo_type, cards + wild[:need], top, top)

    sequences(ComboType.STRAIGHT, 5, 1, lambda r: by_rank.get(r, []))
    for suit in SUIT_ORDER:
        sequences(ComboType.STRAIGHT_FLUSH, 5, 1, lambda r: [c for c in by_rank.get(r, []) if c.suit == suit])
    sequences(ComboType.TUBE, 3, 2, lambda r: by_rank.get(r, []))
    sequences(ComboType.PLATE, 2, 3, lambda r: by_rank.get(r, []))
    return moves

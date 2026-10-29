from typing import List, Optional

from src.CardGames.Core.Card import Card, Rank

__author__ = 'frank.ma'


def card_value(card: Card) -> int:
    # ace counts 1 here, Hand.total promotes one ace to 11 when that does not bust
    if card.rank == Rank.ACE:
        return 1
    if card.rank in (Rank.JACK, Rank.QUEEN, Rank.KING):
        return 10
    if card.is_joker():
        raise ValueError('jokers are not played in blackjack.')
    return card.rank.value


class Hand(object):
    def __init__(self, cards: Optional[List[Card]] = None, bet: float = 1.0, from_split: bool = False):
        self.cards = list(cards) if cards is not None else []  # type: List[Card]
        self.bet = bet
        self.from_split = from_split
        self.doubled = False
        self.surrendered = False
        self.stood = False

    def add(self, card: Card) -> None:
        self.cards.append(card)

    def _hard_total(self) -> int:
        return sum(card_value(card) for card in self.cards)

    @property
    def total(self) -> int:
        hard = self._hard_total()
        has_ace = any(card.rank == Rank.ACE for card in self.cards)
        return hard + 10 if has_ace and hard + 10 <= 21 else hard

    @property
    def is_soft(self) -> bool:
        # an ace is being counted as 11
        hard = self._hard_total()
        return any(card.rank == Rank.ACE for card in self.cards) and hard + 10 <= 21

    @property
    def is_blackjack(self) -> bool:
        # 21 on the first two cards of an unsplit hand
        return len(self.cards) == 2 and not self.from_split and self.total == 21

    @property
    def is_bust(self) -> bool:
        return self._hard_total() > 21

    @property
    def is_pair(self) -> bool:
        # two cards of equal value, so a ten and a king may be split
        return len(self.cards) == 2 and card_value(self.cards[0]) == card_value(self.cards[1])

    @property
    def is_done(self) -> bool:
        return self.stood or self.surrendered or self.doubled or self.is_bust or self.total == 21

    def __str__(self):
        return '%s (%s%i)' % (' '.join(str(card) for card in self.cards), 'soft ' if self.is_soft else '', self.total)

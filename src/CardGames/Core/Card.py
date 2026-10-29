from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

__author__ = 'frank.ma'


class Suit(IntEnum):
    CLUB = 1
    DIAMOND = 2
    HEART = 3
    SPADE = 4


class Rank(IntEnum):
    # natural order with ace high; a game that ranks cards differently (ace low in blackjack, the level card in
    # guandan) supplies its own ordering instead of relying on these values
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    BLACK_JOKER = 16
    RED_JOKER = 17


STANDARD_RANKS = [rank for rank in Rank if rank < Rank.BLACK_JOKER]
JOKERS = [Rank.BLACK_JOKER, Rank.RED_JOKER]

SUIT_SYMBOLS = {Suit.CLUB: 'C', Suit.DIAMOND: 'D', Suit.HEART: 'H', Suit.SPADE: 'S'}
RANK_SYMBOLS = {Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5', Rank.SIX: '6', Rank.SEVEN: '7',
                Rank.EIGHT: '8', Rank.NINE: '9', Rank.TEN: 'T', Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K',
                Rank.ACE: 'A'}
JOKER_SYMBOLS = {Rank.BLACK_JOKER: 'BJ', Rank.RED_JOKER: 'RJ'}

_RANK_BY_SYMBOL = {symbol: rank for rank, symbol in RANK_SYMBOLS.items()}
_RANK_BY_SYMBOL['10'] = Rank.TEN
_SUIT_BY_SYMBOL = {symbol: suit for suit, symbol in SUIT_SYMBOLS.items()}
_JOKER_BY_SYMBOL = {symbol: rank for rank, symbol in JOKER_SYMBOLS.items()}


@dataclass(frozen=True)
class Card(object):
    rank: Rank
    suit: Optional[Suit] = None  # jokers carry no suit

    def __post_init__(self):
        if self.is_joker() != (self.suit is None):
            raise ValueError('a joker has no suit and every other card has one, received %r of %r.'
                             % (self.rank, self.suit))

    def is_joker(self) -> bool:
        return self.rank in JOKERS

    def sort_key(self):
        return self.rank.value, self.suit.value if self.suit is not None else 0

    def __str__(self):
        if self.is_joker():
            return JOKER_SYMBOLS[self.rank]
        return RANK_SYMBOLS[self.rank] + SUIT_SYMBOLS[self.suit]

    def __repr__(self):
        return 'Card(%s)' % self.__str__()

    @staticmethod
    def parse(card_str: str) -> 'Card':
        """
        read a card from its short form, rank then suit, e.g. 'AS', 'TH', '10H', or 'BJ' / 'RJ' for the jokers
        :param card_str: short form of the card, case insensitive
        :return: Card
        """
        text = card_str.strip().upper()
        if text in _JOKER_BY_SYMBOL:
            return Card(_JOKER_BY_SYMBOL[text])
        rank_str, suit_str = text[:-1], text[-1:]
        if rank_str not in _RANK_BY_SYMBOL or suit_str not in _SUIT_BY_SYMBOL:
            raise ValueError('unrecognized card %r, expect rank then suit such as AS, TH or 10H.' % card_str)
        return Card(_RANK_BY_SYMBOL[rank_str], _SUIT_BY_SYMBOL[suit_str])

    @staticmethod
    def parse_many(cards_str: str) -> List['Card']:
        return [Card.parse(token) for token in cards_str.split()]


def standard_deck(jokers: bool = False) -> List[Card]:
    """
    one unshuffled deck of 52 cards, or 54 with the two jokers
    """
    cards = [Card(rank, suit) for suit in Suit for rank in STANDARD_RANKS]
    if jokers:
        cards += [Card(rank) for rank in JOKERS]
    return cards

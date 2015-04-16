from enum import Enum

__author__ = 'frank.ma'


class Card(object):

    class Rank(Enum):
        ace, two, three, four, five, six, seven, eight, nine, ten, jack, queen, king = range(1, 14)

    class Suit(Enum):
        spade, heart, club, diamond = range(1, 5)

    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        # rank string in the str form of A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
        rank_str = self.rank.value.__str__() if self.rank.value in range(2, 11) else self.rank.name[0].__str__().upper()
        suit_str = self.suit.name.__str__().upper()
        return '%s %s' % (rank_str, suit_str)

    @staticmethod
    def read_str(card_str: str):
        # assume a string separated by space with the format of rank suit
        c_s = card_str.split(sep=' ')
        assert c_s.__len__() == 2, 'incorrect string input, expect 2 items, received %i.' % c_s.__len__()
        # TODO: return card class
from enum import Enum

__author__ = 'frank.ma'


class Rank(Enum):
    ace, two, three, four, five, six, seven, eight, nine, ten, jack, queen, king = range(1, 14)


class Suit(Enum):
    spade, heart, club, diamond = range(1, 5)


class Card(object):

    rank_dict = dict({'a': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'j': 11,
                      'q': 12, 'k': 13})
    suit_dict = dict({'spade': 1, 'heart': 2, 'club': 3, 'diamond': 4})

    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        # rank string in the str form of A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
        rank_str = self.rank.value.__str__() if self.rank.value in range(2, 11) else self.rank.name[0].__str__().upper()
        suit_str = self.suit.name.__str__().upper()
        return '%s %s' % (rank_str, suit_str)

    def get_rank_int(self):
        """
        expect to get the integer
        :return: int
        """
        return self.rank.value

    @staticmethod
    def read_str(card_str: str):
        # assume a string separated by space with the the sequence as rank then suit
        c_s = card_str.lower().split(sep=' ')
        assert c_s.__len__() == 2, 'incorrect string input, expect 2 items, received %i.' % c_s.__len__()
        assert c_s[0] in Card.rank_dict, 'incorrect rank input, expect one of %s, received %s.' % \
                                         (Card.rank_dict.keys().__str__(), c_s[0])
        assert c_s[1] in Card.suit_dict, 'incorrect rank input, expect one of %s, received %s.' % \
                                         (Card.suit_dict.keys().__str__(), c_s[1])
        rank = Rank(Card.rank_dict[c_s[0]])
        suit = Suit(Card.suit_dict[c_s[1]])
        return Card(rank, suit)
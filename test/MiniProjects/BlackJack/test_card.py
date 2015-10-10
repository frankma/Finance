from unittest import TestCase

from src.MiniProjects.BlackJack.Card import Card, Rank, Suit

__author__ = 'frank.ma'


class TestCard(TestCase):
    def test_string(self):
        print(Card(Rank.ace, Suit.spade))

    def test_read_str(self):
        for rank in Rank:
            for suit in Suit:
                card = Card(rank, suit)
                card_str = card.__str__()
                card_recovered = Card.read_str(card_str)
                assert card.__str__() == card_recovered.__str__(), 'incorrect card reading: %s' % card_str
        pass

    def test_get_rank(self):
        for rank in Rank:
            for suit in Suit:
                card = Card(rank, suit)
                rank_in = card.get_rank_int()
                rank_out = rank.value
                assert rank_in == rank_out, 'returned integer value %i does not match input %i.' % (rank_in, rank_out)
        pass

from unittest import TestCase
from src.MiniProjects.BlackJack.Card import Card

__author__ = 'frank.ma'


class TestCard(TestCase):

    def test_string(self):
        print(Card(Card.Rank.ace, Card.Suit.diamond))

    def test_read_str(self):
        card = Card(Card.Rank.ace, Card.Suit.diamond)
        card_str = card.__str__()

        card_recovered = Card.read_str(card_str)
from unittest import TestCase
from src.MiniProjects.BlackJack.Deck import Deck

__author__ = 'frank.ma'


class TestDeck(TestCase):

    def test_new_deck(self):
        deck = Deck.new_deck(1)
        pass

    def test___str__(self):
        deck = Deck.new_deck(0)
        # print(deck.__str__())
        deck = Deck.new_deck(1)
        # print(deck.__str__())
        pass

    def test_draw(self):
        deck = Deck.new_deck(1)
        print(deck.draw().__str__(), deck.public)
        pass
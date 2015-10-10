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
        deck = Deck.new_deck(2)
        # first draw a public card
        card = deck.draw()
        public_last = deck.view_public()[-1]
        assert card.rank == public_last.rank and card.suit == public_last.suit, \
            'mismatch between public card %s and drawn one %s.' % (card.__str__(), public_last.__str__())
        card_nonpublic = deck.draw(False)
        public_last = deck.view_public()[-1]
        assert card_nonpublic.rank != public_last.rank and card_nonpublic.suit != public_last.suit, \
            'unexpected match between public card and nonpublic drawn one.'
        pass

    def test_reveal(self):
        deck = Deck.new_deck(1)
        card = deck.draw(False)
        assert deck.public.__len__() == 0, 'no public card should be seen at this stage.'
        assert deck.nonpublic.__len__() == 1, 'one nonpublic card should be expected.'
        non_public = deck.cards[deck.nonpublic][0]
        assert non_public.rank == card.rank and non_public.suit == card.suit
        deck.reveal()
        assert deck.nonpublic.__len__() == 0, 'no nonpublic card should be seen at this stage anymore.'
        assert deck.public.__len__() == 1, 'one public should be expected.'
        pubic = deck.cards[deck.public][0]
        assert pubic.rank == card.rank and pubic.suit == card.suit
        pass

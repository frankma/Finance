from unittest import TestCase

from src.Poker.BlackJack.Deck import Deck

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
        err_msg = 'mismatch between public card %s and drawn one %s.' % (card.__str__(), public_last.__str__())
        self.assertEqual(card.rank, public_last.rank, msg=err_msg)
        self.assertEqual(card.suit, public_last.suit, msg=err_msg)

        # a nonpublic draw must not show up among the public cards; two random cards may still share rank or suit
        public_before = list(deck.public)
        card_nonpublic = deck.draw(False)
        self.assertEqual(deck.public, public_before, msg='nonpublic draw changed the public cards.')
        self.assertEqual(len(deck.nonpublic), 1, msg='one nonpublic card should be expected.')
        drawn = deck.cards[deck.nonpublic[-1]]
        self.assertEqual(drawn.rank, card_nonpublic.rank)
        self.assertEqual(drawn.suit, card_nonpublic.suit)
        pass

    def test_reveal(self):
        deck = Deck.new_deck(1)
        card = deck.draw(False)
        self.assertEqual(deck.public.__len__(), 0, msg='no public card should be seen at this stage.')
        self.assertEqual(deck.nonpublic.__len__(), 1, msg='one nonpublic card should be expected.')
        non_public = deck.cards[deck.nonpublic][0]
        self.assertEqual(non_public.rank, card.rank)
        self.assertEqual(non_public.suit, card.suit)
        deck.reveal()
        self.assertEqual(deck.nonpublic.__len__(), 0, msg='no nonpublic card should be seen at this stage anymore.')
        self.assertEqual(deck.public.__len__(), 1, msg='one public should be expected.')
        pubic = deck.cards[deck.public][0]
        self.assertEqual(pubic.rank, card.rank)
        self.assertEqual(pubic.suit, card.suit)
        pass

from unittest import TestCase

from src.CardGames.BlackJack.Counting import hi_lo_counter
from src.CardGames.Core.Card import Card, Rank, standard_deck
from src.CardGames.Core.CardCounter import CardCounter

__author__ = 'frank.ma'


class TestCardCounter(TestCase):
    def test_unseen(self):
        counter = CardCounter(standard_deck() * 2)
        counter.see_many(Card.parse_many('AS AS KD'))
        self.assertEqual(counter.n_seen, 3)
        self.assertEqual(counter.n_unseen, 101)
        unseen = counter.unseen()
        self.assertNotIn(Card.parse('AS'), unseen)
        self.assertEqual(unseen.count(Card.parse('KD')), 1)
        self.assertEqual(counter.unseen_by_rank()[Rank.ACE], 6)

    def test_cannot_see_more_than_composition(self):
        counter = CardCounter(standard_deck())
        counter.see(Card.parse('AS'))
        with self.assertRaises(ValueError):
            counter.see(Card.parse('AS'))

    def test_hi_lo(self):
        counter = hi_lo_counter(standard_deck() * 2)
        counter.see_many(Card.parse_many('2C 3D 4H 5S 6C 7D 8H 9S'))
        self.assertEqual(counter.running_count, 5.0)
        counter.see_many(Card.parse_many('TC JD QH KS AC'))
        self.assertEqual(counter.running_count, 0.0)
        counter.see_many(Card.parse_many('2D 2H 2S'))
        # 3 over (104 - 16) / 52 decks unseen
        self.assertAlmostEqual(counter.true_count(), 3.0 / (88 / 52))
        counter.reset()
        self.assertEqual((counter.running_count, counter.n_seen), (0.0, 0))

import random
from collections import Counter
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Core.CardCounter import CardCounter
from src.CardGames.Core.Shoe import Shoe

__author__ = 'frank.ma'


class TestShoe(TestCase):
    def test_composition(self):
        self.assertEqual(Shoe(6).n_cards, 312)
        self.assertEqual(Shoe(2, jokers=True).n_cards, 108)
        self.assertEqual(Shoe(cards=Card.parse_many('AS AS KD')).n_cards, 3)
        with self.assertRaises(ValueError):
            Shoe(cards=[])
        with self.assertRaises(ValueError):
            Shoe(penetration=0.0)

    def test_seeded_deal_is_reproducible(self):
        first = Shoe(2, rng=random.Random(7)).draw_many(20)
        second = Shoe(2, rng=random.Random(7)).draw_many(20)
        self.assertEqual(first, second)

    def test_face_up_and_reveal(self):
        shoe = Shoe(1, rng=random.Random(0))
        counter = CardCounter(shoe.composition)
        shoe.watch(counter)
        up = shoe.draw()
        down = shoe.draw(face_up=False)
        self.assertEqual(shoe.face_up, [up])
        self.assertEqual(shoe.face_down, [down])
        self.assertEqual(counter.n_seen, 1, msg='a face-down card must stay unseen.')
        shoe.reveal(down)
        self.assertEqual(shoe.face_up, [up, down])
        self.assertEqual(shoe.face_down, [])
        self.assertEqual(counter.n_seen, 2)
        with self.assertRaises(ValueError):
            shoe.reveal(down)

    def test_penetration_and_reshuffle(self):
        shoe = Shoe(1, penetration=0.5, rng=random.Random(1))
        counter = CardCounter(shoe.composition)
        shoe.watch(counter)
        shoe.draw_many(25)
        self.assertFalse(shoe.needs_shuffle)
        shoe.draw()
        self.assertTrue(shoe.needs_shuffle)
        shoe.reshuffle()
        self.assertEqual(shoe.cards_remaining, 52)
        self.assertEqual(Counter(shoe.cards), Counter(shoe.composition))
        self.assertEqual(counter.n_seen, 0, msg='a reshuffle should reset the watchers.')

    def test_ran_out(self):
        shoe = Shoe(cards=Card.parse_many('AS KD'))
        shoe.draw_many(2)
        with self.assertRaises(ValueError):
            shoe.draw()
        with self.assertRaises(ValueError):
            Shoe(1).draw_many(53)

import random
from unittest import TestCase

from src.CardGames.Core.Card import Card, Rank
from src.CardGames.GuanDan.Combo import ComboType, generate_moves, group_value, interpret, is_wild, sort_hand
from src.CardGames.GuanDan.Round import deal

__author__ = 'frank.ma'

P = Card.parse_many
L2, L7 = Rank.TWO, Rank.SEVEN


def read(cards: str, level: Rank = L7):
    return [(c.type, c.value) for c in interpret(P(cards), level)]


def best(cards: str, level: Rank = L7):
    return interpret(P(cards), level)[0]


class TestLevel(TestCase):
    def test_wild_and_order(self):
        self.assertTrue(is_wild(Card.parse('7H'), L7))
        self.assertFalse(is_wild(Card.parse('7S'), L7))
        self.assertEqual([group_value(r, L7) for r in (14, 7, 16, 17)], [14, 15, 16, 17])
        self.assertEqual(' '.join(map(str, sort_hand(P('RJ 7H 7S AS 2C BJ'), L7))), '2C AS 7S BJ RJ 7H')


class TestInterpret(TestCase):
    def test_groups(self):
        self.assertEqual(read('9S'), [(ComboType.SINGLE, 9)])
        self.assertEqual(read('7S 7D'), [(ComboType.PAIR, 15)], 'the level rank pairs above aces.')
        self.assertEqual(read('BJ BJ'), [(ComboType.PAIR, 16)])
        self.assertEqual(read('BJ RJ'), [], 'one joker of each colour is no pair.')
        self.assertEqual(read('BJ 7H'), [], 'a wild never stands for a joker.')
        self.assertEqual(read('9S 7H'), [(ComboType.PAIR, 9)])
        self.assertEqual(read('9S 9D 9C'), [(ComboType.TRIPLE, 9)])
        self.assertEqual(read('BJ BJ RJ'), [], 'jokers make no triple.')
        self.assertEqual(read('9S 9D 9C 9H 7H'), [(ComboType.BOMB, 9)])
        self.assertEqual(read('BJ BJ RJ RJ'), [(ComboType.JOKERS, 100)])
        self.assertEqual(read('9S 8D'), [])

    def test_full_house(self):
        self.assertEqual(read('9S 9D 9C 4S 4D'), [(ComboType.FULL_HOUSE, 9)])
        self.assertEqual(read('9S 9D 9C BJ BJ'), [(ComboType.FULL_HOUSE, 9)], 'a joker pair may sit under a triple.')
        # a wild completes either the triple or the pair, and both are legal readings, strongest first
        self.assertEqual(read('9S 9D 4S 4D 7H'), [(ComboType.FULL_HOUSE, 9), (ComboType.FULL_HOUSE, 4)])
        self.assertEqual(best('9S 9D 9C 4S 4D').describe(), 'full house of 9s with 4s')

    def test_sequences(self):
        self.assertEqual(read('5S 6D 7C 8S 9H'), [(ComboType.STRAIGHT, 9)], 'the level card plays naturally in a run.')
        self.assertEqual(read('AS 2D 3C 4S 5H'), [(ComboType.STRAIGHT, 5)], 'the ace plays low.')
        self.assertEqual(read('TS JD QC KS AH'), [(ComboType.STRAIGHT, 14)], 'or high.')
        self.assertEqual(read('QS KD AC 2S 3H'), [], 'but never in the middle.')
        self.assertEqual(read('5S 6S 7S 8S 9S')[0], (ComboType.STRAIGHT_FLUSH, 9))
        self.assertEqual(read('5S 6S 8S 9S 7H')[0], (ComboType.STRAIGHT_FLUSH, 9), 'a wild fills the gap in suit.')
        self.assertEqual(read('9S TD JC QS 7H'), [(ComboType.STRAIGHT, 13), (ComboType.STRAIGHT, 12)])
        self.assertEqual(read('QS QD KC KS AH AD'), [(ComboType.TUBE, 14)])
        self.assertEqual(read('AS AD 2C 2S 3H 3D'), [(ComboType.TUBE, 3)])
        self.assertEqual(read('KS KD KC AS AH AD'), [(ComboType.PLATE, 14)])
        self.assertEqual(read('KS KD AC AS AH AD'), [], 'a plate is two triples.')

    def test_beats_and_bomb_ladder(self):
        self.assertTrue(best('9S 9D').beats(best('8S 8D')))
        self.assertFalse(best('9S 9D').beats(best('9C 9H')), 'equal value does not beat.')
        self.assertFalse(best('9S 9D 9C').beats(best('8S 8D')), 'shape must match.')
        self.assertTrue(best('7S').beats(best('AS')), 'the level rank tops the ace.')
        self.assertTrue(best('RJ').beats(best('7S')))
        ladder = ['3S 3D 3C 3H', 'AS AD AC AH', '7S 7D 7C 7C', '2S 2D 2C 2H 2S', '9S TS JS QS KS',
                  '2S 2D 2C 2H 2S 2D', '3S 3D 3C 3H 3S 3D 3C', 'BJ BJ RJ RJ']
        bombs = [best(cards) for cards in ladder]
        for lower, higher in zip(bombs, bombs[1:]):
            self.assertTrue(higher.beats(lower), '%s should beat %s' % (higher, lower))
            self.assertFalse(lower.beats(higher))
        self.assertTrue(bombs[0].beats(best('TS JD QC KS AH')), 'any bomb beats an ordinary combination.')
        self.assertFalse(best('TS JD QC KS AH').beats(bombs[0]))
        self.assertTrue(bombs[0].beats(None), 'anything leads.')


class TestGenerateMoves(TestCase):
    def test_small_hand(self):
        moves = generate_moves(P('9S 9D 9C 4S 4D 7H'), L7)
        kinds = {(m.type, m.value) for m in moves}
        for expected in [(ComboType.SINGLE, 9), (ComboType.SINGLE, 15), (ComboType.PAIR, 4), (ComboType.TRIPLE, 4),
                         (ComboType.BOMB, 9), (ComboType.FULL_HOUSE, 9), (ComboType.FULL_HOUSE, 4)]:
            self.assertIn(expected, kinds)
        self.assertNotIn((ComboType.PAIR, 15), kinds, 'one wild alone makes no pair.')

    def test_every_move_is_a_reading_of_its_cards(self):
        rng = random.Random(1)
        for _ in range(20):
            level = Rank(rng.randint(2, 14))
            hand = deal(rng)[0]
            for move in generate_moves(hand, level):
                readings = {(r.type, r.value, r.rank) for r in interpret(list(move.cards), level)}
                self.assertIn((move.type, move.value, move.rank), readings, str(move))

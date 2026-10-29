import random
from itertools import combinations
from unittest import TestCase

from src.CardGames.Core.Card import Card, standard_deck
from src.CardGames.Holdem.HandEvaluator import HandCategory, best_five, describe, evaluate, evaluate_omaha

__author__ = 'frank.ma'


def rank(cards: str):
    return evaluate(Card.parse_many(cards))


def brute_force(cards):
    # reference: the best of every five-card subset
    return max(evaluate(list(five)) for five in combinations(cards, 5))


class TestHandEvaluator(TestCase):
    def test_categories(self):
        cases = [('AS KH 9C 5D 2S', HandCategory.HIGH_CARD), ('AS AH 9C 5D 2S', HandCategory.PAIR),
                 ('AS AH 9C 9D 2S', HandCategory.TWO_PAIR), ('AS AH AC 5D 2S', HandCategory.THREE_OF_A_KIND),
                 ('6S 5H 4C 3D 2S', HandCategory.STRAIGHT), ('AS 9S 7S 5S 2S', HandCategory.FLUSH),
                 ('AS AH AC 5D 5S', HandCategory.FULL_HOUSE), ('AS AH AC AD 2S', HandCategory.FOUR_OF_A_KIND),
                 ('6S 5S 4S 3S 2S', HandCategory.STRAIGHT_FLUSH)]
        for cards, category in cases:
            self.assertEqual(rank(cards).category, category, cards)

    def test_straights(self):
        self.assertEqual(rank('AS 5H 4C 3D 2S').tiebreakers, (5,), 'the wheel is five high.')
        self.assertEqual(rank('AS KH QC JD TS').tiebreakers, (14,))
        self.assertEqual(rank('KS AH 2C 3D 4S').category, HandCategory.HIGH_CARD, 'no wrap-around straights.')
        self.assertGreater(rank('6S 5H 4C 3D 2S'), rank('AS 5H 4C 3D 2S'))

    def test_tiebreakers(self):
        self.assertGreater(rank('AC AD KC 5H 2H'), rank('AS AH 9C 5D 2S'), 'kicker decides.')
        self.assertEqual(rank('AC AD KC 5H 2H'), rank('AS AH KD 5C 2D'), 'suits never break ties.')
        self.assertGreater(rank('3S 3H 3C 2D 2S'), rank('2S 2H 2C AD AS'), 'trips rank first in a full house.')
        self.assertEqual(rank('AS AH AC AD KS 2H 3C').tiebreakers, (14, 13), 'best five of seven.')
        self.assertEqual(rank('KS KH KC QD QS QH 2C').tiebreakers, (13, 12), 'two trips make a full house.')
        self.assertEqual(rank('KS KH QC QD JS JH 2C').tiebreakers, (13, 12, 11), 'best two of three pairs.')

    def test_two_flush_suits(self):
        # ten cards can hold two flushes; the higher one plays
        self.assertEqual(rank('AS 9S 7S 5S 2S KH QH JH 3H 2H').tiebreakers, (14, 9, 7, 5, 2))
        self.assertEqual(rank('9S 8S 7S 6S 5S 2H 3H 4H 5H 6H').tiebreakers, (9,))

    def test_matches_brute_force(self):
        rng = random.Random(21)
        deck = standard_deck()
        for _ in range(500):
            cards = rng.sample(deck, rng.choice([5, 6, 7]))
            self.assertEqual(evaluate(cards), brute_force(cards), ' '.join(str(card) for card in cards))

    def test_omaha_uses_two_and_three(self):
        hole = Card.parse_many('AS KS QH JH')
        board = Card.parse_many('2S 7S 9S TD 3C')
        # a hold'em reading sees an ace-high flush; the Omaha rule needs two hole spades and three board spades
        self.assertEqual(evaluate(hole + board).category, HandCategory.FLUSH)
        self.assertEqual(evaluate_omaha(hole, board).category, HandCategory.FLUSH)
        one_spade = Card.parse_many('AS KH QH JH')
        self.assertEqual(evaluate(one_spade + board).category, HandCategory.STRAIGHT, 'A K Q J from the hole, T.')
        self.assertEqual(evaluate_omaha(one_spade, board).category, HandCategory.HIGH_CARD,
                         'one hole spade makes no flush, and no two hole cards make a straight.')
        quads_on_board = Card.parse_many('2S 2H 2C 2D 3C')
        self.assertEqual(evaluate_omaha(hole, quads_on_board).category, HandCategory.THREE_OF_A_KIND)
        with self.assertRaises(ValueError):
            evaluate_omaha(hole, board[:2])

    def test_describe_and_best_five(self):
        self.assertEqual(describe(rank('KS KH KC 5D 5S')), 'Full house, kings full of fives')
        self.assertEqual(describe(rank('AS KS QS JS TS')), 'Royal flush')
        self.assertEqual(describe(rank('6S 6H 9C 5D 2S')), 'Pair of sixes')
        cards = Card.parse_many('AS AH AC AD KS 2H 3C')
        self.assertEqual(sorted(map(str, best_five(cards))), ['AC', 'AD', 'AH', 'AS', 'KS'])
        with self.assertRaises(ValueError):
            rank('AS KS')

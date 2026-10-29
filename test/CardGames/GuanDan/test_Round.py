import random
from collections import Counter
from unittest import TestCase

from src.CardGames.Core.Card import Card, Rank, standard_deck
from src.CardGames.Core.Game import play_out
from src.CardGames.Core.Player import RandomStrategy
from src.CardGames.GuanDan.Combo import ComboType, interpret
from src.CardGames.GuanDan.Planner import plan_hand
from src.CardGames.GuanDan.Round import PASS, Round, deal, partner, team

__author__ = 'frank.ma'

P = Card.parse_many
L2 = Rank.TWO


def combo(cards: str, level: Rank = L2):
    return interpret(P(cards), level)[0]


class TestDeal(TestCase):
    def test_deal(self):
        hands = deal(random.Random(0))
        self.assertEqual([len(hand) for hand in hands], [27] * 4)
        self.assertEqual(Counter(card for hand in hands for card in hand), Counter(standard_deck(jokers=True) * 2))
        self.assertEqual((team(0), team(1), team(2), team(3)), (0, 1, 0, 1))
        self.assertEqual(partner(1), 3)


class TestRound(TestCase):
    def test_trick_flow(self):
        game = Round([P('3S 9D'), P('4S 5D'), P('6S 7D'), P('8S TD')], L2, leader=0)
        self.assertNotIn(PASS, game.legal_moves(0), 'the leader must play.')
        self.assertFalse(game.is_legal(0, PASS))
        game.apply(0, combo('3S'))
        self.assertEqual(game.current_seat(), 1)
        self.assertIn(PASS, game.legal_moves(1))
        self.assertEqual(interpret(P('4S 5D'), L2), [], 'two unpaired cards have no reading.')
        game.apply(1, combo('5D'))
        game.apply(2, PASS)
        game.apply(3, PASS)
        game.apply(0, PASS)
        self.assertEqual((game.current_seat(), game.target), (1, None), 'three passes win the trick.')
        with self.assertRaises(ValueError):
            game.apply(1, combo('9D'))

    def test_any_reading_is_legal(self):
        # the generator picks one spade for a single nine; the other nine, or the wild as the level rank, is fine too
        game = Round([P('9S 9D 2H'), P('4S'), P('6S'), P('8S')], L2, leader=0)
        self.assertTrue(game.is_legal(0, combo('9D')))
        self.assertTrue(game.is_legal(0, combo('9D 2H')))
        self.assertFalse(game.is_legal(0, combo('9C')), 'a card not held.')

    def test_catching_the_wind(self):
        # seat 0 goes out with a trick nobody beats, so partner seat 2 leads
        game = Round([P('AS'), P('4S 5D'), P('6S 7D'), P('8S TD')], L2, leader=0)
        game.apply(0, combo('AS'))
        self.assertEqual(game.out, [0])
        game.apply(1, PASS)
        game.apply(2, PASS)
        game.apply(3, PASS)
        self.assertEqual(game.current_seat(), 2)
        self.assertIsNone(game.target)

    def test_hand_result(self):
        # seats 0 and 2 go out first and second: a double win, up three
        game = Round([P('AS'), P('4S 5D'), P('KS'), P('8S TD')], L2, leader=0)
        game.apply(0, combo('AS'))
        for seat in (1, 2, 3):
            game.apply(seat, PASS)
        game.apply(2, combo('KS'))
        self.assertTrue(game.is_terminal())
        self.assertEqual((game.result.order, game.result.winning_team, game.result.levels_up), ((0, 2, 1, 3), 0, 3))
        self.assertEqual(game.payoffs(), [3.0, -3.0, 3.0, -3.0])

    def test_levels_up_by_partner_finish(self):
        # seat 0 out first and its partner last: the team of the first out still wins, but climbs only one
        game = Round([P('AS'), P('KS'), P('3S 4D'), P('QS')], L2, leader=0)
        game.apply(0, combo('AS'))
        for seat in (1, 2, 3):
            game.apply(seat, PASS)
        game.apply(2, combo('3S'))
        game.apply(3, combo('QS'))
        game.apply(1, PASS)
        self.assertEqual(game.out, [0, 3])
        game.apply(2, PASS)
        self.assertEqual(game.current_seat(), 1, 'seat 3 went out on the trick, so its partner leads.')
        game.apply(1, combo('KS'))
        self.assertEqual((game.result.order, game.result.winning_team, game.result.levels_up), ((0, 3, 1, 2), 0, 1))

    def test_random_play_keeps_the_cards(self):
        rng = random.Random(2)
        for _ in range(30):
            level = Rank(rng.randint(2, 14))
            hands = deal(rng)
            game = Round(hands, level, rng.randrange(4))
            play_out(game, [RandomStrategy(rng)] * 4)
            self.assertEqual(Counter(game.played) + Counter(c for hand in game.hands for c in hand),
                             Counter(card for hand in hands for card in hand))
            self.assertEqual(sum(game.payoffs()), 0.0)
            self.assertEqual(sorted(game.result.order), [0, 1, 2, 3])


class TestPlanner(TestCase):
    def test_structure_lowers_cost(self):
        scattered = plan_hand(P('3S 5D 7C 9H JS KD 4C 6H 8S TD'), L2)
        structured = plan_hand(P('3S 4D 5C 6H 7S 9D 9C 9H KS KD'), L2)
        self.assertGreater(scattered.turns, structured.turns)
        self.assertLess(structured.cost, scattered.cost)
        self.assertIn('straight to 7', structured.sequences)
        self.assertEqual(plan_hand(P('9S 9D 9C 9H BJ BJ RJ RJ'), L2).bombs, 2)

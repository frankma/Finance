import random
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Holdem.Equity import call_ev, exact_equity, outs, pot_odds, simulate_equity
from src.CardGames.Holdem.HandEvaluator import HandCategory
from src.CardGames.Holdem.Rules import OMAHA

__author__ = 'frank.ma'

P = Card.parse_many


class TestEquity(TestCase):
    def test_preflop_equity(self):
        # pocket aces: about 85% heads-up and 56% against four random hands
        heads_up = simulate_equity(P('AS AH'), [], 1, 4000, rng=random.Random(1)).equity
        self.assertLess(abs(heads_up.mean - 0.852), 4.0 * heads_up.stderr, str(heads_up))
        five_way = simulate_equity(P('AS AH'), [], 4, 3000, rng=random.Random(2)).equity
        self.assertLess(abs(five_way.mean - 0.56), 4.0 * five_way.stderr, str(five_way))
        self.assertEqual(simulate_equity(P('7S 2H'), [], 0, 10, rng=random.Random(3)).equity.mean, 1.0)

    def test_seeded_and_categories(self):
        first = simulate_equity(P('AS KS'), P('2S 7S 9H'), 2, 300, rng=random.Random(4))
        second = simulate_equity(P('AS KS'), P('2S 7S 9H'), 2, 300, rng=random.Random(4))
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first.categories.values()), 1.0)
        self.assertGreater(first.categories[HandCategory.FLUSH], 0.25, 'nine spades to come on two cards.')

    def test_exact_equity(self):
        aks, queens = exact_equity([P('AS KS'), P('QH QD')], P('2C 7D 9H'))
        self.assertAlmostEqual(aks + queens, 1.0)
        self.assertAlmostEqual(aks, 237 / 990, msg='six outs twice plus runner-runner, over 990 run-outs.')
        # a river board is a single comparison, and identical hands split
        self.assertEqual(exact_equity([P('AS KS'), P('AH KH')], P('2C 7D 9H TC 3S')), [0.5, 0.5])
        monte_carlo = simulate_equity(P('AS KS'), P('2C 7D 9H'), 1, 2000, rng=random.Random(5)).equity
        self.assertGreater(monte_carlo.mean, aks + 0.15, 'a random hand is weaker than queens.')

    def test_outs(self):
        draw = outs(P('AS KS'), P('2S 7S 9H'))
        self.assertEqual(draw.category, HandCategory.HIGH_CARD)
        self.assertEqual(len(draw.by_category[HandCategory.FLUSH]), 9)
        self.assertEqual(len(draw.by_category[HandCategory.PAIR]), 6)
        self.assertEqual(draw.n_unseen, 47)
        self.assertAlmostEqual(draw.p_next_card, 15 / 47)
        self.assertAlmostEqual(draw.p_by_river(2), 1 - (32 / 47) * (31 / 46))
        # a card that only pairs the board helps everyone, so it is not an out
        pair = outs(P('AS KD'), P('2S 7H 9C TD'))
        self.assertNotIn(HandCategory.TWO_PAIR, pair.by_category)
        self.assertEqual(pair.n_outs, 6, 'three aces and three kings; no single card makes a straight.')
        with self.assertRaises(ValueError):
            outs(P('AS KS'), [])

    def test_omaha_equity(self):
        # double-suited aces are a favourite against one random Omaha hand, but much less than in hold'em
        result = simulate_equity(P('AS AH KS QH'), [], 1, 1500, OMAHA, random.Random(6)).equity
        self.assertGreater(result.mean, 0.6)
        self.assertLess(result.mean, 0.75)

    def test_pot_odds(self):
        self.assertAlmostEqual(pot_odds(100, 50), 50 / 150)
        self.assertEqual(pot_odds(100, 0), 0.0)
        self.assertAlmostEqual(call_ev(0.25, 100, 50), 0.25 * 150 - 50)
        self.assertAlmostEqual(call_ev(pot_odds(100, 50), 100, 50), 0.0, msg='pot odds are the break-even equity.')

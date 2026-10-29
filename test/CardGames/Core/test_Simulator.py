import random
from unittest import TestCase

import numpy as np

from src.CardGames.Core.Card import Card, standard_deck
from src.CardGames.Core.CardCounter import CardCounter
from src.CardGames.Core.Simulator import MonteCarloResult, deal_unseen, run_trials

__author__ = 'frank.ma'


class TestSimulator(TestCase):
    def test_from_samples(self):
        result = MonteCarloResult.from_samples([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result.mean, 2.5)
        self.assertAlmostEqual(result.std, np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
        self.assertAlmostEqual(result.stderr, result.std / 2.0)
        low, high = result.confidence_interval()
        self.assertAlmostEqual(high - low, 2.0 * 1.96 * result.stderr)
        self.assertEqual(MonteCarloResult.from_samples([]).n, 0)

    def test_run_trials(self):
        result = run_trials(lambda rng: rng.random(), 20000, random.Random(3))
        self.assertEqual(result.n, 20000)
        self.assertLess(abs(result.mean - 0.5), 4.0 * result.stderr)
        self.assertAlmostEqual(result.std, np.sqrt(1.0 / 12.0), delta=0.01)

    def test_deal_unseen(self):
        counter = CardCounter(standard_deck())
        seen = Card.parse_many('AS AH AD')
        counter.see_many(seen)
        dealt = deal_unseen(counter, 49, random.Random(5))
        self.assertEqual(len(set(dealt)), 49)
        self.assertTrue(set(dealt).isdisjoint(seen))
        with self.assertRaises(ValueError):
            deal_unseen(counter, 50, random.Random(5))

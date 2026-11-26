import math
from unittest import TestCase

import numpy as np

from src.BettingGames.Core.Bet import Bet

__author__ = 'frank.ma'


class TestBet(TestCase):
    def setUp(self):
        self.bet = Bet(0.5, 0.4, 0.5)

    def test_average_gains_while_the_typical_player_loses(self):
        self.assertAlmostEqual(self.bet.mean_return(), 0.05)
        self.assertAlmostEqual(self.bet.growth_rate(), 0.5 * math.log(0.9))
        self.assertAlmostEqual(self.bet.mean_wealth(1.0, 100), 1.05 ** 100)
        self.assertAlmostEqual(self.bet.quantiles(1.0, 100, [0.5])[0], 0.9 ** 50, msg='50 wins and 50 losses.')

    def test_kelly_fraction(self):
        self.assertAlmostEqual(self.bet.kelly_fraction(), 0.25)
        grid = np.linspace(0.0, 1.0, 1001)
        best = grid[np.argmax([self.bet.growth_rate(f) for f in grid])]
        self.assertAlmostEqual(best, 0.25, places=3, msg='the fraction that maximises the growth rate.')
        self.assertAlmostEqual(self.bet.growth_rate(0.25), 0.5 * math.log(1.125) + 0.5 * math.log(0.9))
        self.assertEqual(Bet(0.5, 0.6, 0.5).kelly_fraction(), 0.0, 'no edge, no bet.')
        self.assertEqual(Bet(2.0, 0.1, 0.9).kelly_fraction(), 1.0, 'never more than all of it.')

    def test_exact_distribution(self):
        pmf = self.bet.wins_pmf(100)
        self.assertAlmostEqual(pmf.sum(), 1.0)
        self.assertAlmostEqual(float(np.dot(pmf, self.bet.wealth_after(1.0, 100))), 1.05 ** 100, delta=1e-6 * 1.05 ** 100)
        self.assertAlmostEqual(self.bet.prob_above(1.0, 100), 0.13562651203691772, msg='56 or more wins of 100.')
        self.assertAlmostEqual(self.bet.prob_above(0.0, 100), 0.0, msg='never staking never gains.')

    def test_multiplier(self):
        np.testing.assert_allclose(self.bet.multiplier([1.0, 1.0, 0.25, 0.0], [True, False, False, True]),
                                   [1.5, 0.6, 0.9, 1.0])

    def test_bad_terms(self):
        for args in ((0.0, 0.4, 0.5), (0.5, 1.2, 0.5), (0.5, 0.4, 1.0)):
            with self.assertRaises(ValueError):
                Bet(*args)

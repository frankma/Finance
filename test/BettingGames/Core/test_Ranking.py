from unittest import TestCase

import numpy as np

from src.BettingGames.Core.Ranking import rank_in, share_below, summary, top_share

__author__ = 'frank.ma'


class TestRanking(TestCase):
    def test_rank(self):
        field = np.array([5.0, 1.0, 3.0, 3.0])
        self.assertEqual(rank_in(6.0, field), 1)
        self.assertEqual(rank_in(3.0, field), 2, 'a tie shares the better place.')
        self.assertEqual(rank_in(0.0, field), 5)
        self.assertAlmostEqual(share_below(3.0, field), 0.25)

    def test_top_share_and_summary(self):
        w = np.array([97.0] + [1.0] * 99 + [0.0] * 0)
        self.assertAlmostEqual(top_share(w, 0.01), 97.0 / 196.0)
        s = summary(np.array([0.001, 0.5, 2.0, 4.0]))
        self.assertEqual(s['n'], 4)
        self.assertAlmostEqual(s['above_start'], 0.5)
        self.assertAlmostEqual(s['ruined'], 0.25)
        self.assertAlmostEqual(s['median'], 1.25)

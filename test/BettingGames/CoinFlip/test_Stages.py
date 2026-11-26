import math
from unittest import TestCase

import numpy as np

from src.BettingGames.CoinFlip import Stages

__author__ = 'frank.ma'


class TestStages(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.one = Stages.all_in()
        cls.two = Stages.kelly_outsider(n_trials=200)
        cls.three = Stages.everyone_sensible()

    def test_all_in(self):
        s = self.one['summary']
        self.assertEqual(s['n'], 1000)
        self.assertEqual(len(self.one['ranked']), 1000)
        self.assertEqual(self.one['ranked'], sorted(self.one['ranked'], reverse=True))
        self.assertEqual(self.one['breakeven_wins'], 56)
        self.assertAlmostEqual(s['above_start'], self.one['exact']['above_start'], delta=0.035, msg='about 14%.')
        self.assertLess(s['median'], 0.05, 'the typical player is nearly wiped out.')
        self.assertGreater(s['mean'], 10.0, 'while the average is up.')
        self.assertGreater(s['top1_share'], 0.5, 'and the top 1% hold most of it.')
        self.assertEqual(len(self.one['mean_path']), 101)
        self.assertEqual(self.one['mean_path'][0], 1.0)

    def test_kelly_outsider(self):
        two = self.two
        self.assertEqual(two['fraction'], 0.25)
        self.assertEqual(two['rank'], int(np.sum(np.array(self.one['ranked']) > two['final'])) + 1)
        self.assertAlmostEqual(two['final'], 1.125 ** two['wins'] * 0.9 ** (100 - two['wins']), delta=1e-3 * two['final'])
        t = two['trials']
        self.assertEqual(len(t['ranks']), 200)
        self.assertTrue(all(1 <= r <= 1001 for r in t['ranks']))
        self.assertGreater(t['top_half'], 0.95, 'Kelly nearly always beats the typical all-in player.')
        self.assertAlmostEqual(two['exact_beats'], 0.903, delta=0.002)
        self.assertGreater(two['same_luck']['kelly_beats_all_in'], 0.9, 'on the same coins, Kelly beats all in.')

    def test_everyone_sensible(self):
        by = {s['key']: s for s in self.three['schemes']}
        self.assertEqual(by['all_in']['ranked'], self.one['ranked'], 'the same table as stage 1.')
        kelly, pooled = by['kelly']['summary'], by['pooled']['summary']
        self.assertAlmostEqual(math.log(kelly['median']) / 100, self.three['kelly_growth'], delta=0.002)
        self.assertEqual(kelly['ruined'], 0.0)
        self.assertEqual(pooled['above_start'], 1.0, 'pooled, everyone ends up ahead.')
        self.assertAlmostEqual(pooled['median'], pooled['mean'], delta=1e-6 * pooled['mean'], msg='and equal.')
        self.assertGreater(pooled['median'], 50 * kelly['median'])
        self.assertAlmostEqual(pooled['top1_share'], 0.01)
        pools = self.three['pools']
        self.assertEqual([p['size'] for p in pools], [1, 2, 5, 10, 50, 100, 1000])
        self.assertTrue(all(a['median'] < b['median'] for a, b in zip(pools, pools[1:])), 'bigger pools grow faster.')

    def test_pooled_fraction(self):
        self.assertAlmostEqual(Stages.pooled_fraction(1), 0.25, places=6)
        self.assertGreater(Stages.pooled_fraction(2), 0.45)
        self.assertEqual(Stages.pooled_fraction(1000), 1.0)

    def test_fraction_curve(self):
        curve = Stages.fraction_curve()
        self.assertEqual(len(curve), 101)
        best = max(curve, key=lambda row: row['growth'])
        self.assertEqual(best['fraction'], 0.25)
        self.assertEqual(curve[0]['median'], 1.0)
        self.assertLess(curve[-1]['growth'], 0.0)

    def test_fixed_seeds_repeat(self):
        self.assertEqual(Stages.all_in(), self.one)

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
        pk = by['pooled_kelly']['summary']
        self.assertEqual(by['pooled_kelly']['stake'], 0.25)
        self.assertEqual(pk['above_start'], 1.0, 'pooled at the Kelly stake, everyone ends ahead too.')
        self.assertAlmostEqual(pk['median'], 1.0125 ** 100, delta=0.1, msg='close to the Kelly stake\'s average.')
        self.assertAlmostEqual(pk['median'], kelly['mean'], delta=0.15 * kelly['mean'], msg='everyone gets the average.')
        self.assertLess(pk['median'], pooled['median'] / 10, 'a pool that stakes a quarter leaves most growth unused.')
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


class TestMixedTable(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mix = Stages.mixed_table()
        cls.by = {g['key']: g for g in cls.mix['groups']}

    def test_a_quarter_of_each(self):
        self.assertEqual([g['n'] for g in self.mix['groups']], [250] * 4)
        self.assertEqual(sorted(self.mix['kind_ranked']), sorted([0, 1, 2, 3] * 250))
        self.assertEqual(self.mix['ranked'], sorted(self.mix['ranked'], reverse=True))
        self.assertAlmostEqual(sum(g['wealth_share'] for g in self.mix['groups']), 1.0, places=3)
        self.assertEqual(sum(g['top100'] for g in self.mix['groups']), 100)

    def test_pools_share_only_among_their_own(self):
        for key in ('pooled', 'pooled_kelly'):
            s = self.by[key]['summary']
            self.assertEqual(s['min'], s['max'], '%s: every member holds the same.' % key)
        self.assertGreater(self.by['pooled']['summary']['median'], self.by['pooled_kelly']['summary']['median'])

    def test_who_lands_where(self):
        allin, pooled = self.by['all_in'], self.by['pooled']
        self.assertEqual(allin['best_rank'], 1, 'the luckiest lone all-in player tops the table.')
        self.assertGreater(pooled['wealth_share'], 0.5, 'but the all-in pool holds most of the wealth.')
        self.assertEqual(allin['worst_rank'], 1000, 'and lone all-in players also fill the bottom.')
        self.assertGreater(self.by['kelly']['median_rank'], self.by['pooled_kelly']['median_rank'])


class TestLevers(TestCase):
    def test_the_two_levers(self):
        lv = Stages.levers()
        self.assertAlmostEqual(lv['mu'], 0.05)
        self.assertAlmostEqual(lv['var'], 0.2025)
        self.assertAlmostEqual(lv['kelly_approx'], 0.05 / 0.2025, places=4)
        self.assertAlmostEqual(lv['kelly_exact'], 0.25, msg='the approximation is close to the exact stake.')
        self.assertGreater(lv['pool_kelly_approx'], 100.0, 'a pool of 1000 would take far more than all in.')
        self.assertEqual(lv['pool_kelly_exact'], 1.0, 'so its stake is capped at everything.')
        self.assertLess(lv['growth_all_in'], 0.0)
        self.assertLess(lv['growth_kelly'], lv['growth_ensemble'])

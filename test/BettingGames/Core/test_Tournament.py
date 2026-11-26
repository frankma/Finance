from unittest import TestCase

import numpy as np

from src.BettingGames.Core.Bet import Bet
from src.BettingGames.Core.Strategy import AllIn, FixedFraction, Kelly, Strategy
from src.BettingGames.Core.Tournament import play

__author__ = 'frank.ma'


class TestTournament(TestCase):
    def setUp(self):
        self.bet = Bet()
        self.wins = self.bet.draw(np.random.default_rng(3), (100, 400))

    def test_all_in_wealth_depends_only_on_the_number_of_wins(self):
        result = play(self.bet, AllIn(), self.wins, keep_paths=True)
        k = self.wins.sum(axis=0)
        np.testing.assert_allclose(result.final, 1.5 ** k * 0.6 ** (100 - k))
        np.testing.assert_allclose(result.paths[0], 1.0)
        np.testing.assert_allclose(result.paths[-1], result.final)
        self.assertTrue(np.all(result.stakes == 1.0))

    def test_kelly_stakes_a_quarter(self):
        result = play(self.bet, Kelly(), self.wins, keep_paths=True)
        self.assertTrue(np.allclose(result.stakes, 0.25))
        k = self.wins.sum(axis=0)
        np.testing.assert_allclose(result.final, 1.125 ** k * 0.9 ** (100 - k))
        self.assertTrue(np.allclose(play(self.bet, Kelly(0.5), self.wins, keep_paths=True).stakes, 0.125))

    def test_pooling_shares_every_round(self):
        result = play(self.bet, AllIn(), self.wins, pool_size=100, keep_paths=True)
        groups = result.paths.reshape(101, 4, 100)
        self.assertTrue(np.allclose(groups, groups[:, :, :1]), 'everyone in a pool holds the same.')
        manual = np.ones(4)
        for t in range(100):
            manual = manual * np.where(self.wins[t], 1.5, 0.6).reshape(4, 100).mean(axis=1)
        np.testing.assert_allclose(result.final.reshape(4, 100)[:, 0], manual)
        with self.assertRaises(ValueError):
            play(self.bet, AllIn(), self.wins, pool_size=300)

    def test_monte_carlo_matches_the_closed_form(self):
        wins = self.bet.draw(np.random.default_rng(5), (100, 20000))
        final = play(self.bet, AllIn(), wins).final
        self.assertAlmostEqual(np.mean(final > 1.0), self.bet.prob_above(1.0, 100), delta=0.01)
        logs = np.log(play(self.bet, FixedFraction(0.25), wins).final) / 100
        self.assertAlmostEqual(logs.mean(), self.bet.growth_rate(0.25), delta=0.0005)

    def test_a_strategy_must_stake_within_its_wealth(self):
        class TooMuch(Strategy):
            def fraction(self, view):
                return np.full(view.wealth.shape, 1.5)
        with self.assertRaises(ValueError):
            play(self.bet, TooMuch(), self.wins)

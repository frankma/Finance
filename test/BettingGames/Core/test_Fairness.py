from unittest import TestCase

import numpy as np

from src.BettingGames.Core.Bet import Bet
from src.BettingGames.Core.Fairness import assert_fair_play
from src.BettingGames.Core.Strategy import AllIn, Kelly, Strategy
from src.BettingGames.Core.Tournament import play

__author__ = 'frank.ma'


class AfterALossStakeLess(Strategy):
    # fair: reacts to the player's own past only
    def fraction(self, view):
        if view.round == 0:
            return np.full(view.wealth.shape, 0.25)
        return np.where(view.past_wins[-1], 0.25, 0.1)


class FollowTheTable(Strategy):
    # unfair: stakes more when the other players have been winning
    def fraction(self, view):
        if view.round == 0:
            return np.full(view.wealth.shape, 0.25)
        return np.full(view.wealth.shape, float(view.past_wins.mean()))


class TestFairness(TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.bet = Bet()
        self.a = self.bet.draw(rng, (30, 6))
        self.b = self.a.copy()
        self.b[15:] = ~self.b[15:]  # the future differs for everyone
        self.b[:, 0] = ~self.b[:, 0]  # and one player's whole history differs

    def test_fair_strategies(self):
        for make in (AllIn, Kelly, AfterALossStakeLess):
            assert_fair_play(self.bet, make, self.a, self.b)

    def test_a_strategy_that_reads_the_other_players_fails(self):
        with self.assertRaises(AssertionError):
            assert_fair_play(self.bet, FollowTheTable, self.a, self.b)

    def test_the_view_cannot_see_ahead_or_be_changed(self):
        seen, test = [], self

        class Spy(Strategy):
            def fraction(self, view):
                seen.append(view.past_wins.shape[0])
                with test.assertRaises(ValueError):
                    view.past_wins[...] = True
                with test.assertRaises(ValueError):
                    view.wealth[...] = 0.0
                return np.zeros(view.wealth.shape)
        play(self.bet, Spy(), self.a)
        self.assertEqual(seen, list(range(30)))

import random
from unittest import TestCase

import numpy as np

from src.CardGames.BlackJack.BasicStrategy import BasicStrategy, MimicDealerStrategy
from src.CardGames.BlackJack.Counting import FlatBetSizer, KellyBetSizer, kelly_fraction
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.BlackJack.Table import Table, ev_by_true_count, simulate_bankroll, simulate_ev
from src.CardGames.Core.Card import Rank, standard_deck

__author__ = 'frank.ma'


class TestCounting(TestCase):
    def test_kelly_fraction(self):
        self.assertAlmostEqual(kelly_fraction(0.01, 1.3), 0.01 / 1.3)
        self.assertEqual(kelly_fraction(-0.01, 1.3), 0.0)
        with self.assertRaises(ValueError):
            kelly_fraction(0.01, 0.0)

    def test_kelly_bet_sizer(self):
        sizer = KellyBetSizer(min_bet=10.0, max_bet=200.0)
        self.assertAlmostEqual(sizer.edge(3.0), 0.01)
        self.assertEqual(sizer.bet(10000.0, 0.0), 10.0, 'no edge, table minimum.')
        self.assertAlmostEqual(sizer.bet(10000.0, 3.0), 10000.0 * 0.01 / 1.3)
        self.assertEqual(sizer.bet(10000.0, 20.0), 200.0, 'capped at the table maximum.')
        half = KellyBetSizer(min_bet=10.0, max_bet=200.0, kelly_multiplier=0.5)
        self.assertAlmostEqual(half.bet(10000.0, 3.0), 0.5 * sizer.bet(10000.0, 3.0))
        with self.assertRaises(ValueError):
            KellyBetSizer(min_bet=20.0, max_bet=10.0)


class TestTable(TestCase):
    def test_reshuffles_at_penetration(self):
        rules = Rules(n_decks=1, penetration=0.5)
        table = Table(rules, random.Random(0))
        strategy = BasicStrategy(rules)
        for _ in range(200):
            table.play_round(strategy)
            self.assertGreater(table.shoe.cards_remaining, 0)
            self.assertEqual(table.counter.n_seen, len(table.shoe.face_up))

    def test_basic_strategy_ev(self):
        # six decks, S17, DAS, 3:2: basic strategy gives up about 0.4%, mimicking the dealer about 5.5%
        rules = Rules()
        basic = simulate_ev(rules, BasicStrategy(rules), 40000, random.Random(11))
        mimic = simulate_ev(rules, MimicDealerStrategy(), 40000, random.Random(12))
        self.assertLess(abs(basic.mean + 0.004), 4.0 * basic.stderr, str(basic))
        self.assertLess(abs(mimic.mean + 0.055), 4.0 * mimic.stderr, str(mimic))
        self.assertAlmostEqual(basic.std, 1.15, delta=0.05)

    def test_composition_drives_ev(self):
        # removing the small cards favours the player, removing the tens favours the dealer
        rules = Rules()
        strategy = BasicStrategy(rules)
        decks = standard_deck() * 6
        rich = [card for card in decks if card.rank not in (Rank.FIVE, Rank.SIX)]
        poor = [card for card in decks if card.rank not in (Rank.TEN, Rank.JACK)]
        ev_rich = simulate_ev(rules, strategy, 10000, random.Random(13), rich)
        ev_poor = simulate_ev(rules, strategy, 10000, random.Random(14), poor)
        self.assertGreater(ev_rich.mean - ev_poor.mean, 4.0 * np.hypot(ev_rich.stderr, ev_poor.stderr))

    def test_ev_by_true_count(self):
        rules = Rules()
        buckets = ev_by_true_count(rules, BasicStrategy(rules), 20000, random.Random(15), max_count=3)
        self.assertEqual(sum(result.n for result in buckets.values()), 20000)
        self.assertTrue(set(buckets).issubset(range(-3, 4)))
        self.assertGreater(buckets[0].n, buckets[3].n, 'most rounds are dealt near a zero count.')

    def test_simulate_bankroll(self):
        rules = Rules()
        strategy = BasicStrategy(rules)
        flat = simulate_bankroll(rules, strategy, FlatBetSizer(10.0), 100.0, 300, 20, random.Random(16))
        self.assertEqual(flat.paths.shape, (20, 301))
        self.assertTrue(np.all(flat.paths[:, 0] == 100.0))
        self.assertGreater(flat.ruin_probability, 0.0, 'ten bets of bankroll should not survive 300 rounds.')
        for path, ruined in zip(flat.paths, flat.ruined):
            if ruined:
                self.assertEqual(path[-1], path[-2], 'a ruined path stays flat.')
        deep = simulate_bankroll(rules, strategy, KellyBetSizer(1.0, 50.0), 10000.0, 200, 5, random.Random(17))
        self.assertEqual(deep.ruin_probability, 0.0)

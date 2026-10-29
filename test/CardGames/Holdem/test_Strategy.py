import random
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Core.Player import RandomStrategy
from src.CardGames.Holdem.Round import Action, Move, Observation, Street
from src.CardGames.Holdem.Rules import Rules
from src.CardGames.Holdem.Strategy import CallingStation, MonteCarloStrategy
from src.CardGames.Holdem.Table import Table, simulate_win_rate

__author__ = 'frank.ma'


def observation(hole: str, board: str = '', pot: int = 30, to_call: int = 0, current_bet: int = 0, bet: int = 0,
                stack: int = 1000, n_opponents: int = 1, can_raise: bool = True) -> Observation:
    rules = Rules(5, 10)
    min_to = max(current_bet + 10, 10) if can_raise else None
    max_to = bet + stack if can_raise else None
    return Observation(0, tuple(Card.parse_many(hole)), tuple(Card.parse_many(board)), Street.FLOP, pot, to_call,
                       bet, current_bet, stack, min_to, max_to, n_opponents, 10, rules)


class TestMonteCarloStrategy(TestCase):
    def strategy(self, seed: int = 1) -> MonteCarloStrategy:
        return MonteCarloStrategy(n_sims=400, rng=random.Random(seed))

    def test_bets_strong_hand_for_value(self):
        advice = self.strategy().advise(observation('AS AH', 'AD 7C 2S'))
        self.assertEqual(advice.move.action, Action.RAISE)
        self.assertEqual(advice.move.to % 5, 0, 'sized in whole small blinds.')
        self.assertGreater(advice.equity, 0.9)

    def test_checks_weak_hand(self):
        self.assertEqual(self.strategy().advise(observation('7S 2H', 'AD KC QS')).move, Move(Action.CHECK))

    def test_folds_against_bad_pot_odds(self):
        advice = self.strategy().advise(observation('2C 7D', 'AS KH QC JS', pot=40, to_call=100, current_bet=100))
        self.assertEqual(advice.move, Move(Action.FOLD))
        self.assertLess(advice.call_ev, 0.0)

    def test_calls_a_draw_with_the_price(self):
        # nut flush draw with two overcards for a small bet into a big pot
        advice = self.strategy().advise(observation('AS KS', '2S 7S 9H', pot=200, to_call=20, current_bet=20,
                                                    can_raise=False))
        self.assertEqual(advice.move, Move(Action.CALL, 20))
        self.assertGreater(advice.call_ev, 0.0)
        self.assertIn('pot odds', advice.reason)

    def test_all_in_when_most_of_stack(self):
        advice = self.strategy().advise(observation('AS AH', 'AD 7C 2S', pot=300, stack=200))
        self.assertEqual(advice.move, Move(Action.RAISE, 200))


class TestTable(TestCase):
    def test_play_until_one_left(self):
        table = Table([100, 100, 100, 100], Rules(1, 2), random.Random(7))
        strategies = [RandomStrategy(random.Random(seat)) for seat in range(4)]
        buttons = []
        while table.n_live > 1 and table.n_hands < 1000:
            buttons.append(table.button)
            table.play_hand(strategies)
        self.assertEqual(sum(table.stacks), 400)
        self.assertEqual(table.n_live, 1)
        self.assertTrue(all(table.stacks[b] >= 0 for b in buttons))
        with self.assertRaises(ValueError):
            table.play_hand(strategies)

    def test_win_rate(self):
        # equity-based play beats a seat that calls everything
        rates = simulate_win_rate([MonteCarloStrategy(100, rng=random.Random(8)), CallingStation()], 400, 200,
                                  Rules(1, 2), random.Random(9))
        self.assertAlmostEqual(rates[0].mean + rates[1].mean, 0.0)
        self.assertGreater(rates[0].mean, 0.0)

import random
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Holdem.Equity import simulate_equity
from src.CardGames.Holdem.Range import (FLUSH_DRAW, MADE, NOTHING, TOP_PAIR, LOW_PAIR, WEIGHT_FLOOR, betting_weights,
                                        board_strength, preflop_percentile)
from src.CardGames.Holdem.Round import Action, Move, Round
from src.CardGames.Holdem.Rules import OMAHA, Rules
from src.CardGames.Core.Shoe import Shoe

__author__ = 'frank.ma'

P = Card.parse_many
RULES = Rules(1, 2)


def spot(you: str, them: str, board: str, *moves):
    # heads-up with you on the button: these cards, then these (seat, action, to) moves in order
    y, t, b = P(you), P(them), P(board)
    top = [t[0], y[0], t[1], y[1]]
    shoe = Shoe(1, rng=random.Random(0))
    rest = [card for card in shoe.cards if card not in top + b]
    order = top + [rest[0]] + b[:3] + [rest[1]] + b[3:4] + [rest[2]] + b[4:5]
    shoe.cards = rest[3:] + list(reversed(order))
    game = Round([200, 200], RULES, 0, shoe=shoe)
    for seat, action, to in moves:
        game.apply(seat, Move(action, to))
    return game


class TestPreflop(TestCase):
    def test_order(self):
        self.assertGreater(preflop_percentile(P('AS AH')), 0.99)
        self.assertGreater(preflop_percentile(P('KS QS')), preflop_percentile(P('KS QD')), 'suited is better.')
        self.assertLess(preflop_percentile(P('7S 2H')), 0.05)


class TestBoardStrength(TestCase):
    def test_categories(self):
        for hole, board, expected in (
                ('7S 7D', '7H KC 2D', MADE),  # a set
                ('KS 9D', 'KH 9C 2D', MADE),  # two pair
                ('KS 9D', 'KH 7C 2D', TOP_PAIR),
                ('AS AD', 'KH 7C 2D', TOP_PAIR),  # an overpair
                ('9S 4D', 'KH 9C 2D', LOW_PAIR),
                ('AH 5H', 'KH 9H 2C', FLUSH_DRAW),  # two in the hand, two on the board
                ('AH 5C', 'KH 9H 2H', FLUSH_DRAW),  # one in the hand, three on the board
                ('QD 8C', 'AS 7H 2C', NOTHING),
                ('QD 8C', '7S 7H 2C', NOTHING)):  # the board's own pair is not the hand's
            self.assertEqual(board_strength(P(hole), P(board), RULES), expected, hole + ' on ' + board)

    def test_no_draw_once_the_board_is_complete(self):
        self.assertEqual(board_strength(P('AH 5H'), P('KH 9H 2C 3D 8S'), RULES), NOTHING)


class TestBettingWeights(TestCase):
    def test_no_public_action_means_no_weights(self):
        game = spot('AS KS', 'QC 8D', '2S 7S 9H 3D 4C')
        self.assertIsNone(betting_weights(game.observation(0)))

    def test_a_preflop_raise_favours_strong_hands(self):
        game = spot('7H 7C', 'AS KS', '2S 7S 9H 3D 4C', (0, Action.CALL, 2), (1, Action.RAISE, 8))
        weight = betting_weights(game.observation(0))[0]
        self.assertEqual(weight(P('AD KD')), 1.0)
        self.assertLess(weight(P('7D 2H')), 0.5)
        self.assertGreaterEqual(weight(P('7D 2H')), WEIGHT_FLOOR, 'no hand is ever ruled out.')

    def test_a_bet_favours_what_connects_with_the_board(self):
        game = spot('AH 5H', 'QC 8D', 'KH 9H 2C 4S JD', (0, Action.CALL, 2), (1, Action.CHECK, 0), (1, Action.RAISE, 4))
        weight = betting_weights(game.observation(0))[0]
        self.assertGreater(weight(P('KD 9C')), weight(P('KD 3C')))
        self.assertGreater(weight(P('KD 3C')), weight(P('6D 3C')))

    def test_only_public_moves_are_read(self):
        # the same public play with different hidden cards reads the same way
        moves = ((0, Action.CALL, 2), (1, Action.RAISE, 8))
        one = betting_weights(spot('7H 7C', 'AS KS', '2S 7S 9H 3D 4C', *moves).observation(0))[0]
        two = betting_weights(spot('7H 7C', 'QD 3C', '2S 7S 9H 3D 4C', *moves).observation(0))[0]
        for hand in ('AD KD', '7D 2H', '9C 9D'):
            self.assertEqual(one(P(hand)), two(P(hand)))

    def test_lowers_equity_against_a_big_bet(self):
        game = spot('AC 5D', 'KS TD', 'KH 7S 2C 9D 3H', (0, Action.CALL, 2), (1, Action.CHECK, 0), (1, Action.RAISE, 8))
        o = game.observation(0)
        plain = simulate_equity(list(o.hole), list(o.board), 1, 3000, rng=random.Random(1)).equity.mean
        read = simulate_equity(list(o.hole), list(o.board), 1, 3000, rng=random.Random(1),
                               weights=betting_weights(o)).equity.mean
        self.assertLess(read, plain - 0.05)

    def test_omaha_is_left_alone(self):
        game = Round([200, 200], OMAHA, 0, rng=random.Random(2))
        seat = game.current_seat()
        game.apply(seat, Move(Action.RAISE, game.raise_range(seat)[0]))
        self.assertIsNone(betting_weights(game.observation(game.current_seat())))

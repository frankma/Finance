import random
from typing import List
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Core.Game import play_out
from src.CardGames.Core.Player import RandomStrategy
from src.CardGames.Core.Shoe import Shoe
from src.CardGames.Holdem.Round import Action, Move, Round, Street
from src.CardGames.Holdem.Rules import Rules

__author__ = 'frank.ma'

FOLD, CHECK = Move(Action.FOLD), Move(Action.CHECK)


def call(to: int) -> Move:
    return Move(Action.CALL, to)


def raise_to(to: int) -> Move:
    return Move(Action.RAISE, to)


def stacked(holes: List[str], board: str, button: int = 0) -> Shoe:
    """
    a shoe dealing the given hole cards (by seat) and board, with burns taken from the cards nobody holds
    """
    hole_cards = [Card.parse_many(hole) for hole in holes]
    board_cards = Card.parse_many(board)
    shoe = Shoe(1)
    rest = [card for card in shoe.composition
            if card not in board_cards and all(card not in hole for hole in hole_cards)]
    n = len(holes)
    order = []
    for k in range(len(hole_cards[0])):
        order += [hole_cards[(button + 1 + i) % n][k] for i in range(n)]
    burns = [rest.pop(), rest.pop(), rest.pop()]
    order += [burns[0]] + board_cards[:3] + [burns[1], board_cards[3], burns[2], board_cards[4]]
    shoe.cards = rest + list(reversed(order))
    return shoe


class TestRound(TestCase):
    def test_deal_and_blinds(self):
        game = Round([100, 100, 100], Rules(1, 2), button=0, shoe=stacked(['AS AH', 'KS KH', 'QS QH'], '2C 3D 4H 5S 7C'))
        self.assertEqual([game.hole(i) for i in range(3)], [Card.parse_many(h) for h in ('AS AH', 'KS KH', 'QS QH')])
        self.assertEqual(game.stacks, [100, 99, 98])
        self.assertEqual(game.current_seat(), 0, 'first to act preflop is left of the big blind.')
        self.assertEqual(game.legal_moves(0), [FOLD, call(2), raise_to(4), raise_to(100)])
        self.assertEqual(game.legal_moves(1), [], 'only the seat to act has moves.')

    def test_heads_up_button_posts_small_blind(self):
        game = Round([100, 100], Rules(1, 2), button=1, rng=random.Random(0))
        self.assertEqual(game.stacks, [98, 99])
        self.assertEqual(game.current_seat(), 1, 'the button acts first preflop heads-up.')
        game.apply(1, call(2))
        self.assertEqual(game.legal_moves(0), [CHECK, raise_to(4), raise_to(100)], 'the big blind has the option.')
        game.apply(0, CHECK)
        self.assertEqual(game.street, Street.FLOP)
        self.assertEqual(game.current_seat(), 0, 'after the flop the big blind acts first heads-up.')

    def test_showdown_and_fold(self):
        shoe = stacked(['AS AH', 'KS KH', 'QS QH'], '2C 3D 9H TS 7C')
        game = Round([100, 100, 100], Rules(1, 2), shoe=shoe)
        game.apply(0, call(2))
        game.apply(1, call(2))
        game.apply(2, CHECK)
        for _ in range(3):
            for seat in (1, 2, 0):
                game.apply(seat, CHECK)
        self.assertTrue(game.is_terminal())
        self.assertEqual(game.payoffs(), [4.0, -2.0, -2.0])
        self.assertEqual(game.board, Card.parse_many('2C 3D 9H TS 7C'))
        self.assertEqual(len(shoe.face_up), 11, 'the board and every contender hole card are shown.')

        game = Round([100, 100, 100], Rules(1, 2), rng=random.Random(1))
        game.apply(0, raise_to(10))
        game.apply(1, FOLD)
        game.apply(2, FOLD)
        self.assertEqual(game.payoffs(), [3.0, -1.0, -2.0])

    def test_min_raise(self):
        game = Round([1000, 1000, 1000], Rules(5, 10), rng=random.Random(2))
        game.apply(0, raise_to(30))  # a raise of 20
        self.assertEqual(game.raise_range(1), (50, 1000), 'the next raise must be at least 20 more.')
        self.assertFalse(game.is_legal(1, raise_to(49)))
        self.assertTrue(game.is_legal(1, raise_to(337)))
        with self.assertRaises(ValueError):
            game.apply(1, raise_to(40))

    def test_short_all_in_does_not_reopen(self):
        # seat 2 is the big blind with 25; seat 0 raises to 20, seat 1 calls, seat 2 moves in for 5 more
        game = Round([1000, 1000, 25], Rules(5, 10), rng=random.Random(3))
        game.apply(0, raise_to(20))
        game.apply(1, call(20))
        game.apply(2, raise_to(25))
        self.assertEqual(game.current_seat(), 0)
        self.assertEqual(game.legal_moves(0), [FOLD, call(25)], 'a short all-in does not reopen the raising.')
        game.apply(0, call(25))
        self.assertEqual(game.legal_moves(1), [FOLD, call(25)])

    def test_side_pots(self):
        # seat 2 is all in for 50 with the best hand, seats 0 and 1 play on for a side pot
        shoe = stacked(['KS KH', 'QS QH', 'AS AH'], '2C 3D 9H TS 7C')
        game = Round([200, 200, 50], Rules(5, 10), shoe=shoe)
        game.apply(0, raise_to(50))
        game.apply(1, call(50))
        game.apply(2, call(50))
        self.assertEqual(game.street, Street.FLOP)
        game.apply(1, raise_to(150))
        game.apply(0, call(150))
        self.assertTrue(game.is_terminal(), 'everyone is all in, so the board runs out.')
        self.assertEqual([(pot.amount, pot.eligible, pot.winners) for pot in game.pots],
                         [(150, (0, 1, 2), (2,)), (300, (0, 1), (0,))])
        self.assertEqual(game.payoffs(), [100.0, -200.0, 100.0])

    def test_split_pot_odd_chip(self):
        # both play the board's straight; the odd chip goes to the first winner left of the button
        shoe = stacked(['2S 3H', '2H 3S', '4C 4D'], '9C TD JH QS KC', button=0)
        game = Round([100, 100, 100], Rules(1, 2), shoe=shoe)
        game.apply(0, call(2))
        game.apply(1, FOLD)  # leaves a pot of 5
        game.apply(2, CHECK)
        self.assertEqual(game.street, Street.FLOP)
        for _ in range(3):
            game.apply(2, CHECK)
            game.apply(0, CHECK)
        self.assertEqual(game.pots[0].winners, (2, 0))
        self.assertEqual(game.payoffs(), [0.0, -1.0, 1.0])

    def test_observation_hides_other_hands(self):
        game = Round([100, 100, 100], Rules(1, 2), shoe=stacked(['AS AH', 'KS KH', 'QS QH'], '2C 3D 4H 5S 7C'))
        observation = game.observation(0)
        self.assertEqual(observation.hole, tuple(Card.parse_many('AS AH')))
        self.assertEqual((observation.to_call, observation.pot, observation.n_opponents), (2, 3, 2))
        self.assertEqual((observation.min_raise_to, observation.max_raise_to), (4, 100))
        self.assertIsNone(game.observation(1).min_raise_to, 'raise sizes only for the seat to act.')

    def test_validation(self):
        with self.assertRaises(ValueError):
            Round([100, 0], Rules())
        with self.assertRaises(ValueError):
            Round([100] * 12, Rules(n_hole=4))
        with self.assertRaises(ValueError):
            Rules(n_hole=6)
        game = Round([100, 0, 100], Rules(1, 2), button=1, rng=random.Random(4))
        self.assertEqual(game.hole(1), [], 'a seat without chips sits out.')
        self.assertEqual(game.button, 2)

    def test_chips_are_conserved(self):
        rng = random.Random(5)
        for rules in (Rules(1, 2), Rules(5, 10, n_hole=4, omaha=True), Rules(1, 2, n_hole=5)):
            for _ in range(200):
                n = rng.randint(2, 6)
                stacks = [rng.choice([0, 3, 20, 100, 400]) for _ in range(n)]
                stacks[0] = stacks[1] = 50
                game = Round(stacks, rules, rng.randrange(n), rng)
                payoffs = play_out(game, [RandomStrategy(rng) for _ in range(n)])
                self.assertEqual(sum(payoffs), 0.0)
                self.assertTrue(all(stack >= 0 for stack in game.stacks))
                self.assertEqual(sum(pot.amount for pot in game.pots), sum(seat.total for seat in game.seats))

import json
import random
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Core.Shoe import Shoe
from src.CardGames.Play.HoldemSession import HoldemSession

__author__ = 'frank.ma'


def stack(session: HoldemSession, order: str) -> None:
    # the next hand deals these cards first, in order: hole cards round the table from the small blind, then
    # burn, flop, burn, turn, burn, river
    cards = Card.parse_many(order)

    def shoe():
        made = Shoe(1, rng=random.Random(0))
        rest = [card for card in made.cards if card not in cards]
        made.cards = rest + list(reversed(cards))
        return made
    session._shoe = shoe


def play_coached(session: HoldemSession, n_hands: int) -> None:
    for _ in range(n_hands):
        if session.view()['can_rebuy']:
            session.act({'type': 'rebuy'})
        session.act({'type': 'deal'})
        while session.phase == 'play':
            advice = session.advice()
            session.act({'type': 'move', 'move': advice['move'], 'to': advice['to']})


class TestHoldemSession(TestCase):
    def test_options(self):
        view = HoldemSession({'seats': '6'}, random.Random(0)).view()
        self.assertEqual((view['phase'], view['n_seats'], view['legal'], view['board']), ('deal', 6, [], []))
        self.assertEqual([seat['name'] for seat in view['seats']][:2], ['You', 'Tight 2'])
        for options in ({'seats': 3}, {'stack': 10}):
            with self.assertRaises(ValueError):
                HoldemSession(options)
        session = HoldemSession({}, random.Random(0))
        for action in ({'type': 'shuffle'}, {'type': 'move', 'move': 'check'}, {'type': 'rebuy'}):
            with self.assertRaises(ValueError):
                session.act(action)
        self.assertEqual(session.advice()['kind'], 'wait')

    def test_your_decision(self):
        # heads-up you have the button, post the small blind and act first before the flop
        session = HoldemSession({'seats': 2}, random.Random(1))
        stack(session, 'KD AS KC AH 2C 7D 8S 9H 3C TD 4C JS')
        session.act({'type': 'deal'})
        view = session.view()
        self.assertTrue(view['your_turn'])
        self.assertEqual(view['seats'][0]['cards'], ['AS', 'AH'])
        self.assertIsNone(view['seats'][1]['cards'], 'the other seat is face down.')
        self.assertEqual((view['legal'], view['to_call'], view['call_to']), (['fold', 'call', 'raise'], 1, 2))
        self.assertEqual(view['raise'], {'min': 4, 'max': 200, 'pot': 6})
        with self.assertRaises(ValueError):
            session.act({'type': 'move', 'move': 'raise', 'to': 3})
        with self.assertRaises(ValueError):
            session.act({'type': 'move', 'move': 'raise'})
        with self.assertRaises(ValueError):
            session.act({'type': 'move', 'move': 'check'})
        with self.assertRaises(ValueError):
            session.act({'type': 'deal'})
        advice = session.advice()
        self.assertEqual(advice['move'], 'raise', 'aces heads-up are a raise.')
        self.assertGreater(advice['equity'], 0.8)
        self.assertEqual(session.advice(), advice, 'one answer per decision.')
        session.act({'type': 'move', 'move': 'fold'})
        view = session.view()
        self.assertEqual((view['phase'], view['payoff'], view['seats'][0]['stack']), ('done', -1, 199))
        self.assertIsNone(view['seats'][1]['cards'], 'a hand won without a showdown is never shown.')
        self.assertEqual(view['log'][-1], 'Tight 2 wins 3 with everyone else folding.')
        self.assertEqual((view['stats']['decisions'], view['stats']['followed']), (1, 0))

    def test_showdown(self):
        session = HoldemSession({'seats': 2}, random.Random(2))
        stack(session, 'KD AS KC AH 2C 7D 8S 9H 3C TD 4C JS')
        session.act({'type': 'deal'})
        session.act({'type': 'move', 'move': 'call'})
        while session.phase == 'play':
            session.act({'type': 'move', 'move': 'check' if 'check' in session.view()['legal'] else 'call'})
        view = session.view()
        self.assertEqual(view['board'], ['7D', '8S', '9H', 'TD', 'JS'])
        self.assertEqual(view['seats'][1]['cards'], ['KD', 'KC'], 'hands are turned over at showdown.')
        self.assertEqual(view['seats'][0]['hand'], 'Straight, jack high')
        self.assertEqual([seat['bet'] for seat in view['seats']], [0, 0])
        self.assertEqual(view['pots'][0]['winners'], ['Tight 2', 'You'], 'the board plays: a split pot, odd chip first '
                                                                         'to the left of the button.')
        self.assertEqual(view['payoff'], 0)

    def test_rebuy(self):
        session = HoldemSession({'seats': 4}, random.Random(3))
        session.stacks[0], session.stacks[2] = 1, 0
        self.assertTrue(session.view()['can_rebuy'])
        with self.assertRaises(ValueError):
            session.act({'type': 'deal'})
        session.act({'type': 'rebuy'})
        self.assertEqual((session.stacks[0], session.bought[0]), (200, 399))
        session.act({'type': 'deal'})
        self.assertEqual(session.bought[2], 400, 'a busted computer seat buys back in.')
        self.assertIn('Loose 3 buys back in for 200.', session.log)

    def test_coached_session(self):
        session = HoldemSession({'seats': 6}, random.Random(4))
        play_coached(session, 25)
        view = session.view()
        self.assertEqual(view['stats']['hands'], 25)
        self.assertEqual(sum(session.stacks), sum(session.bought), 'chips are only moved, never made.')
        self.assertEqual(view['stats']['net'], session.stacks[0] - session.bought[0])
        self.assertEqual(view['stats']['followed'], view['stats']['decisions'])
        json.dumps(view)


class TestFairPlay(TestCase):
    def test_the_page_never_gets_another_seats_cards(self):
        # two heads-up tables alike in everything you can see, different in the other seat's cards
        views, advice = [], []
        for other in ('2D 7C', 'KD KC'):
            session = HoldemSession({'seats': 2}, random.Random(5))
            a, b = other.split()
            stack(session, '%s AS %s AH 2C 7D 8S 9H' % (a, b))
            session.act({'type': 'deal'})
            views.append(json.dumps(session.view(), sort_keys=True))
            advice.append(session.advice())
            for card in (a, b):
                self.assertNotIn('"%s"' % card, views[-1])
        self.assertEqual(views[0], views[1])
        self.assertEqual(advice[0], advice[1])

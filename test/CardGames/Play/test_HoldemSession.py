import json
import random
import tempfile
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Core.Player import Strategy
from src.CardGames.Core.Shoe import Shoe
from src.CardGames.Holdem.Equity import simulate_equity
from src.CardGames.Holdem.Round import Action, Street
from src.CardGames.Holdem.Strategy import CallingStation
from src.CardGames.Play.HoldemLessons import _heads_up
from src.CardGames.Play.HoldemSession import HoldemSession
from src.CardGames.Play.Server import Sessions

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
        self.assertEqual(session.round.pots[0].winners, (1, 0), 'the board plays: a split pot, odd chip first to the '
                                                                'left of the button.')
        self.assertEqual(view['log'][-1], 'Tight 2 and You split 20 with straight, jack high.')
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

    def test_the_coach_reads_public_bets_and_nothing_else(self):
        # the same public betting with different cards behind it: your view and the advice are the same
        results = []
        for them in ('QC 8D', 'KS KD'):
            session = _heads_up('AH 5H', them, 'KH 9H 2C 4S JD', {Street.FLOP: 0.5})(random.Random(14))
            before = session.advice()
            self.assertFalse(before['read_betting'], 'nobody has acted yet, so the hands are random.')
            session.act({'type': 'move', 'move': 'call'})
            results.append((json.dumps(session.view(), sort_keys=True), session.advice()))
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[0][1], results[1][1])
        self.assertTrue(results[0][1]['read_betting'], 'the opponent has bet.')

    def test_a_bet_lowers_the_coachs_equity(self):
        session = _heads_up('AH 5D', 'QC 8D', 'KH 9S 2C 4S JD', {Street.FLOP: 1.0})(random.Random(15))
        session.act({'type': 'move', 'move': 'call'})
        o = session.round.observation(0)
        plain = simulate_equity(list(o.hole), list(o.board), 1, 2000, rng=random.Random(1)).equity.mean
        self.assertLess(session.advice()['equity'], plain - 0.03)


class Raiser(Strategy):
    # bets the minimum whenever it may
    def decide(self, observation, legal_moves):
        return next((m for m in legal_moves if m.action == Action.RAISE), legal_moves[0])


class TestOmaha(TestCase):
    def test_a_coached_session(self):
        session = HoldemSession({'variant': 'omaha', 'seats': '4'}, random.Random(21))
        play_coached(session, 4)
        view = session.view()
        self.assertEqual((view['variant'], view['n_hole']), ('omaha', 4))
        self.assertEqual(sum(session.stacks), sum(session.bought), 'chips are only moved, never made.')
        self.assertEqual(len(view['seats'][0]['cards']), 4)
        json.dumps(view)
        with self.assertRaises(ValueError):
            HoldemSession({'variant': 'stud'})

    def test_the_two_plus_three_rule_reaches_the_table(self):
        # a board of four hearts and an ace of hearts alone: hold'em would call this a flush, Omaha does not
        session = HoldemSession({'variant': 'omaha', 'seats': 2}, random.Random(22))
        stack(session, '3S AH 4S 2D 5S 7C 6S 9S 8C KH QH JH 8D TH 8S 3C')
        session.bots[1] = CallingStation()
        session.act({'type': 'deal'})
        session.act({'type': 'move', 'move': 'call'})
        self.assertEqual(session.view()['board'], ['KH', 'QH', 'JH'])
        self.assertEqual(session.view()['made'], 'Ace high', 'one heart in hand is no flush draw made good.')
        self.assertIsNotNone(session.advice()['outs'])

    def test_the_coach_does_not_read_bets(self):
        session = HoldemSession({'variant': 'omaha', 'seats': 2}, random.Random(23))
        session.bots[1] = Raiser()
        session.act({'type': 'deal'})
        session.act({'type': 'move', 'move': 'call'})
        self.assertEqual((session.phase, session.round.history[-1][0]), ('play', 1), 'the other seat has just bet.')
        self.assertEqual(session.view()['to_call'] > 0, True)
        self.assertFalse(session.advice()['read_betting'])

    def test_other_hands_stay_hidden(self):
        # two tables alike in everything you can see, different in the other seat's four cards
        views, advice = [], []
        for other in ('2D 7C 3C 9D', 'KD KC QD QS'):
            session = HoldemSession({'variant': 'omaha', 'seats': 2}, random.Random(24))
            a = other.split()
            stack(session, '%s AS %s AH %s KH %s TS' % (a[0], a[1], a[2], a[3]))
            session.act({'type': 'deal'})
            views.append(json.dumps(session.view(), sort_keys=True))
            advice.append(session.advice())
            self.assertIsNone(session.view()['seats'][1]['cards'])
            for card in a:
                self.assertNotIn('"%s"' % card, views[-1])
        self.assertEqual(views[0], views[1])
        self.assertEqual(advice[0], advice[1])

    def test_progress_keeps_the_game(self):
        # the variant is an option of the table, so a restart rebuilds Omaha, not hold'em
        with tempfile.TemporaryDirectory() as folder:
            path = folder + '/progress.json'
            sid, _ = Sessions(path=path).create('holdem', {'variant': 'omaha', 'seats': '2'})
            self.assertEqual(Sessions(path=path).get('holdem', sid).view()['variant'], 'omaha')

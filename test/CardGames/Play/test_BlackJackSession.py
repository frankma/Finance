import json
import random
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Play.BlackJackSession import BlackJackSession

__author__ = 'frank.ma'


def stack(session: BlackJackSession, order: str) -> None:
    # deal these cards next, in order: player, dealer up, player, dealer hole, then any draws
    cards = Card.parse_many(order)
    rest = list(session.table.shoe.cards)
    for card in cards:
        rest.remove(card)
    session.table.shoe.cards = rest + list(reversed(cards))


class TestBlackJackSession(TestCase):
    def test_betting(self):
        session = BlackJackSession({}, random.Random(0))
        view = session.view()
        self.assertEqual((view['phase'], view['legal'], view['dealer'], view['bankroll']), ('bet', [], None, 1000.0))
        advice = session.advice()
        self.assertEqual((advice['kind'], advice['bet']), ('bet', 5), 'no edge at a zero count: the table minimum.')
        for bet in (0, 4, 501):
            with self.assertRaises(ValueError):
                session.act({'type': 'deal', 'bet': bet})
        with self.assertRaises(ValueError):
            session.act({'type': 'fold'})
        with self.assertRaises(ValueError):
            BlackJackSession({'rules': 'x'})

    def test_round(self):
        session = BlackJackSession({}, random.Random(1))
        stack(session, 'TS 6D 9H KC 8S')
        session.act({'type': 'deal', 'bet': 20})
        view = session.view()
        self.assertEqual(view['phase'], 'play')
        self.assertEqual(view['dealer'], {'cards': ['6D'], 'total': None, 'hidden': True})
        self.assertEqual(view['hands'][0]['total'], 19)
        self.assertEqual(session.advice()['move'], 'stand')
        with self.assertRaises(ValueError):
            session.act({'type': 'deal', 'bet': 10})
        session.act({'type': 'move', 'move': 'stand'})
        view = session.view()
        self.assertEqual(view['phase'], 'done')
        self.assertEqual(view['dealer']['cards'], ['6D', 'KC', '8S'], 'the hole card and draws are public once played.')
        self.assertEqual((view['payoff'], view['bankroll'], view['hands'][0]['outcome']), (20.0, 1020.0, 'win'))
        self.assertEqual((view['stats']['followed'], view['stats']['decisions']), (1, 1))

    def test_bankroll_limits_doubles(self):
        session = BlackJackSession({'bankroll': 15}, random.Random(2))
        stack(session, '6S 6D 5H KC')
        session.act({'type': 'deal', 'bet': 10})
        self.assertEqual(session.view()['legal'], ['stand', 'hit'], 'no money left to double or split.')

    def test_coached_session(self):
        session = BlackJackSession({'rules': 'h17'}, random.Random(3))
        payoffs = 0.0
        for _ in range(120):
            session.act({'type': 'deal', 'bet': session.advice()['bet']})
            while session.phase == 'play':
                session.act({'type': 'move', 'move': session.advice()['move']})
            payoffs += session.payoff
        view = session.view()
        self.assertEqual(view['stats']['rounds'], 120)
        self.assertAlmostEqual(view['bankroll'], 1000.0 + payoffs)
        self.assertEqual(view['stats']['followed'], view['stats']['decisions'])
        json.dumps(view)

    def test_reshuffle_at_the_cut_card(self):
        session = BlackJackSession({}, random.Random(4))
        session.table.shoe.draw_many(250)
        session.act({'type': 'deal', 'bet': 10})
        self.assertIn('reshuffled', session.log[0])


class TestFairPlay(TestCase):
    def test_the_page_never_gets_the_hole_card(self):
        # two tables alike in everything shown, different in the hole card and the shoe behind it
        views, advice = [], []
        for hole, seed in (('5C', 5), ('KC', 6)):
            session = BlackJackSession({}, random.Random(seed))
            stack(session, 'TS 9D 6H ' + hole)
            session.table.counter.reset()
            session.act({'type': 'deal', 'bet': 10})
            view = session.view()
            view['shoe'] = None  # the shoes behind the cards differ in order only, which the page never learns
            views.append(json.dumps(view, sort_keys=True))
            advice.append(session.advice())
            self.assertNotIn('"%s"' % hole, views[-1])
        self.assertEqual(views[0], views[1])
        self.assertEqual(advice[0], advice[1])

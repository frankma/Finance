import json
import random
from unittest import TestCase

from src.CardGames.Play.BlackJackSession import BlackJackSession
from src.CardGames.Play.GuanDanSession import GuanDanSession
from src.CardGames.Play.HoldemSession import HoldemSession
from test.CardGames.Play.helpers import play_hand

__author__ = 'frank.ma'


class TestCoachedPlay(TestCase):
    """
    each game played the coach's way for a while, as a player taking every piece of advice would: the table never
    breaks, the money or levels add up, every decision counts as followed, and the view stays plain JSON for the page
    """

    def played(self, session, view):
        self.assertEqual(view['stats']['followed'], view['stats']['decisions'])
        self.assertGreater(view['stats']['decisions'], 0)
        json.dumps(view)

    def test_blackjack(self):
        session, payoffs = BlackJackSession({'rules': 'h17'}, random.Random(3)), 0.0
        for _ in range(120):
            play_hand(session)
            payoffs += session.payoff
        view = session.view()
        self.assertEqual(view['stats']['rounds'], 120)
        self.assertAlmostEqual(view['bankroll'], 1000.0 + payoffs)
        self.played(session, view)

    def test_holdem_and_omaha(self):
        for options, n_hands, seed in (({'seats': 6}, 25, 4), ({'variant': 'omaha', 'seats': '4'}, 4, 21)):
            with self.subTest(**options):
                session = HoldemSession(options, random.Random(seed))
                for _ in range(n_hands):
                    play_hand(session)
                view = session.view()
                self.assertEqual(view['stats']['hands'], n_hands)
                self.assertEqual(sum(session.stacks), sum(session.bought), 'chips are only moved, never made.')
                self.assertEqual(view['stats']['net'], session.stacks[0] - session.bought[0])
                self.assertEqual(len(view['seats'][0]['cards']), view['n_hole'])
                self.played(session, view)

    def test_guandan_match(self):
        session = GuanDanSession({}, random.Random(6))
        while session.phase != 'over':
            play_hand(session)
        view = session.view()
        self.assertIn(view['match']['winner'], ('you', 'them'))
        self.assertIn('A', view['levels'].values(), 'a match is won at A.')
        with self.assertRaises(ValueError):
            session.act({'type': 'deal'})
        self.played(session, view)

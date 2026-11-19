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
        self.assertEqual(BlackJackSession({'rules': 'h17', 'n_decks': 2}).view()['rules'],
                         '2 decks, dealer hits soft 17, late surrender', 'the label follows the decks in play.')

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


class TestCoachedBets(TestCase):
    def test_the_bet_is_sized_for_the_shoe_it_is_dealt_from(self):
        session = BlackJackSession({}, random.Random(7))
        session.table.counter.running_count = 40.0  # a very high count
        self.assertGreater(session.advice()['bet'], 5, 'with the shoe still in play, the count is worth betting on.')
        session.table.shoe.draw_many(250)  # past the cut card, so the next deal comes from a reshuffled shoe
        session.table.counter.running_count = 40.0
        self.assertTrue(session.view()['shoe']['reshuffle_next'])
        advice = session.advice()
        self.assertEqual(advice['bet'], 5, 'a fresh shoe has no count, so the table minimum.')
        self.assertIn('reshuffled', advice['reason'])

    def test_half_chips_are_not_rounded(self):
        session = BlackJackSession({'rules': 'h17'}, random.Random(8))
        stack(session, 'TS 9D 6H KC')
        session.act({'type': 'deal', 'bet': 5})
        session.act({'type': 'move', 'move': 'surrender'})
        self.assertEqual(session.bankroll, 997.5)
        self.assertTrue(session.log[-1].endswith('surrendered: -2.5.'), session.log[-1])
        stack(session, 'AS 6D KH 9C')
        session.act({'type': 'deal', 'bet': 5})
        self.assertEqual(session.bankroll, 1005.0)
        self.assertTrue(session.log[-1].endswith('blackjack: +7.5.'), session.log[-1])
        stack(session, 'TS 7D 8H KC')
        session.act({'type': 'deal', 'bet': 10})
        session.act({'type': 'move', 'move': 'stand'})
        self.assertTrue(session.log[-1].endswith('win: +10.') or session.log[-1].endswith('push: +0.'), session.log[-1])
        session.bankroll = 7.5
        with self.assertRaises(ValueError) as error:
            session.act({'type': 'deal', 'bet': 10})
        self.assertIn('bankroll of 7.5', str(error.exception))


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

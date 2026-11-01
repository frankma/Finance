import json
import random
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.GuanDan.Round import deal
from src.CardGames.Play.GuanDanSession import GuanDanSession

__author__ = 'frank.ma'


def seat(session: GuanDanSession, hands, leader: int = 0) -> None:
    # the next deal hands out these hands, and the first hand is led by this seat
    session._deal_hands = lambda: [list(hand) for hand in hands]
    session.match.rng.randrange = lambda n: leader


def give(hands, codes: str, to: int = 0):
    # move these cards into one seat's hand, swapping them for cards it had, so every hand keeps 27
    hands = [list(hand) for hand in hands]
    for card in Card.parse_many(codes):
        holder = next(s for s in range(4) if s != to and card in hands[s])
        swap = next(c for c in hands[to] if c not in Card.parse_many(codes))
        hands[holder].remove(card)
        hands[holder].append(swap)
        hands[to].remove(swap)
        hands[to].append(card)
    return hands


def follow_coach(session: GuanDanSession) -> None:
    advice = session.advice()
    if advice['kind'] == 'return':
        session.act({'type': 'return', 'card': advice['card']})
    elif advice['move']['kind'] == 'pass':
        session.act({'type': 'pass'})
    else:
        session.act({'type': 'play', 'cards': advice['move']['cards']})
        if session.choices is not None:
            readings = [combo.describe() for combo in session.choices]
            session.act({'type': 'play', 'cards': advice['move']['cards'],
                         'reading': readings.index(advice['move']['desc'])})


class TestGuanDanSession(TestCase):
    def test_first_hand(self):
        session = GuanDanSession({}, random.Random(0))
        view = session.view()
        self.assertEqual((view['phase'], view['level'], view['levels'], view['hand']),
                         ('deal', '2', {'you': '2', 'them': '2'}, []))
        self.assertEqual(session.advice()['kind'], 'wait')
        seat(session, deal(random.Random(1)))
        session.act({'type': 'deal'})
        view = session.view()
        self.assertTrue(view['your_turn'])
        self.assertEqual((len(view['hand']), view['wild'], view['can_pass'], view['trick']), (27, '2H', False, None))
        self.assertEqual([s['count'] for s in view['seats']], [27] * 4)
        self.assertEqual([s['team'] for s in view['seats']], ['you', 'them', 'you', 'them'])
        with self.assertRaises(ValueError):
            session.act({'type': 'pass'})  # the leader must play
        with self.assertRaises(ValueError):
            session.act({'type': 'play', 'cards': []})
        hand = session.round.hands[0]
        missing = next(card for card in deal(random.Random(9))[0] if card not in hand)
        with self.assertRaises(ValueError):
            session.act({'type': 'play', 'cards': [str(missing)]})
        # two plain cards of different ranks, neither the level rank 2 (a wild 2 would make them a pair)
        plain = [card for card in hand if 2 < card.rank.value <= 14]
        low = min(plain, key=lambda card: card.rank.value)
        high = max(plain, key=lambda card: card.rank.value)
        with self.assertRaises(ValueError) as error:
            session.act({'type': 'play', 'cards': [str(low), str(high)]})
        self.assertIn('is not a combination', str(error.exception))
        session.act({'type': 'play', 'cards': [str(low)]})
        view = session.view()
        self.assertEqual(view['seats'][0]['count'], 26)
        self.assertEqual(view['stats']['decisions'], 1)
        json.dumps(view)

    def test_wilds_read_several_ways(self):
        session = GuanDanSession({}, random.Random(2))
        seat(session, give(deal(random.Random(3)), '2H 5S 6S 7S 8S'))
        session.act({'type': 'deal'})
        cards = ['2H', '5S', '6S', '7S', '8S']
        session.act({'type': 'play', 'cards': cards})
        choices = session.view()['choices']
        self.assertEqual(choices, ['straight flush to 9', 'straight flush to 8', 'straight to 9', 'straight to 8'])
        self.assertEqual(session.view()['seats'][0]['count'], 27, 'nothing is played until you pick a reading.')
        with self.assertRaises(ValueError):
            session.act({'type': 'play', 'cards': cards, 'reading': 7})
        session.act({'type': 'play', 'cards': cards, 'reading': 2})
        self.assertEqual(session.round.history[0][1].describe(), 'straight to 9')
        self.assertIsNone(session.view()['choices'])

    def test_tribute(self):
        # you went out first last hand and West last, so West pays you and you hand a card back
        session = GuanDanSession({}, random.Random(4))
        hands = deal(random.Random(5))
        seat(session, hands)
        session.match.previous_order, session.match.declarer, session.match.levels = (0, 1, 2, 3), 0, [4, 2]
        session.act({'type': 'deal'})
        view = session.view()
        self.assertEqual(view['phase'], 'return')
        self.assertEqual(len(view['hand']), 28)
        paid = view['tribute']['paid']
        self.assertEqual([(t['payer'], t['receiver']) for t in paid], [('West', 'You')])
        advice = session.advice()
        self.assertEqual(advice['kind'], 'return')
        self.assertIn(advice['card'], view['tribute']['options'])
        joker = next((c for c in view['hand'] if c in ('RJ', 'BJ')), None)
        if joker is not None:
            with self.assertRaises(ValueError):
                session.act({'type': 'return', 'card': joker})
        with self.assertRaises(ValueError):
            session.act({'type': 'play', 'cards': [view['hand'][0]]})
        session.act({'type': 'return', 'card': view['tribute']['options'][0]})
        self.assertIn(session.phase, ('play', 'done'))
        self.assertIn('West pays you tribute of %s.' % paid[0]['name'], session.log)
        self.assertEqual(sum(line.startswith('West pays you') for line in session.log), 1, 'announced once.')
        self.assertTrue(any(line.startswith('You hand back') and line.endswith('to West.') for line in session.log))
        self.assertEqual(sum(len(hand) for hand in session.round.hands) + len(session.round.played), 108)

    def test_coached_match(self):
        session = GuanDanSession({}, random.Random(6))
        while session.phase != 'over':
            if session.phase in ('deal', 'done'):
                session.act({'type': 'deal'})
            else:
                follow_coach(session)
        view = session.view()
        self.assertIn(view['match']['winner'], ('you', 'them'))
        self.assertIn('A', view['levels'].values())
        self.assertEqual(view['stats']['followed'], view['stats']['decisions'])
        with self.assertRaises(ValueError):
            session.act({'type': 'deal'})
        json.dumps(view)


class TestFairPlay(TestCase):
    def test_the_page_never_gets_another_seats_cards(self):
        # two tables alike in everything you can see, different in which opponent holds two of the cards
        hands = deal(random.Random(7))
        a = next(card for card in hands[1] if card not in hands[3])
        b = next(card for card in hands[3] if card not in hands[1])
        swapped = [list(hand) for hand in hands]
        swapped[1].remove(a)
        swapped[1].append(b)
        swapped[3].remove(b)
        swapped[3].append(a)
        views, advice = [], []
        for h in (hands, swapped):
            session = GuanDanSession({}, random.Random(8))
            seat(session, h)
            session.act({'type': 'deal'})
            views.append(json.dumps(session.view(), sort_keys=True))
            advice.append(session.advice())
        self.assertEqual(views[0], views[1])
        self.assertEqual(advice[0], advice[1])

import json
import os
import random
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.CardGames.Core.Card import Card
from src.CardGames.Holdem.Equity import outs, simulate_equity
from src.CardGames.Holdem.HandEvaluator import HandCategory, describe
from src.CardGames.Holdem.Round import Action, Street
from src.CardGames.Holdem.Rules import Rules
from src.CardGames.Play import GuanDanLessons, HoldemLessons
from src.CardGames.Play.BlackJackLessons import lessons
from src.CardGames.Play.HoldemLessons import Scripted
from src.CardGames.Play.Lesson import LessonSession, Step
from src.CardGames.Play.Registry import lobby
from src.CardGames.Play.Server import Sessions

__author__ = 'frank.ma'


def follow(session: LessonSession) -> None:
    # take every step as the lesson expects, checking first that the coach would have chosen the same
    test = TestCase()
    lesson = session.lesson
    for step in lesson.steps:
        advice = session.advice()
        if step.expect['type'] == 'move':
            test.assertEqual(advice['move'], step.expect['move'], '%s: the coach agrees with the lesson.' % lesson.key)
            session.act(dict(step.expect, to=advice.get('to')))
        else:
            test.assertGreaterEqual(advice['bet'], step.expect['bet_at_least'], lesson.key)
            session.act({'type': 'deal', 'bet': advice['bet']})
        test.assertEqual(session.feedback['kind'], 'right')


class TestStep(TestCase):
    def test_matches(self):
        step = Step({'type': 'move', 'move': 'hit'}, '', '', '')
        self.assertTrue(step.matches({'type': 'move', 'move': 'hit'}))
        self.assertFalse(step.matches({'type': 'move', 'move': 'stand'}))
        bet = Step({'type': 'deal', 'bet_at_least': 10}, '', '', '')
        self.assertTrue(bet.matches({'type': 'deal', 'bet': 25}))
        for action in ({'type': 'deal', 'bet': 5}, {'type': 'deal', 'bet': 'x'}, {'type': 'move', 'bet': 50}):
            self.assertFalse(bet.matches(action))


class TestBlackJackLessons(TestCase):
    def test_every_lesson_plays_as_written(self):
        session = LessonSession(lessons(), random.Random(0))
        outcomes = {}
        for k, lesson in enumerate(session.lessons):
            session.act({'type': 'lesson', 'index': k})
            first = session.view()
            if first['phase'] == 'play':
                # fair play holds in a lesson too: one dealer card up, the other face down
                self.assertTrue(first['dealer']['hidden'])
                self.assertEqual(len(first['dealer']['cards']), 1)
            follow(session)
            view = session.view()
            self.assertTrue(view['lesson']['complete'])
            self.assertIsNone(view['lesson']['prompt'])
            self.assertTrue(view['lesson']['lessons'][k]['done'])
            if lesson.wrap:
                self.assertIn(lesson.wrap, view['lesson']['feedback']['text'])
            outcomes[lesson.key] = [hand['outcome'] for hand in view['hands']], view.get('payoff')
            json.dumps(view)
        self.assertEqual(outcomes['bust'], (['win'], 10.0), 'the dealer busts drawing from a 6.')
        self.assertEqual(outcomes['sixteen'], (['win'], 10.0))
        self.assertEqual(outcomes['double'], (['win'], 20.0), 'twice the bet won.')
        self.assertEqual(outcomes['soft'], (['win'], 10.0))
        self.assertEqual(outcomes['split'], (['win', 'push'], 10.0))
        self.assertTrue(all(lesson['done'] for lesson in session.view()['lesson']['lessons']))

    def test_a_wrong_move_gets_a_hint_and_is_not_played(self):
        session = LessonSession(lessons(), random.Random(1))
        before = session.view()['hands']
        session.act({'type': 'move', 'move': 'hit'})  # the first lesson wants a stand on 13 against a 6
        view = session.view()
        self.assertEqual(view['hands'], before, 'nothing was played.')
        self.assertEqual((view['lesson']['feedback']['kind'], view['lesson']['mistakes'], view['lesson']['step']),
                         ('hint', 1, 0))
        self.assertIn('Try standing', view['lesson']['feedback']['text'])
        session.act({'type': 'move', 'move': 'stand'})
        self.assertEqual(session.view()['lesson']['feedback']['kind'], 'right')

    def test_the_count_lesson(self):
        session = LessonSession(lessons(), random.Random(2))
        session.act({'type': 'lesson', 'index': len(session.lessons) - 1})
        view = session.view()
        self.assertEqual((view['phase'], view['lesson']['awaiting']), ('bet', 'deal'))
        self.assertGreater(view['count']['true'], 4.0, 'the low cards shown push the count up.')
        session.act({'type': 'deal', 'bet': 5})
        self.assertEqual(session.view()['lesson']['feedback']['kind'], 'hint', 'the minimum ignores the count.')
        session.act({'type': 'deal', 'bet': 20})
        self.assertTrue(session.complete)
        # the lesson is over, so the hand plays out freely
        while session.table.phase == 'play':
            session.act({'type': 'move', 'move': session.advice()['move']})
        self.assertEqual(session.table.phase, 'done')

    def test_choosing_lessons(self):
        session = LessonSession(lessons(), random.Random(3))
        session.act({'type': 'next'})
        self.assertEqual(session.index, 1)
        for bad in ({'type': 'lesson', 'index': 99}, {'type': 'lesson', 'index': 'x'}):
            with self.assertRaises(ValueError):
                session.act(bad)
        with self.assertRaises(ValueError):
            LessonSession([])


class TestHoldemLessons(TestCase):
    def test_every_lesson_plays_as_written(self):
        session = LessonSession(HoldemLessons.lessons(), random.Random(4))
        for k, lesson in enumerate(session.lessons):
            session.act({'type': 'lesson', 'index': k})
            first = session.view()
            self.assertTrue(first['your_turn'])
            self.assertEqual(len(first['seats'][0]['cards']), 2)
            self.assertIsNone(first['seats'][1]['cards'], 'the opponent is face down in a lesson too.')
            follow(session)
            view = session.view()
            self.assertTrue(view['lesson']['complete'], lesson.key)
            if view['phase'] == 'play':
                self.assertIsNone(view['seats'][1]['cards'])
            json.dumps(view)
        self.assertTrue(all(x['done'] for x in session.view()['lesson']['lessons']))

    def test_the_fold_lesson_loses_only_the_blind(self):
        session = LessonSession(HoldemLessons.lessons(), random.Random(5))
        session.act({'type': 'lesson', 'index': [x.key for x in session.lessons].index('fold')})
        session.act({'type': 'move', 'move': 'call'})
        session.act({'type': 'move', 'move': 'call'})  # calling the pot-size bet is the mistake
        self.assertEqual(session.feedback['kind'], 'hint')
        session.act({'type': 'move', 'move': 'fold'})
        view = session.view()
        self.assertEqual((view['phase'], view['payoff']), ('done', -2))

    def test_later_hands_are_shuffled(self):
        session = LessonSession(HoldemLessons.lessons(), random.Random(6))
        session.act({'type': 'lesson', 'index': [x.key for x in session.lessons].index('fold')})
        session.act({'type': 'move', 'move': 'call'})
        session.act({'type': 'move', 'move': 'fold'})
        session.act({'type': 'deal'})  # the lesson is done, so the table deals on freely
        self.assertNotEqual(session.view()['seats'][0]['cards'], ['3C', '2D'], 'the next hand is not stacked.')

    def test_scripted_opponent(self):
        session = HoldemLessons._heads_up('AH 5H', 'QC 8D', 'KH 9H 2C 4S JD', {Street.FLOP: 0.5})(random.Random(7))
        session.act({'type': 'move', 'move': 'call'})
        r = session.round
        seat, street, move = r.history[-1]
        self.assertEqual((seat, street, move.action, move.to), (1, Street.FLOP, Action.RAISE, 2), 'half the pot of 4.')
        self.assertEqual(Scripted().decide(r.observation(0), r.legal_moves(0)).action, Action.CALL)


class TestOmahaLessons(TestCase):
    def setUp(self):
        self.lessons = HoldemLessons.omaha_lessons()
        self.rules = Rules(small_blind=1, big_blind=2, n_hole=4, omaha=True)

    def table(self, key):
        return next(lesson for lesson in self.lessons if lesson.key == key).setup(random.Random(0))

    def test_every_lesson_plays_as_written(self):
        session = LessonSession(self.lessons, random.Random(30))
        for k, lesson in enumerate(session.lessons):
            session.act({'type': 'lesson', 'index': k})
            first = session.view()
            self.assertEqual((first['variant'], first['n_hole']), ('omaha', 4))
            self.assertEqual(len(first['seats'][0]['cards']), 4)
            self.assertIsNone(first['seats'][1]['cards'], 'the opponent is face down in a lesson too.')
            follow(session)
            self.assertTrue(session.view()['lesson']['complete'], lesson.key)
            json.dumps(session.view())
        self.assertTrue(all(x['done'] for x in session.view()['lesson']['lessons']))

    def test_later_hands_are_omaha_and_shuffled(self):
        session = LessonSession(self.lessons, random.Random(31))
        follow(session)  # ends with a fold
        session.act({'type': 'deal'})
        view = session.view()
        self.assertEqual(len(view['seats'][0]['cards']), 4)
        self.assertNotEqual(view['seats'][0]['cards'], ['AH', 'TC', '6D', '3S'], 'the next hand is not stacked.')

    def test_one_heart_never_makes_a_flush(self):
        session = LessonSession(self.lessons, random.Random(32))
        session.act({'type': 'move', 'move': 'call'})
        session.act({'type': 'move', 'move': 'check'})
        view = session.view()
        self.assertEqual((view['pot'], view['to_call']), (8, 4), 'a pot-size bet of 4 into 4, as the prompt says.')
        self.assertEqual(view['made'], 'Ace high', 'four hearts on the board, and the ace of hearts in hand.')
        session.act({'type': 'move', 'move': 'call'})
        self.assertEqual(session.feedback['kind'], 'hint', 'calling is the mistake.')
        session.act({'type': 'move', 'move': 'fold'})
        self.assertEqual((session.view()['phase'], session.view()['payoff']), ('done', -2))

    def test_the_counted_cards_are_as_the_lessons_say(self):
        for hole, board, expected in (('AH 9H 7C 2S', 'KH 8H 3C', {HandCategory.FLUSH: 9, HandCategory.PAIR: 18}),
                                      ('JS TD 7C 6H', '9S 8D 2C', {HandCategory.STRAIGHT: 20, HandCategory.PAIR: 9})):
            found = outs(Card.parse_many(hole), Card.parse_many(board), self.rules)
            self.assertEqual({category: len(cards) for category, cards in found.by_category.items()}, expected)
        wrap = outs(Card.parse_many('JS TD 7C 6H'), Card.parse_many('9S 8D 2C'), self.rules)
        ranks = sorted(card.rank.value for card in wrap.by_category[HandCategory.STRAIGHT])
        self.assertEqual({r: ranks.count(r) for r in set(ranks)}, {12: 4, 11: 3, 10: 3, 7: 3, 6: 3, 5: 4})

    def test_the_hands_made_are_as_the_lessons_say(self):
        two = Card.parse_many('AH 9H 7C 2S')
        self.assertEqual(describe(self.rules.hand_rank(two, Card.parse_many('KH 8H 3C 5H'))), 'Flush, ace high')
        wrap = Card.parse_many('JS TD 7C 6H')
        self.assertEqual(describe(self.rules.hand_rank(wrap, Card.parse_many('9S 8D 2C QH'))), 'Straight, queen high')
        sevens = Card.parse_many('7S 7D 7C 2H')
        self.assertEqual(describe(self.rules.hand_rank(sevens, Card.parse_many('KS QS JS'))), 'Pair of sevens')

    def test_the_equities_quoted_are_about_right(self):
        quoted = (('AH TC 6D 3S', '', 0.46), ('AH TC 6D 3S', '8H JH 2H', 0.21), ('AH TC 6D 3S', '8H JH 2H 5H', 0.16),
                  ('AH 9H 7C 2S', '', 0.49), ('AH 9H 7C 2S', 'KH 8H 3C', 0.54), ('AH 9H 7C 2S', 'KH 8H 3C 5H', 0.96),
                  ('JS TD 7C 6H', '', 0.48), ('JS TD 7C 6H', '9S 8D 2C', 0.62), ('JS TD 7C 6H', '9S 8D 2C QH', 0.92),
                  ('7S 7D 7C 2H', '', 0.32), ('7S 7D 7C 2H', 'KS QS JS', 0.16))
        for hole, board, equity in quoted:
            found = simulate_equity(Card.parse_many(hole), Card.parse_many(board) if board else [], 1, 2000, self.rules,
                                    random.Random(33)).equity.mean
            self.assertAlmostEqual(found, equity, delta=0.04, msg='%s on %s' % (hole, board))

    def test_the_lobby_and_server_offer_them_apart_from_holdem(self):
        entry = next(g for g in lobby() if g['name'] == 'holdem')
        self.assertEqual([(p['label'], p['query'], p['n']) for p in entry['practice']],
                         [("Practice Hold'em", 'mode=learn', 4), ('Practice Omaha', 'mode=learn&variant=omaha', 4)])
        self.assertEqual([p['label'] for p in next(g for g in lobby() if g['name'] == 'guandan')['practice']],
                         ['Practice'])
        sessions = Sessions()
        _, omaha = sessions.create('holdem', {'mode': 'learn', 'variant': 'omaha'})
        self.assertEqual(omaha.view()['lesson']['title'], 'One heart is not a flush')
        _, holdem = sessions.create('holdem', {'mode': 'learn'})
        self.assertEqual(holdem.view()['lesson']['title'], 'Call when the price is right')

    def test_a_restart_keeps_them_omaha(self):
        with TemporaryDirectory() as folder:
            path = os.path.join(folder, 'progress.json')
            sessions = Sessions(path=path)
            sid, session = sessions.create('holdem', {'mode': 'learn', 'variant': 'omaha'})
            session.act({'type': 'lesson', 'index': 2})
            sessions.save('holdem', sid, session)
            view = Sessions(path=path).get('holdem', sid).view()
            self.assertEqual((view['lesson']['title'], view['variant']), ('Four cards make a wrap', 'omaha'))


def follow_guandan(session: LessonSession) -> None:
    # take every step as the lesson expects, checking first that the coach would make the same play
    test = TestCase()
    lesson = session.lesson
    for step in lesson.steps:
        advice = session.advice()
        kind = step.expect['type']
        if kind == 'return':
            test.assertEqual(advice['kind'], 'return')
            session.act({'type': 'return', 'card': advice['card']})
        elif kind == 'pass':
            test.assertEqual(advice['move']['kind'], 'pass', lesson.key)
            session.act({'type': 'pass'})
        else:
            test.assertEqual(sorted(advice['move']['cards']), sorted(step.expect['cards']), lesson.key)
            session.act({'type': 'play', 'cards': advice['move']['cards']})
            if session.table.choices:
                readings = [combo.describe() for combo in session.table.choices]
                session.act({'type': 'play', 'cards': advice['move']['cards'],
                             'reading': readings.index(advice['move']['desc'])})
        test.assertEqual(session.feedback['kind'], 'right', lesson.key)


class TestGuanDanLessons(TestCase):
    def keys(self, session):
        return [lesson.key for lesson in session.lessons]

    def test_every_lesson_plays_as_written(self):
        session = LessonSession(GuanDanLessons.lessons(), random.Random(8))
        for k, lesson in enumerate(session.lessons):
            session.act({'type': 'lesson', 'index': k})
            follow_guandan(session)
            view = session.view()
            self.assertTrue(view['lesson']['complete'], lesson.key)
            json.dumps(view)
        self.assertTrue(all(x['done'] for x in session.view()['lesson']['lessons']))

    def test_an_endgame_spot_is_your_view_only(self):
        session = LessonSession(GuanDanLessons.lessons(), random.Random(9))
        session.act({'type': 'lesson', 'index': self.keys(session).index('wind')})
        view = session.view()
        self.assertEqual((view['hand'], [seat['count'] for seat in view['seats']]), (['RJ'], [1, 3, 4, 3]))
        text = json.dumps(view)
        for card in ('9S', 'AS', 'QC', '8H'):
            self.assertNotIn('"%s"' % card, text, 'the other hands stay hidden.')
        # the rest of both packs counts as played, so what you have not seen is only what the others hold
        self.assertEqual(view['unseen'][0], {'label': 'Red jokers', 'n': 0, 'of': 2})
        self.assertEqual(sum(view['seats'][s]['count'] for s in range(4)) + len(session.table.round.played), 108)

    def test_wrong_plays_get_hints(self):
        session = LessonSession(GuanDanLessons.lessons(), random.Random(10))
        session.act({'type': 'play', 'cards': ['5S']})  # a single cannot beat a pair
        self.assertEqual((session.feedback['kind'], session.view()['seats'][0]['count']), ('hint', 5))
        session.act({'type': 'lesson', 'index': self.keys(session).index('partner')})
        session.act({'type': 'play', 'cards': ['KS', 'KD']})
        self.assertEqual(session.feedback['kind'], 'hint', 'beating your partner is the mistake.')

    def test_the_wild_reading(self):
        session = LessonSession(GuanDanLessons.lessons(), random.Random(11))
        session.act({'type': 'lesson', 'index': self.keys(session).index('wild')})
        cards = ['2H', '5S', '6S', '7S', '8S']
        session.act({'type': 'play', 'cards': cards})
        self.assertEqual(session.feedback['kind'], 'choose')
        readings = session.view()['choices']
        self.assertEqual(readings, ['straight flush to 9', 'straight flush to 8'])
        session.act({'type': 'play', 'cards': cards, 'reading': readings.index('straight flush to 8')})
        self.assertEqual(session.feedback['kind'], 'hint')
        self.assertEqual(session.view()['choices'], readings, 'still waiting for the reading.')
        session.act({'type': 'play', 'cards': cards, 'reading': readings.index('straight flush to 9')})
        self.assertEqual(session.feedback['kind'], 'right')
        self.assertTrue(session.complete)

    def test_tribute_hands_back_only_a_low_card(self):
        session = LessonSession(GuanDanLessons.lessons(), random.Random(12))
        session.act({'type': 'lesson', 'index': self.keys(session).index('tribute')})
        view = session.view()
        self.assertEqual(view['phase'], 'return')
        self.assertEqual(view['tribute']['paid'][0]['payer'], 'West')
        session.act({'type': 'return', 'card': 'BJ'})
        self.assertEqual(session.feedback['kind'], 'hint', 'a joker is never handed back.')
        session.act({'type': 'return', 'card': view['tribute']['options'][0]})
        self.assertEqual(session.feedback['kind'], 'right')

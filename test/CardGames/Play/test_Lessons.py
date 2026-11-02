import json
import random
from unittest import TestCase

from src.CardGames.Holdem.Round import Action, Street
from src.CardGames.Play import HoldemLessons
from src.CardGames.Play.BlackJackLessons import lessons
from src.CardGames.Play.HoldemLessons import Scripted
from src.CardGames.Play.Lesson import LessonSession, Step

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

import json
import random
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
from test.CardGames.Play.helpers import play_coach_move, session_act

__author__ = 'frank.ma'


LESSON_SETS = [  # name, lessons, seed
    ('blackjack', lessons, 0), ("hold'em", HoldemLessons.lessons, 4), ('Omaha', HoldemLessons.omaha_lessons, 30),
    ('GuanDan', GuanDanLessons.lessons, 8)]


def follow(test: TestCase, session: LessonSession) -> None:
    # take every step as the lesson expects, checking first that the coach would make the same move
    for step in session.lesson.steps:
        advice, want = session.advice(), step.expect
        if want['type'] == 'deal':
            test.assertGreaterEqual(advice['bet'], want['bet_at_least'], session.lesson.key)
        elif want['type'] == 'move':
            test.assertEqual(advice['move'], want['move'], '%s: the coach agrees with the lesson.' % session.lesson.key)
        elif want['type'] == 'return':
            test.assertEqual(advice['kind'], 'return', session.lesson.key)
        elif want['type'] == 'pass':
            test.assertEqual(advice['move']['kind'], 'pass', session.lesson.key)
        else:
            test.assertEqual(sorted(advice['move']['cards']), sorted(want['cards']), session.lesson.key)
        play_coach_move(session_act(session), session.view(), advice)
        test.assertEqual(session.feedback['kind'], 'right', session.lesson.key)


class TestEveryLesson(TestCase):
    def test_every_lesson_plays_as_written(self):
        for name, make, seed in LESSON_SETS:
            with self.subTest(name):
                session, outcomes = LessonSession(make(), random.Random(seed)), {}
                for k, lesson in enumerate(session.lessons):
                    session.act({'type': 'lesson', 'index': k})
                    first = session.view()
                    # fair play holds in a lesson too: a face-down card stays face down
                    if first['game'] == 'blackjack' and first['phase'] == 'play':
                        self.assertEqual((first['dealer']['hidden'], len(first['dealer']['cards'])), (True, 1))
                    if first['game'] == 'holdem':
                        self.assertTrue(first['your_turn'])
                        self.assertEqual(len(first['seats'][0]['cards']), first['n_hole'])
                        self.assertIsNone(first['seats'][1]['cards'], 'the opponent is face down.')
                    follow(self, session)
                    view = session.view()
                    self.assertTrue(view['lesson']['complete'], lesson.key)
                    self.assertIsNone(view['lesson']['prompt'])
                    self.assertTrue(view['lesson']['lessons'][k]['done'])
                    if lesson.wrap:
                        self.assertIn(lesson.wrap, view['lesson']['feedback']['text'])
                    if view['game'] == 'holdem' and view['phase'] == 'play':
                        self.assertIsNone(view['seats'][1]['cards'])
                    outcomes[lesson.key] = [hand['outcome'] for hand in view.get('hands', [])], view.get('payoff')
                    json.dumps(view)
                self.assertTrue(all(x['done'] for x in session.view()['lesson']['lessons']))
                if name == 'blackjack':
                    self.assertEqual(outcomes['bust'], (['win'], 10.0), 'the dealer busts drawing from a 6.')
                    self.assertEqual(outcomes['sixteen'], (['win'], 10.0))
                    self.assertEqual(outcomes['double'], (['win'], 20.0), 'twice the bet won.')
                    self.assertEqual(outcomes['soft'], (['win'], 10.0))
                    self.assertEqual(outcomes['split'], (['win', 'push'], 10.0))

    def test_later_hands_are_dealt_freely(self):
        # once a lesson is done the table deals on, shuffled, in the lesson's own game
        for make, steps, stacked in ((HoldemLessons.lessons, ('call', 'call', 'fold'), ['3C', '2D']),
                                     (HoldemLessons.omaha_lessons, ('call', 'check', 'fold'), ['AH', 'TC', '6D', '3S'])):
            session = LessonSession(make(), random.Random(6))
            session.act({'type': 'lesson', 'index': [x.key for x in session.lessons].index(
                'fold' if make is HoldemLessons.lessons else 'one-heart')})
            for move in steps:
                session.act({'type': 'move', 'move': move})
            session.act({'type': 'deal'})
            cards = session.view()['seats'][0]['cards']
            self.assertEqual(len(cards), len(stacked))
            self.assertNotEqual(cards, stacked, 'the next hand is not stacked.')


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
    def test_the_fold_lesson_loses_only_the_blind(self):
        session = LessonSession(HoldemLessons.lessons(), random.Random(5))
        session.act({'type': 'lesson', 'index': [x.key for x in session.lessons].index('fold')})
        session.act({'type': 'move', 'move': 'call'})
        session.act({'type': 'move', 'move': 'call'})  # calling the pot-size bet is the mistake
        self.assertEqual(session.feedback['kind'], 'hint')
        session.act({'type': 'move', 'move': 'fold'})
        view = session.view()
        self.assertEqual((view['phase'], view['payoff']), ('done', -2))

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

    def test_the_cards_and_hands_are_as_the_lessons_say(self):
        for hole, board, expected in (('AH 9H 7C 2S', 'KH 8H 3C', {HandCategory.FLUSH: 9, HandCategory.PAIR: 18}),
                                      ('JS TD 7C 6H', '9S 8D 2C', {HandCategory.STRAIGHT: 20, HandCategory.PAIR: 9})):
            found = outs(Card.parse_many(hole), Card.parse_many(board), self.rules)
            self.assertEqual({category: len(cards) for category, cards in found.by_category.items()}, expected)
        wrap = outs(Card.parse_many('JS TD 7C 6H'), Card.parse_many('9S 8D 2C'), self.rules)
        ranks = sorted(card.rank.value for card in wrap.by_category[HandCategory.STRAIGHT])
        self.assertEqual({r: ranks.count(r) for r in set(ranks)}, {12: 4, 11: 3, 10: 3, 7: 3, 6: 3, 5: 4})
        for hole, board, made in (('AH 9H 7C 2S', 'KH 8H 3C 5H', 'Flush, ace high'),
                                  ('JS TD 7C 6H', '9S 8D 2C QH', 'Straight, queen high'),
                                  ('7S 7D 7C 2H', 'KS QS JS', 'Pair of sevens')):
            self.assertEqual(describe(self.rules.hand_rank(Card.parse_many(hole), Card.parse_many(board))), made)

    def test_the_equities_quoted_are_about_right(self):
        quoted = (('AH TC 6D 3S', '', 0.46), ('AH TC 6D 3S', '8H JH 2H', 0.21), ('AH TC 6D 3S', '8H JH 2H 5H', 0.16),
                  ('AH 9H 7C 2S', '', 0.49), ('AH 9H 7C 2S', 'KH 8H 3C', 0.54), ('AH 9H 7C 2S', 'KH 8H 3C 5H', 0.96),
                  ('JS TD 7C 6H', '', 0.48), ('JS TD 7C 6H', '9S 8D 2C', 0.62), ('JS TD 7C 6H', '9S 8D 2C QH', 0.92),
                  ('7S 7D 7C 2H', '', 0.32), ('7S 7D 7C 2H', 'KS QS JS', 0.16))
        for hole, board, equity in quoted:
            found = simulate_equity(Card.parse_many(hole), Card.parse_many(board) if board else [], 1, 2000, self.rules,
                                    random.Random(33)).equity.mean
            self.assertAlmostEqual(found, equity, delta=0.04, msg='%s on %s' % (hole, board))

class TestGuanDanLessons(TestCase):
    def keys(self, session):
        return [lesson.key for lesson in session.lessons]

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

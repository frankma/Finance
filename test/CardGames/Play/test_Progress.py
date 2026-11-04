import json
import os
import random
import re
import tempfile
import threading
import urllib.request
from unittest import TestCase

from src.CardGames.GuanDan.Round import deal
from src.CardGames.Play.BlackJackLessons import lessons
from src.CardGames.Play.BlackJackSession import BlackJackSession
from src.CardGames.Play.GuanDanSession import GuanDanSession
from src.CardGames.Play.HoldemSession import HoldemSession
from src.CardGames.Play.Lesson import LessonSession
from src.CardGames.Play.Server import MAX_SESSIONS, Sessions, make_server

__author__ = 'frank.ma'

CARD = re.compile(r'"(?:[2-9TJQKA][SHDC]|BJ|RJ)"')


def play_hand(session: HoldemSession) -> None:
    session.act({'type': 'deal'})
    while session.phase == 'play':
        advice = session.advice()
        session.act({'type': 'move', 'move': advice['move'], 'to': advice['to']})


class TestRestore(TestCase):
    def test_blackjack(self):
        session = BlackJackSession({}, random.Random(0))
        session.act({'type': 'deal', 'bet': 20})
        while session.phase == 'play':
            session.act({'type': 'move', 'move': session.advice()['move']})
        back = BlackJackSession({}, random.Random(1))
        back.restore(json.loads(json.dumps(session.progress())))
        self.assertEqual(back.bankroll, session.bankroll)
        self.assertEqual((back.phase, back.rounds, back.last_bet), ('bet', 1, 20))
        self.assertEqual(back.view()['stats']['net'], session.view()['stats']['net'])
        self.assertIn('restarted', back.log[-1])
        self.assertEqual(back.view()['shoe']['cards_left'], back.view()['shoe']['n_cards'], 'a fresh shoe.')

    def test_blackjack_round_in_play_does_not_count(self):
        session = BlackJackSession({}, random.Random(2))
        session.act({'type': 'deal', 'bet': 20})
        self.assertEqual(session.phase, 'play')
        back = BlackJackSession({}, random.Random(3))
        back.restore(session.progress())
        self.assertEqual((back.phase, back.bankroll), ('bet', 1000.0))

    def test_holdem(self):
        session = HoldemSession({'seats': 4}, random.Random(4))
        for _ in range(3):
            play_hand(session)
        back = HoldemSession({'seats': 4}, random.Random(5))
        back.restore(json.loads(json.dumps(session.progress())))
        self.assertEqual((back.stacks, back.bought, back.button, back.hands),
                         (session.stacks, session.bought, session.button, 3))
        self.assertEqual(back.phase, 'deal')
        self.assertEqual(back.view()['stats'], session.view()['stats'])
        with self.assertRaises(ValueError):
            HoldemSession({'seats': 6}, random.Random(6)).restore(session.progress())

    def test_holdem_hand_in_play_does_not_count(self):
        session = HoldemSession({'seats': 2}, random.Random(20))
        play_hand(session)
        settled = session.progress()
        session.act({'type': 'deal'})
        self.assertEqual(session.phase, 'play')
        self.assertEqual(session.progress(), settled, 'the same hands, button and stacks as before the deal.')

    def test_guandan_match_carries_on(self):
        session = GuanDanSession({}, random.Random(7))
        session.match.previous_order, session.match.declarer, session.match.levels = (0, 1, 2, 3), 0, [7, 4]
        session.match.a_fails, session.hands_before, session.won = [0, 1], 5, 3
        back = GuanDanSession({}, random.Random(8))
        back.restore(json.loads(json.dumps(session.progress())))
        view = back.view()
        self.assertEqual((view['phase'], view['level'], view['levels']), ('deal', '7', {'you': '7', 'them': '4'}))
        self.assertEqual(view['match']['hands'], 5)
        self.assertEqual(view['a_fails'], [0, 1])
        back._deal_hands = lambda: deal(random.Random(9))
        back.act({'type': 'deal'})
        self.assertEqual(back.phase, 'return', 'you went out first last hand, so tribute is still owed to you.')
        self.assertTrue(any(line.startswith('Hand 6 at level 7') for line in back.log))

    def test_guandan_finished_match_stays_finished(self):
        session = GuanDanSession({}, random.Random(10))
        session.match.winner, session.match.levels, session.match.declarer = 0, [14, 9], 0
        back = GuanDanSession({}, random.Random(11))
        back.restore(session.progress())
        self.assertEqual((back.phase, back.view()['match']['winner']), ('over', 'you'))

    def test_bad_guandan_progress_is_refused(self):
        good = GuanDanSession({}, random.Random(12)).progress()
        for change in ({'levels': [1, 99]}, {'levels': [2]}, {'previous_order': [0, 0, 1, 2]}, {'declarer': 5}):
            with self.assertRaises(ValueError):
                GuanDanSession({}, random.Random(13)).restore(dict(good, **change))

    def test_lessons(self):
        session = LessonSession(lessons(), random.Random(14))
        session.act({'type': 'move', 'move': 'stand'})
        session.act({'type': 'next'})
        back = LessonSession(lessons(), random.Random(15))
        back.restore(json.loads(json.dumps(session.progress())))
        view = back.view()['lesson']
        self.assertEqual((view['index'], view['step']), (1, 0), 'back at the start of the lesson you were on.')
        self.assertEqual([x['done'] for x in view['lessons']], [True, False, False, False, False, False])
        with self.assertRaises(ValueError):
            back.restore({'done': [True], 'index': 0})


class TestNoCardsAreSaved(TestCase):
    def test_progress_holds_no_card_in_any_session(self):
        blackjack = BlackJackSession({}, random.Random(16))
        blackjack.act({'type': 'deal', 'bet': 10})
        holdem = HoldemSession({'seats': 4}, random.Random(17))
        holdem.act({'type': 'deal'})
        guandan = GuanDanSession({}, random.Random(18))
        guandan.act({'type': 'deal'})
        lesson = LessonSession(lessons(), random.Random(19))
        for session in (blackjack, holdem, guandan, lesson):
            text = json.dumps(session.progress())
            self.assertIsNone(CARD.search(text), text)
            json.loads(text)


class Server(object):
    # a play server on a free port, with its progress in this file
    def __init__(self, path):
        self.server = make_server(port=0, progress=path)
        self.base = 'http://127.0.0.1:%i' % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def call(self, method, route, body=None):
        data = json.dumps(body).encode('utf-8') if body is not None else None
        request = urllib.request.Request(self.base + route, data=data, method=method,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as res:
            return json.loads(res.read())

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class TestAcrossRestarts(TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = os.path.join(self.folder.name, 'nested', 'progress.json')

    def test_a_session_survives_a_restart(self):
        first = Server(self.path)
        sid = first.call('POST', '/api/blackjack/new', {'rules': 'h17'})['id']
        view = first.call('POST', '/api/blackjack/%s/act' % sid, {'type': 'deal', 'bet': 25})
        while view['phase'] == 'play':
            advice = first.call('GET', '/api/blackjack/%s/advice' % sid)
            view = first.call('POST', '/api/blackjack/%s/act' % sid, {'type': 'move', 'move': advice['move']})
        first.stop()
        with open(self.path, encoding='utf-8') as f:
            text = f.read()
        self.assertIsNone(CARD.search(text), 'the file holds no cards.')
        second = Server(self.path)
        try:
            again = second.call('GET', '/api/blackjack/%s' % sid)
            self.assertEqual(again['bankroll'], view['bankroll'])
            self.assertEqual(again['stats']['rounds'], 1)
            self.assertEqual(again['rules'], view['rules'], 'the same table rules.')
            self.assertEqual(second.call('POST', '/api/blackjack/%s/act' % sid, {'type': 'deal', 'bet': 5})['last_bet'], 5)
        finally:
            second.stop()

    def test_lessons_survive_a_restart(self):
        first = Server(self.path)
        sid = first.call('POST', '/api/blackjack/new', {'mode': 'learn'})['id']
        first.call('POST', '/api/blackjack/%s/act' % sid, {'type': 'move', 'move': 'stand'})
        first.stop()
        second = Server(self.path)
        try:
            self.assertTrue(second.call('GET', '/api/blackjack/%s' % sid)['lesson']['lessons'][0]['done'])
        finally:
            second.stop()

    def test_a_session_belongs_to_its_game(self):
        first = Server(self.path)
        sid = first.call('POST', '/api/blackjack/new', {})['id']
        first.stop()
        sessions = Sessions(path=self.path)
        with self.assertRaises(KeyError):
            sessions.get('holdem', sid)
        self.assertIsInstance(sessions.get('blackjack', sid), BlackJackSession)

    def test_no_file_without_a_path(self):
        sessions = Sessions()
        sid, session = sessions.create('blackjack', {})
        sessions.save('blackjack', sid, session)
        self.assertEqual(sessions.saved, {})

    def test_an_unreadable_file_is_set_aside(self):
        os.makedirs(os.path.dirname(self.path))
        for content in ('not json {', '[1, 2]'):
            with open(self.path, 'w', encoding='utf-8') as f:
                f.write(content)
            with self.assertLogs('src.CardGames.Play.Server', 'WARNING'):
                sessions = Sessions(path=self.path)
            self.assertEqual(sessions.saved, {})
            self.assertTrue(os.path.exists(self.path + '.unreadable'), 'kept, in case it can be recovered.')
            self.assertEqual(sessions.create('blackjack', {})[1].phase, 'bet')

    def test_a_bad_record_is_dropped_not_fatal(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump({'bad': {'game': 'blackjack', 'options': {}, 'progress': {'bankroll': 'lots'}},
                       'worse': 'not a record'}, f)
        sessions = Sessions(path=self.path)
        with self.assertLogs('src.CardGames.Play.Server', 'WARNING'), self.assertRaises(KeyError):
            sessions.get('blackjack', 'bad')
        self.assertNotIn('bad', sessions.saved)
        with self.assertRaises(KeyError):
            sessions.get('blackjack', 'worse')

    def test_the_file_is_bounded(self):
        sessions = Sessions(path=self.path)
        sessions.saved.update({'old%i' % k: {'game': 'blackjack', 'options': {}, 'progress': {}}
                               for k in range(MAX_SESSIONS + 5)})
        sid, _ = sessions.create('blackjack', {})
        self.assertEqual(len(sessions.saved), MAX_SESSIONS)
        self.assertIn(sid, sessions.saved)
        self.assertNotIn('old0', sessions.saved)

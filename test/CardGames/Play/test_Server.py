from unittest import TestCase

from src.CardGames.Play.Registry import lobby, playable
from src.CardGames.Play.Server import build_id
from test.CardGames.Play.helpers import Served, coach_actions

__author__ = 'frank.ma'


class TestServer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.served = Served(seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.served.stop()

    def request(self, method, path, body=None):
        return self.served.request(method, path, body)

    def new(self, game, options=None):
        return self.served.call('POST', '/api/%s/new' % game, options or {})

    def test_only_the_game_list_is_shared(self):
        # the Lab, opened as a file, may ask whether the server is up; no other page may read a game in progress
        self.assertEqual(self.request('GET', '/api/games')[2].get('Access-Control-Allow-Origin'), '*')
        sid = self.new('blackjack')['id']
        for path in ('/api/blackjack/%s' % sid, '/api/blackjack/%s/advice' % sid, '/'):
            self.assertIsNone(self.request('GET', path)[2].get('Access-Control-Allow-Origin'), path)

    def test_every_response_names_its_build(self):
        for path in ('/', '/api/games', '/nothing'):
            self.assertEqual(self.request('GET', path)[2].get('X-Card-Games-Build'), build_id(), path)

    def test_pages(self):
        for path in ('/', '/blackjack', '/holdem', '/guandan', '/blackjack/rules', '/holdem/rules', '/guandan/rules',
                     '/static/rules-guandan.html', '/static/rules.html', '/static/holdem.html', '/static/theme.css',
                     '/static/common.js', '/static/common.css', '/static/engine.js', '/lab',
                     # the pages link with a trailing slash, as the static site serves folders
                     '/blackjack/', '/holdem/', '/guandan/', '/blackjack/rules/', '/lab/'):
            self.assertEqual(self.request('GET', path)[0], 200, path)
        for path in ('/poker', '/poker/rules', '/blackjack/rules/x', '/static/Server.py', '/static/..%2FServer.py',
                     '/nothing'):
            self.assertEqual(self.request('GET', path)[0], 404, path)

    def test_a_hand_of_each_game_over_the_api(self):
        # deal, then ask the coach and play its move until the hand is over; a bad action is a 400 with a reason
        self.assertEqual(self.request('GET', '/api/games')[1], lobby())
        self.assertEqual(playable(), ['blackjack', 'holdem', 'guandan'])
        for game, options, bad, error in (
                ('blackjack', {'rules': 'h17'}, {'type': 'move', 'move': 'fly'}, "unknown move 'fly'."),
                ('holdem', {'seats': '2'}, {'type': 'move', 'move': 'raise', 'to': 1}, 'it is not your turn.'),
                ('guandan', {}, {'type': 'play', 'cards': ['ZZ']}, "unknown cards ['ZZ'].")):
            with self.subTest(game):
                made = self.new(game, options)
                sid, view = made['id'], made['view']
                route = '/api/%s/%s' % (game, sid)
                for _ in range(200 if game != 'guandan' else 3):
                    advice = self.served.call('GET', route + '/advice')
                    if advice['kind'] == 'wait' or (view['phase'] == 'done' and game != 'blackjack'):
                        view = self.served.call('POST', route + '/act', {'type': 'deal'})
                        continue
                    if advice['kind'] == 'move' and game != 'guandan':
                        self.assertIn(advice['move'], view['legal'])
                    action = coach_actions(view, advice)[0]
                    view = self.served.call('POST', route + '/act', action)
                    if view.get('choices'):
                        view = self.served.call('POST', route + '/act',
                                                dict(action, reading=view['choices'].index(advice['move']['desc'])))
                    self.assertEqual(self.served.call('GET', route), view, 'what the page reads back is what it got.')
                    if view['phase'] == 'done':
                        break
                status, body, _ = self.request('POST', route + '/act', bad)
                self.assertEqual((status, body['error']), (400, error))
                self.assertEqual(self.request('GET', '/api/%s/%s' % ('blackjack' if game != 'blackjack' else 'holdem',
                                                                     sid))[0], 404, 'a session belongs to its game.')
        self.assertEqual(self.request('GET', '/api/blackjack/unknown')[0], 404)
        self.assertEqual(self.request('POST', '/api/poker/new', {})[0], 404)

    def test_lessons(self):
        made = self.new('blackjack', {'mode': 'learn'})
        self.assertEqual(made['view']['lesson']['index'], 0)
        view = self.served.call('POST', '/api/blackjack/%s/act' % made['id'], {'type': 'move', 'move': 'stand'})
        self.assertEqual(view['lesson']['feedback']['kind'], 'right')
        self.assertEqual(self.new('guandan', {'mode': 'learn'})['view']['lesson']['title'], 'Beat it or pass')
        # Omaha's lessons are apart from hold'em's, each with its own link in the lobby
        self.assertEqual(self.new('holdem', {'mode': 'learn', 'variant': 'omaha'})['view']['lesson']['title'],
                         'One heart is not a flush')
        self.assertEqual(self.new('holdem', {'mode': 'learn'})['view']['lesson']['title'],
                         'Call when the price is right')
        games = {g['name']: g for g in lobby()}
        self.assertEqual([games[name]['lessons'] for name in playable()], [6, 4, 7])
        self.assertEqual([(p['label'], p['query'], p['n']) for p in games['holdem']['practice']],
                         [("Practice Hold'em", 'mode=learn', 4), ('Practice Omaha', 'mode=learn&variant=omaha', 4)])
        self.assertEqual([p['label'] for p in games['guandan']['practice']], ['Practice'])

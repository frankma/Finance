import json
import threading
import urllib.error
import urllib.request
from unittest import TestCase

from src.CardGames.Play.Registry import lobby, playable
from src.CardGames.Play.Server import make_server

__author__ = 'frank.ma'


class TestServer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(port=0, seed=1)
        cls.base = 'http://127.0.0.1:%i' % cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=False):
        data = json.dumps(body).encode('utf-8') if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as res:
                return (res.status, res.read(), res.headers) if headers else (res.status, res.read())
        except urllib.error.HTTPError as e:
            return (e.code, e.read(), e.headers) if headers else (e.code, e.read())

    def test_only_the_game_list_is_shared(self):
        # the Lab, opened as a file, may ask whether the server is up; no other page may read a game in progress
        self.assertEqual(self.request('GET', '/api/games', headers=True)[2].get('Access-Control-Allow-Origin'), '*')
        sid = json.loads(self.request('POST', '/api/blackjack/new', {})[1])['id']
        for path in ('/api/blackjack/%s' % sid, '/api/blackjack/%s/advice' % sid, '/'):
            self.assertIsNone(self.request('GET', path, headers=True)[2].get('Access-Control-Allow-Origin'), path)

    def test_pages(self):
        for path in ('/', '/blackjack', '/holdem', '/guandan', '/static/holdem.html', '/static/theme.css', '/static/common.js', '/static/common.css', '/lab'):
            self.assertEqual(self.request('GET', path)[0], 200, path)
        for path in ('/poker', '/static/Server.py', '/static/..%2FServer.py', '/nothing'):
            self.assertEqual(self.request('GET', path)[0], 404, path)

    def test_api(self):
        status, body = self.request('GET', '/api/games')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), lobby())
        self.assertEqual(playable(), ['blackjack', 'holdem', 'guandan'])
        status, body = self.request('POST', '/api/blackjack/new', {'rules': 'h17'})
        made = json.loads(body)
        sid = made['id']
        self.assertEqual(made['view']['phase'], 'bet')
        status, body = self.request('POST', '/api/blackjack/%s/act' % sid, {'type': 'deal', 'bet': 10})
        self.assertEqual(status, 200)
        view = json.loads(body)
        self.assertIn(view['phase'], ('play', 'done'))
        self.assertEqual(json.loads(self.request('GET', '/api/blackjack/%s' % sid)[1]), view)
        advice = json.loads(self.request('GET', '/api/blackjack/%s/advice' % sid)[1])
        self.assertIn('reason', advice)
        status, body = self.request('POST', '/api/blackjack/%s/act' % sid, {'type': 'move', 'move': 'fly'})
        self.assertEqual((status, json.loads(body)['error']), (400, "unknown move 'fly'."))
        self.assertEqual(self.request('GET', '/api/blackjack/unknown')[0], 404)
        self.assertEqual(self.request('POST', '/api/poker/new', {})[0], 404)
        self.assertEqual(self.request('GET', '/api/holdem/%s' % sid)[0], 404, 'a session belongs to its game.')

    def test_holdem_api(self):
        sid = json.loads(self.request('POST', '/api/holdem/new', {'seats': '2'})[1])['id']
        view = json.loads(self.request('POST', '/api/holdem/%s/act' % sid, {'type': 'deal'})[1])
        while view['phase'] == 'play':
            advice = json.loads(self.request('GET', '/api/holdem/%s/advice' % sid)[1])
            self.assertIn(advice['move'], view['legal'])
            status, body = self.request('POST', '/api/holdem/%s/act' % sid,
                                        {'type': 'move', 'move': advice['move'], 'to': advice['to']})
            self.assertEqual(status, 200)
            view = json.loads(body)
        self.assertEqual(view['phase'], 'done')
        status, body = self.request('POST', '/api/holdem/%s/act' % sid, {'type': 'move', 'move': 'raise', 'to': 1})
        self.assertEqual((status, json.loads(body)['error']), (400, 'it is not your turn.'))

    def test_guandan_api(self):
        sid = json.loads(self.request('POST', '/api/guandan/new', {})[1])['id']
        view = json.loads(self.request('POST', '/api/guandan/%s/act' % sid, {'type': 'deal'})[1])
        for _ in range(3):
            if not view['your_turn']:
                break
            advice = json.loads(self.request('GET', '/api/guandan/%s/advice' % sid)[1])
            move = advice['move']
            action = {'type': 'pass'} if move['kind'] == 'pass' else {'type': 'play', 'cards': move['cards']}
            status, body = self.request('POST', '/api/guandan/%s/act' % sid, action)
            self.assertEqual(status, 200)
            view = json.loads(body)
            if view['choices']:
                action['reading'] = view['choices'].index(move['desc'])
                view = json.loads(self.request('POST', '/api/guandan/%s/act' % sid, action)[1])
        status, body = self.request('POST', '/api/guandan/%s/act' % sid, {'type': 'play', 'cards': ['ZZ']})
        self.assertEqual(status, 400)

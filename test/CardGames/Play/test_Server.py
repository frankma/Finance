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

    def request(self, method, path, body=None):
        data = json.dumps(body).encode('utf-8') if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as res:
                return res.status, res.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_pages(self):
        for path in ('/', '/blackjack', '/static/theme.css', '/static/common.js', '/static/common.css', '/lab'):
            self.assertEqual(self.request('GET', path)[0], 200, path)
        for path in ('/holdem', '/static/Server.py', '/static/..%2FServer.py', '/nothing'):
            self.assertEqual(self.request('GET', path)[0], 404, path)

    def test_api(self):
        status, body = self.request('GET', '/api/games')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), lobby())
        self.assertEqual(playable(), ['blackjack'])
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
        self.assertEqual(self.request('POST', '/api/holdem/new', {})[0], 404)
        self.assertEqual(self.request('GET', '/api/holdem/%s' % sid)[0], 404, 'a session belongs to its game.')

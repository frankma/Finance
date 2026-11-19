import json
import os
import re
import subprocess
import sys
import zipfile
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.CardGames.Play.Api import MemoryStore, Sessions, handle
from src.CardGames.Play.Registry import lobby
from src.CardGames.Play.Server import STATIC
from src.CardGames.Play.StaticSite import build, page

__author__ = 'frank.ma'

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
PAGES = ['lobby.html', 'rules.html', 'blackjack.html', 'holdem.html', 'guandan.html']


def read(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()


class TestPages(TestCase):
    def test_links_are_relative_to_the_site_root(self):
        # the same pages serve the local server at / and the static site under /<repository>/, so every link is
        # relative to the root <base> names: none may start with /, nor be a bare ?query or #fragment in markup
        for name in PAGES:
            html = read(os.path.join(STATIC, name))
            self.assertEqual(html.count('<base href="/">'), 1, name)
            self.assertEqual(re.findall(r'(?:href|src)="/[^"]*"', html.replace('<base href="/">', '')), [], name)
            self.assertEqual(re.findall(r'href="\?[^"]*"', html), [], name)
        for name in os.listdir(STATIC):
            if name.endswith(('.html', '.js')):
                text = read(os.path.join(STATIC, name))
                self.assertEqual(re.findall(r"""(?:fetch|api)\([^)]*['"`]/(?:api|static)""", text), [], name)

    def test_every_word_about_the_server_has_web_wording(self):
        # on the static site the game runs in the browser, so no page may claim the server keeps the cards
        for name in os.listdir(STATIC):
            if name.endswith('.html'):
                html = read(os.path.join(STATIC, name))
                plain = re.sub(r'<[^>]+data-static="[^"]*"[^>]*>.*?</[a-z]+>', '', html, flags=re.S)
                self.assertNotIn('server never sends', plain, name)
                self.assertNotIn("Python server", plain, name)

    def test_a_page_moves_under_the_site_base(self):
        html = page('<html lang="en">\n<head>\n<base href="/">', '/Finance/')
        self.assertIn('<base href="/Finance/">', html)
        self.assertIn('<html lang="en" data-mode="static">', html)
        with self.assertRaises(ValueError):
            page('<html lang="en"><head></head>', '/Finance/')


class TestBuild(TestCase):
    def test_the_site(self):
        with TemporaryDirectory() as folder:
            out = os.path.join(folder, 'site')
            build(out, '/Finance/')
            for path in ('index.html', 'blackjack/index.html', 'holdem/index.html', 'guandan/index.html',
                         'blackjack/rules/index.html', 'lab/index.html', 'static/engine.js', 'static/common.js',
                         'static/theme.css', 'static/rules-guandan.html', 'static/cardgames.zip', '.nojekyll'):
                self.assertTrue(os.path.exists(os.path.join(out, path)), path)
            table = read(os.path.join(out, 'holdem', 'index.html'))
            self.assertIn('<base href="/Finance/">', table)
            self.assertIn('data-mode="static"', table)
            self.assertEqual(json.loads(read(os.path.join(out, 'api', 'games'))), lobby(),
                             'the Lab reads the game list there, as from a running server.')
            with zipfile.ZipFile(os.path.join(out, 'static', 'cardgames.zip')) as z:
                names = set(z.namelist())
            for module in ('src/__init__.py', 'src/CardGames/Play/Api.py', 'src/CardGames/Holdem/Equity.py',
                           'src/CardGames/GuanDan/Strategy.py'):
                self.assertIn(module, names)
            self.assertFalse(any('__pycache__' in name or not name.endswith('.py') for name in names))
            build(out, '/')  # a second build replaces the first
            self.assertIn('<base href="/">', read(os.path.join(out, 'index.html')))
            with self.assertRaises(ValueError):
                build(out, 'Finance')

    def test_the_browser_side_needs_no_server(self):
        # the browser imports Api, not the HTTP server, and Pyodide has no use for sockets
        code = 'import sys, src.CardGames.Play.Api; print(sorted(m for m in ("http.server", "socketserver") ' \
               'if m in sys.modules))'
        done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(done.stdout.strip(), '[]', done.stderr)


class TestInTheBrowser(TestCase):
    def test_progress_is_handed_to_the_page_and_back(self):
        # as the web version runs: progress goes out through a function, to the browser's storage, and comes back in
        saved = []
        sessions = Sessions(store=MemoryStore(None, saved.append))
        status, made = handle(sessions, 'POST', 'api/blackjack/new', {'rules': 'h17'})
        self.assertEqual(status, 200)
        status, view = handle(sessions, 'POST', 'api/blackjack/%s/act' % made['id'], {'type': 'deal', 'bet': 25})
        self.assertEqual(status, 200)
        while view['phase'] == 'play':
            advice = handle(sessions, 'GET', 'api/blackjack/%s/advice' % made['id'])[1]
            view = handle(sessions, 'POST', 'api/blackjack/%s/act' % made['id'],
                          {'type': 'move', 'move': advice['move']})[1]
        self.assertTrue(saved)
        again = Sessions(store=MemoryStore(saved[-1]))
        status, back = handle(again, 'GET', 'api/blackjack/%s' % made['id'])
        self.assertEqual((status, back['bankroll'], back['stats']['rounds']), (200, view['bankroll'], 1))

    def test_routes(self):
        sessions = Sessions()
        self.assertEqual(handle(sessions, 'GET', '/api/games'), (200, lobby()))
        self.assertEqual(handle(sessions, 'GET', 'api/blackjack/nobody')[0], 404)
        self.assertEqual(handle(sessions, 'GET', 'api/nothing/at/all/here')[0], 404)
        sid = handle(sessions, 'POST', 'api/holdem/new', {'seats': 2})[1]['id']
        self.assertEqual(handle(sessions, 'POST', 'api/holdem/%s/act' % sid, {'type': 'fly'}),
                         (400, {'error': "unknown action 'fly'."}))
        self.assertEqual(handle(sessions, 'POST', 'api/holdem/new', {'seats': 'x'})[0], 400)
        with self.assertLogs('src.CardGames.Play.Api', 'WARNING'):
            sessions = Sessions(store=MemoryStore('not json'))
        self.assertEqual(sessions.saved, {}, 'a bad save in the browser is dropped, never fatal.')

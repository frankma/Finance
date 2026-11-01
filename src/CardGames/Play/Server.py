import json
import os
import random
import secrets
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from src.CardGames.Play.Registry import BY_NAME, lobby
from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
# the project's one copy of the paper theme, shared with viz/
VIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'viz')
THEME = os.path.join(VIZ, 'theme.css')
LAB = os.path.join(VIZ, 'card_games_lab.html')
TYPES = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
         '.js': 'text/javascript; charset=utf-8'}
MAX_SESSIONS = 200


class Sessions(object):
    """
    sessions in memory by id, oldest dropped first; one server process holds every table
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.items = OrderedDict()  # type: OrderedDict

    def create(self, game: str, options: dict) -> Tuple[str, Session]:
        entry = BY_NAME.get(game)
        if entry is None or entry.start is None:
            raise KeyError(game)
        with self.lock:
            session = entry.start(options, random.Random(self.rng.random()))
            sid = secrets.token_hex(8)
            self.items[sid] = (game, session)
            while len(self.items) > MAX_SESSIONS:
                self.items.popitem(last=False)
        return sid, session

    def get(self, game: str, sid: str) -> Session:
        with self.lock:
            found = self.items.get(sid)
        if found is None or found[0] != game:
            raise KeyError(sid)
        return found[1]


def _static_files() -> dict:
    files = {name: os.path.join(STATIC, name) for name in os.listdir(STATIC)}
    files['theme.css'] = THEME
    return files


class Handler(BaseHTTPRequestHandler):
    """
    pages: /, /<game>, /lab, /static/<file>; JSON: GET /api/games, POST /api/<game>/new,
    GET /api/<game>/<id>, POST /api/<game>/<id>/act, GET /api/<game>/<id>/advice
    """
    sessions = None  # type: Sessions
    quiet = True

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload).encode('utf-8'), 'application/json')

    def _file(self, path: str) -> None:
        with open(path, 'rb') as f:
            self._send(200, f.read(), TYPES.get(os.path.splitext(path)[1], 'application/octet-stream'))

    def _body(self) -> dict:
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return {}
        payload = json.loads(self.rfile.read(length).decode('utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('expect a JSON object.')
        return payload

    def _parts(self):
        return [p for p in self.path.split('?', 1)[0].split('/') if p]

    def do_GET(self):
        parts = self._parts()
        try:
            if not parts:
                return self._file(os.path.join(STATIC, 'lobby.html'))
            if parts == ['lab']:
                return self._file(LAB)
            if len(parts) == 2 and parts[0] == 'static':
                files = _static_files()  # only these names, so no path can reach outside
                if parts[1] in files:
                    return self._file(files[parts[1]])
            elif len(parts) == 1 and parts[0] in BY_NAME and BY_NAME[parts[0]].start is not None:
                return self._file(os.path.join(STATIC, BY_NAME[parts[0]].page))
            elif parts == ['api', 'games']:
                return self._json(200, lobby())
            elif len(parts) == 3 and parts[0] == 'api':
                session = self.sessions.get(parts[1], parts[2])
                with self.sessions.lock:
                    return self._json(200, session.view())
            elif len(parts) == 4 and parts[0] == 'api' and parts[3] == 'advice':
                session = self.sessions.get(parts[1], parts[2])
                with self.sessions.lock:
                    return self._json(200, session.advice())
            self._json(404, {'error': 'not found'})
        except KeyError:
            self._json(404, {'error': 'no such game or session; start a new one'})

    def do_POST(self):
        parts = self._parts()
        try:
            if len(parts) == 3 and parts[0] == 'api' and parts[2] == 'new':
                sid, session = self.sessions.create(parts[1], self._body())
                return self._json(200, {'id': sid, 'view': session.view()})
            if len(parts) == 4 and parts[0] == 'api' and parts[3] == 'act':
                session = self.sessions.get(parts[1], parts[2])
                with self.sessions.lock:
                    session.act(self._body())
                    view = session.view()
                return self._json(200, view)
            self._json(404, {'error': 'not found'})
        except KeyError:
            self._json(404, {'error': 'no such game or session; start a new one'})
        except (ValueError, TypeError) as e:
            self._json(400, {'error': str(e)})


def make_server(host: str = '127.0.0.1', port: int = 8765, seed: Optional[int] = None,
                quiet: bool = True) -> ThreadingHTTPServer:
    """
    a play server on host:port (port 0 picks a free one); local only by default
    """
    handler = type('PlayHandler', (Handler,), {'sessions': Sessions(seed), 'quiet': quiet})
    return ThreadingHTTPServer((host, port), handler)

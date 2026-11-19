import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from src.CardGames.Play.Api import Sessions, handle
from src.CardGames.Play.Registry import BY_NAME

__author__ = 'frank.ma'

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
# the project's one copy of the paper theme, shared with viz/
VIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'viz')
THEME = os.path.join(VIZ, 'theme.css')
LAB = os.path.join(VIZ, 'card_games_lab.html')
TYPES = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
         '.js': 'text/javascript; charset=utf-8'}
PROGRESS = os.path.join(VIZ, '..', '.card_games')  # git-ignored, next to the code; holds no cards
CARD_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BUILD_HEADER = 'X-Card-Games-Build'
SHUTDOWN_HEADER = 'X-Card-Games-Shutdown'


def build_id(root: str = CARD_GAMES) -> str:
    """
    a short fingerprint of the Python code under root, so a launcher can tell a server running older code from one
    running this code; pages and other static files are read from disk on every request, so they never go stale
    """
    digest = hashlib.sha1()
    for folder, dirs, files in sorted(os.walk(root)):
        dirs.sort()
        for name in sorted(f for f in files if f.endswith('.py')):
            path = os.path.join(folder, name)
            digest.update(os.path.relpath(path, root).replace(os.sep, '/').encode('utf-8'))
            with open(path, 'rb') as f:
                digest.update(f.read())
    return digest.hexdigest()[:12]


def progress_path(port: int) -> str:
    # one file per port, as a browser keeps its session ids per address
    return os.path.join(PROGRESS, 'progress-%i.json' % port)


def _static_files() -> dict:
    files = {name: os.path.join(STATIC, name) for name in os.listdir(STATIC)}
    files['theme.css'] = THEME
    return files


class Handler(BaseHTTPRequestHandler):
    """
    pages: /, /<game>, /<game>/rules, /lab, /static/<file>; JSON: GET /api/games, POST /api/<game>/new,
    GET /api/<game>/<id>, POST /api/<game>/<id>/act, GET /api/<game>/<id>/advice, and POST /api/shutdown for a
    launcher replacing this server with newer code

    every response carries the build of the code it runs; the shutdown needs a custom header, which a page on another
    site cannot send without the browser first asking this server's permission, which it never gives
    """
    sessions = None  # type: Sessions
    quiet = True
    build = ''

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str, shared: bool = False) -> None:
        self.send_response(status)
        self.send_header(BUILD_HEADER, self.build)
        if shared:
            # the read-only game list may be read by any page, e.g. the Lab opened as a file, to see the server is up
            self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload, shared: bool = False) -> None:
        self._send(status, json.dumps(payload).encode('utf-8'), 'application/json', shared)

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

    def _api(self, method: str, body: Optional[dict] = None) -> None:
        # the JSON routes, answered by the same code the web version runs in the browser
        status, payload = handle(self.sessions, method, self.path, body)
        self._json(status, payload, shared=self._parts() == ['api', 'games'])

    def do_GET(self):
        parts = self._parts()
        if parts[:1] == ['api']:
            return self._api('GET')
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
        elif len(parts) == 2 and parts[0] in BY_NAME and parts[1] == 'rules':
            return self._file(os.path.join(STATIC, 'rules.html'))
        self._json(404, {'error': 'not found'})

    def do_POST(self):
        parts = self._parts()
        if parts == ['api', 'shutdown']:
            return self._shutdown()
        try:
            body = self._body()
        except (ValueError, TypeError) as e:
            return self._json(400, {'error': str(e)})
        self._api('POST', body)

    def _shutdown(self) -> None:
        if self.headers.get(SHUTDOWN_HEADER) != self.build:
            return self._json(403, {'error': 'a shutdown needs the %s header naming this build.' % SHUTDOWN_HEADER})
        self._json(200, {'build': self.build})
        self.server.replaced = True
        # serve_forever is waiting on this thread's request to finish, so stop it from another thread
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def make_server(host: str = '127.0.0.1', port: int = 8765, seed: Optional[int] = None,
                quiet: bool = True, build: Optional[str] = None,
                progress: Optional[str] = None) -> ThreadingHTTPServer:
    """
    a play server on host:port (port 0 picks a free one); local only by default
    :param build: the build it reports, by default the fingerprint of the code it runs
    :param progress: a file to keep each session's progress in across restarts, by default none
    """
    handler = type('PlayHandler', (Handler,), {'sessions': Sessions(seed, progress), 'quiet': quiet,
                                               'build': build if build is not None else build_id()})
    server = ThreadingHTTPServer((host, port), handler)
    server.replaced = False  # set once a newer server has asked this one to stop
    return server

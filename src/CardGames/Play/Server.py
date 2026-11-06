import hashlib
import json
import logging
import os
import random
import secrets
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from src.CardGames.Play.Lesson import LessonSession
from src.CardGames.Play.Registry import BY_NAME, lessons_for, lobby
from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
# the project's one copy of the paper theme, shared with viz/
VIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'viz')
THEME = os.path.join(VIZ, 'theme.css')
LAB = os.path.join(VIZ, 'card_games_lab.html')
TYPES = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
         '.js': 'text/javascript; charset=utf-8'}
MAX_SESSIONS = 200
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


class Sessions(object):
    """
    sessions in memory by id, oldest dropped first; one server process holds every table

    with a path, each session's progress (Session.progress: numbers, never a card) is kept in that file, so a session
    asked for after the server restarted is rebuilt from it, between hands
    """

    def __init__(self, seed: Optional[int] = None, path: Optional[str] = None):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.items = OrderedDict()  # type: OrderedDict
        self.path = path
        self.saved = self._read()  # type: OrderedDict

    def _read(self) -> OrderedDict:
        if self.path is None or not os.path.exists(self.path):
            return OrderedDict()
        try:
            with open(self.path, encoding='utf-8') as f:
                records = json.load(f)
            if not isinstance(records, dict):
                raise ValueError('expect an object of saved sessions.')
            return OrderedDict(records)
        except (OSError, ValueError) as e:
            # never stop the games over a bad file, and never overwrite what might be recoverable from it
            logger.warning('could not read saved progress in %s: %s', self.path, e)
            try:
                os.replace(self.path, self.path + '.unreadable')
            except OSError:
                pass
            return OrderedDict()

    def _write(self) -> None:
        # to a side file first, so a crash mid-write cannot leave a half-written save
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            side = self.path + '.tmp'
            with open(side, 'w', encoding='utf-8') as f:
                json.dump(self.saved, f)
            os.replace(side, self.path)
        except OSError as e:
            logger.warning('could not save progress in %s: %s', self.path, e)

    def _record(self, game: str, sid: str, session: Session, options: Optional[dict] = None) -> None:
        # the caller holds the lock
        if self.path is None:
            return
        progress = session.progress()
        record = self.saved.get(sid)
        if record is not None and record['progress'] == progress:
            return
        self.saved[sid] = {'game': game, 'options': record['options'] if record else (options or {}),
                           'progress': progress}
        self.saved.move_to_end(sid)
        while len(self.saved) > MAX_SESSIONS:
            self.saved.popitem(last=False)
        self._write()

    def _start(self, game: str, options: dict) -> Session:
        entry = BY_NAME.get(game)
        if entry is None or entry.start is None:
            raise KeyError(game)
        rng = random.Random(self.rng.random())
        if options.get('mode') == 'learn':
            # guided practice: the game's lessons, each on a table set up for it
            lessons = lessons_for(entry, options)
            if lessons is None:
                raise ValueError('%s has no lessons yet.' % entry.title)
            return LessonSession(lessons(), rng)
        return entry.start(options, rng)

    def _keep(self, sid: str, game: str, session: Session) -> None:
        self.items[sid] = (game, session)
        while len(self.items) > MAX_SESSIONS:
            self.items.popitem(last=False)

    def create(self, game: str, options: dict) -> Tuple[str, Session]:
        with self.lock:
            session = self._start(game, options)
            sid = secrets.token_hex(8)
            self._keep(sid, game, session)
            self._record(game, sid, session, options)
        return sid, session

    def save(self, game: str, sid: str, session: Session) -> None:
        # after an action; the caller holds the lock
        self._record(game, sid, session)

    def _restore(self, game: str, sid: str) -> Optional[Session]:
        # the caller holds the lock
        record = self.saved.get(sid)
        if not isinstance(record, dict) or record.get('game') != game:
            return None
        try:
            session = self._start(game, record['options'])
            session.restore(record['progress'])
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning('dropping saved session %s: %r', sid, e)
            del self.saved[sid]
            return None
        self._keep(sid, game, session)
        return session

    def get(self, game: str, sid: str) -> Session:
        with self.lock:
            found = self.items.get(sid)
            if found is None:
                restored = self._restore(game, sid)
                if restored is not None:
                    return restored
        if found is None or found[0] != game:
            raise KeyError(sid)
        return found[1]


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
            elif len(parts) == 2 and parts[0] in BY_NAME and parts[1] == 'rules':
                return self._file(os.path.join(STATIC, 'rules.html'))
            elif parts == ['api', 'games']:
                return self._json(200, lobby(), shared=True)
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
        if parts == ['api', 'shutdown']:
            return self._shutdown()
        try:
            if len(parts) == 3 and parts[0] == 'api' and parts[2] == 'new':
                sid, session = self.sessions.create(parts[1], self._body())
                return self._json(200, {'id': sid, 'view': session.view()})
            if len(parts) == 4 and parts[0] == 'api' and parts[3] == 'act':
                session = self.sessions.get(parts[1], parts[2])
                with self.sessions.lock:
                    session.act(self._body())
                    view = session.view()
                    self.sessions.save(parts[1], parts[2], session)
                return self._json(200, view)
            self._json(404, {'error': 'not found'})
        except KeyError:
            self._json(404, {'error': 'no such game or session; start a new one'})
        except (ValueError, TypeError) as e:
            self._json(400, {'error': str(e)})


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

"""
the play server's game side without HTTP: the sessions and the JSON routes the pages call. The local server answers
these routes over the network (Server.py); the web version runs this same module in the browser, with Pyodide, so both
play the same Python
"""
import json
import logging
import os
import random
import secrets
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple

from src.CardGames.Play.Lesson import LessonSession
from src.CardGames.Play.Registry import BY_NAME, lessons_for, lobby
from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)

MAX_SESSIONS = 200


class FileStore(object):
    """
    where the local server keeps progress: a JSON file, written to a side file first so a crash mid-write cannot leave
    half of it, and set aside rather than overwritten when it cannot be read
    """

    def __init__(self, path: str):
        self.path = path

    def read(self) -> Optional[str]:
        if not os.path.exists(self.path):
            return None
        with open(self.path, encoding='utf-8') as f:
            return f.read()

    def write(self, text: str) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        side = self.path + '.tmp'
        with open(side, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(side, self.path)

    def set_aside(self) -> None:
        os.replace(self.path, self.path + '.unreadable')

    def __str__(self):
        return self.path


class MemoryStore(object):
    """
    where the web version keeps progress: the text the page read from the browser's storage, and a function that
    hands each new save back to the page
    """

    def __init__(self, text: Optional[str] = None, on_write: Optional[Callable[[str], Any]] = None):
        self.text = text
        self.on_write = on_write

    def read(self) -> Optional[str]:
        return self.text

    def write(self, text: str) -> None:
        self.text = text
        if self.on_write is not None:
            self.on_write(text)

    def set_aside(self) -> None:
        self.text = None

    def __str__(self):
        return 'the browser'


class Sessions(object):
    """
    sessions in memory by id, oldest dropped first; one server process holds every table

    with a store, each session's progress (Session.progress: numbers, never a card) is kept there, so a session asked
    for after the server restarted is rebuilt from it, between hands
    """

    def __init__(self, seed: Optional[int] = None, path: Optional[str] = None, store=None):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.items = OrderedDict()  # type: OrderedDict
        self.store = store if store is not None else FileStore(path) if path is not None else None
        self.saved = self._read()  # type: OrderedDict

    def _read(self) -> OrderedDict:
        if self.store is None:
            return OrderedDict()
        try:
            text = self.store.read()
            if text is None:
                return OrderedDict()
            records = json.loads(text)
            if not isinstance(records, dict):
                raise ValueError('expect an object of saved sessions.')
            return OrderedDict(records)
        except (OSError, ValueError) as e:
            # never stop the games over a bad file, and never overwrite what might be recoverable from it
            logger.warning('could not read saved progress in %s: %s', self.store, e)
            try:
                self.store.set_aside()
            except OSError:
                pass
            return OrderedDict()

    def _write(self) -> None:
        try:
            self.store.write(json.dumps(self.saved))
        except OSError as e:
            logger.warning('could not save progress in %s: %s', self.store, e)

    def _record(self, game: str, sid: str, session: Session, options: Optional[dict] = None) -> None:
        # the caller holds the lock
        if self.store is None:
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


def handle(sessions: Sessions, method: str, path: str, body: Optional[dict] = None) -> Tuple[int, Any]:
    """
    answer one JSON route: GET api/games, POST api/<game>/new, GET api/<game>/<id>, POST api/<game>/<id>/act and
    GET api/<game>/<id>/advice
    :return: the HTTP status and the JSON payload, an error message on 400 and 404
    """
    parts = [p for p in path.split('?', 1)[0].split('/') if p]
    if parts[:1] == ['api']:
        parts = parts[1:]
    try:
        if method == 'GET' and parts == ['games']:
            return 200, lobby()
        if method == 'GET' and len(parts) in (2, 3) and parts[2:] in ([], ['advice']):
            session = sessions.get(parts[0], parts[1])
            with sessions.lock:
                return 200, session.advice() if parts[2:] else session.view()
        if method == 'POST' and len(parts) == 2 and parts[1] == 'new':
            sid, session = sessions.create(parts[0], body or {})
            return 200, {'id': sid, 'view': session.view()}
        if method == 'POST' and len(parts) == 3 and parts[2] == 'act':
            session = sessions.get(parts[0], parts[1])
            with sessions.lock:
                session.act(body or {})
                view = session.view()
                sessions.save(parts[0], parts[1], session)
            return 200, view
        return 404, {'error': 'not found'}
    except KeyError:
        return 404, {'error': 'no such game or session; start a new one'}
    except (ValueError, TypeError) as e:
        return 400, {'error': str(e)}

"""
play the card games in your browser, against the computer and with a coach that sees only what your seat sees

    python -m src.CardGames.Play               # the lobby
    python -m src.CardGames.Play blackjack     # straight to a table
"""
import argparse
import errno
import json
import sys
import time
import urllib.request
import webbrowser
from http.server import ThreadingHTTPServer
from typing import List, Optional, Tuple

from src.CardGames.Play.Registry import lobby, playable
from src.CardGames.Play.Server import BUILD_HEADER, SHUTDOWN_HEADER, build_id, make_server

__author__ = 'frank.ma'


def probe(port: int) -> Optional[str]:
    """
    the build of the card games server on this port, '' for one from before servers reported their build, or None
    when the program there is not the card games server
    """
    try:
        with urllib.request.urlopen('http://127.0.0.1:%i/api/games' % port, timeout=2) as res:
            if [game['name'] for game in json.loads(res.read())] != [game['name'] for game in lobby()]:
                return None
            return res.headers.get(BUILD_HEADER) or ''
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _replace(port: int, running: str, start, wait: float = 5.0) -> Optional[ThreadingHTTPServer]:
    # ask the server running other code to stop, then take the port as soon as it is free
    request = urllib.request.Request('http://127.0.0.1:%i/api/shutdown' % port, data=b'', method='POST',
                                     headers={SHUTDOWN_HEADER: running})
    try:
        urllib.request.urlopen(request, timeout=2).close()
    except OSError:
        return None
    deadline = time.monotonic() + wait
    while True:
        try:
            return start()
        except OSError as e:
            if e.errno != errno.EADDRINUSE or time.monotonic() > deadline:
                return None
            time.sleep(0.1)


def claim(port: int, quiet: bool = True) -> Tuple[Optional[ThreadingHTTPServer], str]:
    """
    a server running this code on the port, and how it was got: 'started' on a free port, 'replaced' after asking a
    card games server running other code to stop; or None and why not: 'running' when this code is already serving
    there, 'old' for a card games server too old to be asked, 'stuck' when one asked to stop did not, 'busy' when
    another program holds the port
    """
    current = build_id()

    def start():
        return make_server(port=port, quiet=quiet, build=current)
    try:
        return start(), 'started'
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
    running = probe(port)
    if running is None:
        return None, 'busy'
    if running == current:
        return None, 'running'
    if not running:
        return None, 'old'
    server = _replace(port, running, start)
    return (server, 'replaced') if server is not None else (None, 'stuck')


MESSAGES = {
    'busy': 'Port {port} is in use by another program. Pick another one, e.g. --port {next}',
    'old': 'An older card games server is running on port {port}, from before servers could be replaced. Close its '
           'window or press Ctrl+C in it, then start again.',
    'stuck': 'The older card games server on port {port} did not stop. Close its window or press Ctrl+C in it, then '
             'start again.',
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('game', nargs='?', choices=playable(), help='open this table instead of the lobby')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true', help='do not open a browser window')
    parser.add_argument('--verbose', action='store_true', help='log every request')
    args = parser.parse_args(argv)
    server, how = claim(args.port, quiet=not args.verbose)
    url = 'http://127.0.0.1:%i/%s' % (server.server_address[1] if server else args.port, args.game or '')
    if how == 'running':
        # a server from an earlier run, on this same code, is still up: use it rather than start another
        print('Card games are already running at %s' % url)
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    if server is None:
        print(MESSAGES[how].format(port=args.port, next=args.port + 1), file=sys.stderr)
        return 1
    if how == 'replaced':
        print('Replaced an older card games server on port %i with this code.' % args.port)
    print('Card games at %s  (Ctrl+C to stop)' % url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    if server.replaced:
        print('A newer card games server took over this port, so this one has stopped. You can close this window.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
play the card games in your browser, against the computer and with a coach that sees only what your seat sees

    python -m src.CardGames.Play               # the lobby
    python -m src.CardGames.Play blackjack     # straight to a table
"""
import argparse
import errno
import json
import sys
import urllib.request
import webbrowser
from typing import List, Optional

from src.CardGames.Play.Registry import lobby, playable
from src.CardGames.Play.Server import make_server

__author__ = 'frank.ma'


def already_serving(port: int) -> bool:
    """
    whether the card games server is the program on this port, by asking it for its game list
    """
    try:
        with urllib.request.urlopen('http://127.0.0.1:%i/api/games' % port, timeout=2) as res:
            return [game['name'] for game in json.loads(res.read())] == [game['name'] for game in lobby()]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('game', nargs='?', choices=playable(), help='open this table instead of the lobby')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true', help='do not open a browser window')
    parser.add_argument('--verbose', action='store_true', help='log every request')
    args = parser.parse_args(argv)
    try:
        server = make_server(port=args.port, quiet=not args.verbose)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        url = 'http://127.0.0.1:%i/%s' % (args.port, args.game or '')
        if already_serving(args.port):
            # a server from an earlier run is still up: use it rather than fail
            print('Card games are already running at %s' % url)
            if not args.no_browser:
                webbrowser.open(url)
            return 0
        print('Port %i is in use by another program. Pick another one, e.g. --port %i' % (args.port, args.port + 1),
              file=sys.stderr)
        return 1
    url = 'http://127.0.0.1:%i/%s' % (server.server_address[1], args.game or '')
    print('Card games at %s  (Ctrl+C to stop)' % url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())

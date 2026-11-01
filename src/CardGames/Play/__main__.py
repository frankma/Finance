"""
play the card games in your browser, against the computer and with a coach that sees only what your seat sees

    python -m src.CardGames.Play               # the lobby
    python -m src.CardGames.Play blackjack     # straight to a table
"""
import argparse
import webbrowser

from src.CardGames.Play.Registry import playable
from src.CardGames.Play.Server import make_server

__author__ = 'frank.ma'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('game', nargs='?', choices=playable(), help='open this table instead of the lobby')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true', help='do not open a browser window')
    parser.add_argument('--verbose', action='store_true', help='log every request')
    args = parser.parse_args()
    server = make_server(port=args.port, quiet=not args.verbose)
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


if __name__ == '__main__':
    main()

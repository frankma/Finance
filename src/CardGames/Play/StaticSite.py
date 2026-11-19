"""
the card games as a static web site, e.g. for GitHub Pages: the same pages as the local server, with the same Python
run in the browser by Pyodide (static/engine.js), so play and lessons work with no server

    python -m src.CardGames.Play.StaticSite _site --base /Finance/

the base is the path the site is served under: /<repository>/ on GitHub Pages, / at the root of a host. Fair play
differs in one way: the whole game runs in the browser, so hidden cards sit in the page's memory, where a determined
player could find them. The coach and the computer seats still use only what their seat can see, and the pages say so
"""
import argparse
import json
import os
import shutil
import sys
import zipfile
from typing import List, Optional

from src.CardGames.Play.Registry import GAMES, lobby
from src.CardGames.Play.Server import LAB, STATIC, THEME

__author__ = 'frank.ma'

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')  # the src package
LOCAL_BASE = '<base href="/">'
HTML_TAG = '<html lang="en">'


def page(html: str, base: str) -> str:
    """
    a page of the local server as the static site serves it: under the site's base, and marked so its scripts run the
    game in the browser
    """
    if LOCAL_BASE not in html or HTML_TAG not in html:
        raise ValueError('expect a page with %s and %s.' % (LOCAL_BASE, HTML_TAG))
    return html.replace(LOCAL_BASE, '<base href="%s">' % base, 1).replace(
        HTML_TAG, '<html lang="en" data-mode="static">', 1)


def code_zip(path: str) -> List[str]:
    """
    the card games' Python, as the package src.CardGames, for the browser to import
    :return: the files packed
    """
    names = [os.path.join('src', '__init__.py')]
    for folder, dirs, files in os.walk(os.path.join(SRC, 'CardGames')):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        names += [os.path.relpath(os.path.join(folder, f), os.path.join(SRC, '..')) for f in sorted(files)
                  if f.endswith('.py')]
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(os.path.join(SRC, '..', name), name.replace(os.sep, '/'))
    return names


def build(out: str, base: str = '/') -> None:
    """
    write the site to out, replacing what was there
    """
    if not (base.startswith('/') and base.endswith('/')):
        raise ValueError('base %r should start and end with /, e.g. /Finance/.' % base)
    if os.path.exists(out):
        shutil.rmtree(out)

    def write(rel: str, text: str) -> None:
        path = os.path.join(out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    def read(path: str) -> str:
        with open(path, encoding='utf-8') as f:
            return f.read()

    # each address the local server answers, as a folder with an index page
    write('index.html', page(read(os.path.join(STATIC, 'lobby.html')), base))
    for game in GAMES:
        if game.start is not None:
            write(os.path.join(game.name, 'index.html'), page(read(os.path.join(STATIC, game.page)), base))
        write(os.path.join(game.name, 'rules', 'index.html'), page(read(os.path.join(STATIC, 'rules.html')), base))
    write(os.path.join('lab', 'index.html'), read(LAB))
    # the files under static/, the theme among them, and the game code for the browser
    os.makedirs(os.path.join(out, 'static'))
    for name in os.listdir(STATIC):
        shutil.copy(os.path.join(STATIC, name), os.path.join(out, 'static', name))
    shutil.copy(THEME, os.path.join(out, 'static', 'theme.css'))
    code_zip(os.path.join(out, 'static', 'cardgames.zip'))
    # the game list as a file, so the Lab sees the site's tables as it sees a running server's
    write(os.path.join('api', 'games'), json.dumps(lobby()))
    write('.nojekyll', '')  # serve the files as they are


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', help='the folder to write the site to, replaced if it exists')
    parser.add_argument('--base', default='/', help='the path the site is served under, e.g. /Finance/')
    args = parser.parse_args(argv)
    build(args.out, args.base)
    print('Card games site written to %s, to be served under %s' % (args.out, args.base))
    return 0


if __name__ == '__main__':
    sys.exit(main())

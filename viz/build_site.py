"""
build the GitHub Pages site: a front page at the root that links each project, and each project in a folder of its own

    python -m viz.build_site _site --base /Finance/

to add a page, add a Project to PROJECTS: a single self-contained page needs only its source file
"""
import argparse
import html
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.CardGames.Play import StaticSite

__author__ = 'frank.ma'

HERE = os.path.dirname(os.path.abspath(__file__))
THEME = os.path.join(HERE, 'theme.css')
TEMPLATE = os.path.join(HERE, 'site_index.template.html')


@dataclass(frozen=True)
class Project(object):
    slug: str  # its folder on the site, e.g. card-games
    area: str  # the label beside it on the front page
    title: str
    blurb: str
    build: Callable[[str, str, str], None]  # (folder to write, base it is served under, the site's front page)
    enter: str = 'Open'  # the front page's link to it


def single_page(source: str) -> Callable[[str, str, str], None]:
    """
    a project that is one self-contained page, served as its folder's index
    """
    def build(out: str, base: str, home: str) -> None:
        os.makedirs(out, exist_ok=True)
        shutil.copy(source, os.path.join(out, 'index.html'))
    return build


def card_games(out: str, base: str, home: str) -> None:
    StaticSite.build(out, base, home)


PROJECTS = [
    Project('card-games', 'Games', 'Card games',
            'Blackjack, Texas hold\'em, Omaha and GuanDan against the computer, with a coach that uses only what '
            'your seat can see. Guided lessons, the rules of each game, and a lab of the simulations behind the coach.',
            card_games, 'Play'),
    Project('betting-games', 'Games', 'Betting games',
            'A thousand players, a hundred flips of a coin that pays +50% or -40%. Betting everything makes the '
            'average rich and almost everyone poor; a Kelly stake, or pooling the luck, turns the same coin into '
            'growth.', single_page(os.path.join(HERE, 'betting_games_lab.html')), 'Open'),
    Project('sabr-smile', 'Volatility', 'SABR smile lab',
            'Move the SABR parameters and watch Hagan\'s smile, the implied density, the backbone and the vol surface '
            'respond.', single_page(os.path.join(HERE, 'sabr_smile_lab.html'))),
]


def front_page(projects: List[Project]) -> str:
    with open(TEMPLATE, encoding='utf-8') as f:
        page = f.read()
    with open(THEME, encoding='utf-8') as f:
        theme = f.read()
    items = '\n'.join(
        '    <li class="project"><span class="area">%s</span><div><h2><a href="%s/">%s</a></h2><p>%s</p></div>'
        '<a class="enter" href="%s/" aria-label="%s %s">%s</a></li>'
        % (html.escape(p.area), p.slug, html.escape(p.title), html.escape(p.blurb), p.slug, p.enter,
           html.escape(p.title), p.enter) for p in projects)
    return page.replace('/*__THEME__*/', theme).replace('<!--__PROJECTS__-->', items)


# every page of every project carries the same way home: a small "All projects" at the top right of its column
WRAP = '<div class="wrap">'
HOME_LINK = ('<nav class="site-home" aria-label="Site" style="display: flex; justify-content: flex-end; '
             'margin: -10px 0 4px; font: 400 13px/1.5 var(--sans);">'
             '<a href="%s" style="color: var(--muted);">All projects</a></nav>')


def add_home_link(page: str, home: str) -> str:
    """
    the page with the site's link home as the first thing in its column; every page of the site has one column,
    <div class="wrap">, so the link sits at the column's top right however wide the page is
    """
    if page.count(WRAP) != 1:
        raise ValueError('expect exactly one %s to place the link home in.' % WRAP)
    return page.replace(WRAP, WRAP + '\n  ' + HOME_LINK % html.escape(home, quote=True), 1)


def link_home(folder: str, home: str) -> List[str]:
    """
    add the link home to every page a project wrote: each index.html under its folder (other .html files are
    fragments and copies that are never opened as pages)
    :return: the pages changed, relative to the folder
    """
    changed = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        if 'index.html' in files:
            path = os.path.join(root, 'index.html')
            with open(path, encoding='utf-8') as f:
                text = f.read()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(add_home_link(text, home))
            changed.append(os.path.relpath(path, folder))
    return changed


def build(out: str, base: str = '/', projects: Optional[List[Project]] = None) -> None:
    """
    write the whole site to out, replacing what was there
    :param base: the path the site is served under: /<repository>/ on GitHub Pages
    """
    if not (base.startswith('/') and base.endswith('/')):
        raise ValueError('base %r should start and end with /, e.g. /Finance/.' % base)
    projects = PROJECTS if projects is None else projects
    slugs = [p.slug for p in projects]
    if len(set(slugs)) != len(slugs) or not all(s and s == s.lower() and '/' not in s for s in slugs):
        raise ValueError('project folders should be unique lower-case names, received %s.' % slugs)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for p in projects:
        p.build(os.path.join(out, p.slug), base + p.slug + '/', base)
        link_home(os.path.join(out, p.slug), base)
    with open(os.path.join(out, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(front_page(projects))
    open(os.path.join(out, '.nojekyll'), 'w').close()  # serve the files as they are


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', help='the folder to write the site to, replaced if it exists')
    parser.add_argument('--base', default='/', help='the path the site is served under, e.g. /Finance/')
    args = parser.parse_args(argv)
    build(args.out, args.base)
    print('Site written to %s, to be served under %s: %s' % (args.out, args.base,
                                                            ', '.join(p.slug for p in PROJECTS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

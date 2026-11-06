import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.CardGames.Play import BlackJackLessons, GuanDanLessons, HoldemLessons
from src.CardGames.Play.BlackJackSession import PRESETS, BlackJackSession, describe
from src.CardGames.Play.GuanDanSession import GuanDanSession
from src.CardGames.Play.HoldemSession import SEATS, VARIANTS, HoldemSession
from src.CardGames.Play.Lesson import Lesson
from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'


@dataclass(frozen=True)
class GameEntry(object):
    name: str  # in the URL: /blackjack, /api/blackjack/...
    title: str
    blurb: str
    page: str  # the table page under static/
    start: Optional[Callable[[dict, random.Random], Session]]  # None until the game can be played
    options: tuple = ()  # (key, label, choices as (value, label) pairs) for the lobby
    lessons: Optional[Callable[[], List[Lesson]]] = None  # guided practice hands, played with mode=learn
    variant_lessons: tuple = ()  # (variant, title, lessons) for a variant with guided hands of its own


GAMES = [
    GameEntry('blackjack', 'Blackjack',
              'You against the dealer from a six-deck shoe. The coach plays basic strategy and sizes bets on the count.',
              'blackjack.html', BlackJackSession,
              (('rules', 'Rules', tuple((key, describe(rules)) for key, rules in PRESETS.items())),),
              BlackJackLessons.lessons),
    GameEntry('holdem', "Hold'em", "No-limit Texas hold'em or Omaha against computer seats. The coach weighs your "
              'equity against the pot odds, from your cards and the public play only.',
              'holdem.html', HoldemSession, (('variant', 'Game', VARIANTS),
                                             ('seats', 'Table', tuple((str(n), label) for n, label in SEATS))),
              HoldemLessons.lessons, (('omaha', 'Omaha', HoldemLessons.omaha_lessons),)),
    GameEntry('guandan', 'GuanDan', 'The partnership climbing game from 2 to A, with a computer partner against two '
              'computer opponents. The coach plans your hand from your cards and the public play.',
              'guandan.html', GuanDanSession, (), GuanDanLessons.lessons),
]
BY_NAME = {game.name: game for game in GAMES}  # type: Dict[str, GameEntry]


def playable() -> List[str]:
    return [game.name for game in GAMES if game.start is not None]


def lessons_for(game: GameEntry, options: dict) -> Optional[Callable[[], List[Lesson]]]:
    for variant, _, lessons in game.variant_lessons:
        if options.get('variant') == variant:
            return lessons
    return game.lessons


def _practice(game: GameEntry) -> List[dict]:
    # the links to a game's guided hands; only a game with variants needs to say which one each link is for
    if game.lessons is None:
        return []
    out = [{'label': 'Practice %s' % game.title if game.variant_lessons else 'Practice', 'query': 'mode=learn',
            'n': len(game.lessons())}]
    for variant, title, lessons in game.variant_lessons:
        out.append({'label': 'Practice %s' % title, 'query': 'mode=learn&variant=%s' % variant, 'n': len(lessons())})
    return out


def lobby() -> List[dict]:
    return [{'name': g.name, 'title': g.title, 'blurb': g.blurb, 'available': g.start is not None,
             'lessons': len(g.lessons()) if g.lessons else 0, 'practice': _practice(g),
             'options': [{'key': k, 'label': label, 'choices': [{'value': v, 'label': t} for v, t in choices]}
                         for k, label, choices in g.options]} for g in GAMES]

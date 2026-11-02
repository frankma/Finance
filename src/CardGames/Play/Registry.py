import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.CardGames.Play import BlackJackLessons, HoldemLessons
from src.CardGames.Play.BlackJackSession import PRESETS, BlackJackSession, describe
from src.CardGames.Play.GuanDanSession import GuanDanSession
from src.CardGames.Play.HoldemSession import SEATS, HoldemSession
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


GAMES = [
    GameEntry('blackjack', 'Blackjack',
              'You against the dealer from a six-deck shoe. The coach plays basic strategy and sizes bets on the count.',
              'blackjack.html', BlackJackSession,
              (('rules', 'Rules', tuple((key, describe(rules)) for key, rules in PRESETS.items())),),
              BlackJackLessons.lessons),
    GameEntry('holdem', "Hold'em", "No-limit Texas hold'em against computer seats. The coach weighs your equity "
              'against the pot odds, from your cards and the board only.',
              'holdem.html', HoldemSession, (('seats', 'Table', tuple((str(n), label) for n, label in SEATS)),),
              HoldemLessons.lessons),
    GameEntry('guandan', 'GuanDan', 'The partnership climbing game from 2 to A, with a computer partner against two '
              'computer opponents. The coach plans your hand from your cards and the public play.',
              'guandan.html', GuanDanSession),
]
BY_NAME = {game.name: game for game in GAMES}  # type: Dict[str, GameEntry]


def playable() -> List[str]:
    return [game.name for game in GAMES if game.start is not None]


def lobby() -> List[dict]:
    return [{'name': g.name, 'title': g.title, 'blurb': g.blurb, 'available': g.start is not None,
             'lessons': len(g.lessons()) if g.lessons else 0,
             'options': [{'key': k, 'label': label, 'choices': [{'value': v, 'label': t} for v, t in choices]}
                         for k, label, choices in g.options]} for g in GAMES]

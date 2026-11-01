import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.CardGames.Play.BlackJackSession import PRESETS, BlackJackSession
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


GAMES = [
    GameEntry('blackjack', 'Blackjack',
              'You against the dealer from a six-deck shoe. The coach plays basic strategy and sizes bets on the count.',
              'blackjack.html', BlackJackSession,
              (('rules', 'Rules', tuple((key, label) for key, (label, _) in PRESETS.items())),)),
    GameEntry('holdem', "Hold'em", 'No-limit Texas hold\'em against computer seats, with equity and pot odds.',
              'holdem.html', None),
    GameEntry('guandan', 'GuanDan', 'The partnership climbing game, with a computer partner and two opponents.',
              'guandan.html', None),
]
BY_NAME = {game.name: game for game in GAMES}  # type: Dict[str, GameEntry]


def playable() -> List[str]:
    return [game.name for game in GAMES if game.start is not None]


def lobby() -> List[dict]:
    return [{'name': g.name, 'title': g.title, 'blurb': g.blurb, 'available': g.start is not None,
             'options': [{'key': k, 'label': label, 'choices': [{'value': v, 'label': t} for v, t in choices]}
                         for k, label, choices in g.options]} for g in GAMES]

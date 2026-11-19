import json
import random
import threading
import urllib.error
import urllib.request
from typing import List, Optional

from src.CardGames.Core.Card import Card
from src.CardGames.Core.Shoe import Shoe
from src.CardGames.Play.Server import make_server

__author__ = 'frank.ma'

# shared by the play tests: a server on a free port, cards stacked at a table, and playing the coach's moves


# ---- a play server ----

class Served(object):
    """
    a play server on a free port, serving in the background until stop; build '' stands for a server from before
    servers reported their build, and progress is the file it keeps sessions in
    """

    def __init__(self, build: Optional[str] = None, progress: Optional[str] = None, seed: Optional[int] = None):
        self.server = make_server(port=0, build=build, progress=progress, seed=seed)
        self.port = self.server.server_address[1]
        self.base = 'http://127.0.0.1:%i' % self.port

        def run():
            # as main does: serve until stopped, then free the port
            self.server.serve_forever()
            self.server.server_close()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def request(self, method: str, path: str, body=None):
        """
        (status, body, headers), the body read as JSON when it is JSON
        """
        data = json.dumps(body).encode('utf-8') if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as res:
                status, raw, headers = res.status, res.read(), res.headers
        except urllib.error.HTTPError as e:
            status, raw, headers = e.code, e.read(), e.headers
        if headers.get('Content-Type') == 'application/json':
            raw = json.loads(raw)
        return status, raw, headers

    def call(self, method: str, path: str, body=None):
        # the JSON of a request that must succeed
        status, data, _ = self.request(method, path, body)
        if status != 200:
            raise AssertionError('%s %s answered %i: %r' % (method, path, status, data))
        return data

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


# ---- stacked cards ----

def stack_blackjack(session, order: str) -> None:
    # deal these cards next, in order: player, dealer up, player, dealer hole, then any draws
    cards = Card.parse_many(order)
    rest = list(session.table.shoe.cards)
    for card in cards:
        rest.remove(card)
    session.table.shoe.cards = rest + list(reversed(cards))


def stack_holdem(session, order: str) -> None:
    # the next hand deals these cards first, in order: hole cards round the table from the seat after the button,
    # then burn, flop, burn, turn, burn, river
    cards = Card.parse_many(order)

    def shoe():
        made = Shoe(1, rng=random.Random(0))
        rest = [card for card in made.cards if card not in cards]
        made.cards = rest + list(reversed(cards))
        return made
    session._shoe = shoe


def stack_guandan(session, hands, leader: int = 0) -> None:
    # the next deal hands out these hands, and the first hand is led by this seat
    session._deal_hands = lambda: [list(hand) for hand in hands]
    session.match.rng.randrange = lambda n: leader


def give(hands, codes: str, to: int = 0):
    # GuanDan: move these cards into one seat's hand, swapping them for cards it had, so every hand keeps 27
    hands = [list(hand) for hand in hands]
    for card in Card.parse_many(codes):
        holder = next(s for s in range(4) if s != to and card in hands[s])
        swap = next(c for c in hands[to] if c not in Card.parse_many(codes))
        hands[holder].remove(card)
        hands[holder].append(swap)
        hands[to].remove(swap)
        hands[to].append(card)
    return hands


# ---- the coach's moves ----

def coach_actions(view: dict, advice: dict) -> List[dict]:
    """
    the actions that make the coach's move, for any game: a GuanDan play that reads several ways needs a second
    action naming the reading, which play_coach_move sends once the table asks for it
    """
    kind = advice['kind']
    if kind == 'bet':
        return [{'type': 'deal', 'bet': advice['bet']}]
    if kind == 'return':
        return [{'type': 'return', 'card': advice['card']}]
    move = advice['move']
    if view['game'] == 'guandan':
        return [{'type': 'pass'} if move['kind'] == 'pass' else {'type': 'play', 'cards': move['cards']}]
    return [{'type': 'move', 'move': move, 'to': advice.get('to')}]


def play_coach_move(act, view: dict, advice: dict) -> dict:
    """
    make the coach's move through act (a session's act, or a request to the server) and return the view after it
    """
    action = coach_actions(view, advice)[0]
    view = act(action)
    if view.get('choices') and action.get('cards'):
        view = act(dict(action, reading=view['choices'].index(advice['move']['desc'])))
    return view


def session_act(session):
    # act on a session in memory and get its view back, as the server would answer
    def act(action):
        session.act(action)
        return session.view()
    return act


def coach_move(session) -> None:
    # make the coach's move at a session in memory
    play_coach_move(session_act(session), session.view(), session.advice())


def play_hand(session) -> None:
    """
    deal the next hand or round and play it out the coach's way; a busted hold'em stack buys back in first
    """
    view = session.view()
    if view.get('can_rebuy'):
        session.act({'type': 'rebuy'})
    act = session_act(session)
    if view['game'] == 'blackjack':
        view = play_coach_move(act, view, session.advice())
    else:
        view = act({'type': 'deal'})
    while view['phase'] in ('play', 'return'):
        view = play_coach_move(act, view, session.advice())

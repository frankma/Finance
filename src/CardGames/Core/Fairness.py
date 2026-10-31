from typing import Callable

from src.CardGames.Core.Game import Game
from src.CardGames.Core.Player import Strategy

__author__ = 'frank.ma'


def assert_fair_play(game_a: Game, game_b: Game, seat: int, make_strategy: Callable[[], Strategy]) -> None:
    """
    check the fair-play rule on two games that agree on everything the seat can know and differ only in cards it
    cannot see (another seat's hand, a face-down card, the undealt order): the seat must get equal observations and
    legal moves, and a strategy must choose the same move in both, so no assistant can be using the hidden cards
    :param make_strategy: builds a fresh strategy, seeded alike each time if it is random
    :raises AssertionError: naming what differs
    """
    if game_a.current_seat() != seat or game_b.current_seat() != seat:
        raise AssertionError('seat %i should be the one to act in both games.' % seat)
    view_a, view_b = game_a.observation(seat), game_b.observation(seat)
    try:
        hash(view_a)
    except TypeError:
        raise AssertionError('the observation %r is not an immutable snapshot.' % type(view_a).__name__)
    if view_a != view_b:
        raise AssertionError('the observations differ although only hidden cards do:\n%r\n%r' % (view_a, view_b))
    legal_a, legal_b = game_a.legal_moves(seat), game_b.legal_moves(seat)
    if legal_a != legal_b:
        raise AssertionError('the legal moves differ although only hidden cards do.')
    move_a = make_strategy().decide(view_a, legal_a)
    move_b = make_strategy().decide(view_b, legal_b)
    if move_a != move_b:
        raise AssertionError('the strategy chose %r and %r from the same view.' % (move_a, move_b))

from abc import ABC, abstractmethod
from typing import Any, List, Optional

__author__ = 'frank.ma'


class Game(ABC):
    """
    turn-based card game with hidden information: one seat acts at a time, choosing among legal moves using only its
    own observation; each game supplies its rules by implementing these methods
    """

    @property
    @abstractmethod
    def n_seats(self) -> int:
        pass

    @abstractmethod
    def current_seat(self) -> Optional[int]:
        """
        the seat to act, None once the game is over
        """
        pass

    @abstractmethod
    def legal_moves(self, seat: int) -> List[Any]:
        pass

    def is_legal(self, seat: int, move: Any) -> bool:
        """
        whether the seat may make the move; a game whose moves take a size from a range (a raise) overrides this, and
        its legal_moves lists representative sizes only
        """
        return move in self.legal_moves(seat)

    @abstractmethod
    def apply(self, seat: int, move: Any) -> None:
        pass

    @abstractmethod
    def observation(self, seat: int) -> Any:
        """
        what the seat may see: its own cards and the public ones, never another seat's hidden cards
        """
        pass

    @abstractmethod
    def payoffs(self) -> List[float]:
        """
        net result per seat, defined once the game is over
        """
        pass

    def is_terminal(self) -> bool:
        return self.current_seat() is None


def play_out(game: Game, strategies: list) -> List[float]:
    """
    drive a game to the end with one strategy per seat
    :param game: a game in progress
    :param strategies: Strategy for each seat, indexed by seat
    :return: payoffs per seat
    """
    if len(strategies) != game.n_seats:
        raise ValueError('expect %i strategies, received %i.' % (game.n_seats, len(strategies)))
    while not game.is_terminal():
        seat = game.current_seat()
        legal = game.legal_moves(seat)
        move = strategies[seat].decide(game.observation(seat), legal)
        if not game.is_legal(seat, move):
            raise ValueError('seat %i chose %r, not among the legal moves %r.' % (seat, move, legal))
        game.apply(seat, move)
    return game.payoffs()

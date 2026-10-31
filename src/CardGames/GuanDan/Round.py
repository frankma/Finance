import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.CardGames.Core.Card import Card, Rank
from src.CardGames.Core.Game import Game
from src.CardGames.Core.Shoe import Shoe
from src.CardGames.GuanDan.Combo import Combo, generate_moves, interpret, sort_hand

__author__ = 'frank.ma'

N_SEATS = 4
HAND_SIZE = 27
PASS = None  # the move that passes on a trick; a seat leading a trick may not pass
LEVELS_UP = {1: 3, 2: 2, 3: 1}  # partner of the first out finishing 2nd, 3rd or 4th


def team(seat: int) -> int:
    # partners sit opposite: seats 0 and 2 against 1 and 3
    return seat % 2


def partner(seat: int) -> int:
    return (seat + 2) % N_SEATS


def deal(rng: Optional[random.Random] = None) -> List[List[Card]]:
    # two packs with their four jokers, 27 cards to each seat
    shoe = Shoe(2, jokers=True, rng=rng)
    return [shoe.draw_many(HAND_SIZE, face_up=False) for _ in range(N_SEATS)]


@dataclass(frozen=True)
class Observation(object):
    """
    what a seat knows: its own hand and every public fact (the trick to beat, how many cards each seat holds, every
    card played and who played or passed what, who is out), never the cards in another seat's hand
    """
    seat: int
    hand: Tuple[Card, ...]
    level: Rank
    target: Optional[Combo]  # the combination to beat, None when leading
    target_seat: Optional[int]
    counts: Tuple[int, ...]  # cards left in each seat's hand
    played: Tuple[Card, ...]  # every card played so far this hand
    history: Tuple[Tuple[int, Optional[Combo]], ...] = ()  # (seat, combination or None for a pass), in order
    out: Tuple[int, ...] = ()  # seats out so far, in finishing order


@dataclass(frozen=True)
class HandResult(object):
    order: Tuple[int, ...]  # finishing order, the seats still holding cards after the ones out, in seat order
    winning_team: int
    levels_up: int


class Round(Game):
    """
    one hand: the leader plays any combination, then each seat in turn beats it or passes; when everyone else still
    holding cards passes, the last player wins the trick and leads, and if they are already out their partner leads
    (catching the wind), else the next seat with cards; the hand ends once both players of a team are out
    """

    def __init__(self, hands: List[List[Card]], level: Rank, leader: int):
        if len(hands) != N_SEATS:
            raise ValueError('expect %i hands, received %i.' % (N_SEATS, len(hands)))
        self.hands = [list(hand) for hand in hands]
        self.level = level
        self.played = []  # type: List[Card]
        self.target = None  # type: Optional[Combo]
        self.target_seat = None  # type: Optional[int]
        self.passes = 0
        self.out = []  # type: List[int]
        self.to_act = leader
        self.result = None  # type: Optional[HandResult]
        self.history = []  # type: List[Tuple[int, Optional[Combo]]]

    @property
    def n_seats(self) -> int:
        return N_SEATS

    def current_seat(self) -> Optional[int]:
        return None if self.result is not None else self.to_act

    def legal_moves(self, seat: int) -> List[Optional[Combo]]:
        if self.result is not None or seat != self.to_act:
            return []
        moves = generate_moves(self.hands[seat], self.level)
        if self.target is None:
            return moves
        return [PASS] + [move for move in moves if move.beats(self.target)]

    def is_legal(self, seat: int, move: Optional[Combo]) -> bool:
        # any valid reading of cards held is legal, not only the card choices legal_moves lists
        if self.result is not None or seat != self.to_act:
            return False
        if move is PASS:
            return self.target is not None
        if Counter(move.cards) - Counter(self.hands[seat]):
            return False
        readings = interpret(list(move.cards), self.level)
        if not any((r.type, r.value, r.rank) == (move.type, move.value, move.rank) for r in readings):
            return False
        return move.beats(self.target)

    def apply(self, seat: int, move: Optional[Combo]) -> None:
        if not self.is_legal(seat, move):
            raise ValueError('%s is not a legal move for seat %i now.' % (move, seat))
        self.history.append((seat, move))
        if move is PASS:
            self.passes += 1
        else:
            for card in move.cards:
                self.hands[seat].remove(card)
            self.played += move.cards
            self.target, self.target_seat, self.passes = move, seat, 0
            if not self.hands[seat]:
                self.out.append(seat)
                if self._team_out(team(seat)):
                    self._finish()
                    return
        others = [s for s in range(N_SEATS) if self.hands[s] and s != self.target_seat]
        if self.passes >= len(others):
            winner = self.target_seat
            if self.hands[winner]:
                self.to_act = winner
            elif self.hands[partner(winner)]:
                self.to_act = partner(winner)  # catching the wind
            else:
                self.to_act = self._next_with_cards(winner)
            self.target, self.target_seat, self.passes = None, None, 0
        else:
            self.to_act = self._next_with_cards(seat)

    def _next_with_cards(self, seat: int) -> int:
        for k in range(1, N_SEATS + 1):
            s = (seat + k) % N_SEATS
            if self.hands[s]:
                return s
        return seat

    def _team_out(self, t: int) -> bool:
        return sum(team(s) == t for s in self.out) == 2

    def _finish(self) -> None:
        order = tuple(self.out + [s for s in range(N_SEATS) if s not in self.out])
        first = order[0]
        self.result = HandResult(order, team(first), LEVELS_UP[order.index(partner(first))])
        self.to_act = None

    def observation(self, seat: int) -> Observation:
        return Observation(seat, tuple(sort_hand(self.hands[seat], self.level)), self.level, self.target,
                           self.target_seat, tuple(len(hand) for hand in self.hands), tuple(self.played),
                           tuple(self.history), tuple(self.out))

    def payoffs(self) -> List[float]:
        if self.result is None:
            raise ValueError('the hand is not over yet.')
        up = float(self.result.levels_up)
        return [up if team(s) == self.result.winning_team else -up for s in range(N_SEATS)]

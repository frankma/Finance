import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from src.CardGames.Core.Card import Card
from src.CardGames.Core.Game import Game
from src.CardGames.Core.Shoe import Shoe
from src.CardGames.Holdem.HandEvaluator import HandRank
from src.CardGames.Holdem.Rules import Rules

__author__ = 'frank.ma'


class Action(Enum):
    FOLD = 'fold'
    CHECK = 'check'
    CALL = 'call'
    RAISE = 'raise'  # a bet when nothing has been bet on this street


class Street(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass(frozen=True)
class Move(object):
    action: Action
    to: int = 0  # for a call or raise, the seat's total bet on this street after the move


@dataclass(frozen=True)
class Observation(object):
    seat: int
    hole: Tuple[Card, ...]
    board: Tuple[Card, ...]
    street: Street
    pot: int  # every chip put in this hand, the current street included
    to_call: int  # chips needed to call, capped at the stack
    bet: int  # this seat's bet on this street
    current_bet: int  # the bet to match on this street
    stack: int
    min_raise_to: Optional[int]  # None when the seat may not raise
    max_raise_to: Optional[int]
    n_opponents: int  # opponents still in the hand
    big_blind: int
    rules: Rules


@dataclass(frozen=True)
class Pot(object):
    amount: int
    eligible: Tuple[int, ...]
    winners: Tuple[int, ...]


class _Seat(object):
    def __init__(self, stack: int):
        self.stack = stack
        self.start = stack
        self.hole = []  # type: List[Card]
        self.bet = 0  # this street
        self.total = 0  # this hand
        self.out = stack <= 0  # sitting out with no chips
        self.folded = self.out
        self.all_in = False
        self.acted = False
        self.faced = 0  # full raises seen when it last acted

    @property
    def in_hand(self) -> bool:
        return not self.folded

    @property
    def can_act(self) -> bool:
        return not self.folded and not self.all_in


class Round(Game):
    """
    one no-limit hand, dealt as soon as it is created and played through Game: legal_moves, apply, observation

    a full raise reopens the betting while a short all-in only asks the others to call the extra, the button posts
    the small blind heads-up, and the showdown splits side pots by how much each player put in, with the odd chip
    going to the first winner left of the button
    """

    def __init__(self, stacks: List[int], rules: Rules = Rules(), button: int = 0, rng: Optional[random.Random] = None,
                 shoe: Optional[Shoe] = None):
        self.rules = rules
        self.seats = [_Seat(stack) for stack in stacks]
        if sum(not seat.out for seat in self.seats) < 2:
            raise ValueError('need at least two seats with chips.')
        if len(self.seats) > rules.max_seats():
            raise ValueError('%i seats with %i hole cards each do not fit in one deck.' % (len(stacks), rules.n_hole))
        self.shoe = shoe if shoe is not None else Shoe(1, rng=rng)
        self.button = button if not self.seats[button].out else self._next(button, lambda s: not s.out)
        self.board = []  # type: List[Card]
        self.street = Street.PREFLOP
        self.current_bet = 0
        self.min_raise = rules.big_blind
        self.full_raises = 0
        self.to_act = None  # type: Optional[int]
        self.finished = False
        self.pots = []  # type: List[Pot]
        self.ranks = {}  # type: dict  # seat -> HandRank at showdown

        seated = lambda s: not s.out
        for _ in range(rules.n_hole):
            i = self.button
            for _ in range(sum(seated(seat) for seat in self.seats)):
                i = self._next(i, seated)
                self.seats[i].hole.append(self.shoe.draw(face_up=False))
        heads_up = sum(seated(seat) for seat in self.seats) == 2
        self.small_blind_seat = self.button if heads_up else self._next(self.button, seated)
        self.big_blind_seat = self._next(self.small_blind_seat, seated)
        self._post(self.small_blind_seat, rules.small_blind)
        self._post(self.big_blind_seat, rules.big_blind)
        self.current_bet = rules.big_blind
        self.to_act = self.big_blind_seat
        self._advance()

    # ---- seat helpers ----

    def _next(self, i: int, predicate) -> Optional[int]:
        n = len(self.seats)
        for k in range(1, n + 1):
            j = (i + k) % n
            if predicate(self.seats[j]):
                return j
        return None

    def _post(self, i: int, amount: int) -> int:
        seat = self.seats[i]
        amount = min(amount, seat.stack)
        seat.stack -= amount
        seat.bet += amount
        seat.total += amount
        if seat.stack == 0:
            seat.all_in = True
        return amount

    def _needs_to_act(self, seat: _Seat) -> bool:
        return seat.can_act and (not seat.acted or seat.bet < self.current_bet)

    @property
    def pot(self) -> int:
        return sum(seat.total for seat in self.seats)

    # ---- Game ----

    @property
    def n_seats(self) -> int:
        return len(self.seats)

    def current_seat(self) -> Optional[int]:
        return None if self.finished else self.to_act

    def raise_range(self, i: int) -> Optional[Tuple[int, int]]:
        """
        smallest and largest legal raise-to amounts, None when the seat may not raise: it cannot cover more than the
        call, nobody else is left to call, or it already acted and only a short all-in has come since
        """
        seat = self.seats[i]
        to_call = max(0, self.current_bet - seat.bet)
        others = sum(other.can_act for j, other in enumerate(self.seats) if j != i)
        reopened = not seat.acted or seat.faced < self.full_raises
        if seat.stack <= to_call or others == 0 or not reopened:
            return None
        max_to = seat.bet + seat.stack
        return min(max_to, self.current_bet + self.min_raise), max_to

    def legal_moves(self, seat: int) -> List[Move]:
        if self.finished or seat != self.to_act:
            return []
        s = self.seats[seat]
        to_call = max(0, self.current_bet - s.bet)
        moves = [Move(Action.FOLD), Move(Action.CALL, s.bet + min(to_call, s.stack))] if to_call > 0 \
            else [Move(Action.CHECK)]
        sizes = self.raise_range(seat)
        if sizes is not None:
            moves += [Move(Action.RAISE, to) for to in sorted(set(sizes))]
        return moves

    def is_legal(self, seat: int, move: Move) -> bool:
        if move.action != Action.RAISE:
            return move in self.legal_moves(seat)
        if self.finished or seat != self.to_act:
            return False
        sizes = self.raise_range(seat)
        return sizes is not None and sizes[0] <= move.to <= sizes[1]

    def apply(self, seat: int, move: Move) -> None:
        if not self.is_legal(seat, move):
            raise ValueError('%r is not a legal move for seat %i now.' % (move, seat))
        s = self.seats[seat]
        if move.action == Action.FOLD:
            s.folded = True
        elif move.action == Action.CALL:
            self._post(seat, self.current_bet - s.bet)
        elif move.action == Action.RAISE:
            self._post(seat, move.to - s.bet)
            increment = move.to - self.current_bet
            if increment >= self.min_raise:
                # a full raise reopens the betting for everyone who already acted
                self.min_raise = increment
                self.full_raises += 1
                for other in self.seats:
                    if other is not s:
                        other.acted = False
            self.current_bet = max(self.current_bet, move.to)
        s.acted = True
        s.faced = self.full_raises
        self._advance()

    def _advance(self) -> None:
        while True:
            live = [i for i, seat in enumerate(self.seats) if seat.in_hand]
            if len(live) == 1:
                self._award([Pot(self.pot, (live[0],), (live[0],))])
                return
            nxt = self._next(self.to_act, self._needs_to_act) if self.to_act is not None else None
            if nxt is not None:
                self.to_act = nxt
                return
            if self.street == Street.RIVER:
                self._showdown()
                return
            self._deal_street()
            if sum(seat.can_act for seat in self.seats) <= 1:
                self.to_act = None  # no more betting, run the board out
            else:
                self.to_act = self.button

    def _deal_street(self) -> None:
        self.shoe.draw(face_up=False)  # burn
        self.board += self.shoe.draw_many(3 if self.street == Street.PREFLOP else 1)
        self.street = Street(self.street.value + 1)
        self.current_bet = 0
        self.min_raise = self.rules.big_blind
        for seat in self.seats:
            seat.bet = 0
            seat.acted = False

    def _showdown(self) -> None:
        contenders = [i for i, seat in enumerate(self.seats) if seat.in_hand]
        for i in contenders:
            for card in self.seats[i].hole:
                self.shoe.reveal(card)
            self.ranks[i] = self.rules.hand_rank(self.seats[i].hole, self.board)
        # one layer per distinct amount a contender put in; folded chips fall into the layers they reached
        levels = sorted({self.seats[i].total for i in contenders})
        pots, previous = [], 0
        for level in levels:
            amount = sum(max(0, min(seat.total, level) - min(seat.total, previous)) for seat in self.seats)
            eligible = tuple(i for i in contenders if self.seats[i].total >= level)
            best = max(self.ranks[i] for i in eligible)
            n = len(self.seats)
            winners = tuple(sorted((i for i in eligible if self.ranks[i] == best),
                                   key=lambda i: (i - self.button - 1) % n))
            pots.append(Pot(amount, eligible, winners))
            previous = level
        extra = sum(max(0, seat.total - previous) for seat in self.seats)  # folded over the top contender
        if extra:
            last = pots[-1]
            pots[-1] = Pot(last.amount + extra, last.eligible, last.winners)
        self._award(pots)

    def _award(self, pots: List[Pot]) -> None:
        for pot in pots:
            share = pot.amount // len(pot.winners)
            for k, i in enumerate(pot.winners):
                self.seats[i].stack += share + (pot.amount - share * len(pot.winners) if k == 0 else 0)
        self.pots = pots
        self.to_act = None
        self.finished = True

    def observation(self, seat: int) -> Observation:
        s = self.seats[seat]
        sizes = self.raise_range(seat) if seat == self.to_act and not self.finished else None
        n_opponents = sum(other.in_hand for j, other in enumerate(self.seats) if j != seat)
        return Observation(seat, tuple(s.hole), tuple(self.board), self.street, self.pot,
                           min(max(0, self.current_bet - s.bet), s.stack), s.bet, self.current_bet, s.stack,
                           sizes[0] if sizes else None, sizes[1] if sizes else None, n_opponents,
                           self.rules.big_blind, self.rules)

    def payoffs(self) -> List[float]:
        if not self.finished:
            raise ValueError('the hand is not over yet.')
        return [float(seat.stack - seat.start) for seat in self.seats]

    @property
    def stacks(self) -> List[int]:
        return [seat.stack for seat in self.seats]

    def hole(self, seat: int) -> List[Card]:
        return list(self.seats[seat].hole)

    def rank(self, seat: int) -> Optional[HandRank]:
        return self.ranks.get(seat)

    def __str__(self):
        board = ' '.join(str(card) for card in self.board) or '-'
        return '%s, board %s, pot %i, stacks %s' % (self.street.name.lower(), board, self.pot, self.stacks)

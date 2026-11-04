import random
from typing import List, Optional

from src.CardGames.Core.Shoe import Shoe
from src.CardGames.Holdem.Equity import outs
from src.CardGames.Holdem.HandEvaluator import describe
from src.CardGames.Holdem.Range import betting_weights
from src.CardGames.Holdem.Round import Action, Move, Round, Street
from src.CardGames.Holdem.Rules import Rules
from src.CardGames.Holdem.Strategy import MonteCarloStrategy
from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'

# the computer seats, in the order they sit down; a style is how a seat plays, which the table can read over time
STYLES = [
    ('Tight', dict(fold_slack=0.02, aggression=0.7, bluff=0.0)),
    ('Loose', dict(fold_slack=0.15, aggression=1.2, bluff=0.15)),
    ('Solid', dict(fold_slack=0.08, aggression=1.0, bluff=0.05)),
    ('Aggressive', dict(fold_slack=0.06, aggression=1.5, bluff=0.2)),
    ('Passive', dict(fold_slack=0.12, aggression=0.5, bluff=0.0)),
]
SEATS = ((4, 'Four-handed'), (6, 'Six-handed'), (2, 'Heads-up'))  # the lobby offers the first by default
VERBS = {Action.FOLD: 'fold', Action.CHECK: 'check', Action.CALL: 'call', Action.RAISE: 'raise'}
BOT_SIMS, COACH_SIMS = 400, 2000


class HoldemSession(Session):
    """
    you in seat 0 at a no-limit hold'em table against computer seats, hand after hand with the button moving left;
    a computer seat that busts buys back in, and so may you. The computer seats play from equity against random hands
    dealt from the cards they have not seen; the coach does the same, but favours hands that fit the public betting
    """

    def __init__(self, options: Optional[dict] = None, rng: Optional[random.Random] = None):
        options = options or {}
        n_seats = int(options.get('seats', 4))
        if n_seats not in dict(SEATS):
            raise ValueError('seats %i should be one of %s.' % (n_seats, sorted(dict(SEATS))))
        self.rules = Rules(small_blind=1, big_blind=2)
        self.buy_in = int(options.get('stack', 200))
        if not 20 <= self.buy_in <= 2000:
            raise ValueError('stack %i should be between 20 and 2000.' % self.buy_in)
        self.rng = rng if rng is not None else random.Random()
        self.seed = self.rng.getrandbits(32)  # seeds the coach, so asking twice at one decision gives one answer
        self.names = ['You'] + ['%s %i' % (STYLES[k][0], k + 2) for k in range(n_seats - 1)]
        self.bots = [None] + [MonteCarloStrategy(BOT_SIMS, rng=random.Random(self.rng.random()), **STYLES[k][1])
                              for k in range(n_seats - 1)]
        self.stacks = [self.buy_in] * n_seats
        self.bought = [self.buy_in] * n_seats  # chips each seat has bought in for
        self.button = n_seats - 1  # moves to you for the first hand
        self.round = None  # type: Optional[Round]
        self.hands = 0
        self.payoff = 0
        self.followed = 0  # decisions where you took the coach's move
        self.decisions = 0
        self.log = []  # type: List[str]
        self._advice = None  # (decision key, Advice) for the decision in front of you

    # ---- state ----

    @property
    def phase(self) -> str:
        if self.round is None:
            return 'deal'
        return 'done' if self.round.finished else 'play'

    def _shoe(self) -> Shoe:
        return Shoe(1, rng=self.rng)

    def _your_turn(self) -> bool:
        return self.phase == 'play' and self.round.current_seat() == 0

    # ---- actions ----

    def act(self, action: dict) -> None:
        kind = action.get('type')
        if kind == 'deal':
            self._deal()
        elif kind == 'move':
            self._move(action.get('move'), action.get('to'))
        elif kind == 'rebuy':
            self._rebuy()
        else:
            raise ValueError('unknown action %r.' % kind)

    def _rebuy(self) -> None:
        if self.phase == 'play':
            raise ValueError('buy back in between hands.')
        if self.stacks[0] >= self.rules.big_blind:
            raise ValueError('you can buy back in once you are below the big blind.')
        self.bought[0] += self.buy_in - self.stacks[0]
        self.stacks[0] = self.buy_in
        self.log.append('You buy back in for %i.' % self.buy_in)

    def _deal(self) -> None:
        if self.phase == 'play':
            raise ValueError('finish the hand in play before dealing again.')
        if self.stacks[0] < self.rules.big_blind:
            raise ValueError('your stack is below the big blind: buy back in to play on.')
        for i in range(1, len(self.stacks)):
            if self.stacks[i] < self.rules.big_blind:
                self.bought[i] += self.buy_in - self.stacks[i]
                self.stacks[i] = self.buy_in
                self.log.append('%s buys back in for %i.' % (self.names[i], self.buy_in))
        self.button = (self.button + 1) % len(self.stacks)
        self.round = Round(self.stacks, self.rules, self.button, shoe=self._shoe())
        self.hands += 1
        self._advice = None
        r = self.round
        posts = lambda seat: '%s post%s' % (self.names[seat], '' if seat == 0 else 's')
        self.log.append('Hand %i: %s on the button; %s %i, %s %i.' % (
            self.hands, self.names[r.button], posts(r.small_blind_seat), self.rules.small_blind,
            posts(r.big_blind_seat), self.rules.big_blind))
        self._run()

    def _parse(self, name: Optional[str], to) -> Move:
        try:
            action = Action(name)
        except ValueError:
            raise ValueError('unknown move %r.' % name)
        if action == Action.RAISE:
            try:
                return Move(action, int(to))
            except (TypeError, ValueError):
                raise ValueError('a raise needs a whole-chip amount to raise to, received %r.' % to)
        for move in self.round.legal_moves(0):
            if move.action == action:
                return move
        raise ValueError('%s is not allowed now.' % name)

    def _move(self, name: Optional[str], to) -> None:
        if not self._your_turn():
            raise ValueError('it is not your turn.')
        move = self._parse(name, to)
        if not self.round.is_legal(0, move):
            low, high = self.round.raise_range(0) or (None, None)
            raise ValueError('raise to %i is not allowed now; raise to between %s and %s.' % (move.to, low, high)
                             if low is not None else 'raising is not allowed now.')
        advised = self._coach().move
        self.decisions += 1
        self.followed += move.action == advised.action
        self._apply(0, move)
        self._run()

    def _apply(self, seat: int, move: Move) -> None:
        street, n_board, opened = self.round.street, len(self.round.board), self.round.current_bet == 0
        self.round.apply(seat, move)
        verb = 'bet' if move.action == Action.RAISE and opened else VERBS[move.action]
        self.log.append('%s %s%s%s.' % (self.names[seat], verb, '' if seat == 0 else 's',
                                        ' %i' % move.to if verb == 'bet' else
                                        ' to %i' % move.to if move.action == Action.RAISE else ''))
        if len(self.round.board) > n_board:
            board = ' '.join(str(card) for card in self.round.board)
            jumped = self.round.street.value - street.value > 1  # all in: the rest of the board comes at once
            self.log.append('%s: %s.' % ('Board runs out' if jumped else self.round.street.name.capitalize(), board))

    def _run(self) -> None:
        # the computer seats move until it is your turn or the hand is over
        r = self.round
        while not r.finished and r.current_seat() != 0:
            seat = r.current_seat()
            if seat is None:  # everyone left is all in: the board runs out on its own
                break
            self._apply(seat, self.bots[seat].decide(r.observation(seat), r.legal_moves(seat)))
        if r.finished:
            self._settle()

    def _settle(self) -> None:
        r = self.round
        start = list(self.stacks)
        self.stacks = r.stacks
        self.payoff = self.stacks[0] - start[0]
        for pot in r.pots:
            names = ' and '.join(self.names[i] for i in pot.winners)
            verb = 'split' if len(pot.winners) > 1 else 'win' if pot.winners == (0,) else 'wins'
            how = describe(r.rank(pot.winners[0])).lower() if r.ranks else 'everyone else folding'
            self.log.append('%s %s %i with %s.' % (names, verb, pot.amount, how))

    # ---- what you see ----

    def _coach(self):
        # one Monte Carlo answer per decision, from your observation only
        key = len(self.round.history)
        if self._advice is None or self._advice[0] != (self.hands, key):
            coach = MonteCarloStrategy(COACH_SIMS, rng=random.Random('%i-%i-%i' % (self.seed, self.hands, key)),
                                       reads_betting=True)
            self._advice = ((self.hands, key), coach.advise(self.round.observation(0), self.round.legal_moves(0)))
        return self._advice[1]

    def _last_moves(self, history) -> dict:
        # each seat's latest public move, a raise that opens the betting after the flop called a bet
        last, raised = {}, set()
        for seat, street, move in history:
            action = move.action.value
            if move.action == Action.RAISE and street != Street.PREFLOP and street not in raised:
                action = 'bet'
            if move.action == Action.RAISE:
                raised.add(street)
            last[seat] = {'street': street.name.lower(), 'action': action, 'to': move.to}
        return last

    def _seat_views(self) -> List[dict]:
        r = self.round
        shown = set(r.ranks) if r is not None else set()  # the hands turned over at showdown
        o = r.observation(0) if r is not None else None
        last = self._last_moves(o.history) if o is not None else {}
        out = []
        for i, name in enumerate(self.names):
            seat = {'name': name, 'stack': self.stacks[i], 'bet': 0, 'in_hand': r is not None, 'button': False,
                    'blind': None, 'cards': None, 'hand': None, 'last': last.get(i), 'to_act': False}
            if r is not None:
                seat.update(stack=o.stacks[i], bet=0 if r.finished else o.bets[i], in_hand=o.in_hand[i],
                            button=i == o.button, to_act=r.current_seat() == i,
                            blind='SB' if i == r.small_blind_seat else 'BB' if i == r.big_blind_seat else None)
                if i == 0:
                    seat['cards'] = [str(card) for card in o.hole]
                elif i in shown:
                    seat['cards'] = [str(card) for card in r.hole(i)]
                if i in shown:
                    seat['hand'] = describe(r.rank(i))
            out.append(seat)
        return out

    def view(self) -> dict:
        n = len(self.stacks)
        out = {
            'game': 'holdem', 'phase': self.phase, 'blinds': [self.rules.small_blind, self.rules.big_blind],
            'buy_in': self.buy_in, 'seats': self._seat_views(), 'board': [], 'pot': 0, 'street': None,
            'legal': [], 'to_call': 0, 'raise': None, 'your_turn': False, 'made': None,
            'payoff': None, 'log': self.log[-14:], 'can_rebuy': self.phase != 'play' and self.stacks[0] < self.rules.big_blind,
            # chips won so far, counted between hands: during one, your stack is the one you sat down with
            'stats': {'hands': self.hands, 'net': self.stacks[0] - self.bought[0], 'bought': self.bought[0],
                      'followed': self.followed, 'decisions': self.decisions},
            'n_seats': n,
        }
        out['stats']['bb_per_100'] = 100.0 * out['stats']['net'] / self.rules.big_blind / self.hands \
            if self.hands else 0.0
        if self.round is None:
            return out
        o = self.round.observation(0)
        out.update(board=[str(card) for card in o.board], pot=o.pot, street=o.street.name.lower(),
                   your_turn=self._your_turn())
        if len(o.board) >= 3:
            out['made'] = describe(self.rules.hand_rank(list(o.hole), list(o.board)))
        if out['your_turn']:
            moves = self.round.legal_moves(0)
            out['legal'] = sorted({m.action.value for m in moves}, key=[a.value for a in Action].index)
            out['to_call'] = o.to_call
            out['call_to'] = o.bet + o.to_call
            if o.min_raise_to is not None:
                out['raise'] = {'min': o.min_raise_to, 'max': o.max_raise_to,
                                'pot': min(o.max_raise_to, max(o.min_raise_to, o.current_bet + o.pot + o.to_call))}
        if self.phase == 'done':
            out['payoff'] = self.payoff
        return out

    def advice(self) -> dict:
        if not self._your_turn():
            return {'kind': 'wait', 'reason': 'deal the next hand' if self.phase != 'play' else 'it is not your turn'}
        advice = self._coach()
        o = self.round.observation(0)
        out = {'kind': 'move', 'move': advice.move.action.value, 'to': advice.move.to, 'reason': advice.reason,
               'equity': advice.equity, 'pot_odds': advice.pot_odds, 'call_ev': advice.call_ev,
               'n_opponents': o.n_opponents, 'read_betting': betting_weights(o) is not None, 'outs': None}
        if o.street in (Street.FLOP, Street.TURN):
            found = outs(list(o.hole), list(o.board), self.rules)
            out['outs'] = {'n': found.n_outs, 'p_river': found.p_by_river(2 if o.street == Street.FLOP else 1)}
        return out

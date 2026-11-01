import random
from collections import Counter
from typing import List, Optional, Tuple

from src.CardGames.Core.Card import Card, Rank, Suit, standard_deck
from src.CardGames.GuanDan.Combo import BLACK_JOKER, RED_JOKER, Combo, interpret, is_wild, rank_name, sort_hand
from src.CardGames.GuanDan.Match import Match, Tribute, choose_return
from src.CardGames.GuanDan.Round import N_SEATS, PASS, Round, team
from src.CardGames.GuanDan.Strategy import PlannerStrategy
from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'

# play passes to the next seat, so East follows you, then your partner opposite, then West
NAMES = ['You', 'East', 'Partner', 'West']
PLACES = ['1st', '2nd', '3rd', '4th']


def _level_name(value: int) -> str:
    return rank_name(value)


def _name(card: Card) -> str:
    # 'red joker', '10♥': how a card reads in the log
    if card.is_joker():
        return rank_name(card.rank.value)
    return rank_name(card.rank.value) + {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}[str(card)[-1]]


def _cards(cards) -> List[str]:
    return [str(card) for card in cards]


def _move(move: Optional[Combo]) -> dict:
    if move is PASS:
        return {'kind': 'pass', 'cards': [], 'desc': 'pass'}
    return {'kind': 'play', 'cards': _cards(move.cards), 'desc': move.describe(), 'bomb': move.is_bomb}


class _Returner(object):
    """
    stands in for your seat while tribute is paid: notes the hand and the cards you may hand back, and returns the
    card you picked, or the coach's pick on a dry run
    """

    def __init__(self, pick: Optional[Card] = None):
        self.pick = pick
        self.seen = None  # type: Optional[Tuple[List[Card], List[Card]]]

    def choose_return(self, hand: List[Card], level: Rank, options: List[Card]) -> Card:
        self.seen = (list(hand), list(options))
        return self.pick if self.pick is not None else choose_return(hand, level)


class GuanDanSession(Session):
    """
    you and a computer partner (seats 0 and 2) against two computer opponents, hand after hand through a whole match:
    levels from 2 to A, tribute before each hand after the first, and catching the wind. The computer seats and the
    coach plan from their own hand and the public play only: card counts, the cards played and the trick
    """

    def __init__(self, options: Optional[dict] = None, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()
        self.match = Match(self.rng)
        self.bots = [None] + [PlannerStrategy() for _ in range(N_SEATS - 1)]
        self.coach = PlannerStrategy()
        self.round = None  # type: Optional[Round]
        self.leader = 0
        self.tributes = []  # type: List[Tribute]
        # hands dealt, your hand with the tribute in it, the cards you may hand back, and the tributes paid (public)
        self.waiting = None  # type: Optional[Tuple[List[List[Card]], List[Card], List[Card], List[Tribute]]]
        self.choices = None  # type: Optional[List[Combo]]  # your selection reads several ways: pick one
        self.trick = []  # type: List[Tuple[int, Optional[Combo]]]  # moves in the trick in play
        self.last_trick = []  # type: List[Tuple[int, Optional[Combo]]]
        self.won = 0  # hands your team won
        self.followed = 0  # decisions where you made the coach's move
        self.decisions = 0
        self.log = []  # type: List[str]

    # ---- state ----

    @property
    def phase(self) -> str:
        if self.waiting is not None:
            return 'return'
        if self.round is None:
            return 'deal'
        if self.round.result is None:
            return 'play'
        return 'over' if self.match.is_over else 'done'

    def _your_turn(self) -> bool:
        return self.phase == 'play' and self.round.current_seat() == 0

    def _deal_hands(self) -> List[List[Card]]:
        return self.match.deal()

    # ---- actions ----

    def act(self, action: dict) -> None:
        kind = action.get('type')
        if kind == 'deal':
            self._deal()
        elif kind == 'return':
            self._return(action.get('card'))
        elif kind == 'play':
            self._play(action.get('cards'), action.get('reading'))
        elif kind == 'pass':
            self._pass()
        else:
            raise ValueError('unknown action %r.' % kind)

    def _deal(self) -> None:
        if self.phase in ('play', 'return'):
            raise ValueError('finish the hand in play before dealing again.')
        if self.phase == 'over':
            raise ValueError('the match is over: start a new session for another.')
        hands = self._deal_hands()
        self.log.append('Hand %i at level %s, %s' % (
            len(self.match.records) + 1, _level_name(self.match.level.value),
            'the first hand' if self.match.declarer is None else
            'your team\'s level' if self.match.declarer == 0 else 'the opponents\' level'))
        if self.match.previous_order is not None:
            # a dry run on a copy finds out whether you receive tribute and so hand a card back
            dry = _Returner()
            _, paid = self.match.tribute([list(hand) for hand in hands], self._strategies(dry))
            if dry.seen is not None:
                self.waiting = (hands, dry.seen[0], dry.seen[1], paid)
                for t in paid:
                    self.log.append('%s pay%s %s tribute of %s.' % (
                        NAMES[t.payer], '' if t.payer == 0 else 's', 'you' if t.receiver == 0 else NAMES[t.receiver],
                        _name(t.card)))
                self.log.append('Pick a card to hand back.')
                return
        self._start(hands, _Returner())

    def _strategies(self, you: _Returner) -> list:
        return [you] + self.bots[1:]

    def _return(self, code) -> None:
        if self.waiting is None:
            raise ValueError('there is no tribute to return now.')
        card = self._parse([code])[0] if code is not None else None
        hands, _, options, _ = self.waiting
        if card not in options:
            raise ValueError('hand back one of %s.' % ', '.join(_cards(sorted(set(options), key=str))))
        self.waiting = None
        self._start(hands, _Returner(card), announced=True)

    def _start(self, hands: List[List[Card]], you: _Returner, announced: bool = False) -> None:
        # announced: the tributes paid are in the log already, from before you picked your card to hand back
        self.leader, self.tributes = self.match.tribute(hands, self._strategies(you))
        for t in self.tributes:
            paid = '' if announced else '%s pay%s %s tribute of %s; ' % (
                NAMES[t.payer], '' if t.payer == 0 else 's', 'you' if t.receiver == 0 else NAMES[t.receiver],
                _name(t.card))
            back = '%s hand%s back %s' % ('you' if t.receiver == 0 else NAMES[t.receiver],
                                          '' if t.receiver == 0 else 's', _name(t.returned))
            self.log.append(paid + (back[0].upper() + back[1:] if announced else back) +
                            (' to %s.' % ('you' if t.payer == 0 else NAMES[t.payer]) if announced else '.'))
        if self.match.previous_order is not None and not self.tributes:
            self.log.append('Tribute is resisted: the payers hold both red jokers.')
        self.round = Round(hands, self.match.level, self.leader)
        self.trick, self.last_trick, self.choices = [], [], None
        self.log.append('%s lead%s.' % (NAMES[self.leader], '' if self.leader == 0 else 's'))
        self._run()

    @staticmethod
    def _parse(codes) -> List[Card]:
        if not isinstance(codes, list) or not codes:
            raise ValueError('pick the cards to play.')
        try:
            return [Card.parse(str(code)) for code in codes]
        except (KeyError, ValueError, IndexError):
            raise ValueError('unknown cards %r.' % codes)

    def _play(self, codes, reading) -> None:
        if not self._your_turn():
            raise ValueError('it is not your turn.')
        cards = self._parse(codes)
        if Counter(cards) - Counter(self.round.hands[0]):
            raise ValueError('you do not hold %s.' % ' '.join(_cards(cards)))
        level = self.round.level
        readings = interpret(cards, level)
        if not readings:
            raise ValueError('%s is not a combination.' % ' '.join(_cards(cards)))
        legal = [combo for combo in readings if self.round.is_legal(0, combo)]
        if not legal:
            raise ValueError('a %s does not beat the %s.' % (readings[0].describe(), self.round.target.describe()))
        if len(legal) > 1 and reading is None:
            self.choices = legal  # wild cards make it read several ways: you say which
            return
        try:
            move = legal[int(reading or 0)]
        except (IndexError, TypeError, ValueError):
            raise ValueError('reading %r is not one of the %i choices.' % (reading, len(legal)))
        self._decide(move)

    def _pass(self) -> None:
        if not self._your_turn():
            raise ValueError('it is not your turn.')
        if self.round.target is None:
            raise ValueError('you lead this trick, so you must play.')
        self._decide(PASS)

    def _decide(self, move: Optional[Combo]) -> None:
        advised = self._advice()[0].move
        same = move is PASS if advised is PASS else move is not PASS and \
            (move.type, move.n, move.value) == (advised.type, advised.n, advised.value)
        self.decisions += 1
        self.followed += same
        self.choices = None
        self._apply(0, move)
        self._run()

    def _apply(self, seat: int, move: Optional[Combo]) -> None:
        r = self.round
        r.apply(seat, move)
        self.trick.append((seat, move))
        if move is not PASS and not r.hands[seat]:
            self.log.append('%s %s out %s with %s.' % (NAMES[seat], 'go' if seat == 0 else 'goes',
                                                       PLACES[len(r.out) - 1], move.describe()))
        if r.result is None and r.target is None:
            # everyone else passed: the trick is won and a new one starts
            winner = next(s for s, m in reversed(self.trick) if m is not PASS)
            self.log.append('%s win%s the trick with %s.' % (NAMES[winner], '' if winner == 0 else 's',
                                                           next(m for s, m in reversed(self.trick)
                                                                if m is not PASS).describe()))
            if not r.hands[winner]:
                self.log.append('%s %s, so %s lead%s (catching the wind).' % (
                    NAMES[winner], 'are out' if winner == 0 else 'is out',
                    'you' if r.to_act == 0 else NAMES[r.to_act], '' if r.to_act == 0 else 's'))
            self.last_trick, self.trick = self.trick, []

    def _run(self) -> None:
        # the computer seats play until it is your turn or the hand is over
        r = self.round
        while r.result is None and r.current_seat() != 0:
            seat = r.current_seat()
            self._apply(seat, self.bots[seat].decide(r.observation(seat), r.legal_moves(seat)))
        if r.result is not None:
            self._settle()

    def _settle(self) -> None:
        result = self.round.result
        record = self.match.record(self.round, self.leader, self.tributes)
        ours = result.winning_team == 0
        self.won += ours
        self.log.append('%s win the hand and go up %i level%s: you %s, them %s.' % (
            'You and your partner' if ours else 'East and West', result.levels_up,
            '' if result.levels_up == 1 else 's', _level_name(record.levels_after[0]),
            _level_name(record.levels_after[1])))
        if self.match.is_over:
            self.log.append('%s the match.' % ('You and your partner win' if self.match.winner == 0
                                               else 'East and West win'))

    # ---- what you see ----

    def _advice(self):
        o = self.round.observation(0)
        return self.coach.rank(o, self.round.legal_moves(0))

    def _unseen(self, hand: List[Card], played: List[Card], level: Rank) -> List[dict]:
        # the cards that matter most, still out of your sight: in the other three hands, wherever they are
        unseen = Counter(standard_deck(jokers=True) * 2) - Counter(hand) - Counter(played)
        rows = [('Red jokers', lambda c: c.rank.value == RED_JOKER, 2),
                ('Black jokers', lambda c: c.rank.value == BLACK_JOKER, 2),
                ('Wild cards (%s of hearts)' % _level_name(level.value), lambda c: is_wild(c, level), 2),
                ('Other %ss' % _level_name(level.value), lambda c: c.rank == level and not is_wild(c, level), 6)]
        if level != Rank.ACE:
            rows.append(('Aces', lambda c: c.rank == Rank.ACE, 8))
        return [{'label': label, 'n': sum(n for card, n in unseen.items() if test(card)), 'of': of}
                for label, test, of in rows]

    def view(self) -> dict:
        m = self.match
        out = {
            'game': 'guandan', 'phase': self.phase, 'names': NAMES,
            'levels': {'you': _level_name(m.levels[0]), 'them': _level_name(m.levels[1])},
            'level': _level_name(m.level.value), 'declarer': None if m.declarer is None else
            ('you' if m.declarer == 0 else 'them'), 'a_fails': list(m.a_fails),
            'hand': [], 'wild': None, 'seats': [], 'trick': None, 'last_trick': [], 'your_turn': False,
            'can_pass': False, 'choices': None, 'tribute': None, 'result': None, 'unseen': [],
            'match': {'hands': len(m.records), 'won': self.won, 'winner': None if m.winner is None else
                      ('you' if m.winner == 0 else 'them')},
            'stats': {'followed': self.followed, 'decisions': self.decisions}, 'log': self.log[-14:],
        }
        if self.waiting is not None:
            _, yours, options, paid = self.waiting
            # only what is paid is shown, not what the other receiver will hand back
            out.update(hand=_cards(sort_hand(yours, m.level)), wild=str(Card(m.level, Suit.HEART)),
                       tribute={'options': sorted(set(_cards(options))),
                                'paid': [{'payer': NAMES[t.payer], 'receiver': NAMES[t.receiver], 'card': str(t.card),
                                          'name': _name(t.card)} for t in paid]})
            return out
        if self.round is None:
            return out
        r, o = self.round, self.round.observation(0)
        out['hand'] = _cards(o.hand)
        out['wild'] = str(Card(r.level, Suit.HEART))
        moves = {s: _move(move) for s, move in self.trick}  # each seat's latest move in this trick
        out['seats'] = [{'name': NAMES[s], 'team': 'you' if team(s) == 0 else 'them', 'count': o.counts[s],
                         'place': PLACES[o.out.index(s)] if s in o.out else None,
                         'to_act': r.current_seat() == s, 'move': moves.get(s)} for s in range(N_SEATS)]
        if o.target is not None:
            out['trick'] = {'seat': NAMES[o.target_seat], **_move(o.target)}
        out['last_trick'] = [{'seat': NAMES[s], **_move(move)} for s, move in self.last_trick]
        out['unseen'] = self._unseen(list(o.hand), list(o.played), r.level)
        if self._your_turn():
            out.update(your_turn=True, can_pass=o.target is not None,
                       choices=[c.describe() for c in self.choices] if self.choices else None)
        if r.result is not None:
            res = r.result
            out['result'] = {'order': [NAMES[s] for s in res.order],
                             'winners': 'you' if res.winning_team == 0 else 'them', 'levels_up': res.levels_up}
        return out

    def advice(self) -> dict:
        if self.waiting is not None:
            _, yours, _, _ = self.waiting
            card = choose_return(yours, self.match.level)
            return {'kind': 'return', 'card': str(card),
                    'name': _name(card), 'reason': 'of the cards you may hand back, losing the %s leaves your hand '
                                                   'easiest to play out' % _name(card)}
        if not self._your_turn():
            return {'kind': 'wait', 'reason': 'deal the next hand' if self.phase != 'play' else 'it is not your turn'}
        ranked = self._advice()
        top = [{'move': _move(c.move), 'reason': c.reason, 'turns': c.after.turns} for c in ranked[:3]]
        return {'kind': 'move', **top[0], 'alternatives': top[1:]}

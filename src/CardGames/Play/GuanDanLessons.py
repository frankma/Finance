import random
from collections import Counter
from typing import List, Optional

from src.CardGames.Core.Card import Card, Rank, standard_deck
from src.CardGames.Core.Player import Strategy
from src.CardGames.GuanDan.Combo import Combo, interpret
from src.CardGames.GuanDan.Round import PASS, Observation, Round, deal
from src.CardGames.GuanDan.Strategy import PlannerStrategy
from src.CardGames.Play.GuanDanSession import GuanDanSession
from src.CardGames.Play.Lesson import Lesson, Step
from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'

LEVEL = Rank.TWO


class Scripted(Strategy):
    """
    a lesson's computer seat: makes these plays in turn ('-' passes), then plays on like any computer seat; like
    every seat it sees only its own hand and the public play
    """

    def __init__(self, plays: Optional[List[str]] = None):
        self.plays = list(plays or [])
        self.planner = PlannerStrategy()

    def decide(self, observation: Observation, legal_moves: List[Optional[Combo]]) -> Optional[Combo]:
        if self.plays:
            play = self.plays.pop(0)
            if play == '-':
                return PASS
            readings = interpret(Card.parse_many(play), observation.level)
            return next(r for r in readings if r.beats(observation.target))
        return self.planner.decide(observation, legal_moves)


def _spot(hands: List[str], leader: int, scripts: Optional[dict] = None, note: str = ''):
    """
    a hand near its end at level 2: these cards still held (you, East, your partner, West), every other card of the
    two packs already played, and the seats in scripts making those plays first; dealt and played up to your turn
    """
    def setup(rng: random.Random) -> GuanDanSession:
        session = GuanDanSession({}, rng)
        held = [Card.parse_many(h) for h in hands]
        for seat, plays in (scripts or {}).items():
            session.bots[seat] = Scripted(plays)
        session.round = Round(held, LEVEL, leader)
        # the rest of both packs has been played already, so what you have not seen is only in the other hands
        session.round.played = list((Counter(standard_deck(jokers=True) * 2) - Counter(c for h in held for c in h))
                                    .elements())
        session.leader = leader
        session.log.append(note or 'A hand near its end, at level 2: everything else has been played.')
        session._run()
        return session
    return setup


def _tribute(rng: random.Random) -> GuanDanSession:
    # the hand after one you went out first in and West last: West pays you, and you hand a card back
    session = GuanDanSession({}, rng)
    hands = deal(random.Random(5))  # a deal where West does not hold both red jokers, so tribute is paid
    session._deal_hands = lambda: [list(hand) for hand in hands]
    session.match.previous_order, session.match.declarer, session.match.levels = (0, 1, 2, 3), 0, [4, 2]
    session.act({'type': 'deal'})
    return session


def _play(cards: str) -> dict:
    return {'type': 'play', 'cards': cards.split()}


def _read_as(desc: str):
    # a play with a wild must be read this way once you are asked which reading you mean
    def check(action: dict, table: Session) -> bool:
        choices = getattr(table, 'choices', None)
        if 'reading' not in action or not choices:
            return True
        try:
            return choices[int(action['reading'])].describe() == desc
        except (IndexError, TypeError, ValueError):
            return False
    return check


def _returnable(action: dict, table: Session) -> bool:
    # one of the cards the table offers to hand back
    waiting = getattr(table, 'waiting', None)
    try:
        return waiting is not None and Card.parse(str(action.get('card'))) in waiting[2]
    except (KeyError, ValueError, IndexError):
        return False


def lessons() -> List[Lesson]:
    return [
        Lesson('beat', 'Beat it or pass', 'Only the same kind of play, higher, beats a play.',
               _spot(['9C 9D 5S 6H JC', '3D 4H KS', 'QH QS TD', '7S 7D AH'], 3, {3: ['7S 7D']}), (
                   Step(_play('9C 9D'),
                        'West leads a pair of 7s. To beat a pair you need a higher pair, or a bomb; a single or three '
                        'cards will not do. Select your two 9s and play them.',
                        'Right: a pair of 9s beats a pair of 7s: the same kind of play, the same number of cards, '
                        'higher.',
                        'A pair can only be beaten by a higher pair or a bomb. Select both 9s and play them.'),
               ), 'Each trick keeps the shape of its lead: singles, pairs, straights, and so on.', ('tricks',)),
        Lesson('shape', 'Lead your weakest shape', 'Play the group that makes the rest of your hand easiest.',
               _spot(['3S 4D 5C 6H 7S KD KC', '8D 9S TC JH', 'AS AD QC', '6D 6S 9H'], 0), (
                   Step(_play('3S 4D 5C 6H 7S'),
                        'You lead. Your hand falls into two groups: a straight from 3 to 7, and a pair of kings. Lead '
                        'the weaker group and keep the kings to win a later trick.',
                        'Right: lead the low straight. Five cards go in one play, and the pair of kings stays back as '
                        'a strong play for later.',
                        'Splitting the straight or leading the kings leaves your hand harder to play out. Lead the '
                        'straight from 3 to 7.'),
               ), 'Plan the whole hand: lead low groups first, keep the strong ones for when you need the lead.',
               ('tips', 'combos')),
        Lesson('partner', 'Let your partner win', 'Do not beat your partner when they are winning a trick.',
               _spot(['KS KD 4C 8H', '5D 6D 9C', 'QS QD 7H', 'TC JD'], 2, {2: ['QS QD'], 3: ['-']}), (
                   Step({'type': 'pass'},
                        'Your partner leads a pair of queens and West passes. You hold a pair of kings that would beat '
                        'it. Play them or pass?',
                        'Right: pass. Your partner is winning the trick, and beating them only spends your kings for a '
                        'trick your team already has.',
                        'Your partner is winning this trick for your team. Keep your kings: pass.'),
               ), 'Help your partner instead: when they are down to one or two cards, lead low to let them out.',
               ('tips',)),
        Lesson('bomb', 'When to spend a bomb', 'Bomb when an opponent is about to go out.',
               _spot(['5S 5H 5D 5C 8C 9D', 'AS AD JC QH', 'TD 4S 3H', '7H 6C'], 1, {1: ['AS AD'], 2: ['-'], 3: ['-']}), (
                   Step(_play('5S 5H 5D 5C'),
                        'East plays a pair of aces and has only two cards left. Your partner and West pass. Nothing '
                        'but a bomb beats the aces. Bomb or pass?',
                        'Right: bomb. If East keeps the lead with two cards left, East may go out next. Four 5s take '
                        'the lead away.',
                        'Passing lets East lead again with only two cards left, likely going out. This is the moment '
                        'a bomb is for: play the four 5s.'),
               ), 'Keep bombs back early, and spend them when an opponent is close to going out.', ('bombs', 'tips')),
        Lesson('wild', 'Wild cards', 'The heart of the level rank stands in for any card.',
               _spot(['2H 5S 6S 7S 8S KC', '9H 9D 4C', 'JS 3C', 'KS KH KD KC 3D'], 3, {3: ['KS KH KD KC']}), (
                   Step(_play('2H 5S 6S 7S 8S'),
                        'West plays four kings, a bomb. At level 2 your 2♥ is wild: with 5, 6, 7 and 8 of spades it '
                        'makes a straight flush, which beats any four of a kind. Select the five cards and play them. '
                        'The wild can stand for the 4 or the 9 of spades, so you will be asked which you mean: take '
                        'the higher, a straight flush to 9.',
                        'Right: the wild 2♥ fills a straight flush to 9, and a straight flush beats a bomb of four.',
                        'Only a bigger bomb beats four kings: the wild 2♥ with your 5 to 8 of spades makes a straight '
                        'flush. Select those five cards, and read them as the straight flush to 9, the higher one.',
                        check=_read_as('straight flush to 9')),
               ), 'A wild can complete straights, pairs, triples and bombs, but never stands for a joker.',
               ('level', 'bombs')),
        Lesson('wind', 'Catching the wind', 'Go out on a trick nobody beats, and your partner leads.',
               _spot(['RJ', '9S 9D 4H', 'AS AH QD QC', '8D 8H 3S'], 0), (
                   Step(_play('RJ'),
                        'You lead with your last card, the red joker, the highest single there is. Play it.',
                        'Right: nobody can beat the red joker, so you go out first, and as the trick ends your partner '
                        'leads the next one: catching the wind.',
                        'Your last card is the red joker: play it to go out.'),
               ), 'Going out on a trick that stands hands the lead to your partner instead of an opponent.',
               ('wind',)),
        Lesson('tribute', 'Hand back a tribute card', 'After a hand, the winners receive tribute and return a card.',
               _tribute, (
                   Step({'type': 'return'},
                        'You went out first last hand and West went out last, so West pays you their highest card. You '
                        'hand one card back: a 10 or lower, never a wild or a joker. The cards you may return are '
                        'outlined. Pick the one your hand can spare best; the coach can suggest one.',
                        'Right: a low card back, and you keep the tribute.',
                        'You may hand back only a 10 or lower, never a wild or a joker: pick one of the outlined '
                        'cards.', check=_returnable),
               ), 'The losers pay their best card; the winners return a low one. Then the payer of the bigger '
                  'tribute leads.', ('tribute',)),
    ]

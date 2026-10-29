import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.CardGames.Core.Card import Card, Rank
from src.CardGames.Core.Game import play_out
from src.CardGames.GuanDan.Combo import RED_JOKER, group_value, is_wild
from src.CardGames.GuanDan.Planner import plan_hand
from src.CardGames.GuanDan.Round import N_SEATS, HandResult, Round, deal, team

__author__ = 'frank.ma'

MAX_A_FAILS = 3


@dataclass(frozen=True)
class Tribute(object):
    payer: int
    receiver: int
    card: Card
    returned: Card


def returnable(hand: List[Card], level: Rank) -> List[Card]:
    # the card handed back is a 10 or lower when the hand has one, never a wild or a joker
    low = [card for card in hand if card.rank.value <= 10 and not is_wild(card, level)]
    return low or [card for card in hand if not card.is_joker() and not is_wild(card, level)]


def choose_return(hand: List[Card], level: Rank) -> Card:
    """
    the returnable card whose loss hurts the hand's plan least, the lower one on a tie
    """
    def loss(card):
        rest = list(hand)
        rest.remove(card)
        return plan_hand(rest, level).cost + group_value(card.rank.value, level) * 0.05
    return min(returnable(hand, level), key=loss)


def pay_tribute(hands: List[List[Card]], level: Rank, order: Tuple[int, ...], strategies=None) \
        -> Tuple[int, List[Tribute]]:
    """
    tribute before a hand, in place on hands, after the previous hand finished in order

    the last out pays its highest card that is not wild to the first out, or after a double win both losers pay and
    the bigger card goes to the first out; each receiver returns one card; if the payers hold both red jokers the
    tribute is resisted and the first out leads, otherwise the payer of the bigger tribute leads
    :param strategies: optional per seat; a strategy with a choose_return(hand, level, options) method picks the card
    :return: the leader and the tributes paid
    """
    first = order[0]
    double = team(order[1]) == team(first)
    payers = [order[2], order[3]] if double else [order[3]]
    receivers = [first, order[1]] if double else [first]
    if sum(card.rank.value == RED_JOKER for p in payers for card in hands[p]) >= 2:
        return first, []

    def highest(seat):
        return max((card for card in hands[seat] if not is_wild(card, level)),
                   key=lambda card: group_value(card.rank.value, level))

    paid = sorted(((p, highest(p)) for p in payers), key=lambda pc: -group_value(pc[1].rank.value, level))
    for (payer, card), receiver in zip(paid, receivers):
        hands[payer].remove(card)
        hands[receiver].append(card)
    tributes = []
    for (payer, card), receiver in zip(paid, receivers):
        chooser = getattr(strategies[receiver], 'choose_return', None) if strategies else None
        options = returnable(hands[receiver], level)
        back = chooser(list(hands[receiver]), level, options) if chooser else choose_return(hands[receiver], level)
        if back not in options:
            raise ValueError('seat %i may not return %s.' % (receiver, back))
        hands[receiver].remove(back)
        hands[payer].append(back)
        tributes.append(Tribute(payer, receiver, card, back))
    return paid[0][0], tributes


@dataclass(frozen=True)
class HandRecord(object):
    level: Rank
    leader: int
    tributes: Tuple[Tribute, ...]
    result: HandResult
    levels_after: Tuple[int, int]


class Match(object):
    """
    hands played until a team wins at level A: both teams start at 2, each hand is played at the level of the team
    that won the last one, the winners climb 3, 2 or 1 levels (never past A), and a team at A wins the match by
    winning a hand there with its partner 2nd or 3rd; one that fails at A three times drops back to 2
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()
        self.levels = [2, 2]
        self.declarer = None  # type: Optional[int]  # team whose level is played
        self.a_fails = [0, 0]
        self.previous_order = None  # type: Optional[Tuple[int, ...]]
        self.winner = None  # type: Optional[int]
        self.records = []  # type: List[HandRecord]

    @property
    def level(self) -> Rank:
        return Rank(2 if self.declarer is None else self.levels[self.declarer])

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    def play_hand(self, strategies: list) -> HandRecord:
        if self.is_over:
            raise ValueError('the match is over.')
        level = self.level
        hands = deal(self.rng)
        tributes = []
        if self.previous_order is None:
            leader = self.rng.randrange(N_SEATS)
        else:
            leader, tributes = pay_tribute(hands, level, self.previous_order, strategies)
        game = Round(hands, level, leader)
        play_out(game, strategies)
        self._score(game.result)
        record = HandRecord(level, leader, tuple(tributes), game.result, (self.levels[0], self.levels[1]))
        self.records.append(record)
        return record

    def _score(self, result: HandResult) -> None:
        winners = result.winning_team
        if self.declarer is not None and self.levels[self.declarer] == 14:
            if winners == self.declarer and result.levels_up >= 2:
                self.winner = winners
            else:
                self.a_fails[self.declarer] += 1
                if self.a_fails[self.declarer] >= MAX_A_FAILS:
                    self.levels[self.declarer], self.a_fails[self.declarer] = 2, 0
        if self.winner is None:
            self.levels[winners] = min(14, self.levels[winners] + result.levels_up)
        self.declarer = winners
        self.previous_order = result.order

    def play(self, strategies: list, max_hands: int = 200) -> Optional[int]:
        while not self.is_over and len(self.records) < max_hands:
            self.play_hand(strategies)
        return self.winner


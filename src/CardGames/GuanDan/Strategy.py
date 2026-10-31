from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from src.CardGames.Core.Card import Card, Rank, standard_deck
from src.CardGames.Core.Player import Advice, Advisor
from src.CardGames.GuanDan.Combo import Combo, ComboType, group_value, is_wild
from src.CardGames.GuanDan.Match import choose_return
from src.CardGames.GuanDan.Planner import HandPlan, plan_hand
from src.CardGames.GuanDan.Round import N_SEATS, PASS, Observation, partner

__author__ = 'frank.ma'


@dataclass(frozen=True)
class Candidate(Advice):
    # move is a combination, or None to pass
    cost: float  # lower is better
    after: HandPlan  # the plan for what is left


def _unseen_max(observation: Observation) -> int:
    # the highest group value among cards this seat has not seen, i.e. what could still top a single
    unseen = Counter(standard_deck(jokers=True) * 2) - Counter(observation.hand) - Counter(observation.played)
    return max((group_value(card.rank.value, observation.level) for card in unseen), default=0)


def _cards(n: int) -> str:
    return '%i card%s' % (n, '' if n == 1 else 's')


class PlannerStrategy(Advisor):
    """
    the browser table's computer seat and coach: score every legal play (and passing) by the plan cost of the hand it
    leaves, plus adjustments for the trick and the card counts, and play the cheapest

    leading sheds the weakest shape, a bomb is kept back unless an opponent is close to going out, a partner's winning
    trick is not overtaken, a wild is spent only to finish or to make a bomb, and a low single or pair invites a
    partner with one or two cards left; it reads only its own hand and the public play (card counts, cards played,
    the trick), so what it treats as unseen is every card it has not seen, wherever those cards are
    """

    def rank(self, observation: Observation, legal_moves: List[Optional[Combo]]) -> List[Candidate]:
        o = observation
        hand, level, counts, seat = list(o.hand), o.level, o.counts, o.seat
        mate = partner(seat)
        opponents = [s for s in ((seat + 1) % N_SEATS, (seat + 3) % N_SEATS) if counts[s] > 0]
        lead = o.target is None
        mate_winning = not lead and o.target_seat == mate
        threat = counts[o.target_seat] if not lead and not mate_winning else 99
        opponent_min = min((counts[s] for s in opponents), default=99)
        next_opponent = (seat + 1) % N_SEATS if counts[(seat + 1) % N_SEATS] > 0 else (seat + 3) % N_SEATS
        unseen_max = _unseen_max(o)
        base = plan_hand(hand, level)
        plays = [move for move in legal_moves if move is not PASS]
        candidates = []

        if not lead:
            cost = base.cost
            if mate_winning:
                reason = 'partner is winning this trick, so there is no need to spend cards'
            elif not plays:
                reason = 'nothing in hand beats the %s' % o.target.describe()
            else:
                cost += 3.0
                if threat <= 8:
                    cost += (9 - threat) * 3.0
                    reason = 'passing lets seat %i (only %s) keep control, which is risky' % (o.target_seat,
                                                                                             _cards(threat))
                else:
                    reason = 'beating the %s would cost more than it gains' % o.target.describe()
            candidates.append(Candidate(PASS, reason, cost, base))

        for move in plays:
            rest = list(hand)
            for card in move.cards:
                rest.remove(card)
            after = plan_hand(rest, level)
            cost, why = after.cost, []
            wilds = sum(is_wild(card, level) for card in move.cards)
            spends_wild = wilds and rest and not move.is_bomb
            if spends_wild:
                cost += 6.0 * wilds
            if not rest:
                cost -= 1000.0
                why.append('plays out the last cards')
            elif lead:
                cost += (0.12 if move.n >= 5 else 0.25) * move.value - move.n * 0.6
                if move.is_bomb:
                    cost += 14.0 + move.power / 200.0
                    why.append('leads a bomb to keep control')
                single_low = move.type == ComboType.SINGLE and move.value < unseen_max
                if counts[next_opponent] == 1 and single_low:
                    cost += 12.0
                elif opponent_min == 1 and single_low:
                    cost += 8.0
                if opponent_min == 2 and move.type == ComboType.PAIR:
                    cost += 6.0
                if 0 < counts[mate] <= 3 < opponent_min:
                    if move.type == ComboType.SINGLE and counts[mate] == 1 and move.value <= 9:
                        cost -= 6.0
                        why.append('partner has 1 card left, so a low single invites them out')
                    if move.type == ComboType.PAIR and counts[mate] == 2 and move.value <= 9:
                        cost -= 6.0
                        why.append('partner has 2 cards, so a low pair may let them finish')
                if not why:
                    why.append('leads the weakest shape in the plan' if after.turns < base.turns
                               else 'leads, though it splits a bigger group')
            elif mate_winning:
                cost += 14.0 + (20.0 if move.is_bomb else 0.0)
                why.append('overtakes partner')
            else:
                cost += move.value * 0.1
                if move.value >= 14 and not move.is_bomb and threat > 10:
                    cost += 4.0
                if move.is_bomb:
                    if threat <= 6:
                        cost += 3.0
                        why.append('seat %i has only %s, so a bomb is worth spending' % (o.target_seat, _cards(threat)))
                    elif after.turns <= 1:
                        cost += 2.0
                        why.append('a bomb takes the lead and the rest of the hand can follow')
                    else:
                        cost += 16.0 + move.power / 200.0
                        why.append('spends a bomb early')
                else:
                    why.append('beats it with a group the plan wanted to play anyway' if after.turns < base.turns
                               else 'beats it, but splits a bigger group')
            if spends_wild:
                why.append('it spends %s, which is costly' % ('both wild cards' if wilds > 1 else 'a wild card'))
            candidates.append(Candidate(move, '; '.join(why), cost, after))
        return sorted(candidates, key=lambda c: c.cost)

    def advise(self, observation: Observation, legal_moves: List[Optional[Combo]]) -> Candidate:
        return self.rank(observation, legal_moves)[0]

    def choose_return(self, hand: List[Card], level: Rank, options: List[Card]) -> Card:
        return choose_return(hand, level)

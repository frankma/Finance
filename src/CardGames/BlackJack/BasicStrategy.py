from typing import Dict, List

from src.CardGames.BlackJack.Hand import card_value
from src.CardGames.BlackJack.Round import Action, Observation
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.Core.Player import Strategy

__author__ = 'frank.ma'

# total-dependent basic strategy for a multi-deck shoe, one column per dealer up card 2, 3, ..., 10, A
#   H hit, S stand, D double else hit, B double else stand, R surrender else hit, Q surrender else stand
# pair rows say only whether to split (P) or play the pair as a total (-)

_HARD_S17 = {
    9: 'HDDDDHHHHH',
    10: 'DDDDDDDDHH',
    11: 'DDDDDDDDDH',
    12: 'HHSSSHHHHH',
    13: 'SSSSSHHHHH',
    14: 'SSSSSHHHHH',
    15: 'SSSSSHHHRH',
    16: 'SSSSSHHRRR',
}
_SOFT_S17 = {
    13: 'HHHDDHHHHH',
    14: 'HHHDDHHHHH',
    15: 'HHDDDHHHHH',
    16: 'HHDDDHHHHH',
    17: 'HDDDDHHHHH',
    18: 'SBBBBSSHHH',
}
_PAIRS_DAS = {
    2: 'PPPPPP----',
    3: 'PPPPPP----',
    4: '---PP-----',
    6: 'PPPPP-----',
    7: 'PPPPPP----',
    8: 'PPPPPPPPPP',
    9: 'PPPPP-PP--',
    1: 'PPPPPPPPPP',  # aces
}
# without double after split the small pairs split less often
_PAIRS_NO_DAS = {**_PAIRS_DAS, 2: '--PPPP----', 3: '--PPPP----', 4: '----------', 6: '-PPPP-----'}
# a dealer hitting soft 17 is weaker against doubles and makes some stiff hands worth surrendering
_HARD_H17 = {**_HARD_S17, 11: 'DDDDDDDDDD', 15: 'SSSSSHHHRR', 17: 'SSSSSSSSSQ'}
_SOFT_H17 = {**_SOFT_S17, 18: 'BBBBBSSHHH', 19: 'SSSSBSSSSS'}


def _column(dealer_value: int) -> int:
    return 9 if dealer_value == 1 else dealer_value - 2


class BasicStrategy(Strategy):
    """
    the standard total-dependent strategy for the given rules; it ignores the count and the composition of the hand
    """

    def __init__(self, rules: Rules):
        self.rules = rules
        self.hard = _HARD_H17 if rules.dealer_hits_soft_17 else _HARD_S17  # type: Dict[int, str]
        self.soft = _SOFT_H17 if rules.dealer_hits_soft_17 else _SOFT_S17  # type: Dict[int, str]
        self.pairs = _PAIRS_DAS if rules.double_after_split else _PAIRS_NO_DAS  # type: Dict[int, str]

    def code(self, observation: Observation) -> str:
        """
        the table entry for the hand, before falling back on what is legal
        """
        hand = observation.hand
        if hand.is_pair:
            row = self.pairs.get(card_value(hand.cards[0]))
            if row is not None and row[_column(card_value(observation.dealer_up))] == 'P':
                return 'P'
        return self._total_code(observation)

    def decide(self, observation: Observation, legal_moves: List[Action]) -> Action:
        code = self.code(observation)
        if code == 'P' and Action.SPLIT in legal_moves:
            return Action.SPLIT
        if code == 'P':
            # splitting is no longer allowed, so play the pair as a total
            code = self._total_code(observation)
        if code in ('R', 'Q') and Action.SURRENDER in legal_moves:
            return Action.SURRENDER
        if code in ('D', 'B') and Action.DOUBLE in legal_moves:
            return Action.DOUBLE
        move = Action.STAND if code in ('S', 'B', 'Q') else Action.HIT
        return move if move in legal_moves else Action.STAND

    def _total_code(self, observation: Observation) -> str:
        # the hard or soft table entry, ignoring whether the hand is a pair
        hand = observation.hand
        column = _column(card_value(observation.dealer_up))
        table = self.soft if hand.is_soft else self.hard
        row = table.get(hand.total)
        if row is not None:
            return row[column]
        # off the tables: small hard totals and soft 12 (a pair of aces kept together) hit, the rest stand
        return 'H' if hand.total <= (12 if hand.is_soft else 8) else 'S'


class MimicDealerStrategy(Strategy):
    """
    baseline that plays like the dealer: hit below 17, never double, split or surrender
    """

    def decide(self, observation: Observation, legal_moves: List[Action]) -> Action:
        return Action.HIT if observation.hand.total < 17 and Action.HIT in legal_moves else Action.STAND

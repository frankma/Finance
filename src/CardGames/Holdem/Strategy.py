import random
from dataclasses import dataclass
from typing import List, Optional

from src.CardGames.Core.Player import Advice as CoreAdvice, Advisor, Strategy
from src.CardGames.Holdem.Equity import call_ev, pot_odds, simulate_equity
from src.CardGames.Holdem.Round import Action, Move, Observation

__author__ = 'frank.ma'


@dataclass(frozen=True)
class Advice(CoreAdvice):
    equity: float  # against random hands, from the seat's own cards and the board only
    pot_odds: float
    call_ev: float  # chips gained by calling over folding, 0 when there is nothing to call


class MonteCarloStrategy(Advisor):
    """
    equity against random hands, compared with the pot odds to call and with a value threshold to bet; the
    opponents' cards are dealt at random from the cards this seat has not seen, never read from the table

    the value threshold scales with the field: a fair share is 1 / players in the hand, and the seat bets once its
    equity clears that by 0.15 * aggression; it folds when equity is short of the pot odds by more than fold_slack
    """

    def __init__(self, n_sims: int = 500, fold_slack: float = 0.08, aggression: float = 1.0, bluff: float = 0.0,
                 rng: Optional[random.Random] = None):
        self.n_sims = n_sims
        self.fold_slack = fold_slack
        self.aggression = aggression
        self.bluff = bluff  # chance of betting without equity when checked to, heads-up or three-handed
        self.rng = rng if rng is not None else random.Random()

    def _raise_to(self, observation: Observation, fraction: float) -> int:
        # raise by a fraction of the pot after calling, in whole small blinds, within the legal range
        unit = observation.rules.small_blind
        to = observation.current_bet + fraction * (observation.pot + observation.to_call)
        to = int(round(to / unit)) * unit
        to = max(observation.min_raise_to, min(observation.max_raise_to, to))
        if to - observation.bet >= 0.8 * observation.stack:
            to = observation.max_raise_to  # most of the stack anyway, so move it all in
        return to

    def advise(self, observation: Observation, legal_moves: Optional[List[Move]] = None) -> Advice:
        o = observation
        equity = simulate_equity(list(o.hole), list(o.board), o.n_opponents, self.n_sims, o.rules, self.rng) \
            .equity.mean
        odds = pot_odds(o.pot, o.to_call)
        ev = call_ev(equity, o.pot, o.to_call) if o.to_call > 0 else 0.0
        fair = 1.0 / (o.n_opponents + 1)
        value_at = min(0.9, fair + 0.15 * self.aggression)
        strong_at = min(0.95, value_at + 0.15)
        can_raise = o.min_raise_to is not None
        size = 0.75 if equity > strong_at else 0.5

        def advice(move, reason):
            return Advice(move, reason, equity, odds, ev)

        if o.to_call == 0:
            if can_raise and equity > value_at:
                return advice(Move(Action.RAISE, self._raise_to(o, size)),
                              'equity %.0f%% is above the %.0f%% wanted to bet for value' % (100 * equity, 100 * value_at))
            if can_raise and o.board and o.n_opponents <= 2 and self.rng.random() < self.bluff:
                return advice(Move(Action.RAISE, self._raise_to(o, 0.5)),
                              'a bluff: equity is only %.0f%%, but a bet may take the pot now' % (100 * equity))
            return advice(Move(Action.CHECK),
                          'equity %.0f%% is below the %.0f%% wanted to bet, and checking is free'
                          % (100 * equity, 100 * value_at))
        if equity < odds - self.fold_slack:
            return advice(Move(Action.FOLD),
                          'equity %.0f%% is short of the %.0f%% the pot odds ask for' % (100 * equity, 100 * odds))
        if can_raise and equity > strong_at:
            return advice(Move(Action.RAISE, self._raise_to(o, size)),
                          'equity %.0f%% is well clear of the %.0f%% pot odds, so raise for value'
                          % (100 * equity, 100 * odds))
        return advice(Move(Action.CALL, o.bet + o.to_call),
                      'equity %.0f%% against %.0f%% pot odds, calling is worth %+.1f chips'
                      % (100 * equity, 100 * odds, ev))


class CallingStation(Strategy):
    # baseline that never folds and never raises
    def decide(self, observation: Observation, legal_moves: List[Move]) -> Move:
        for move in legal_moves:
            if move.action in (Action.CHECK, Action.CALL):
                return move
        return legal_moves[0]

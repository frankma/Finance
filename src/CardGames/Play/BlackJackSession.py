import dataclasses
import random
from typing import List, Optional

from src.CardGames.BlackJack.BasicStrategy import BasicStrategy
from src.CardGames.BlackJack.Counting import KellyBetSizer
from src.CardGames.BlackJack.Hand import Hand
from src.CardGames.BlackJack.Round import Action, Round
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.BlackJack.Table import Table
from src.CardGames.Play.Session import Session

__author__ = 'frank.ma'

PRESETS = {'s17': Rules(), 'h17': Rules(dealer_hits_soft_17=True, surrender=True)}


def describe(rules: Rules) -> str:
    return '%i deck%s, dealer %s soft 17%s' % (rules.n_decks, '' if rules.n_decks == 1 else 's',
                                              'hits' if rules.dealer_hits_soft_17 else 'stands on',
                                              ', late surrender' if rules.surrender else '')


def _outcome(hand: Hand, dealer: Hand, payoff: float) -> str:
    if hand.surrendered:
        return 'surrendered'
    if hand.is_bust:
        return 'bust'
    if dealer.is_blackjack and not hand.is_blackjack:
        return 'dealer blackjack'
    if hand.is_blackjack and payoff > 0:
        return 'blackjack'
    return 'win' if payoff > 0 else 'lose' if payoff < 0 else 'push'


class BlackJackSession(Session):
    """
    you against the dealer, round after round from one shoe, with a bankroll, a hi-lo count of the cards shown and a
    coach: basic strategy for each hand, Kelly sizing on the true count for each bet
    """

    def __init__(self, options: Optional[dict] = None, rng: Optional[random.Random] = None):
        options = options or {}
        preset = options.get('rules', 's17')
        if preset not in PRESETS:
            raise ValueError('unknown rules %r, expect one of %s.' % (preset, sorted(PRESETS)))
        rules = PRESETS[preset]
        n_decks = int(options.get('n_decks', rules.n_decks))
        if not 1 <= n_decks <= 8:
            raise ValueError('n_decks %i should be between 1 and 8.' % n_decks)
        self.rules = dataclasses.replace(rules, n_decks=n_decks)
        self.min_bet, self.max_bet = 5, 500
        self.bankroll = float(options.get('bankroll', 1000))
        self.bought = self.bankroll  # what you sat down with
        self.table = Table(self.rules, rng)
        self.coach = BasicStrategy(self.rules)
        self.sizer = KellyBetSizer(self.min_bet, self.max_bet)
        self.round = None  # type: Optional[Round]
        self.last_bet = 10
        self.payoff = 0.0
        self.outcomes = []  # type: List[str]
        self.rounds = 0
        self.followed = 0  # decisions where you took the coach's advice
        self.decisions = 0
        self.log = []  # type: List[str]

    # ---- state ----

    @property
    def phase(self) -> str:
        if self.round is None:
            return 'bet'
        return 'done' if self.round.is_terminal() else 'play'

    def _committed(self) -> float:
        return sum(hand.bet for hand in self.round.hands) if self.round is not None else 0.0

    def legal(self) -> List[Action]:
        """
        the round's legal moves, less a double or split your bankroll cannot cover
        """
        if self.phase != 'play':
            return []
        moves = self.round.legal_moves(0)
        spare = self.bankroll - self._committed()
        stake = self.round.current_hand.bet
        return [m for m in moves if m not in (Action.DOUBLE, Action.SPLIT) or spare >= stake]

    # ---- actions ----

    def act(self, action: dict) -> None:
        kind = action.get('type')
        if kind == 'deal':
            self._deal(int(action.get('bet', self.last_bet)))
        elif kind == 'move':
            self._move(action.get('move'))
        else:
            raise ValueError('unknown action %r.' % kind)

    def _deal(self, bet: int) -> None:
        if self.phase == 'play':
            raise ValueError('finish the round in play before dealing again.')
        if not self.min_bet <= bet <= self.max_bet:
            raise ValueError('bet %i should be between %i and %i.' % (bet, self.min_bet, self.max_bet))
        if bet > self.bankroll:
            raise ValueError('bet %i is more than the bankroll of %.0f.' % (bet, self.bankroll))
        if self.table.shoe.needs_shuffle:
            self.table.shoe.reshuffle()
            self.log.append('The cut card came out: the shoe is reshuffled and the count starts again.')
        self.last_bet = bet
        self.round = Round(self.table.shoe, self.rules, bet, self.table.counter)
        self.log.append('Round %i: bet %i.' % (self.rounds + 1, bet))
        if self.round.is_terminal():
            self._settle()

    def _move(self, name: Optional[str]) -> None:
        try:
            move = Action(name)
        except ValueError:
            raise ValueError('unknown move %r.' % name)
        if move not in self.legal():
            raise ValueError('%s is not allowed now.' % move.value)
        advised = self.coach.advise(self.round.observation(0), self.legal()).move
        self.decisions += 1
        self.followed += move == advised
        self.round.apply(0, move)
        if self.round.is_terminal():
            self._settle()

    def _settle(self) -> None:
        self.payoff = self.round.payoffs()[0]
        self.bankroll += self.payoff
        self.rounds += 1
        dealer = self.round.dealer_hand
        self.outcomes = [_outcome(hand, dealer, self.round.hand_payoff(k)) for k, hand in enumerate(self.round.hands)]
        self.log.append('Dealer %s; %s: %+.0f.' % (
            'busts' if dealer.is_bust else 'has blackjack' if dealer.is_blackjack else 'makes %i' % dealer.total,
            ', '.join(self.outcomes), self.payoff))

    # ---- what you see ----

    def _count(self, observation=None) -> dict:
        # from your observation in a round, else from the counter, which has only seen face-up cards
        counter = self.table.counter
        if observation is not None:
            running, true = observation.running_count, observation.true_count
            n_unseen = sum(n for _, n in observation.unseen)
        else:
            running, true, n_unseen = counter.running_count, counter.true_count(), counter.n_unseen
        return {'running': running, 'true': true, 'decks_left': n_unseen / 52.0,
                'edge': self.sizer.edge(true)}

    def view(self) -> dict:
        shoe = self.table.shoe
        out = {
            'game': 'blackjack', 'phase': self.phase, 'rules': describe(self.rules),
            'bankroll': self.bankroll, 'min_bet': self.min_bet,
            'max_bet': min(self.max_bet, int(self.bankroll)), 'last_bet': self.last_bet,
            'shoe': {'cards_left': shoe.cards_remaining, 'n_cards': shoe.n_cards, 'reshuffle_next': shoe.needs_shuffle},
            'stats': {'rounds': self.rounds, 'net': self.bankroll - self.bought,
                      'followed': self.followed, 'decisions': self.decisions},
            'log': self.log[-12:], 'legal': [m.value for m in self.legal()],
            'broke': self.phase != 'play' and self.bankroll < self.min_bet,
        }
        if self.round is None:
            out.update(hands=[], active=None, dealer=None, count=self._count())
            return out
        observation = self.round.observation(0)
        hands = []
        for k, (cards, bet) in enumerate(zip(observation.hands, observation.bets)):
            hand = Hand(list(cards))
            hands.append({'cards': [str(c) for c in cards], 'total': hand.total, 'soft': hand.is_soft, 'bet': bet,
                          'outcome': self.outcomes[k] if self.phase == 'done' else None})
        if self.phase == 'done':
            # the round is over and the hole card has been turned over: the dealer's whole hand is public
            dealer = self.round.dealer_hand
            out['dealer'] = {'cards': [str(c) for c in dealer.cards], 'total': dealer.total, 'hidden': False}
            out['payoff'] = self.payoff
        else:
            out['dealer'] = {'cards': [str(observation.dealer_up)], 'total': None, 'hidden': True}
        out.update(hands=hands, active=observation.hand_index if self.phase == 'play' else None,
                   count=self._count(observation))
        return out

    def advice(self) -> dict:
        if self.phase == 'play':
            advice = self.coach.advise(self.round.observation(0), self.legal())
            return {'kind': 'move', 'move': advice.move.value, 'reason': advice.reason}
        true = self.table.counter.true_count()
        edge = self.sizer.edge(true)
        stake = self.sizer.bet(self.bankroll, true)
        bet = int(min(self.bankroll, max(self.min_bet, round(stake / self.min_bet) * self.min_bet)))
        if edge <= 0:
            reason = ('true count %+.1f: the estimated edge is %+.2f%%, against you, so Kelly bets nothing and you '
                      'bet the table minimum of %i' % (true, 100 * edge, self.min_bet))
        else:
            reason = ('true count %+.1f: the estimated edge is %+.2f%%; Kelly stakes edge / variance = %.2f%% of the '
                      'bankroll of %.0f, about %i' % (true, 100 * edge, 100 * edge / self.sizer.variance,
                                                      self.bankroll, bet))
        return {'kind': 'bet', 'bet': bet, 'reason': reason}

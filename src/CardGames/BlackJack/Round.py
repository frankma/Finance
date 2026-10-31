from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from src.CardGames.BlackJack.Dealer import Dealer
from src.CardGames.BlackJack.Hand import Hand, card_value
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.Core.Card import Card, Rank
from src.CardGames.Core.CardCounter import CardCounter
from src.CardGames.Core.Game import Game
from src.CardGames.Core.Shoe import Shoe

__author__ = 'frank.ma'


class Action(Enum):
    STAND = 'stand'
    HIT = 'hit'
    DOUBLE = 'double'
    SPLIT = 'split'
    SURRENDER = 'surrender'


@dataclass(frozen=True)
class Observation(object):
    """
    what the player knows: the hand being played, every hand it holds after splits, the dealer's up card (never the
    hole card), the rules, and, when counting, the count over every card shown since the shuffle
    """
    cards: Tuple[Card, ...]  # the hand being played
    from_split: bool
    dealer_up: Card
    hand_index: int
    hands: Tuple[Tuple[Card, ...], ...]
    rules: Rules
    running_count: Optional[float] = None  # None when nobody counts
    true_count: Optional[float] = None
    unseen: Optional[Tuple[Tuple[Rank, int], ...]] = None  # cards not yet shown, by rank

    @property
    def hand(self) -> Hand:
        # a fresh copy to evaluate; changing it cannot touch the game
        return Hand(list(self.cards), from_split=self.from_split)

    @property
    def n_hands(self) -> int:
        return len(self.hands)


class Round(Game):
    """
    one round of a single player against the dealer, dealt from the shoe as soon as it is created

    seat 0 is the player; the dealer follows fixed rules and never chooses, so it is not a seat
    """

    def __init__(self, shoe: Shoe, rules: Rules, bet: float = 1.0, counter: Optional[CardCounter] = None):
        if bet <= 0.0:
            raise ValueError('bet %r should be strictly positive.' % bet)
        self.shoe = shoe
        self.rules = rules
        self.dealer = Dealer(rules)
        self.counter = counter
        self.bet = bet
        self.hands = [Hand(bet=bet)]  # type: List[Hand]
        self.dealer_hand = Hand()
        self.active = 0
        self.finished = False

        player = self.hands[0]
        player.add(shoe.draw())
        self.dealer_hand.add(shoe.draw())
        player.add(shoe.draw())
        self.hole = shoe.draw(face_up=False)
        self.dealer_hand.add(self.hole)

        if self._dealer_shows_blackjack() or player.is_blackjack:
            self._finish(dealer_draws=False)

    @property
    def dealer_up(self) -> Card:
        return self.dealer_hand.cards[0]

    def _dealer_shows_blackjack(self) -> bool:
        # the dealer peeks only under an ace or a ten-value card
        if card_value(self.dealer_up) not in (1, 10):
            return False
        return self.dealer_hand.is_blackjack

    @property
    def n_seats(self) -> int:
        return 1

    def current_seat(self) -> Optional[int]:
        return None if self.finished else 0

    @property
    def current_hand(self) -> Hand:
        return self.hands[self.active]

    def legal_moves(self, seat: int = 0) -> List[Action]:
        if self.finished:
            return []
        hand = self.current_hand
        split_ace = hand.from_split and hand.cards[0].rank == Rank.ACE
        can_split = hand.is_pair and len(self.hands) < self.rules.max_hands and \
            not (split_ace and not self.rules.resplit_aces)
        if split_ace and not self.rules.hit_split_aces:
            # only a split ace that paired again is still live here, and it may only stand or resplit
            return [Action.STAND, Action.SPLIT] if can_split else [Action.STAND]
        moves = [Action.STAND, Action.HIT]
        if len(hand.cards) == 2 and (not hand.from_split or self.rules.double_after_split):
            moves.append(Action.DOUBLE)
        if can_split:
            moves.append(Action.SPLIT)
        if self.rules.surrender and len(self.hands) == 1 and len(hand.cards) == 2:
            moves.append(Action.SURRENDER)
        return moves

    def apply(self, seat: int, move: Action) -> None:
        if move not in self.legal_moves(seat):
            raise ValueError('%r is not a legal move now.' % move)
        hand = self.current_hand
        if move == Action.STAND:
            hand.stood = True
        elif move == Action.HIT:
            hand.add(self.shoe.draw())
        elif move == Action.DOUBLE:
            hand.bet *= 2.0
            hand.add(self.shoe.draw())
            hand.doubled = True
        elif move == Action.SPLIT:
            self._split()
        elif move == Action.SURRENDER:
            hand.surrendered = True
        self._advance()

    def _split(self) -> None:
        hand = self.current_hand
        aces = hand.cards[0].rank == Rank.ACE
        second = Hand([hand.cards.pop()], bet=hand.bet, from_split=True)
        hand.from_split = True
        self.hands.insert(self.active + 1, second)
        for new_hand in (hand, second):
            new_hand.add(self.shoe.draw())
            if aces and not self.rules.hit_split_aces and not new_hand.is_pair:
                new_hand.stood = True  # one card to each split ace, unless it pairs again and may be resplit
        if aces and not self.rules.hit_split_aces and not self.rules.resplit_aces:
            hand.stood = second.stood = True

    def _advance(self) -> None:
        while self.active < len(self.hands) and self.hands[self.active].is_done:
            self.active += 1
        if self.active == len(self.hands):
            self.active = len(self.hands) - 1
            live = any(not (hand.is_bust or hand.surrendered) for hand in self.hands)
            self._finish(dealer_draws=live)

    def _finish(self, dealer_draws: bool) -> None:
        self.shoe.reveal(self.hole)
        if dealer_draws:
            self.dealer.play(self.dealer_hand, self.shoe)
        self.finished = True

    def observation(self, seat: int = 0) -> Observation:
        hand, counter = self.current_hand, self.counter
        counts = {}
        if counter is not None:
            counts = dict(running_count=counter.running_count, true_count=counter.true_count(),
                          unseen=tuple(sorted(counter.unseen_by_rank().items())))
        return Observation(tuple(hand.cards), hand.from_split, self.dealer_up, self.active,
                           tuple(tuple(h.cards) for h in self.hands), self.rules, **counts)

    def payoffs(self) -> List[float]:
        if not self.finished:
            raise ValueError('the round is not over yet.')
        return [sum(self._settle(hand) for hand in self.hands)]

    def _settle(self, hand: Hand) -> float:
        dealer = self.dealer_hand
        if dealer.is_blackjack:
            return 0.0 if hand.is_blackjack else -hand.bet
        if hand.is_blackjack:
            return self.rules.blackjack_pays * hand.bet
        if hand.surrendered:
            return -0.5 * hand.bet
        if hand.is_bust:
            return -hand.bet
        if dealer.is_bust or hand.total > dealer.total:
            return hand.bet
        if hand.total < dealer.total:
            return -hand.bet
        return 0.0

    def __str__(self):
        hands = ', '.join(str(hand) for hand in self.hands)
        dealer = str(self.dealer_hand) if self.finished else '%s ??' % self.dealer_up
        return 'player: %s; dealer: %s' % (hands, dealer)

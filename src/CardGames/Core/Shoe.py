import random
from typing import List, Optional

from src.CardGames.Core.Card import Card, standard_deck

__author__ = 'frank.ma'


class Shoe(object):
    """
    one or more decks dealt from the top, tracking which dealt cards are face up and which are face down

    watchers (e.g. a CardCounter) are told about every card as it becomes visible and about every reshuffle, so they
    see exactly what a player at the table sees
    """

    def __init__(self, n_decks: int = 1, jokers: bool = False, penetration: float = 1.0,
                 rng: Optional[random.Random] = None, cards: Optional[List[Card]] = None):
        """
        :param n_decks: number of standard decks, ignored when cards is given
        :param jokers: include two jokers per deck, ignored when cards is given
        :param penetration: fraction of the shoe dealt before needs_shuffle turns true, in (0, 1]
        :param rng: random source, pass a seeded one for reproducible deals
        :param cards: explicit composition, e.g. a shoe with some ranks removed
        """
        if not 0.0 < penetration <= 1.0:
            raise ValueError('penetration %r should be in (0, 1].' % penetration)
        self.composition = list(cards) if cards is not None \
            else [card for _ in range(n_decks) for card in standard_deck(jokers)]  # type: List[Card]
        if not self.composition:
            raise ValueError('a shoe needs at least one card.')
        self.penetration = penetration
        self.rng = rng if rng is not None else random.Random()
        self.watchers = []  # type: list
        self.cards = []  # type: List[Card]
        self.face_up = []  # type: List[Card]
        self.face_down = []  # type: List[Card]
        self.reshuffle()

    def watch(self, watcher) -> None:
        """
        register an object with see(card) and reset() methods
        """
        self.watchers.append(watcher)

    def reshuffle(self) -> None:
        # gather every card back, the dealt ones included, and shuffle the full composition
        self.cards = list(self.composition)
        self.rng.shuffle(self.cards)
        self.face_up = []
        self.face_down = []
        for watcher in self.watchers:
            watcher.reset()

    @property
    def n_cards(self) -> int:
        return len(self.composition)

    @property
    def cards_remaining(self) -> int:
        return len(self.cards)

    @property
    def needs_shuffle(self) -> bool:
        return self.n_cards - self.cards_remaining >= self.penetration * self.n_cards

    def draw(self, face_up: bool = True) -> Card:
        if not self.cards:
            raise ValueError('ran out of cards.')
        card = self.cards.pop()
        if face_up:
            self._show(card)
        else:
            self.face_down.append(card)
        return card

    def draw_many(self, n: int, face_up: bool = True) -> List[Card]:
        if n > len(self.cards):
            raise ValueError('cannot draw %i cards, only %i remain.' % (n, len(self.cards)))
        return [self.draw(face_up) for _ in range(n)]

    def reveal(self, card: Card) -> None:
        # turn a face-down card over
        if card not in self.face_down:
            raise ValueError('%s is not among the face-down cards.' % card)
        self.face_down.remove(card)
        self._show(card)

    def _show(self, card: Card) -> None:
        self.face_up.append(card)
        for watcher in self.watchers:
            watcher.see(card)

    def __str__(self):
        return 'Shoe of %i cards: %i remaining, %i face up, %i face down' % \
               (self.n_cards, self.cards_remaining, len(self.face_up), len(self.face_down))

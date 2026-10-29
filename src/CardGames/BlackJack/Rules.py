from dataclasses import dataclass

__author__ = 'frank.ma'


@dataclass(frozen=True)
class Rules(object):
    """
    table rules; the defaults are a common six-deck shoe game: dealer stands on soft 17, blackjack pays 3:2, double on
    any two cards and after splits, split up to four hands, one card to each split ace, no surrender

    the dealer always peeks for blackjack under an ace or a ten, so a dealer blackjack only takes the original bet;
    insurance is not offered since basic strategy never takes it
    """
    n_decks: int = 6
    dealer_hits_soft_17: bool = False
    blackjack_pays: float = 1.5
    double_after_split: bool = True
    max_hands: int = 4  # hands a player may hold after splitting
    resplit_aces: bool = False
    hit_split_aces: bool = False
    surrender: bool = False  # late surrender, after the dealer peeks
    penetration: float = 0.75  # fraction of the shoe dealt before reshuffling

    def __post_init__(self):
        if self.n_decks < 1:
            raise ValueError('need at least one deck, received %i.' % self.n_decks)
        if self.max_hands < 1:
            raise ValueError('max_hands %i should be at least 1.' % self.max_hands)

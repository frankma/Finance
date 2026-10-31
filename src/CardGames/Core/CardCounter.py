from collections import Counter
from typing import Dict, List, Optional

from src.CardGames.Core.Card import Card, Rank

__author__ = 'frank.ma'


class CardCounter(object):
    """
    what one seat has seen out of a known composition: the unseen cards, and a running count under an optional
    tagging system (rank -> weight, e.g. hi-lo)
    """

    def __init__(self, composition: List[Card], tags: Optional[Dict[Rank, float]] = None):
        self.composition = Counter(composition)
        self.tags = dict(tags) if tags is not None else {}
        self._n_cards = sum(self.composition.values())
        self._by_rank = Counter()  # type: Counter
        for card, n in self.composition.items():
            self._by_rank[card.rank] += n
        self.reset()

    def reset(self) -> None:
        self.seen = Counter()  # type: Counter
        self._seen_by_rank = Counter()  # type: Counter
        self._n_seen = 0
        self.running_count = 0.0

    def see(self, card: Card) -> None:
        if self.seen[card] >= self.composition[card]:
            raise ValueError('%s seen more often than the composition holds it.' % card)
        self.seen[card] += 1
        # tallies kept as cards are seen, so a per-decision snapshot stays cheap
        self._seen_by_rank[card.rank] += 1
        self._n_seen += 1
        self.running_count += self.tags.get(card.rank, 0.0)

    def see_many(self, cards: List[Card]) -> None:
        for card in cards:
            self.see(card)

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def n_unseen(self) -> int:
        return self._n_cards - self._n_seen

    def unseen(self) -> List[Card]:
        return list((self.composition - self.seen).elements())

    def unseen_by_rank(self) -> Dict[Rank, int]:
        return {rank: n - self._seen_by_rank[rank] for rank, n in self._by_rank.items()
                if n > self._seen_by_rank[rank]}

    def true_count(self, cards_per_deck: int = 52) -> float:
        """
        running count per deck of unseen cards, the quantity a count-based bet or play keys off
        """
        decks_unseen = self.n_unseen / cards_per_deck
        return self.running_count / decks_unseen if decks_unseen > 0.0 else 0.0

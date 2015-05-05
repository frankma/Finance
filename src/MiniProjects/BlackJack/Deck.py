import numpy as np
from src.MiniProjects.BlackJack.Card import Card, Rank, Suit
from random import shuffle

__author__ = 'frank.ma'


class Deck(object):

    def __init__(self, cards: list):
        for card in cards:
            assert isinstance(card, Card), 'unrecognized card entry, %s' % card.__str__()
        self.cards = np.array(cards)
        self.indices = list(range(cards.__len__()))  # anchor the cards but shuffle index of them
        self.public = []  # nothing is public at initiation
        self.nonpublic = []  # nothing is non-public at initiation
        self.cur_idx = 0  # track realized cards from indices
        self.shuffle()

    def shuffle(self):
        # this method should only shuffle live cards
        realized = self.indices[:self.cur_idx]
        unrealized = self.indices[self.cur_idx:]
        shuffle(unrealized)
        self.indices = realized + unrealized
        pass

    def draw(self, public=True):
        assert self.cur_idx < self.cards.__len__(), 'ran out of cards.'
        idx = self.indices[self.cur_idx]
        card = self.cards[idx]
        if public:
            self.public.append(idx)
        else:
            self.nonpublic.append(idx)
        self.cur_idx += 1
        return card

    def reveal(self):
        # this always assume one non-public to be revealed
        assert self.nonpublic.__len__() > 0, 'no non-public drawn is available.'
        self.public.append(self.nonpublic.pop())

    def view_public(self):
        return self.cards[self.public]

    @staticmethod
    def new_deck(n_sets=4):
        """
        create a new none-shuffled deck with given set of cards
        :param n_sets: number of sets of cards expected to provide
        :return: a deck of none-shuffled cards
        """
        card_set = [Card(rank, suit) for _ in range(n_sets) for rank in Rank for suit in Suit]
        return Deck(card_set)

    def __str__(self):
        str_builder = ''
        str_builder += 'All cards:\n'
        all_cards = ', '.join([card.__str__() for card in self.cards])
        str_builder += all_cards + '\n'
        str_builder += 'Used public:\n'
        public_cards = ''
        # todo: add public used cards
        return str_builder